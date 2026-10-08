from pgvector import HalfVector
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from agent_harness.jev_client import build_relation_edges
from agent_harness.storage.connection import connect_db


CONSOLIDATION_THRESHOLD = 0.85


def choose_consolidation_action(
    decision: dict,
    threshold: float = CONSOLIDATION_THRESHOLD,
) -> str:
    if decision["contradiction_score"] >= threshold:
        return "CONTRADICTION_LINKED"

    current_supersedes = (
        decision["current_supersedes_candidate_score"]
        >= threshold
    )
    candidate_supersedes = (
        decision["candidate_supersedes_current_score"]
        >= threshold
    )

    if current_supersedes and candidate_supersedes:
        return "DEFERRED"

    if current_supersedes or candidate_supersedes:
        return "SUPERSEDED"

    if decision["redundancy_score"] >= threshold:
        return "CANONICALIZED"

    if (
        decision["representation"] == "merge"
        and decision["representation_probability"] >= threshold
    ):
        return "MERGED"

    return "NONE"


def _load_locked_decision(conn, decision_id: int) -> dict:
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            """
            SELECT *
            FROM memory_consolidation_decisions
            WHERE id = %s
            FOR UPDATE
            """,
            (decision_id,),
        )
        decision = cursor.fetchone()

    if decision is None:
        raise ValueError(
            f"Consolidation decision {decision_id} does not exist"
        )

    return decision


def _load_locked_pair_memories(
    conn,
    decision: dict,
) -> dict[int, dict]:
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            """
            SELECT
                id,
                text,
                entities,
                embedding,
                status,
                replaced_by_memory_id,
                created_at
            FROM memories
            WHERE id = ANY(%s)
            ORDER BY id
            FOR UPDATE
            """,
            ([
                decision["memory_a_id"],
                decision["memory_b_id"],
            ],),
        )
        rows = cursor.fetchall()

    if len(rows) != 2:
        raise ValueError(
            f"Decision {decision['id']} does not reference two memories"
        )

    return {row["id"]: row for row in rows}


def _mark_decision_applied(
    conn,
    decision_id: int,
    action: str,
    result_memory_id: int | None = None,
) -> None:
    conn.execute(
        """
        UPDATE memory_consolidation_decisions
        SET applied_action = %s,
            result_memory_id = %s,
            applied_at = now(),
            action_error = NULL
        WHERE id = %s
        """,
        (action, result_memory_id, decision_id),
    )


def _save_lifecycle_edge(
    conn,
    *,
    source_memory_id: int,
    target_memory_id: int,
    relation_type: str,
    score: float,
    decision_id: int,
) -> None:
    if source_memory_id == target_memory_id:
        return

    if relation_type == "CONTRADICTS":
        source_memory_id, target_memory_id = sorted(
            (source_memory_id, target_memory_id)
        )

    conn.execute(
        """
        INSERT INTO memory_lifecycle_edges (
            source_memory_id,
            target_memory_id,
            relation_type,
            score,
            decision_id
        )
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (
            source_memory_id,
            target_memory_id,
            relation_type
        )
        DO UPDATE SET
            decision_id = CASE
                WHEN EXCLUDED.score > memory_lifecycle_edges.score
                    THEN EXCLUDED.decision_id
                ELSE memory_lifecycle_edges.decision_id
            END,
            score = GREATEST(
                memory_lifecycle_edges.score,
                EXCLUDED.score
            )
        """,
        (
            source_memory_id,
            target_memory_id,
            relation_type,
            score,
            decision_id,
        ),
    )


def _save_knowledge_edge(
    conn,
    *,
    source_memory_id: int,
    target_memory_id: int,
    relation_type: str,
    score: float,
) -> None:
    if source_memory_id == target_memory_id:
        return

    if relation_type != "CAUSES":
        source_memory_id, target_memory_id = sorted(
            (source_memory_id, target_memory_id)
        )

    conn.execute(
        """
        INSERT INTO memory_edges (
            source_memory_id,
            target_memory_id,
            relation_type,
            score
        )
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (
            source_memory_id,
            target_memory_id,
            relation_type
        )
        DO UPDATE SET
            score = GREATEST(
                memory_edges.score,
                EXCLUDED.score
            )
        """,
        (
            source_memory_id,
            target_memory_id,
            relation_type,
            score,
        ),
    )


def _resolve_active_memory_id(
    conn,
    memory_id: int,
) -> int:
    visited = set()
    current_id = memory_id

    while True:
        if current_id in visited:
            raise ValueError(
                f"Replacement cycle detected at memory {current_id}"
            )

        visited.add(current_id)
        row = conn.execute(
            """
            SELECT status, replaced_by_memory_id
            FROM memories
            WHERE id = %s
            """,
            (current_id,),
        ).fetchone()

        if row is None:
            raise ValueError(f"Memory {current_id} does not exist")

        status, replacement_id = row

        if status == "ACTIVE":
            if replacement_id is not None:
                raise ValueError(
                    f"Active memory {current_id} has a replacement"
                )
            return current_id

        if replacement_id is None:
            raise ValueError(
                f"Inactive memory {current_id} has no replacement"
            )

        current_id = replacement_id


def _migrate_knowledge_edges(
    conn,
    inactive_memory_id: int,
    replacement_memory_id: int,
) -> None:
    rows = conn.execute(
        """
        SELECT
            source_memory_id,
            target_memory_id,
            relation_type,
            score
        FROM memory_edges
        WHERE source_memory_id = %s
           OR target_memory_id = %s
        ORDER BY id
        """,
        (inactive_memory_id, inactive_memory_id),
    ).fetchall()

    for source_id, target_id, relation_type, score in rows:
        if source_id == inactive_memory_id:
            source_id = replacement_memory_id
            target_id = _resolve_active_memory_id(conn, target_id)
        else:
            source_id = _resolve_active_memory_id(conn, source_id)
            target_id = replacement_memory_id

        _save_knowledge_edge(
            conn,
            source_memory_id=source_id,
            target_memory_id=target_id,
            relation_type=relation_type,
            score=score,
        )


def _rewire_contradictions(
    conn,
    inactive_memory_id: int,
    replacement_memory_id: int,
) -> None:
    rows = conn.execute(
        """
        SELECT
            source_memory_id,
            target_memory_id,
            score,
            decision_id
        FROM memory_lifecycle_edges
        WHERE relation_type = 'CONTRADICTS'
          AND (
              source_memory_id = %s
              OR target_memory_id = %s
          )
        ORDER BY id
        """,
        (inactive_memory_id, inactive_memory_id),
    ).fetchall()

    for source_id, target_id, score, decision_id in rows:
        counterpart_id = (
            target_id
            if source_id == inactive_memory_id
            else source_id
        )
        counterpart_id = _resolve_active_memory_id(
            conn,
            counterpart_id,
        )

        _save_lifecycle_edge(
            conn,
            source_memory_id=replacement_memory_id,
            target_memory_id=counterpart_id,
            relation_type="CONTRADICTS",
            score=score,
            decision_id=decision_id,
        )


def _deactivate_memory(
    conn,
    *,
    inactive_memory_id: int,
    replacement_memory_id: int,
    status: str,
    relation_type: str,
    score: float,
    decision_id: int,
) -> None:
    replacement_id = _resolve_active_memory_id(
        conn,
        replacement_memory_id,
    )

    if inactive_memory_id == replacement_id:
        raise ValueError("A memory cannot replace itself")

    _migrate_knowledge_edges(
        conn,
        inactive_memory_id,
        replacement_id,
    )
    _rewire_contradictions(
        conn,
        inactive_memory_id,
        replacement_id,
    )

    conn.execute(
        """
        UPDATE memories
        SET replaced_by_memory_id = %s
        WHERE replaced_by_memory_id = %s
        """,
        (replacement_id, inactive_memory_id),
    )

    if relation_type == "SUPERSEDES":
        source_id = replacement_id
        target_id = inactive_memory_id
    else:
        source_id = inactive_memory_id
        target_id = replacement_id

    _save_lifecycle_edge(
        conn,
        source_memory_id=source_id,
        target_memory_id=target_id,
        relation_type=relation_type,
        score=score,
        decision_id=decision_id,
    )

    result = conn.execute(
        """
        UPDATE memories
        SET status = %s,
            replaced_by_memory_id = %s
        WHERE id = %s
          AND status = 'ACTIVE'
        """,
        (
            status,
            replacement_id,
            inactive_memory_id,
        ),
    )

    if result.rowcount != 1:
        raise ValueError(
            f"Memory {inactive_memory_id} is no longer active"
        )


def apply_non_merge_decision(decision_id: int) -> dict:
    with connect_db() as conn:
        decision = _load_locked_decision(conn, decision_id)

        if decision["applied_action"] != "PENDING":
            return {
                "decision_id": decision_id,
                "action": decision["applied_action"],
                "result_memory_id": decision["result_memory_id"],
            }

        action = choose_consolidation_action(decision)

        if action == "MERGED":
            raise ValueError(
                "A merge decision requires generated merge preparation"
            )

        memories = _load_locked_pair_memories(conn, decision)

        if any(
            memory["status"] != "ACTIVE"
            for memory in memories.values()
        ):
            action = "DEFERRED"

        result_memory_id = None

        if action == "CONTRADICTION_LINKED":
            _save_lifecycle_edge(
                conn,
                source_memory_id=decision["memory_a_id"],
                target_memory_id=decision["memory_b_id"],
                relation_type="CONTRADICTS",
                score=decision["contradiction_score"],
                decision_id=decision_id,
            )

        elif action == "SUPERSEDED":
            current_id = decision["current_memory_id"]
            candidate_id = decision["candidate_memory_id"]
            current_wins = (
                decision["current_supersedes_candidate_score"]
                >= CONSOLIDATION_THRESHOLD
            )

            if current_wins:
                replacement_id = current_id
                inactive_id = candidate_id
                score = decision[
                    "current_supersedes_candidate_score"
                ]
            else:
                replacement_id = candidate_id
                inactive_id = current_id
                score = decision[
                    "candidate_supersedes_current_score"
                ]

            _deactivate_memory(
                conn,
                inactive_memory_id=inactive_id,
                replacement_memory_id=replacement_id,
                status="SUPERSEDED",
                relation_type="SUPERSEDES",
                score=score,
                decision_id=decision_id,
            )
            result_memory_id = replacement_id

        elif action == "CANONICALIZED":
            ordered = sorted(
                memories.values(),
                key=lambda memory: (
                    memory["created_at"],
                    memory["id"],
                ),
            )
            canonical_id = ordered[0]["id"]
            duplicate_id = ordered[1]["id"]

            _deactivate_memory(
                conn,
                inactive_memory_id=duplicate_id,
                replacement_memory_id=canonical_id,
                status="REDUNDANT",
                relation_type="REDUNDANT_OF",
                score=decision["redundancy_score"],
                decision_id=decision_id,
            )
            result_memory_id = canonical_id

        _mark_decision_applied(
            conn,
            decision_id,
            action,
            result_memory_id,
        )

        return {
            "decision_id": decision_id,
            "action": action,
            "result_memory_id": result_memory_id,
        }


def _save_prepared_relation_edges(
    conn,
    new_memory_id: int,
    relation_decisions: list[dict],
) -> None:
    existing_ids = {
        decision["existing_memory_id"]
        for decision in relation_decisions
    }

    if existing_ids:
        active_ids = {
            row[0]
            for row in conn.execute(
                """
                SELECT id
                FROM memories
                WHERE id = ANY(%s)
                  AND status = 'ACTIVE'
                """,
                (sorted(existing_ids),),
            ).fetchall()
        }
    else:
        active_ids = set()

    current_decisions = [
        decision
        for decision in relation_decisions
        if decision["existing_memory_id"] in active_ids
    ]

    for edge in build_relation_edges(
        new_memory_id,
        current_decisions,
    ):
        _save_knowledge_edge(conn, **edge)


def apply_merge_decision(
    decision_id: int,
    prepared_merge: dict,
) -> dict:
    with connect_db() as conn:
        decision = _load_locked_decision(conn, decision_id)

        if decision["applied_action"] != "PENDING":
            return {
                "decision_id": decision_id,
                "action": decision["applied_action"],
                "result_memory_id": decision["result_memory_id"],
            }

        if choose_consolidation_action(decision) != "MERGED":
            raise ValueError(
                f"Decision {decision_id} no longer authorizes a merge"
            )

        memories = _load_locked_pair_memories(conn, decision)

        if any(
            memory["status"] != "ACTIVE"
            for memory in memories.values()
        ):
            _mark_decision_applied(
                conn,
                decision_id,
                "DEFERRED",
            )
            return {
                "decision_id": decision_id,
                "action": "DEFERRED",
                "result_memory_id": None,
            }

        row = conn.execute(
            """
            INSERT INTO memories (
                text,
                source_message_id,
                entities,
                embedding,
                origin_type,
                last_consolidated_version,
                last_consolidated_at
            )
            VALUES (
                %s,
                NULL,
                %s,
                %s,
                'MERGED',
                %s,
                now()
            )
            RETURNING id
            """,
            (
                prepared_merge["text"],
                Jsonb(prepared_merge["entities"]),
                HalfVector(prepared_merge["embedding"]),
                decision["decision_version"],
            ),
        ).fetchone()
        merged_memory_id = row[0]

        _save_prepared_relation_edges(
            conn,
            merged_memory_id,
            prepared_merge["relation_decisions"],
        )

        for source_memory_id in sorted(memories):
            _deactivate_memory(
                conn,
                inactive_memory_id=source_memory_id,
                replacement_memory_id=merged_memory_id,
                status="MERGED_SOURCE",
                relation_type="MERGED_INTO",
                score=decision["representation_probability"],
                decision_id=decision_id,
            )

        _mark_decision_applied(
            conn,
            decision_id,
            "MERGED",
            merged_memory_id,
        )

        return {
            "decision_id": decision_id,
            "action": "MERGED",
            "result_memory_id": merged_memory_id,
        }
