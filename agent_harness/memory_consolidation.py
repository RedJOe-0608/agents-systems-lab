import json

from pgvector import HalfVector
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from agent_harness.config import API_KEY, BASE_URL, MODEL_ID
from agent_harness.storage.connection import connect_db
from agent_harness.storage.consolidation import (
    count_pending_consolidation_seeds,
    create_consolidation_run,
    fail_consolidation_run,
    find_consolidation_candidates,
    finish_consolidation_run,
    load_consolidation_seeds,
    load_evaluated_pair_keys,
    load_pending_consolidation_decisions,
    mark_consolidation_seeds_processed,
    record_consolidation_run_seeds,
    save_consolidation_decisions,
)
from agent_harness.storage.memories import find_write_candidates
from agent_harness.embedding_client import embed_text
from agent_harness.jev_client import (
    build_relation_edges,
    evaluate_consolidation_pairs,
    evaluate_memory_relations,
)
from agent_harness.model_client import request_chat_completion


CONSOLIDATION_DECISION_VERSION = "v1"
CONSOLIDATION_MODEL_ID = "jev-latest"
CONSOLIDATION_THRESHOLD = 0.85
CONSOLIDATION_EVALUATION_BATCH_SIZE = 10

MERGE_SYSTEM_PROMPT = """
Combine the supplied memories into one self-contained durable fact.

Use only details explicitly supported by the supplied memories. Preserve every
compatible detail that will matter for future recall. Do not infer a broader
preference, habit, personality trait, or general pattern. Do not mention memory
IDs or the act of merging.

Return only valid JSON, with no Markdown, in this shape:
{"text": "One atomic merged memory."}
""".strip()


def discover_consolidation_pairs(
    seed_rows: list[tuple],
    candidate_limit: int,
    evaluated_pair_keys: set[tuple[int, int]] | None = None,
) -> list[tuple]:
    pairs = []
    seen_pair_keys = set(evaluated_pair_keys or set())

    for seed_row in seed_rows:
        candidates = find_consolidation_candidates(
            seed_row,
            limit=candidate_limit,
        )

        for candidate_row in candidates:
            pair_key = tuple(sorted(
                (seed_row[0], candidate_row[0])
            ))

            if pair_key in seen_pair_keys:
                continue

            seen_pair_keys.add(pair_key)
            pairs.append((seed_row, candidate_row))

    return pairs


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


def record_consolidation_decision_error(
    decision_id: int,
    error: Exception,
) -> None:
    with connect_db() as conn:
        conn.execute(
            """
            UPDATE memory_consolidation_decisions
            SET action_error = %s
            WHERE id = %s
              AND applied_action = 'PENDING'
            """,
            (str(error), decision_id),
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


def _load_decision_snapshot(decision_id: int) -> dict:
    with connect_db() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT *
                FROM memory_consolidation_decisions
                WHERE id = %s
                """,
                (decision_id,),
            )
            decision = cursor.fetchone()

    if decision is None:
        raise ValueError(
            f"Consolidation decision {decision_id} does not exist"
        )

    return decision


def _load_pair_memory_snapshots(decision: dict) -> list[dict]:
    with connect_db() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT
                    id,
                    text,
                    entities,
                    status,
                    created_at
                FROM memories
                WHERE id = ANY(%s)
                ORDER BY id
                """,
                ([
                    decision["memory_a_id"],
                    decision["memory_b_id"],
                ],),
            )
            memories = cursor.fetchall()

    if len(memories) != 2:
        raise ValueError(
            f"Decision {decision['id']} does not reference two memories"
        )

    return memories


def parse_merged_memory_text(raw_response: str) -> str:
    try:
        data = json.loads(raw_response)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError("Merge generator returned invalid JSON") from error

    if not isinstance(data, dict):
        raise ValueError("Merge generator response must be an object")

    text = data.get("text")

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Merged memory text must be non-empty")

    return text.strip()


def generate_merged_memory_text(memories: list[dict]) -> str:
    response = request_chat_completion(
        base_url=BASE_URL,
        api_key=API_KEY,
        model_id=MODEL_ID,
        messages=[
            {"role": "system", "content": MERGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps({
                    "memories": [
                        {
                            "memory_id": memory["id"],
                            "text": memory["text"],
                        }
                        for memory in memories
                    ],
                }),
            },
        ],
    )

    try:
        raw_text = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(
            "Merge generator returned no message content"
        ) from error

    return parse_merged_memory_text(raw_text)


def merge_memory_entities(memories: list[dict]) -> list[dict]:
    merged = []
    seen_names = set()

    for memory in memories:
        for entity in memory["entities"]:
            normalized_name = " ".join(
                entity["name"].casefold().split()
            )

            if normalized_name in seen_names:
                continue

            seen_names.add(normalized_name)
            merged.append({
                "name": entity["name"],
                "context": entity["context"],
            })

    return merged


def prepare_merge(
    decision_id: int,
    candidate_limit: int = 10,
) -> dict | None:
    decision = _load_decision_snapshot(decision_id)

    if decision["applied_action"] != "PENDING":
        raise ValueError(
            f"Decision {decision_id} is already applied"
        )

    if choose_consolidation_action(decision) != "MERGED":
        raise ValueError(
            f"Decision {decision_id} does not authorize a merge"
        )

    memories = _load_pair_memory_snapshots(decision)

    if any(memory["status"] != "ACTIVE" for memory in memories):
        return None

    merged_text = generate_merged_memory_text(memories)
    merged_entities = merge_memory_entities(memories)
    merged_embedding = embed_text(merged_text)
    source_ids = {memory["id"] for memory in memories}

    write_candidates = find_write_candidates(
        merged_text,
        merged_entities,
        merged_embedding,
        limit=candidate_limit,
        exclude_memory_ids=source_ids,
    )
    relation_decisions = evaluate_memory_relations(
        merged_text,
        merged_entities,
        write_candidates,
    )

    return {
        "text": merged_text,
        "entities": merged_entities,
        "embedding": merged_embedding,
        "relation_decisions": relation_decisions,
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


def apply_consolidation_decision(
    decision: dict,
    candidate_limit: int = 10,
) -> dict:
    action = choose_consolidation_action(decision)

    if action == "MERGED":
        prepared_merge = prepare_merge(
            decision["id"],
            candidate_limit=candidate_limit,
        )
        return apply_merge_decision(
            decision["id"],
            prepared_merge or {},
        )

    return apply_non_merge_decision(decision["id"])


def _chunks(items: list, chunk_size: int):
    for start in range(0, len(items), chunk_size):
        yield items[start:start + chunk_size]


def run_consolidation_batch(
    *,
    batch_limit: int = 20,
    candidate_limit: int = 10,
    decision_version: str = CONSOLIDATION_DECISION_VERSION,
    evaluation_batch_size: int = (
        CONSOLIDATION_EVALUATION_BATCH_SIZE
    ),
    apply_actions: bool = True,
) -> dict:
    for name, value in (
        ("batch_limit", batch_limit),
        ("candidate_limit", candidate_limit),
        ("evaluation_batch_size", evaluation_batch_size),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    if not isinstance(decision_version, str) or not decision_version:
        raise ValueError("decision_version must be non-empty text")

    run_id = create_consolidation_run(
        decision_version,
        batch_limit,
        candidate_limit,
    )
    seed_count = 0
    proposed_pair_count = 0
    evaluated_pair_count = 0
    applied_action_count = 0

    try:
        seeds = load_consolidation_seeds(
            decision_version,
            batch_limit,
        )
        seed_ids = [seed[0] for seed in seeds]
        seed_count = len(seeds)
        record_consolidation_run_seeds(run_id, seed_ids)

        evaluated_pair_keys = load_evaluated_pair_keys(
            decision_version
        )
        pairs = discover_consolidation_pairs(
            seeds,
            candidate_limit,
            evaluated_pair_keys,
        )
        proposed_pair_count = len(pairs)

        for pair_batch in _chunks(
            pairs,
            evaluation_batch_size,
        ):
            decisions = evaluate_consolidation_pairs(pair_batch)
            decision_ids = save_consolidation_decisions(
                run_id,
                decision_version,
                CONSOLIDATION_MODEL_ID,
                decisions,
            )
            evaluated_pair_count += len(decision_ids)

        action_results = []
        action_failures = []

        if apply_actions:
            pending_decisions = (
                load_pending_consolidation_decisions(
                    decision_version
                )
            )

            for decision in pending_decisions:
                try:
                    result = apply_consolidation_decision(
                        decision,
                        candidate_limit=candidate_limit,
                    )
                    action_results.append(result)

                    if result["action"] not in {
                        "NONE",
                        "DEFERRED",
                    }:
                        applied_action_count += 1
                except Exception as error:
                    record_consolidation_decision_error(
                        decision["id"],
                        error,
                    )
                    action_failures.append({
                        "decision_id": decision["id"],
                        "error": str(error),
                    })

        if action_failures:
            error_message = (
                f"{len(action_failures)} consolidation action(s) "
                "remain pending"
            )
            fail_consolidation_run(
                run_id,
                error_message,
                seed_count=seed_count,
                proposed_pair_count=proposed_pair_count,
                evaluated_pair_count=evaluated_pair_count,
                applied_action_count=applied_action_count,
            )
            return {
                "run_id": run_id,
                "status": "FAILED",
                "decision_version": decision_version,
                "seed_count": seed_count,
                "proposed_pair_count": proposed_pair_count,
                "evaluated_pair_count": evaluated_pair_count,
                "applied_action_count": applied_action_count,
                "action_results": action_results,
                "action_failures": action_failures,
                "remaining_seed_count": (
                    count_pending_consolidation_seeds(
                        decision_version
                    )
                ),
            }

        mark_consolidation_seeds_processed(
            run_id,
            seed_ids,
            decision_version,
        )
        finish_consolidation_run(
            run_id,
            seed_count=seed_count,
            proposed_pair_count=proposed_pair_count,
            evaluated_pair_count=evaluated_pair_count,
            applied_action_count=applied_action_count,
        )

        return {
            "run_id": run_id,
            "status": "COMPLETED",
            "decision_version": decision_version,
            "seed_count": seed_count,
            "proposed_pair_count": proposed_pair_count,
            "evaluated_pair_count": evaluated_pair_count,
            "applied_action_count": applied_action_count,
            "action_results": action_results,
            "action_failures": [],
            "remaining_seed_count": (
                count_pending_consolidation_seeds(
                    decision_version
                )
            ),
        }

    except Exception as error:
        fail_consolidation_run(
            run_id,
            str(error),
            seed_count=seed_count,
            proposed_pair_count=proposed_pair_count,
            evaluated_pair_count=evaluated_pair_count,
            applied_action_count=applied_action_count,
        )
        raise
