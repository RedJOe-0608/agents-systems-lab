from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from agent_harness.storage.connection import connect_db
from agent_harness.storage.memories import find_write_candidates


def load_consolidation_seeds(
    decision_version: str,
    batch_limit: int,
):
    with connect_db() as conn:
        return conn.execute(
            """
            SELECT id, text, entities, created_at, embedding
            FROM memories
            WHERE status = 'ACTIVE'
              AND last_consolidated_version
                  IS DISTINCT FROM %s
            ORDER BY
                last_consolidated_at NULLS FIRST,
                created_at,
                id
            LIMIT %s
            """,
            (decision_version, batch_limit),
        ).fetchall()


def count_pending_consolidation_seeds(
    decision_version: str,
) -> int:
    with connect_db() as conn:
        row = conn.execute(
            """
            SELECT count(*)
            FROM memories
            WHERE status = 'ACTIVE'
              AND last_consolidated_version
                  IS DISTINCT FROM %s
            """,
            (decision_version,),
        ).fetchone()
        return row[0]


def find_consolidation_candidates(
    seed_row: tuple,
    limit: int = 10,
):
    seed_id = seed_row[0]
    seed_text = seed_row[1]
    seed_embedding = seed_row[4]

    if seed_embedding is None:
        raise ValueError(
            f"Memory {seed_id} has no embedding"
        )

    ranked_candidates = find_write_candidates(
        seed_text,
        seed_embedding.to_list(),
        limit=limit,
        exclude_memory_id=seed_id,
    )
    candidate_ids = [
        row[0]
        for row in ranked_candidates
    ]

    if not candidate_ids:
        return []

    with connect_db() as conn:
        rows = conn.execute(
            """
            SELECT id, text, entities, created_at, embedding
            FROM memories
            WHERE id = ANY(%s)
              AND status = 'ACTIVE'
            """,
            (candidate_ids,),
        ).fetchall()

    rows_by_id = {row[0]: row for row in rows}
    return [
        rows_by_id[memory_id]
        for memory_id in candidate_ids
        if memory_id in rows_by_id
    ]


def create_consolidation_run(
    decision_version: str,
    batch_limit: int,
    candidate_limit: int,
) -> int:
    with connect_db() as conn:
        row = conn.execute(
            """
            INSERT INTO memory_consolidation_runs (
                decision_version,
                batch_limit,
                candidate_limit
            )
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            (decision_version, batch_limit, candidate_limit),
        ).fetchone()
        return row[0]


def record_consolidation_run_seeds(
    run_id: int,
    memory_ids: list[int],
) -> None:
    if not memory_ids:
        return

    with connect_db() as conn:
        conn.executemany(
            """
            INSERT INTO memory_consolidation_run_memories (
                run_id,
                memory_id
            )
            VALUES (%s, %s)
            ON CONFLICT DO NOTHING
            """,
            [
                (run_id, memory_id)
                for memory_id in memory_ids
            ],
        )


def load_evaluated_pair_keys(
    decision_version: str,
) -> set[tuple[int, int]]:
    with connect_db() as conn:
        rows = conn.execute(
            """
            SELECT memory_a_id, memory_b_id
            FROM memory_consolidation_decisions
            WHERE decision_version = %s
            """,
            (decision_version,),
        ).fetchall()

    return {
        (memory_a_id, memory_b_id)
        for memory_a_id, memory_b_id in rows
    }


def save_consolidation_decisions(
    run_id: int,
    decision_version: str,
    model_id: str,
    decisions: list[dict],
) -> list[int]:
    decision_ids = []

    with connect_db() as conn:
        for decision in decisions:
            current_id = decision["current_memory_id"]
            candidate_id = decision["candidate_memory_id"]
            memory_a_id, memory_b_id = sorted(
                (current_id, candidate_id)
            )
            representation = decision["representation"]

            row = conn.execute(
                """
                INSERT INTO memory_consolidation_decisions (
                    run_id,
                    memory_a_id,
                    memory_b_id,
                    current_memory_id,
                    candidate_memory_id,
                    redundancy_score,
                    contradiction_score,
                    current_supersedes_candidate_score,
                    candidate_supersedes_current_score,
                    representation,
                    representation_probability,
                    representation_probabilities,
                    decision_version,
                    model_id
                )
                VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s, %s
                )
                ON CONFLICT (
                    memory_a_id,
                    memory_b_id,
                    decision_version
                )
                DO NOTHING
                RETURNING id
                """,
                (
                    run_id,
                    memory_a_id,
                    memory_b_id,
                    current_id,
                    candidate_id,
                    decision["redundancy"],
                    decision["contradiction"],
                    decision["current_supersedes_candidate"],
                    decision["candidate_supersedes_current"],
                    representation["choice"],
                    representation["probability"],
                    Jsonb(representation["probabilities"]),
                    decision_version,
                    model_id,
                ),
            ).fetchone()

            if row is None:
                row = conn.execute(
                    """
                    SELECT id
                    FROM memory_consolidation_decisions
                    WHERE memory_a_id = %s
                      AND memory_b_id = %s
                      AND decision_version = %s
                    """,
                    (
                        memory_a_id,
                        memory_b_id,
                        decision_version,
                    ),
                ).fetchone()

            decision_ids.append(row[0])

    return decision_ids


def load_pending_consolidation_decisions(
    decision_version: str,
) -> list[dict]:
    with connect_db() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT *
                FROM memory_consolidation_decisions
                WHERE decision_version = %s
                  AND applied_action = 'PENDING'
                ORDER BY id
                """,
                (decision_version,),
            )
            return cursor.fetchall()


def load_consolidation_decision(decision_id: int) -> dict:
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


def load_pair_memory_snapshots(decision: dict) -> list[dict]:
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


def mark_consolidation_seeds_processed(
    run_id: int,
    memory_ids: list[int],
    decision_version: str,
) -> None:
    if not memory_ids:
        return

    with connect_db() as conn:
        conn.execute(
            """
            UPDATE memories
            SET last_consolidated_version = %s,
                last_consolidated_at = now()
            WHERE id = ANY(%s)
              AND status = 'ACTIVE'
            """,
            (decision_version, memory_ids),
        )
        conn.execute(
            """
            UPDATE memory_consolidation_run_memories
            SET status = 'COMPLETED',
                processed_at = now(),
                error_message = NULL
            WHERE run_id = %s
              AND memory_id = ANY(%s)
            """,
            (run_id, memory_ids),
        )


def finish_consolidation_run(
    run_id: int,
    *,
    seed_count: int,
    proposed_pair_count: int,
    evaluated_pair_count: int,
    applied_action_count: int,
) -> None:
    with connect_db() as conn:
        conn.execute(
            """
            UPDATE memory_consolidation_runs
            SET status = 'COMPLETED',
                completed_at = now(),
                seed_count = %s,
                proposed_pair_count = %s,
                evaluated_pair_count = %s,
                applied_action_count = %s,
                error_message = NULL
            WHERE id = %s
            """,
            (
                seed_count,
                proposed_pair_count,
                evaluated_pair_count,
                applied_action_count,
                run_id,
            ),
        )


def fail_consolidation_run(
    run_id: int,
    error_message: str,
    *,
    seed_count: int = 0,
    proposed_pair_count: int = 0,
    evaluated_pair_count: int = 0,
    applied_action_count: int = 0,
) -> None:
    with connect_db() as conn:
        conn.execute(
            """
            UPDATE memory_consolidation_runs
            SET status = 'FAILED',
                completed_at = now(),
                seed_count = %s,
                proposed_pair_count = %s,
                evaluated_pair_count = %s,
                applied_action_count = %s,
                error_message = %s
            WHERE id = %s
            """,
            (
                seed_count,
                proposed_pair_count,
                evaluated_pair_count,
                applied_action_count,
                error_message,
                run_id,
            ),
        )
        conn.execute(
            """
            UPDATE memory_consolidation_run_memories
            SET status = 'FAILED',
                processed_at = now(),
                error_message = %s
            WHERE run_id = %s
              AND status = 'PENDING'
            """,
            (error_message, run_id),
        )


def load_active_contradiction_counterparts(
    memory_ids: set[int],
) -> list[tuple]:
    if not memory_ids:
        return []

    ordered_ids = sorted(memory_ids)

    with connect_db() as conn:
        return conn.execute(
            """
            WITH counterparts AS (
                SELECT
                    CASE
                        WHEN edge.source_memory_id = ANY(%s)
                            THEN edge.target_memory_id
                        ELSE edge.source_memory_id
                    END AS memory_id,
                    edge.score,
                    edge.id AS edge_id
                FROM memory_lifecycle_edges AS edge
                WHERE edge.relation_type = 'CONTRADICTS'
                  AND (
                      edge.source_memory_id = ANY(%s)
                      OR edge.target_memory_id = ANY(%s)
                  )
            )
            SELECT DISTINCT ON (memory.id)
                memory.id,
                memory.text,
                memory.entities,
                memory.source_message_id,
                message.role,
                message.payload
            FROM counterparts
            JOIN memories AS memory
                ON memory.id = counterparts.memory_id
            LEFT JOIN messages AS message
                ON message.id = memory.source_message_id
            WHERE memory.status = 'ACTIVE'
              AND NOT (memory.id = ANY(%s))
            ORDER BY
                memory.id,
                counterparts.score DESC,
                counterparts.edge_id
            """,
            (
                ordered_ids,
                ordered_ids,
                ordered_ids,
                ordered_ids,
            ),
        ).fetchall()
