import re
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from agent_harness.config import DATABASE_URL
from pgvector.psycopg import register_vector
from pgvector import HalfVector
from rank_bm25 import BM25Okapi


def connect_db():
    conn = psycopg.connect(DATABASE_URL)
    register_vector(conn)
    return conn

def create_session() -> int:
    with connect_db() as conn:
        row = conn.execute(
            "INSERT INTO sessions DEFAULT VALUES RETURNING id"
        ).fetchone()
        return row[0]

def save_message(session_id: int, sequence_no: int, message: dict) -> int:
    with connect_db() as conn:
        row = conn.execute(
            """
            INSERT INTO messages (session_id, sequence_no, role, payload)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (session_id, sequence_no, message["role"], Jsonb(message)),
        ).fetchone()
        return row[0]

def end_session(session_id: int) -> None:
    with connect_db() as conn:
        conn.execute(
            "UPDATE sessions SET ended_at = now() WHERE id = %s",
            (session_id,),
        )

def load_session_messages(session_id: int):
    with connect_db() as conn:
        return conn.execute(
            """
            SELECT id, sequence_no, role, payload
            FROM messages
            WHERE session_id = %s
            ORDER BY sequence_no
            """,
            (session_id,),
        ).fetchall()

def save_memory(
    text: str,
    source_message_id: int,
    entities: list[dict],
    embedding: list[float],
) -> int:
    with connect_db() as conn:
        row = conn.execute(
            """
            INSERT INTO memories (text, source_message_id, entities, embedding)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (text, source_message_id, Jsonb(entities), HalfVector(embedding)),
        ).fetchone()
        return row[0]

def load_memories_with_sources():
    with connect_db() as conn:
        return conn.execute(
            """
            SELECT m.id, m.text, m.source_message_id, m.entities,
                   msg.role, msg.payload
            FROM memories AS m
            LEFT JOIN messages AS msg
                ON msg.id = m.source_message_id
            WHERE m.status = 'ACTIVE'
            ORDER BY m.id
            """
        ).fetchall()

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

def search_vector_candidates(
    embedding: list[float],
    limit: int = 30,
):
    query_vector = HalfVector(embedding)

    with connect_db() as conn:
        return conn.execute(
            """
            SELECT m.id, m.text, m.entities, m.source_message_id,
                   msg.role, msg.payload,
                   1 - (m.embedding <=> %s) AS similarity
            FROM memories AS m
            LEFT JOIN messages AS msg
                ON msg.id = m.source_message_id
            WHERE m.status = 'ACTIVE'
              AND m.embedding IS NOT NULL
            ORDER BY m.embedding <=> %s
            LIMIT %s
            """,
            (query_vector, query_vector, limit),
        ).fetchall()

def find_entity_candidate_ids(entities: list[dict]) -> list[int]:
    names = normalized_entity_names(entities)
    if not names:
        return []

    with connect_db() as conn:
        rows = conn.execute(
            """
            SELECT id, entities
            FROM memories
            WHERE status = 'ACTIVE'
            """
        ).fetchall()

    return [
        memory_id
        for memory_id, stored_entities in rows
        if names & normalized_entity_names(stored_entities)
    ]

def load_candidates_by_ids(
    candidate_ids: set[int],
    embedding: list[float],
):
    if not candidate_ids:
        return []

    query_vector = HalfVector(embedding)

    with connect_db() as conn:
        return conn.execute(
            """
            SELECT m.id, m.text, m.entities, m.source_message_id,
                   msg.role, msg.payload,
                   CASE
                       WHEN m.embedding IS NULL THEN 0.0
                       ELSE 1 - (m.embedding <=> %s)
                   END AS similarity
            FROM memories AS m
            LEFT JOIN messages AS msg
                ON msg.id = m.source_message_id
            WHERE m.id = ANY(%s)
              AND m.status = 'ACTIVE'
            ORDER BY m.id
            """,
            (query_vector, sorted(candidate_ids)),
        ).fetchall()

def tokenize_memory(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())

MEMORY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for",
    "from", "in", "is", "it", "of", "on", "or", "that",
    "the", "this", "to", "was", "were", "with", "user",
}

# This is for jaccard = number of shared words ÷ number of distinct words across both texts
def word_overlap(first: str, second: str) -> float:
    first_words = set(tokenize_memory(first)) - MEMORY_STOPWORDS
    second_words = set(tokenize_memory(second)) - MEMORY_STOPWORDS
    all_words = first_words | second_words

    if not all_words:
        return 0.0

    return len(first_words & second_words) / len(all_words)

def normalized_entity_names(entities: list[dict]) -> set[str]:
    return {
        " ".join(entity["name"].casefold().split())
        for entity in entities
    }

def rank_write_candidates(
    new_text: str,
    new_entities: list[dict],
    candidates: list[tuple],
    limit: int = 10,
):
    new_names = normalized_entity_names(new_entities)
    ranked = []

    for row in candidates:
        stored_names = normalized_entity_names(row[2])
        entity_overlap = (
            len(new_names & stored_names) / len(new_names)
            if new_names else 0.0
        )
        similarity = max(0.0, min(1.0, float(row[6])))
        overlap = word_overlap(new_text, row[1])

        score = (
            0.70 * similarity
            + 0.20 * entity_overlap
            + 0.10 * overlap
        )
        ranked.append((row, score))

    ranked.sort(key=lambda item: (-item[1], item[0][0]))
    return ranked[:limit]

def find_write_candidates(
    text: str,
    entities: list[dict],
    embedding: list[float],
    limit: int = 10,
    exclude_memory_id: int | None = None,
    exclude_memory_ids: set[int] | None = None,
):
    candidate_ids = {
        row[0] for row in search_vector_candidates(embedding)
    }
    candidate_ids.update(find_entity_candidate_ids(entities))

    if exclude_memory_id is not None:
        candidate_ids.discard(exclude_memory_id)

    if exclude_memory_ids:
        candidate_ids.difference_update(exclude_memory_ids)

    rows = load_candidates_by_ids(candidate_ids, embedding)
    return rank_write_candidates(text, entities, rows, limit)

def find_consolidation_candidates(
    seed_row: tuple,
    limit: int = 10,
):
    seed_id = seed_row[0]
    seed_text = seed_row[1]
    seed_entities = seed_row[2]
    seed_embedding = seed_row[4]

    if seed_embedding is None:
        raise ValueError(
            f"Memory {seed_id} has no embedding"
        )

    ranked_candidates = find_write_candidates(
        seed_text,
        seed_entities,
        seed_embedding.to_list(),
        limit=limit,
        exclude_memory_id=seed_id,
    )

    candidate_ids = [
        row[0]
        for row, _score in ranked_candidates
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

    rows_by_id = {
        row[0]: row
        for row in rows
    }

    return [
        rows_by_id[memory_id]
        for memory_id in candidate_ids
        if memory_id in rows_by_id
    ]

def search_bm25_candidates(
    text: str, exclude_memory_id: int | None = None, limit: int = 10
):
    rows = load_memories_with_sources()

    if exclude_memory_id is not None:
        rows = [
            row
            for row in rows
            if row[0] != exclude_memory_id
        ]

    query_tokens = sorted(set(tokenize_memory(text)))
    documents = [tokenize_memory(row[1]) for row in rows]

    if not query_tokens or not any(documents):
        return []

    scores = BM25Okapi(documents).get_scores(query_tokens)
    matches = [
        (row, float(score))
        for row, tokens, score in zip(rows, documents, scores)
        if not set(tokens).isdisjoint(query_tokens)
    ]
    matches.sort(key=lambda match: (-match[1], -match[0][0]))
    return matches[:limit]

def save_memory_edges(edges: list[dict]) -> list[int]:
    if not edges:
        return []

    edge_ids = []

    with connect_db() as conn:
        for edge in edges:
            row = conn.execute(
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
                RETURNING id
                """,
                (
                    edge["source_memory_id"],
                    edge["target_memory_id"],
                    edge["relation_type"],
                    edge["score"],
                ),
            ).fetchone()

            edge_ids.append(row[0])

    return edge_ids

def reciprocal_rank_fusion(
    ranked_id_lists: list[list[int]],
    limit: int,
    rank_constant: int = 60,
) -> list[int]:
    scores = {}

    for ranked_ids in ranked_id_lists:
        for rank, memory_id in enumerate(ranked_ids, start=1):
            contribution = 1.0 / (rank_constant + rank)

            scores[memory_id] = (
                scores.get(memory_id, 0.0)
                + contribution
            )

    ordered_ids = sorted(
        scores,
        key=lambda memory_id: (
            -scores[memory_id],
            memory_id,
        ),
    )

    return ordered_ids[:limit]

def find_query_anchors(
    query: str,
    embedding: list[float],
    limit: int = 10,
    search_limit: int = 30,
):
    vector_rows = search_vector_candidates(
        embedding,
        limit=search_limit,
    )

    keyword_matches = search_bm25_candidates(
        query,
        limit=search_limit,
    )

    vector_ids = [
        row[0]
        for row in vector_rows
    ]

    keyword_ids = [
        row[0]
        for row, _score in keyword_matches
    ]

    anchor_ids = reciprocal_rank_fusion(
        [vector_ids, keyword_ids],
        limit=limit,
    )

    if not anchor_ids:
        return []

    rows = load_candidates_by_ids(
        set(anchor_ids),
        embedding,
    )

    rows_by_id = {
        row[0]: row
        for row in rows
    }

    return [
        rows_by_id[memory_id]
        for memory_id in anchor_ids
        if memory_id in rows_by_id
    ]

def load_graph_neighbors(
    frontier_ids: set[int],
    active_graphs: dict,
):
    if not frontier_ids or not active_graphs:
        return []

    graph_relation_types = {
        "semantic": "RELATED_TO",
        "causal": "CAUSES",
        "entity": "SHARED_ENTITY",
    }

    relation_types = [
        graph_relation_types[graph]
        for graph in active_graphs
    ]

    frontier_list = sorted(frontier_ids)

    with connect_db() as conn:
        return conn.execute(
            """
            SELECT
                neighbor.id,
                neighbor.text,
                neighbor.entities,
                neighbor.source_message_id,
                msg.role,
                msg.payload,
                edge.relation_type,
                edge.score,
                edge.source_memory_id,
                edge.target_memory_id
            FROM memory_edges AS edge
            JOIN memories AS neighbor
                ON neighbor.id = CASE
                    WHEN edge.source_memory_id = ANY(%s)
                        THEN edge.target_memory_id
                    ELSE edge.source_memory_id
                END
            LEFT JOIN messages AS msg
                ON msg.id = neighbor.source_message_id
            WHERE neighbor.status = 'ACTIVE'
                AND edge.relation_type = ANY(%s)
              AND (
                  edge.source_memory_id = ANY(%s)
                  OR edge.target_memory_id = ANY(%s)
              )
            ORDER BY edge.score DESC, edge.id
            """,
            (
                frontier_list,
                relation_types,
                frontier_list,
                frontier_list,
            ),
        ).fetchall()

def select_graph_expansion_candidates(
    neighbor_rows: list[tuple],
    visited_ids: set[int],
    graph_budgets: dict,
    graph_budget_used: dict,
):
    relation_graphs = {
        "RELATED_TO": "semantic",
        "CAUSES": "causal",
        "SHARED_ENTITY": "entity",
    }

    selected = []
    selected_ids = set()
    updated_usage = dict(graph_budget_used)

    for row in neighbor_rows:
        neighbor_id = row[0]
        relation_type = row[6]
        graph = relation_graphs[relation_type]

        if neighbor_id in visited_ids:
            continue

        if neighbor_id in selected_ids:
            continue

        if updated_usage[graph] >= graph_budgets[graph]:
            continue

        selected.append(row)
        selected_ids.add(neighbor_id)
        updated_usage[graph] += 1

    return selected, updated_usage

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
