import re

from pgvector import HalfVector
from psycopg.types.json import Jsonb
from rank_bm25 import BM25Okapi

from agent_harness.storage.connection import connect_db


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
            (
                text,
                source_message_id,
                Jsonb(entities),
                HalfVector(embedding),
            ),
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


def find_write_candidates(
    text: str,
    embedding: list[float],
    limit: int = 10,
    exclude_memory_id: int | None = None,
    exclude_memory_ids: set[int] | None = None,
):
    search_limit = 30
    candidate_ids = _find_hybrid_candidate_ids(
        text,
        embedding,
        limit=search_limit * 2,
        search_limit=search_limit,
    )

    if exclude_memory_id is not None:
        candidate_ids = [
            memory_id
            for memory_id in candidate_ids
            if memory_id != exclude_memory_id
        ]

    if exclude_memory_ids:
        candidate_ids = [
            memory_id
            for memory_id in candidate_ids
            if memory_id not in exclude_memory_ids
        ]

    candidate_ids = candidate_ids[:limit]
    rows = load_candidates_by_ids(set(candidate_ids), embedding)
    rows_by_id = {row[0]: row for row in rows}
    return [
        rows_by_id[memory_id]
        for memory_id in candidate_ids[:limit]
        if memory_id in rows_by_id
    ]


def search_bm25_candidates(
    text: str,
    limit: int = 10,
):
    rows = load_memories_with_sources()

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


def _find_hybrid_candidate_ids(
    text: str,
    embedding: list[float],
    limit: int,
    search_limit: int,
) -> list[int]:
    vector_ids = [
        row[0]
        for row in search_vector_candidates(
            embedding,
            limit=search_limit,
        )
    ]
    keyword_ids = [
        row[0]
        for row, _score in search_bm25_candidates(
            text,
            limit=search_limit,
        )
    ]
    return reciprocal_rank_fusion(
        [vector_ids, keyword_ids],
        limit=limit,
    )


def find_query_anchors(
    query: str,
    embedding: list[float],
    limit: int = 10,
    search_limit: int = 30,
):
    anchor_ids = _find_hybrid_candidate_ids(
        query,
        embedding,
        limit=limit,
        search_limit=search_limit,
    )

    if not anchor_ids:
        return []

    rows = load_candidates_by_ids(
        set(anchor_ids),
        embedding,
    )
    rows_by_id = {row[0]: row for row in rows}

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
