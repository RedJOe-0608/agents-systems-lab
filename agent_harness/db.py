import re
import psycopg
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
            JOIN messages AS msg ON msg.id = m.source_message_id
            ORDER BY m.id
            """
        ).fetchall()

def search_vector_candidates(embedding: list[float], limit: int = 30):
    query_vector = HalfVector(embedding)

    with connect_db() as conn:
        return conn.execute(
            """
            SELECT m.id, m.text, m.entities, m.source_message_id,
                   msg.role, msg.payload,
                   1 - (m.embedding <=> %s) AS similarity
            FROM memories AS m
            JOIN messages AS msg ON msg.id = m.source_message_id
            WHERE m.embedding IS NOT NULL
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
            "SELECT id, entities FROM memories"
        ).fetchall()

    return [
        memory_id
        for memory_id, stored_entities in rows
        if names & normalized_entity_names(stored_entities)
    ]

def load_candidates_by_ids(
    candidate_ids: set[int], embedding: list[float]
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
            JOIN messages AS msg ON msg.id = m.source_message_id
            WHERE m.id = ANY(%s)
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
):
    candidate_ids = {
        row[0] for row in search_vector_candidates(embedding)
    }
    candidate_ids.update(find_entity_candidate_ids(entities))

    rows = load_candidates_by_ids(candidate_ids, embedding)
    return rank_write_candidates(text, entities, rows, limit)

def search_bm25_candidates(
    text: str, exclude_memory_id: int, limit: int = 10
):
    rows = [
        row for row in load_memories_with_sources()
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