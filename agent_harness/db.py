import psycopg
from psycopg.types.json import Jsonb
from agent_harness.config import DATABASE_URL
from pgvector.psycopg import register_vector
from pgvector import HalfVector


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

def search_vector_candidates(embedding: list[float], exclude_memory_id: int, limit: int = 10):
    query_vector = HalfVector(embedding)

    with connect_db() as conn:
        return conn.execute(
            """
            SELECT m.id, m.text, m.entities, m.source_message_id,
                   msg.role, msg.payload,
                   1 - (m.embedding <=> %s) AS similarity
            FROM memories AS m
            JOIN messages AS msg ON msg.id = m.source_message_id
            WHERE m.id <> %s
              AND m.embedding IS NOT NULL
            ORDER BY m.embedding <=> %s
            LIMIT %s
            """,
            (query_vector, exclude_memory_id, query_vector, limit),
        ).fetchall()