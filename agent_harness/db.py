import psycopg
from psycopg.types.json import Jsonb
from agent_harness.config import DATABASE_URL


def connect_db():
    return psycopg.connect(DATABASE_URL)

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