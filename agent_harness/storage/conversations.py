from psycopg.types.json import Jsonb

from agent_harness.storage.connection import connect_db


def create_session() -> int:
    with connect_db() as conn:
        row = conn.execute(
            "INSERT INTO sessions DEFAULT VALUES RETURNING id"
        ).fetchone()
        return row[0]


def save_message(
    session_id: int,
    sequence_no: int,
    message: dict,
) -> int:
    with connect_db() as conn:
        row = conn.execute(
            """
            INSERT INTO messages (session_id, sequence_no, role, payload)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (
                session_id,
                sequence_no,
                message["role"],
                Jsonb(message),
            ),
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
