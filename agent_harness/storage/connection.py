import psycopg
from pgvector.psycopg import register_vector

from agent_harness.config import DATABASE_URL


def connect_db():
    conn = psycopg.connect(DATABASE_URL)
    register_vector(conn)
    return conn
