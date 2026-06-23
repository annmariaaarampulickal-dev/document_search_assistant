import psycopg
import os
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    conn = psycopg.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        row_factory=dict_row
    )
    register_vector(conn)
    return conn