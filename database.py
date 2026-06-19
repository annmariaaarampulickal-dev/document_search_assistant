import psycopg
import os
from psycopg.rows import dict_row
from dotenv import load_dotenv

# Loads the .env file into environment variables
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
    return conn