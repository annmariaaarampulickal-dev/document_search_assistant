import psycopg
from psycopg.rows import dict_row
 
# Database connection details
# Replace 'your_password' with the master password you created for pgAdmin!
DB_PARAMS = "dbname=document_db user=postgres password=root host=localhost port=5432"
 
def get_db_connection():
    """Creates and returns a fresh connection to our PostgreSQL database."""
    conn = psycopg.connect(DB_PARAMS, row_factory=dict_row)
    return conn