from database import get_db_connection
 
def reset_database():
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                print("🗑️ Clearing database tables...")
                # Clear the tables in the correct order to respect constraints
                cur.execute("DELETE FROM document_chunks;")
                cur.execute("DELETE FROM documents;")
                conn.commit()
                print("✅ Database cleared successfully!")
    except Exception as e:
        print(f"❌ Error clearing database: {e}")
 
if __name__ == "__main__":
    reset_database()