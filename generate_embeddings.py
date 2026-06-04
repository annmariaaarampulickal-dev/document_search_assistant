import psycopg
from psycopg.rows import dict_row
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from database import get_db_connection
 
print(" Loading the SentenceTransformer AI model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print(" Model loaded successfully!")
 
def build_local_vector_index():
    try:
        with get_db_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                # 1. Fetch text chunks from your existing PostgreSQL tables
                cur.execute("SELECT id, chunk_text FROM document_chunks;")
                chunks = cur.fetchall()
                
                if not chunks:
                    print(" No text chunks found in the database. Please upload a PDF via your API playground first!")
                    return
 
                print(f" Found {len(chunks)} text chunks. Generating embeddings...")
 
                texts = [c["chunk_text"] for c in chunks]
                chunk_ids = [c["id"] for c in chunks]
 
                # 2. Convert text arrays into 384-dimensional vector mathematics
                embeddings = model.encode(texts)
 
                # 3. Build a fast flat vector space search index
                index = faiss.IndexFlatL2(384)
                index.add(embeddings)
 
                # 4. Save files directly into your project directory (No admin rights required!)
                faiss.write_index(index, "vector_index.faiss")
 
                with open("chunk_ids.pkl", "wb") as f:
                    pickle.dump(chunk_ids, f)
 
                print(" Success! Generated 'vector_index.faiss' and 'chunk_ids.pkl' locally.")
 
    except Exception as e:
        print(f" Error during local embedding generation: {str(e)}")
 
if __name__ == "__main__":
    build_local_vector_index()
 