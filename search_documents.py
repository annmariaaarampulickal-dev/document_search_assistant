import faiss
import pickle
from psycopg.rows import dict_row
from sentence_transformers import SentenceTransformer
from database import get_db_connection
 
print("⏳ Initializing AI Search Engine...")
# 1. Load the exact same AI model used for generating embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
 
# 2. Load our local vector space and ID mapping files
index = faiss.read_index("vector_index.faiss")
with open("chunk_ids.pkl", "rb") as f:
    chunk_ids = pickle.load(f)
print("✅ Search Engine Ready!")
 
def semantic_search(query_text, top_k=3):
    try:
        # 3. Convert the user's question into a 384-dimensional vector
        query_vector = model.encode([query_text])
 
        # Ask FAISS for a wider net of results (top 15) so we have plenty of room to filter duplicates
        distances, indices = index.search(query_vector, 15)
 
        # Gather all matching database IDs in order
        matched_db_ids = [chunk_ids[idx] for idx in indices[0] if idx != -1]
 
        if not matched_db_ids:
            print("🔍 No matching documents found.")
            return
 
        # 4. Retrieve the paragraph text from PostgreSQL
        with get_db_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                format_ids = ", ".join(str(idx) for idx in matched_db_ids)
 
                query = f"""
                SELECT dc.chunk_text, dc.page_number, d.filename
                FROM document_chunks dc
                JOIN documents d ON dc.document_id = d.id
                WHERE dc.id IN ({format_ids})
                ORDER BY array_position(ARRAY[{format_ids}]::integer[], dc.id);
                """
 
                cur.execute(query)
                raw_results = cur.fetchall()
 
                # 🛠️ THE ULTIMATE FIX: De-duplicate based on the actual content string!
                unique_results = []
                seen_texts = set()
 
                for row in raw_results:
                    # Clean the text up a bit to ensure whitespace differences don't trick the filter
                    cleaned_text = " ".join(row['chunk_text'].split())
                    
                    if cleaned_text not in seen_texts:
                        seen_texts.add(cleaned_text)
                        unique_results.append(row)
                    
                    # Stop as soon as we have our top_k unique results
                    if len(unique_results) == top_k:
                        break
 
                print(f"\n🎯 Top {len(unique_results)} Most Relevant Results for: '{query_text}'\n" + "="*60)
                for i, row in enumerate(unique_results, 1):
                    print(f"📄 Result #{i} | File: {row['filename']} (Page {row['page_number']})")
                    print(f"💬 Text: {row['chunk_text'].strip()}")
                    print("-" * 60)
 
    except Exception as e:
        print(f"❌ Search error: {str(e)}")
 
if __name__ == "__main__":
    user_query = input("\n🤔 Enter your search question: ")
    semantic_search(user_query)
