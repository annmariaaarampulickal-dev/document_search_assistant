import streamlit as st
import faiss
import pickle
from psycopg.rows import dict_row
from sentence_transformers import SentenceTransformer
from database import get_db_connection
 
# Page configuration
st.set_page_config(page_title="Document Search Assistant", page_icon="🔍", layout="wide")
 
# Cache resources so the app doesn't re-load the heavy AI model on every click
@st.cache_resource
def load_search_engine():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("vector_index.faiss")
    with open("chunk_ids.pkl", "rb") as f:
        chunk_ids = pickle.load(f)
    return model, index, chunk_ids
 
try:
    model, index, chunk_ids = load_search_engine()
    st.sidebar.success("✅ AI Search Engine Loaded")
except Exception as e:
    st.sidebar.error(f"❌ Initialization Error: {e}")
 
# Application Title Layout
st.title("🔍 Document Search Assistant")
st.markdown("Query your uploaded  documents using semantic AI mapping.")
 
# Search Input
query_text = st.text_input("🤔 What information are you looking for?", placeholder="e.g., What is organizational behaviour?")
 
if query_text:
    with st.spinner("Searching through vector space..."):
        try:
            # 1. Generate query vector
            query_vector = model.encode([query_text])
 
            # 2. Search FAISS index (grabbing up to 15 chunks to allow content de-duplication)
            distances, indices = index.search(query_vector, 15)
            matched_db_ids = [chunk_ids[idx] for idx in indices[0] if idx != -1]
 
            if not matched_db_ids:
                st.warning("🔍 No matching documents found in the database.")
            else:
                # 3. Pull content from PostgreSQL
                with get_db_connection() as conn:
                    with conn.cursor(row_factory=dict_row) as cur:
                        format_ids = ", ".join(str(idx) for idx in matched_db_ids)
                        query_sql = f"""
                        SELECT dc.chunk_text, dc.page_number, d.filename
                        FROM document_chunks dc
                        JOIN documents d ON dc.document_id = d.id
                        WHERE dc.id IN ({format_ids})
                        ORDER BY array_position(ARRAY[{format_ids}]::integer[], dc.id);
                        """
                        cur.execute(query_sql)
                        raw_results = cur.fetchall()
 
                        # 4. Text-Based De-duplication
                        unique_results = []
                        seen_texts = set()
 
                        for row in raw_results:
                            cleaned_text = " ".join(row['chunk_text'].split())
                            if cleaned_text not in seen_texts:
                                seen_texts.add(cleaned_text)
                                unique_results.append(row)
                            if len(unique_results) == 3: # Keep top 3 unique chunks
                                break
 
                        # 5. Display results neatly in the Web UI
                        st.subheader(f"🎯 Top {len(unique_results)} Most Relevant Chunks")
                        
                        for i, row in enumerate(unique_results, 1):
                            with st.container():
                                # Create a nice card style using columns
                                st.markdown(f"### 📄 Result #{i}")
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.caption(f"**File:** {row['filename']}")
                                with col2:
                                    st.caption(f"📌 **Page Number:** {row['page_number']}")
                                
                                # Info box container for the exact passage text
                                st.info(row['chunk_text'].strip())
                                st.markdown("---")
 
        except Exception as e:
            st.error(f"❌ Search Error encountered: {str(e)}")