import streamlit as st
import faiss
import pickle
from psycopg.rows import dict_row
from sentence_transformers import SentenceTransformer
from database import get_db_connection
 
# 1. Page Configuration
st.set_page_config(
    page_title="Document Search Assistant",
    page_icon="🔍",
    layout="wide"
)
 
# 2. Cache Heavy Resources (Loads once when the app starts)
@st.cache_resource
def load_search_engine():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("vector_index.faiss")
    with open("chunk_ids.pkl", "rb") as f:
        chunk_ids = pickle.load(f)
    return model, index, chunk_ids
 
# Try initializing the engine and log status to the sidebar
try:
    model, index, chunk_ids = load_search_engine()
    st.sidebar.success("✅ AI Search Engine Loaded")
except Exception as e:
    st.sidebar.error(f"❌ Initialization Error: {e}")
 
# 3. Main UI Header Area
st.title("🔍 Document Search Assistant")
st.markdown("Query your uploaded documents using semantic AI search.")
 
# 4. Search Input Field
query_text = st.text_input("🤔 What information are you looking for?", key="query")
 
# 5. Search Execution Pipeline
if query_text:
    with st.spinner("Searching database..."):
        try:
            # Step A: Convert plain text query into a vector embedding
            query_vector = model.encode([query_text])
 
            # Step B: Query FAISS index for top 15 nearest vector matches
            distances, indices = index.search(query_vector, 15)
            matched_db_ids = [chunk_ids[idx] for idx in indices[0] if idx != -1]
 
            if not matched_db_ids:
                st.warning("🔍 No matching documents found in the vector index.")
            else:
                # Step C: Retrieve actual text content from PostgreSQL
                with get_db_connection() as conn:
                    with conn.cursor(row_factory=dict_row) as cur:
                        format_ids = ", ".join(str(idx) for idx in matched_db_ids)
                        
                        # Fetch chunks keeping the exact order of relevance from FAISS
                        query_sql = f"""
                        SELECT dc.chunk_text, dc.page_number, d.filename
                        FROM document_chunks dc
                        JOIN documents d ON dc.document_id = d.id
                        WHERE dc.id IN ({format_ids})
                        ORDER BY array_position(ARRAY[{format_ids}]::integer[], dc.id);
                        """
                        cur.execute(query_sql)
                        raw_results = cur.fetchall()
 
                # Step D: Results De-duplication
                unique_results = []
                seen_texts = set()
                
                for row in raw_results:
                    # Normalize spacing to compare content accurately
                    cleaned_text = " ".join(row['chunk_text'].split())
                    if cleaned_text not in seen_texts:
                        seen_texts.add(cleaned_text)
                        unique_results.append(row)
                    
                    # Cap display at the top 3 most relevant unique segments
                    if len(unique_results) == 3:
                        break
 
                # Step E: Render Cleaned Results to the UI
                st.subheader(f"🎯 Top {len(unique_results)} Most Relevant Results")
 
                for i, row in enumerate(unique_results, start=1):
                    header_label = f"📄 Result #{i} — {row['filename']} (Page {row['page_number']})"
                    
                    with st.expander(header_label, expanded=True):
                        # Clean up text spacing for UI display
                        display_text = " ".join(row['chunk_text'].strip().split())
                        
                        # Render the text block in a clean, high-visibility container
                        st.info(display_text)
                        
                        # Meta information footer
                        st.caption(f"**Source File:** {row['filename']} | **Page Number:** {row['page_number']}")
 
        except Exception as e:
            st.error(f"❌ Search Error: {str(e)}")