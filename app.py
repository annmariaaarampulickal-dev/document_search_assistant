import streamlit as st
import faiss
import pickle
import requests
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
 
import requests # Make sure this import is at the top of your app.py file!
 
# 4. New Document Upload UI Section
st.header("📄 Upload a New Document")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
 
if uploaded_file is not None:
    if st.button("Upload to System"):
        with st.spinner("Processing and chunking PDF..."):
            try:
                # Transmit the file securely to your existing FastAPI upload route
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                upload_response = requests.post("http://127.0.0.1:8000/documents/upload", files=files)
                
                if upload_response.status_code == 201:
                    st.success(f"✅ Successfully processed and stored: {uploaded_file.name}")
                else:
                    st.error(f"❌ Upload failed: {upload_response.text}")
            except Exception as e:
                st.error(f"Could not connect to backend server: {e}")
 
st.write("---")
 
# 5. Search Input Field & Execution Pipeline
st.header("🔍 Ask Questions")
query_text = st.text_input("🤔 What information are you looking for?", key="query")
 
if query_text:
    with st.spinner("Asking Backend API..."):
        try:
            # Route the user question payload directly to your new FastAPI endpoint
            backend_url = "http://127.0.0.1:8000/ask"
            response = requests.post(backend_url, json={"question": query_text})
            
            if response.status_code == 200:
                data = response.json()
                unique_results = data.get("top_3_chunks", [])
                
                if not unique_results:
                    st.warning("🔍 No matching documents found.")
                else:
                    st.subheader(f"🎯 Top {len(unique_results)} Most Relevant Results")
 
                    for i, row in enumerate(unique_results, start=1):
                        header_label = f"📄 Result #{i} — {row['file_name']} (Page {row['page_number']})"
 
                        with st.expander(header_label, expanded=True):
                            display_text = " ".join(row['text'].strip().split())
                            st.info(display_text)
                            st.caption(f"**Source File:** {row['file_name']} | **Page Number:** {row['page_number']}")
            else:
                st.error(f"❌ Backend error: {response.text}")
                
        except Exception as e:
            st.error(f"❌ Connection Error: Could not reach the FastAPI server. {str(e)}")
 