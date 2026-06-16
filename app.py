import streamlit as st
import requests
 
# Page Configuration
st.set_page_config(
    page_title="FAISS Document Search Assistant",
    page_icon="🔍",
    layout="wide"
)
 
st.title("🔍 FAISS Document Search Assistant")
st.markdown("Query your uploaded documents using advanced semantic FAISS index search mapping.")
 
# --- SIDEBAR: SYSTEM RESET ACTIONS ---
st.sidebar.header("⚙️ System Management")
st.sidebar.markdown("Use this to clear out stale test records or reset your workspace cleanly.")
 
if st.sidebar.button("🗑️ Reset Entire Vector System", type="primary"):
    try:
        # Step A: Clear out PostgreSQL Master/Child document tables via endpoint iteration
        get_res = requests.get("http://127.0.0.1:8000/documents")
        if get_res.status_code == 200:
            for doc in get_res.json():
                # Extract ID cleanly regardless of list or dictionary mapping format
                doc_id = doc[0] if isinstance(doc, list) else doc.get("id")
                requests.delete(f"http://127.0.0.1:8000/documents/{doc_id}")
        
        st.sidebar.success("Database wiped! Delete vector_index.faiss and chunk_ids.pkl from folder to complete reset.")
    except Exception as e:
        st.sidebar.error(f"Reset Connection Failure: {e}")
 
 
# --- MAIN PAGE: INTERACTIVE MULTI-UPLOAD ---
st.header("📄 Upload Documents")
uploaded_files = st.file_uploader(
    "Choose PDF files to store and index into FAISS",
    type=["pdf"],
    accept_multiple_files=True
)
 
if uploaded_files:
    if st.button("Upload All Files to FAISS"):
        for uploaded_file in uploaded_files:
            with st.spinner(f"Index parsing {uploaded_file.name}..."):
                try:
                    # Send payload data streams safely to FastAPI
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    res = requests.post("http://127.0.0.1:8000/documents/upload", files=files)
                    
                    if res.status_code == 201:
                        st.success(f"✅ Successfully Indexed into FAISS: {uploaded_file.name}")
                    else:
                        st.error(f"❌ Upload error on {uploaded_file.name}: {res.text}")
                except Exception as e:
                    st.error(f"Unable to establish communication with API server: {e}")
 
st.write("---")
 
 
# --- MAIN PAGE: GLOBAL VECTOR SEARCH ---
st.header("🔍 Search Documents via FAISS Index")
query_text = st.text_input("🤔 What information are you looking for?", key="query")
 
if query_text:
    with st.spinner("Executing FAISS Semantic Search Mapping..."):
        try:
            # Send question block to FastAPI /ask endpoint
            response = requests.post("http://127.0.0.1:8000/ask", json={"question": query_text})
            
            if response.status_code == 200:
                unique_results = response.json().get("top_3_chunks", [])
                
                if not unique_results:
                    st.warning("🔍 No relevant matches found within the localized FAISS indexes.")
                else:
                    st.subheader(f"🎯 Top {len(unique_results)} Most Relevant Matches")
                    
                    for i, row in enumerate(unique_results, start=1):
                        header_label = f"📄 Match #{i} — {row['file_name']} (Page {row['page_number']})"
                        
                        with st.expander(header_label, expanded=True):
                            # Clean up multi-spacing layout quirks for display container
                            display_text = " ".join(row['text'].strip().split())
                            st.info(display_text)
                            
                            # Document tracking trace footer
                            st.caption(f"**Source Target:** {row['file_name']} | **Page:** {row['page_number']}")
            else:
                st.error(f"❌ API Processing Error: {response.text}")
                
        except Exception as e:
            st.error(f"❌ Connection Error: Could not reach the FastAPI server. {str(e)}")