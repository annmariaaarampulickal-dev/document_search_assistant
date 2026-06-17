import streamlit as st
import requests
 
st.set_page_config(
    page_title="FAISS Document Search Assistant",
    page_icon="🔍",
    layout="wide"
)
 
st.title("🔍 FAISS Document Search Assistant")
st.markdown("Query your uploaded documents using advanced semantic FAISS index search mapping.")
 
# --- SIDEBAR CONTROL PANEL ---
st.sidebar.header("⚙️ System Management")
st.sidebar.markdown("Use this panel to clean out your workspace files or reset database tracking registries.")
 
if st.sidebar.button("🗑️ Reset Entire Vector System", type="primary"):
    try:
        get_res = requests.get("http://127.0.0.1:8000/documents")
        if get_res.status_code == 200:
            for doc in get_res.json():
                # Supports dictionary parsing safely
                doc_id = doc.get("id") if isinstance(doc, dict) else doc[0]
                requests.delete(f"http://127.0.0.1:8000/documents/{doc_id}")
        st.sidebar.success("Database records cleared successfully!")
    except Exception as e:
        st.sidebar.error(f"System communication connection crash error: {e}")
 
# --- DOCUMENT UPLOADER CONTAINER BAR ---
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
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    res = requests.post("http://127.0.0.1:8000/documents/upload", files=files)
                    
                    if res.status_code == 201:
                        st.success(f"✅ Successfully Indexed into FAISS: {uploaded_file.name}")
                    else:
                        st.error(f"❌ Upload Error on {uploaded_file.name}: {res.json().get('detail', res.text)}")
                except Exception as e:
                    st.error(f"Unable to reach the FastAPI server connection: {e}")
 
st.write("---")
 
# --- CONSOLE SEMANTIC ENTRY SEARCH BAR ---
st.header("🔍 Search Documents via FAISS Index")
query_text = st.text_input("🤔 What information are you looking for?", key="query")
 
if query_text:
    with st.spinner("Executing FAISS Semantic Search Mapping..."):
        try:
            # Matches your exact QuestionRequest Pydantic structure
            response = requests.post("http://127.0.0.1:8000/ask", json={"question": query_text})
            
            if response.status_code == 200:
                unique_results = response.json().get("top_3_chunks", [])
                
                if not unique_results:
                    st.warning("🔍 No relevant passage matches discovered within the active FAISS indices.")
                else:
                    st.subheader(f"🎯 Top {len(unique_results)} Most Relevant Matches")
                    
                    # FIXED LOOP BLOCK: Cleanly unpacks as individual matches to eliminate tuple syntax error
                    for i, match in enumerate(unique_results, start=1):
                        header_label = f"Passage Match #{i}"
                        
                        with st.expander(header_label, expanded=True):
                            # Cleans double spaces safely out of the dictionary string
                            display_text = " ".join(match['text'].strip().split())
                            st.info(display_text)
                            
                            # REQUIRED EXPLICIT CITATION WITH PAGE NUMBERS
                            st.markdown(f"**Source:** `{match['file_name']}`, page {match['page_number']}")
            else:
                st.error(f"❌ API Engine Error: {response.json().get('detail', response.text)}")
                
        except Exception as e:
            st.error(f"❌ Connection Error: Could not establish contact with the FastAPI server backend. {str(e)}")