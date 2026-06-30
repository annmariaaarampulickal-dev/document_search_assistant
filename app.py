import streamlit as st
import requests
import os

# --- DOCKER NETWORK RESOLUTION ---
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Document Search Assistant",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Document Search Assistant")
st.markdown("Query your uploaded documents using advanced semantic pgvector search.")

st.header("📄 Upload Documents")
st.info("ℹ️ Only PDF files are accepted. Other file types will be automatically rejected.")

if st.button("🔄 Clear Uploader"):
    st.session_state["file_uploader_key"] = st.session_state.get("file_uploader_key", 0) + 1

uploaded_files = st.file_uploader(
    "Choose PDF files to store and index",
    type=["pdf"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.get('file_uploader_key', 0)}"
)

if uploaded_files:
    if st.button("Upload All Files ⬆️"):
        for uploaded_file in uploaded_files:
            with st.spinner(f"Index parsing {uploaded_file.name}..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    res = requests.post(f"{BACKEND_URL}/documents/upload", files=files)

                    if res.status_code == 201:
                        st.success(f"✅ Successfully Indexed: {uploaded_file.name}")
                    else:
                        st.error(f"❌ Upload Error on {uploaded_file.name}: {res.json().get('detail', res.text)}")
                except Exception as e:
                    st.error(f"Unable to reach the FastAPI server: {e}")

st.write("---")

# --- SEARCH SECTION ---
st.header("🔍 Search Documents")

with st.form("search_form"):
    query_text = st.text_input("🤔 What information are you looking for?", key="query")

    # ☑ AI Answer checkbox
    use_ai = st.checkbox("🤖 Generate AI written answer", value=False)

    submitted = st.form_submit_button("🔍 Search")

if submitted and query_text:

    # --- STEP 1: Always run regular semantic search ---
    with st.spinner("Executing Semantic Search..."):
        try:
            response = requests.post(f"{BACKEND_URL}/ask", json={"question": query_text})

            if response.status_code == 200:
                unique_results = response.json().get("top_3_chunks", [])

                if not unique_results:
                    st.warning("🔍 No relevant passage matches found.")
                else:
                    # --- STEP 2: If checkbox ticked, call /ask-ai first ---
                    if use_ai:
                        st.subheader("🤖 AI Answer")
                        with st.spinner("Generating AI answer..."):
                            try:
                                ai_response = requests.post(
                                    f"{BACKEND_URL}/ask-ai",
                                    json={"question": query_text},
                                    timeout=35
                                )

                                if ai_response.status_code == 200:
                                    ai_data = ai_response.json()
                                    st.success(ai_data.get("ai_answer", "No answer returned."))

                                    # Show which sources the AI used
                                    sources = ai_data.get("sources_used", [])
                                    if sources:
                                        st.markdown("**📎 Sources used by AI:**")
                                        for src in sources:
                                            st.markdown(f"- `{src['file_name']}`, page {src['page_number']}")
                                else:
                                    detail = ai_response.json().get("detail", ai_response.text)
                                    st.warning(f"⚠️ AI answer unavailable on this network. Showing document passages instead.\n\n_{detail}_")

                            except requests.exceptions.ConnectionError:
                                st.warning("⚠️ AI answer unavailable on this network. Showing document passages instead.")
                            except requests.exceptions.Timeout:
                                st.warning("⚠️ AI service timed out. Showing document passages instead.")
                            except Exception as e:
                                st.warning(f"⚠️ AI answer unavailable. Showing document passages instead.\n\n_{str(e)}_")

                        st.write("---")

                    # --- STEP 3: Always show raw passages below ---
                    st.subheader(f"📄 Top {len(unique_results)} Most Relevant Passages")

                    for i, match in enumerate(unique_results, start=1):
                        with st.expander(f"Passage Match #{i}", expanded=True):
                            display_text = " ".join(match['text'].strip().split())
                            st.info(display_text)
                            st.markdown(f"**Source:** `{match['file_name']}`, page {match['page_number']}")
                            st.markdown(f"**Similarity Score:** `{match['similarity_score']}`")

            else:
                st.error(f"❌ API Engine Error: {response.json().get('detail', response.text)}")

        except Exception as e:
            st.error(f"❌ Connection Error: Could not reach the FastAPI server. {str(e)}")