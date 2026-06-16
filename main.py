import os
import faiss
import pickle
import fitz  # PyMuPDF
import psycopg
from fastapi import FastAPI, HTTPException, status, File, UploadFile
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from psycopg.rows import dict_row
 
# Import your custom database helper
from database import get_db_connection
from utils import split_text_into_chunks
 
# 1. Initialize the FastAPI web application and Embedding Model
app = FastAPI(title="Document Search Assistant")
model = SentenceTransformer("all-MiniLM-L6-v2")
 
# Global paths for FAISS storage files
FAISS_INDEX_FILE = "vector_index.faiss"
PKL_MAPPING_FILE = "chunk_ids.pkl"
 
# Schemas for incoming request data
class DocumentCreate(BaseModel):
    filename: str
 
class QuestionRequest(BaseModel):
    question: str
 
 
@app.get("/")
def read_root():
    return {"message": "Welcome to the Document Search Assistant API!"}
 
 
# 2. POST /documents - Add a new document record safely
@app.post("/documents", status_code=status.HTTP_201_CREATED)
def create_document(payload: DocumentCreate):
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO documents (filename) VALUES (%s) RETURNING id, filename, upload_date;",
                    (payload.filename,)
                )
                new_doc = cur.fetchone()
                conn.commit()
                return new_doc
    except psycopg.OperationalError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database connection error: {e}"
        )
 
 
# 3. GET /documents - List all documents
@app.get("/documents")
def list_documents():
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM documents ORDER BY upload_date DESC;")
                return cur.fetchall()
    except psycopg.OperationalError:
        raise HTTPException(status_code=500, detail="Database connection failed.")
 
 
# 4. GET /documents/{id} - Get a single document by its ID
@app.get("/documents/{id}")
def get_document(id: int):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM documents WHERE id = %s;", (id,))
            doc = cur.fetchone()
            if not doc:
                raise HTTPException(status_code=404, detail="Document not found")
            return doc
 
 
# 5. DELETE /documents/{id} - Delete a document record
@app.delete("/documents/{id}")
def delete_document(id: int):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM documents WHERE id = %s;", (id,))
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="Document not found")
 
            cur.execute("DELETE FROM documents WHERE id = %s;", (id,))
            conn.commit()
            return {"message": f"Document {id} successfully deleted"}
 
 
# 6. POST /documents/upload - Accept PDF, chunk text, and dynamically update FAISS
@app.post("/documents/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported!")
 
    try:
        file_bytes = await file.read()
        pdf = fitz.open(stream=file_bytes, filetype="pdf")
 
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Insert master record
                cur.execute(
                    "INSERT INTO documents (filename) VALUES (%s) RETURNING id;",
                    (file.filename,)
                )
                doc_id = cur.fetchone()["id"]
 
                chunk_order = 1
                new_chunks_texts = []
                inserted_chunk_ids = []
 
                # Loop through pages page by page
                for page_idx, page in enumerate(pdf):
                    page_number = page_idx + 1
                    page_text = page.get_text()
 
                    # Split page text into overlapping segments
                    page_chunks = split_text_into_chunks(page_text, chunk_size=500, overlap=100)
 
                    # Save each small chunk into the database and collect values for FAISS
                    for chunk_text in page_chunks:
                        cur.execute(
                            """
                            INSERT INTO document_chunks (document_id, chunk_text, chunk_order, page_number)
                            VALUES (%s, %s, %s, %s) RETURNING id;
                            """,
                            (doc_id, chunk_text, chunk_order, page_number)
                        )
                        new_chunk_id = cur.fetchone()["id"]
                        inserted_chunk_ids.append(new_chunk_id)
                        new_chunks_texts.append(chunk_text)
                        chunk_order += 1
 
                conn.commit()
 
        # DYNAMIC FAISS LOGIC: Update vector index files immediately on upload
        if new_chunks_texts:
            new_vectors = model.encode(new_chunks_texts)
            
            # Load existing FAISS components or build a fresh base if missing
            if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(PKL_MAPPING_FILE):
                index = faiss.read_index(FAISS_INDEX_FILE)
                with open(PKL_MAPPING_FILE, "rb") as f:
                    chunk_ids = pickle.load(f)
            else:
                index = faiss.IndexFlatL2(384)  # 384 is the MiniLM dimension
                chunk_ids = []
 
            # Append the fresh vectors and associate them with database primary keys
            index.add(new_vectors)
            chunk_ids.extend(inserted_chunk_ids)
 
            # Persist states safely to local storage
            faiss.write_index(index, FAISS_INDEX_FILE)
            with open(PKL_MAPPING_FILE, "wb") as f:
                pickle.dump(chunk_ids, f)
 
        return {
            "message": f"Successfully processed '{file.filename}' and updated FAISS index dynamically.",
            "document_id": doc_id,
            "total_pages": len(pdf),
            "total_chunks_saved": len(new_chunks_texts)
        }
 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")
 
 
# 7. GET /documents/{id}/chunks - See all processed text chunks for a document
@app.get("/documents/{id}/chunks")
def get_document_chunks(id: int):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM documents WHERE id = %s;", (id,))
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="Document not found")
 
            cur.execute(
                "SELECT id, chunk_text, chunk_order, page_number FROM document_chunks WHERE document_id = %s ORDER BY chunk_order ASC;",
                (id,)
            )
            return cur.fetchall()
 
 
# 8. POST /ask - Semantic Search Endpoint routing requests through FAISS
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    if not os.path.exists(FAISS_INDEX_FILE) or not os.path.exists(PKL_MAPPING_FILE):
        return {"top_3_chunks": [], "message": "No vector system indexed yet. Please upload files first."}
 
    try:
        # Load index and database ID tracks
        index = faiss.read_index(FAISS_INDEX_FILE)
        with open(PKL_MAPPING_FILE, "rb") as f:
            chunk_ids = pickle.load(f)
 
        # Vector conversion and querying via FAISS
        query_vector = model.encode([request.question])
        distances, indices = index.search(query_vector, 15)
        matched_db_ids = [chunk_ids[idx] for idx in indices[0] if idx != -1]
 
        if not matched_db_ids:
            return {"top_3_chunks": []}
 
        # Query the operational database maintaining tracking order
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
 
        # Text stream de-duplication strategy
        unique_results = []
        seen_texts = set()
        for row in raw_results:
            cleaned_text = " ".join(row['chunk_text'].split())
            if cleaned_text not in seen_texts:
                seen_texts.add(cleaned_text)
                unique_results.append({
                    "file_name": row['filename'],
                    "page_number": row['page_number'],
                    "text": row['chunk_text']
                })
            if len(unique_results) == 3:
                break
 
        return {"top_3_chunks": unique_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search pipeline broken: {str(e)}")