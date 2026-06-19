import os
import faiss
import pickle
import logging  # NEW FEATURE REQUIREMENT
import fitz # PyMuPDF
import psycopg
from fastapi import FastAPI, HTTPException, status, File, UploadFile
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from psycopg.rows import dict_row
 
# Import your exact custom database helpers and chunking tools
from database import get_db_connection
from utils import split_text_into_chunks
 
# 1. SETUP PYTHON LOGGING HANDLERS (Saves to file and streams to command terminal console)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app_system.log"),
        logging.StreamHandler()
    ]
)
 
# Initialize the FastAPI web application and Embedding Model exactly as before
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
    logging.info("Root base API verification check pinged.")
    return {"message": "Welcome to the Document Search Assistant API!"}
 
# 2. POST /documents - Add a new document record safely
@app.post("/documents", status_code=status.HTTP_201_CREATED)
def create_document(payload: DocumentCreate):
    logging.info(f"Manually creating database entry record for: '{payload.filename}'")
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO documents (filename) VALUES (%s) RETURNING id, filename, upload_date;",
                    (payload.filename,)
                )
                new_doc = cur.fetchone()
                conn.commit()
                logging.info(f"Manual entry saved successfully with assigned attributes: {new_doc}")
                return new_doc
    except psycopg.OperationalError as e:
        logging.error(f"Database access failure during manual document entry registration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database connection error: {e}"
        )
 
# 3. GET /documents - List all documents
@app.get("/documents")
def list_documents():
    logging.info("Retrieving full registry list of all cataloged documents.")
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM documents ORDER BY upload_date DESC;")
                return cur.fetchall()
    except psycopg.OperationalError as e:
        logging.error(f"Database list retrieval dropped connection link: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed.")
 
# 4. GET /documents/{id} - Get a single document by its ID
@app.get("/documents/{id}")
def get_document(id: int):
    logging.info(f"Targeted inquiry lookup for Document Registry ID: {id}")
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM documents WHERE id = %s;", (id,))
            doc = cur.fetchone()
            if not doc:
                logging.warning(f"Lookup aborted: Document tracking ID {id} not present in database.")
                raise HTTPException(status_code=404, detail="Document not found")
            return doc
 
# 5. DELETE /documents/{id} - Delete a document record
@app.delete("/documents/{id}")
def delete_document(id: int):
    logging.info(f"Targeted deletion execution request for Document ID: {id}")
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM documents WHERE id = %s;", (id,))
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="Document not found")

            # Delete from Postgres — ON DELETE CASCADE removes chunks automatically
            cur.execute("DELETE FROM documents WHERE id = %s;", (id,))
            conn.commit()
            logging.info(f"Document ID {id} deleted from Postgres.")

            # Rebuild FAISS from whatever chunks still remain in the database
            cur.execute("SELECT id, chunk_text FROM document_chunks ORDER BY id;")
            remaining_chunks = cur.fetchall()

    if remaining_chunks:
        logging.info(f"Rebuilding FAISS index with {len(remaining_chunks)} remaining chunks.")
        texts = [row["chunk_text"] for row in remaining_chunks]
        ids = [row["id"] for row in remaining_chunks]

        new_vectors = model.encode(texts,normalize_embeddings=True)
        new_index = faiss.IndexFlatIP(384)
        new_index.add(new_vectors)

        faiss.write_index(new_index, FAISS_INDEX_FILE)
        with open(PKL_MAPPING_FILE, "wb") as f:
            pickle.dump(ids, f)

        logging.info("FAISS index rebuilt and saved successfully.")
    else:
        # No chunks left at all — delete the files entirely so /ask
        # correctly returns "no index yet" rather than searching an empty index
        if os.path.exists(FAISS_INDEX_FILE):
            os.remove(FAISS_INDEX_FILE)
        if os.path.exists(PKL_MAPPING_FILE):
            os.remove(PKL_MAPPING_FILE)
        logging.info("No chunks remaining — FAISS index files cleared.")

    return {"message": f"Document {id} successfully deleted and FAISS index updated."}
# 6. POST /documents/upload - Accept PDF, chunk text, and dynamically update FAISS
@app.post("/documents/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    logging.info(f"Incoming storage network packet processing for document file: '{file.filename}'")
    
    # REQUIREMENT ERROR HANDLING GUARD A: Verify file extension type matches PDF bounds
    if not file.filename.lower().endswith('.pdf'):
        logging.error(f"Processing Aborted: File '{file.filename}' is not a valid PDF file container.")
        raise HTTPException(status_code=400, detail="Only PDF files are supported!")
 
    try:
        file_bytes = await file.read()
        
        # REQUIREMENT ERROR HANDLING GUARD B: Block empty 0-byte file stream allocations
        if len(file_bytes) == 0:
            logging.error(f"Processing Aborted: Incoming stream package for '{file.filename}' contains 0 bytes.")
            raise HTTPException(status_code=400, detail="The uploaded PDF file is empty and cannot be indexed.")
 
        pdf = fitz.open(stream=file_bytes, filetype="pdf")
        
        # REQUIREMENT ERROR HANDLING GUARD C: Verify PDF contains readable pages with extractable characters
        if len(pdf) == 0:
            logging.error(f"Processing Aborted: Document structure '{file.filename}' contains 0 printable pages.")
            raise HTTPException(status_code=400, detail="The uploaded file contains no readable document layout pages.")
 
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Insert master record exactly as you designed it
                cur.execute(
                    "INSERT INTO documents (filename) VALUES (%s) RETURNING id;",
                    (file.filename,)
                )
                doc_id = cur.fetchone()["id"]
 
                chunk_order = 1
                new_chunks_texts = []
                inserted_chunk_ids = []
 
                # Loop through pages page by page using your exact loop logic
                for page_idx, page in enumerate(pdf):
                    page_number = page_idx + 1
                    page_text = page.get_text()
 
                    # Split page text into overlapping segments using your custom token utils
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

            conn.commit()

        # REQUIREMENT ERROR HANDLING GUARD D: Block scanned image-only PDFs with no extractable text
        if not new_chunks_texts:
            logging.error(f"Processing Aborted: '{file.filename}' contains no extractable text content.")
            raise HTTPException(
                status_code=400,
                detail="The PDF contains no extractable text. It may be a scanned image-only PDF."
            )
 
        # DYNAMIC FAISS LOGIC: Update vector index files immediately on upload
        if new_chunks_texts:
            logging.info(f"Vectorizing {len(new_chunks_texts)} extracted text chunks via SentenceTransformers.")
            new_vectors = model.encode(new_chunks_texts,normalize_embeddings=True)
 
            # Load existing FAISS components or build a fresh base if missing
            if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(PKL_MAPPING_FILE):
                index = faiss.read_index(FAISS_INDEX_FILE)
                with open(PKL_MAPPING_FILE, "rb") as f:
                    chunk_ids = pickle.load(f)
            else:
                index = faiss.IndexFlatIP(384) # 384 is the MiniLM dimension
                chunk_ids = []
 
            # Append the fresh vectors and associate them with database primary keys
            index.add(new_vectors)
            chunk_ids.extend(inserted_chunk_ids)
 
            # Persist states safely to local storage
            faiss.write_index(index, FAISS_INDEX_FILE)
            with open(PKL_MAPPING_FILE, "wb") as f:
                pickle.dump(chunk_ids, f)
 
        logging.info(f"Successfully processed, vectorized, and index-mapped file: {file.filename}")
        return {
            "message": f"Successfully processed '{file.filename}' and updated FAISS index dynamically.",
            "document_id": doc_id,
            "total_pages": len(pdf),
            "total_chunks_saved": len(new_chunks_texts)
        }
 
    except HTTPException as he:
        raise he
    except Exception as e:
        logging.critical(f"Unexpected processing system breakdown running data ingestion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")
 
# 7. GET /documents/{id}/chunks - See all processed text chunks for a document
@app.get("/documents/{id}/chunks")
def get_document_chunks(id: int):
    logging.info(f"Requesting exhaustive list of structural text chunks for Document ID: {id}")
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM documents WHERE id = %s;", (id,))
            if not cur.fetchone():
                logging.warning(f"Chunks lookup cancelled: ID {id} does not map to an existing record.")
                raise HTTPException(status_code=404, detail="Document not found")
 
            cur.execute(
                "SELECT id, chunk_text, chunk_order, page_number FROM document_chunks WHERE document_id = %s ORDER BY chunk_order ASC;",
                (id,)
            )
            return cur.fetchall()
 
# 8. POST /ask - Semantic Search Endpoint routing requests through FAISS
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    logging.info(f"Running semantic search query for query input: '{request.question}'")
    
    if not os.path.exists(FAISS_INDEX_FILE) or not os.path.exists(PKL_MAPPING_FILE):
        logging.warning("Search halted: Local indices files missing from directory cache.")
        return {"top_3_chunks": [], "message": "No vector system indexed yet. Please upload files first."}
 
    try:
        # Load index and database ID tracks
        index = faiss.read_index(FAISS_INDEX_FILE)
        with open(PKL_MAPPING_FILE, "rb") as f:
            chunk_ids = pickle.load(f)
 
        # Vector conversion and querying via FAISS using your exact model call
        query_vector = model.encode([request.question],normalize_embeddings=True)
        distances, indices = index.search(query_vector, 15)
        matched_db_ids = [chunk_ids[idx] for idx in indices[0] if idx != -1]
 
        if not matched_db_ids:
            logging.info(f"No coordinate boundaries located inside FAISS space for string query: '{request.question}'")
            return {"top_3_chunks": []}
 
        # Query the operational database maintaining tracking order via your custom SQL sorting parameters
        with get_db_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                format_ids = ", ".join(str(idx) for idx in matched_db_ids)
                query_sql = f"""
                    SELECT dc.chunk_text, dc.page_number, d.filename, dc.id
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id
                    WHERE dc.id IN ({format_ids})
                    ORDER BY array_position(ARRAY[{format_ids}]::integer[], dc.id);
                """
                cur.execute(query_sql)
                raw_results = cur.fetchall()
 
        # Text stream de-duplication strategy exactly as you designed it
        unique_results = []
        seen_texts = set()
        
        for row in raw_results:
            # REQUIREMENT UNKNOWN ID FILTER: Safely skip ghost entries if they are removed from SQL
            if not row:
                continue
                
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
 
        logging.info(f"Successfully located {len(unique_results)} relevant citations for query input.")
        return {"top_3_chunks": unique_results}
        
    except Exception as e:
        logging.error(f"Search retrieval execution circuit failure: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search pipeline broken: {str(e)}")