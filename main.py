import logging
import fitz # PyMuPDF
import psycopg
from fastapi import FastAPI, HTTPException, status, File, UploadFile
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from psycopg.rows import dict_row

from database import get_db_connection
from utils import split_text_into_chunks

# 1. SETUP PYTHON LOGGING HANDLERS
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app_system.log"),
        logging.StreamHandler()
    ]
)

# Initialize the FastAPI web application and Embedding Model
app = FastAPI(title="Document Search Assistant")
model = SentenceTransformer("all-MiniLM-L6-v2")

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

            cur.execute("DELETE FROM documents WHERE id = %s;", (id,))
            conn.commit()
            logging.info(f"Document ID {id} and all its embeddings deleted cleanly via CASCADE.")
            return {"message": f"Document {id} successfully deleted"}

# 6. POST /documents/upload - Accept PDF, chunk text, store embeddings in pgvector
@app.post("/documents/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    logging.info(f"Incoming storage network packet processing for document file: '{file.filename}'")

    # GUARD A: Verify file extension is PDF
    if not file.filename.lower().endswith('.pdf'):
        logging.error(f"Processing Aborted: File '{file.filename}' is not a valid PDF file container.")
        raise HTTPException(status_code=400, detail="Only PDF files are supported!")

    try:
        file_bytes = await file.read()

        # GUARD B: Block empty 0-byte files
        if len(file_bytes) == 0:
            logging.error(f"Processing Aborted: Incoming stream package for '{file.filename}' contains 0 bytes.")
            raise HTTPException(status_code=400, detail="The uploaded PDF file is empty and cannot be indexed.")

        pdf = fitz.open(stream=file_bytes, filetype="pdf")

        # GUARD C: Verify PDF contains pages
        if len(pdf) == 0:
            logging.error(f"Processing Aborted: Document structure '{file.filename}' contains 0 printable pages.")
            raise HTTPException(status_code=400, detail="The uploaded file contains no readable document layout pages.")

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Insert master document record
                cur.execute(
                    "INSERT INTO documents (filename) VALUES (%s) RETURNING id;",
                    (file.filename,)
                )
                doc_id = cur.fetchone()["id"]

                chunk_order = 1
                total_chunks = 0

                # Loop through pages page by page
                for page_idx, page in enumerate(pdf):
                    page_number = page_idx + 1
                    page_text = page.get_text()
                    page_chunks = split_text_into_chunks(page_text, chunk_size=500, overlap=100)

                    for chunk_text in page_chunks:
                        # PGVECTOR: encode each chunk and store embedding directly in INSERT
                        embedding = model.encode(
                            chunk_text,
                            normalize_embeddings=True
                        ).tolist()

                        cur.execute(
                            """
                            INSERT INTO document_chunks
                                (document_id, chunk_text, chunk_order, page_number, embedding)
                            VALUES (%s, %s, %s, %s, %s)
                            RETURNING id;
                            """,
                            (doc_id, chunk_text, chunk_order, page_number, embedding)
                        )
                        chunk_order += 1
                        total_chunks += 1

                conn.commit()

        # GUARD D: Block scanned image-only PDFs with no extractable text
        if total_chunks == 0:
            logging.error(f"Processing Aborted: '{file.filename}' contains no extractable text content.")
            raise HTTPException(
                status_code=400,
                detail="The PDF contains no extractable text. It may be a scanned image-only PDF."
            )

        logging.info(f"Successfully processed and stored embeddings for: {file.filename}")
        return {
            "message": f"Successfully processed '{file.filename}' with pgvector embeddings.",
            "document_id": doc_id,
            "total_pages": len(pdf),
            "total_chunks_saved": total_chunks
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

# 8. POST /ask - Semantic Search Endpoint using pgvector cosine similarity
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    logging.info(f"Running semantic search query for query input: '{request.question}'")

    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        # Step 1: Encode the question into a normalized vector
        query_embedding = model.encode(
            request.question,
            normalize_embeddings=True
        ).tolist()

        # Step 2: ONE SQL query — cosine similarity search directly in Postgres
        with get_db_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT
                        dc.chunk_text,
                        dc.page_number,
                        d.filename,
                        dc.id,
                        1 - (dc.embedding <=> %s::vector) AS similarity_score
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id
                    WHERE dc.embedding IS NOT NULL
                    ORDER BY dc.embedding <=> %s::vector
                    LIMIT 15;
                    """,
                    (query_embedding, query_embedding)
                )
                raw_results = cur.fetchall()

        if not raw_results:
            logging.info("No results found in pgvector search.")
            return {"top_3_chunks": [], "message": "No relevant results found. Please upload documents first."}

        # Step 3: Deduplicate by normalized text content
        unique_results = []
        seen_texts = set()

        for row in raw_results:
            if not row:
                continue
            cleaned_text = " ".join(row['chunk_text'].split())
            if cleaned_text not in seen_texts:
                seen_texts.add(cleaned_text)
                unique_results.append({
                    "file_name": row['filename'],
                    "page_number": row['page_number'],
                    "text": row['chunk_text'],
                    "similarity_score": round(float(row['similarity_score']), 4)
                })
            if len(unique_results) == 3:
                break

        logging.info(f"Successfully located {len(unique_results)} relevant citations for query input.")
        return {"top_3_chunks": unique_results}

    except Exception as e:
        logging.error(f"Search retrieval execution circuit failure: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search pipeline broken: {str(e)}")