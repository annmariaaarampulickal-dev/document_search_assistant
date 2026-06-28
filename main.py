import logging
import fitz # PyMuPDF
import psycopg
import httpx
import os
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

# Groq API Key from environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

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

# 8. POST /ask - Semantic Search Endpoint using pgvector cosine similarity (UNCHANGED)
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


# 9. POST /ask-ai - NEW: AI-powered answer using top passages as context
@app.post("/ask-ai")
async def ask_ai(request: QuestionRequest):
    logging.info(f"AI answer generation requested for query: '{request.question}'")

    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if not GROQ_API_KEY:
        logging.error("GROQ_API_KEY is not set in environment variables.")
        raise HTTPException(status_code=500, detail="AI service is not configured. GROQ_API_KEY missing.")

    try:
        # Step 1: Re-use the same semantic search to get top 3 passages
        query_embedding = model.encode(
            request.question,
            normalize_embeddings=True
        ).tolist()

        with get_db_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT
                        dc.chunk_text,
                        dc.page_number,
                        d.filename,
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
            raise HTTPException(status_code=404, detail="No documents found. Please upload documents first.")

        # Step 2: Deduplicate and take top 3 passages
        unique_chunks = []
        seen_texts = set()
        for row in raw_results:
            cleaned_text = " ".join(row['chunk_text'].split())
            if cleaned_text not in seen_texts:
                seen_texts.add(cleaned_text)
                unique_chunks.append(row)
            if len(unique_chunks) == 3:
                break

        # Step 3: Build context string from the top passages
        context_parts = []
        for i, chunk in enumerate(unique_chunks, start=1):
            context_parts.append(
                f"[Passage {i} — {chunk['filename']}, page {chunk['page_number']}]\n{chunk['chunk_text'].strip()}"
            )
        context = "\n\n".join(context_parts)

        # Step 4: Call OpenAI API with context + question
        prompt = (
            f"You are a helpful assistant. Answer the user's question using ONLY the document passages provided below. "
            f"If the answer is not in the passages, say 'I could not find the answer in the uploaded documents.'\n\n"
            f"Document Passages:\n{context}\n\n"
            f"Question: {request.question}\n\n"
            f"Answer:"
        )

        async with httpx.AsyncClient(timeout=30.0) as client:
            ai_response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 500,
                    "temperature": 0.2
                }
            )

        if ai_response.status_code != 200:
            logging.error(f"Groq API returned error: {ai_response.text}")
            raise HTTPException(status_code=502, detail="AI service returned an error. Please try again.")

        ai_answer = ai_response.json()["choices"][0]["message"]["content"].strip()
        logging.info("AI answer generated successfully.")

        return {
            "ai_answer": ai_answer,
            "sources_used": [
                {"file_name": c['filename'], "page_number": c['page_number']}
                for c in unique_chunks
            ]
        }

    except HTTPException as he:
        raise he
    except httpx.ConnectError:
        logging.error("Could not connect to Groq API — network may be restricted.")
        raise HTTPException(status_code=503, detail="Could not connect to AI service. Network may be restricted.")
    except httpx.TimeoutException:
        logging.error("Groq API request timed out.")
        raise HTTPException(status_code=504, detail="AI service timed out. Please try again.")
    except Exception as e:
        logging.critical(f"Unexpected error in /ask-ai endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI processing failed: {str(e)}")