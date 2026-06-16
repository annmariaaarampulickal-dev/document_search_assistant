from fastapi import FastAPI, HTTPException, status, File, UploadFile
from pydantic import BaseModel
from database import get_db_connection
from utils import split_text_into_chunks
import psycopg
import fitz  # PyMuPDF
 
# 1. Initialize the FastAPI web application (Crucial line!)
app = FastAPI(title="Document Search Assistant")
 
# Define a schema for incoming POST data
class DocumentCreate(BaseModel):
    filename: str
 
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
            detail=f"Database connection error: Make sure your password is correct! Details: {e}"
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
 
# 6. POST /documents/upload - Accept PDF, chunk text with overlap, and save to DB
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
                total_chunks_created = 0
                
                # Loop through pages page by page
                for page_idx, page in enumerate(pdf):
                    page_number = page_idx + 1
                    page_text = page.get_text()
                    
                    # Split this specific page's text into small overlapping pieces
                    page_chunks = split_text_into_chunks(page_text, chunk_size=500, overlap=100)
                    
                    # Save each small chunk into the document_chunks table
                    for chunk_text in page_chunks:
                        cur.execute(
                            """
                            INSERT INTO document_chunks (document_id, chunk_text, chunk_order, page_number)
                            VALUES (%s, %s, %s, %s);
                            """,
                            (doc_id, chunk_text, chunk_order, page_number)
                        )
                        chunk_order += 1
                        total_chunks_created += 1
                
                conn.commit()
                
        return {
            "message": f"Successfully processed '{file.filename}'",
            "document_id": doc_id,
            "total_pages": len(pdf),
            "total_chunks_saved": total_chunks_created
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
 

# =====================================================================
# NEW ADDITION: POST /ask Endpoint using NumPy Cosine Similarity
# =====================================================================
from pydantic import BaseModel
import numpy as np
from sentence_transformers import SentenceTransformer
from psycopg.rows import dict_row
 
# Define the input schema for the endpoint
class QuestionRequest(BaseModel):
    question: str
 
# Re-initialize the model globally here if it's not at the top of your file
model = SentenceTransformer("all-MiniLM-L6-v2")
 
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
 
    try:
        # Step 1: Turn the live user question into an embedding vector
        query_embedding = model.encode(request.question)
 
        # Step 2: Grab ALL chunks currently sitting in your live database
        with get_db_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("""
                    SELECT dc.chunk_text, dc.page_number, d.filename
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id
                """)
                stored_chunks = cur.fetchall()
 
        if not stored_chunks:
            return {"top_3_chunks": [], "message": "No document chunks found in database. Please upload a PDF first."}
 
        # Step 3: Extract texts and generate vector embeddings for them
        texts = [row["chunk_text"] for row in stored_chunks]
        chunk_embeddings = model.encode(texts)
        
        # Step 4: Calculate Cosine Similarity via NumPy Dot Product
        # (This is the "couple of lines" mentioned in the assignment tip!)
        scores = np.dot(chunk_embeddings, query_embedding)
        
        # Step 5: Extract the top 3 highest scoring indices (reversing to descending order)
        top_indices = np.argsort(scores)[-3:][::-1]
 
        # Step 6: Package the exact matches from your uploaded file
        unique_results = []
        for idx in top_indices:
            unique_results.append({
                "file_name": stored_chunks[idx]["filename"],
                "page_number": stored_chunks[idx]["page_number"],
                "text": stored_chunks[idx]["chunk_text"]
            })
 
        return {"top_3_chunks": unique_results}
 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")