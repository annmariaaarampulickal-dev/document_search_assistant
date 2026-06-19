# Document Search Assistant

A semantic document search backend that lets you upload PDF files and search their contents using natural language questions. Built with FastAPI, PostgreSQL, FAISS, and SentenceTransformers — upload a PDF, ask a question in plain English, and get back the three most relevant passages with page citations.

---

## Table of Contents

1. [What It Does](#what-it-does)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Database Setup](#database-setup)
5. [Running the App](#running-the-app)
6. [Usage Example](#usage-example)
7. [How It Works](#how-it-works)
8. [API Reference](#api-reference)
9. [Running the Tests](#running-the-tests)
10. [Limitations and Next Steps](#limitations-and-next-steps)

---

## What It Does

Upload any PDF and the system extracts its text, splits it into sentence-aware overlapping chunks, converts each chunk into a semantic embedding vector using `all-MiniLM-L6-v2`, and stores everything in PostgreSQL (text and metadata) and a FAISS vector index (embeddings). When you ask a question, it encodes your question the same way and finds the chunks whose meaning is closest using **cosine similarity** — returning the top 3 matching passages with their source filename and page number.

---

## Prerequisites

Make sure the following are installed before starting:

- **Python 3.10 or higher**
- **PostgreSQL 14 or higher** with a database named `document_db`
- **pgAdmin** (optional) for viewing the database visually
- **pip** Python package manager

---

## Installation

**Step 1 — Clone the repository:**
```bash
git clone https://github.com/your-username/document-search-assistant.git
cd document-search-assistant
```

**Step 2 — Create and activate a virtual environment:**
```bash
# Create the environment
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on Mac/Linux:
source venv/bin/activate
```

**Step 3 — Install all dependencies:**
```bash
pip install -r requirements.txt
```

The `requirements.txt` should contain:
```
fastapi
uvicorn
psycopg[binary]
pydantic
sentence-transformers
faiss-cpu
PyMuPDF
nltk
streamlit
requests
pytest
python-dotenv
```

**Step 4 — Download the NLTK sentence tokenizer data:**
```bash
python -c "import nltk; nltk.download('punkt')"
```

**Step 5 — Set up your environment variables:**

Create a `.env` file in the project root folder with your database credentials:
```
DB_NAME=document_db
DB_USER=postgres
DB_PASSWORD=your_password_here
DB_HOST=localhost
DB_PORT=5432
```
Replace `your_password_here` with your actual PostgreSQL password. This file is listed in `.gitignore` and will never be pushed to GitHub.

---

## Database Setup

**Step 1 — Open pgAdmin** or any PostgreSQL client and connect to your server.

**Step 2 — Create the database:**
```sql
CREATE DATABASE document_db;
```

**Step 3 — Connect to `document_db` and run the schema:**
```sql
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    upload_date TIMESTAMP DEFAULT now()
);

CREATE TABLE document_chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    chunk_order INTEGER NOT NULL,
    page_number INTEGER
);
```

The `ON DELETE CASCADE` means deleting a document automatically deletes all its chunks from the database. The FAISS index is also automatically rebuilt after every deletion to stay in sync.

---

## Running the App

You need **two separate terminals** running at the same time.

**Terminal 1 — Start the FastAPI backend:**
```bash
uvicorn main:app --reload
```
API available at: `http://127.0.0.1:8000`
Interactive API docs at: `http://127.0.0.1:8000/docs`

**Terminal 2 — Start the Streamlit frontend:**
```bash
streamlit run app.py
```
UI opens automatically at: `http://localhost:8501`

> Both processes must be running simultaneously. The Streamlit frontend communicates with the FastAPI backend over HTTP — if the backend is not running, uploads and searches will fail with a connection error.

---

## Usage Example

**Uploading a PDF:**

1. Open `http://localhost:8501` in your browser
2. Under "Upload Documents", click "Browse files" and select one or more PDF files
3. Click "Upload All Files to FAISS"
4. Wait for the green success message — the PDF has been parsed, chunked, embedded, and indexed

**Searching:**

1. Under "Search Documents via FAISS Index", type a natural language question
   - Example: `What are the main responsibilities of the board of directors?`
   - Example: `What does the policy say about data privacy?`
2. Up to 3 matching passages appear automatically, each showing the source filename and page number

**Resetting everything:**

Click "Reset Entire Vector System" in the left sidebar to delete all document records and clear the index. Use this to start fresh before re-uploading documents.

---

## How It Works

### Ingest Flow (Upload)

```
PDF file uploaded
        │
        ├── Guard A: Not a .pdf extension?              → 400 error
        ├── Guard B: File is 0 bytes?                   → 400 error
        ├── Guard C: PDF has 0 pages?                   → 400 error
        │
        ├── Insert row into documents table
        │
        ├── For each page:
        │     Extract raw text with PyMuPDF
        │     Split into overlapping chunks (NLTK sentence tokenizer)
        │     Insert each chunk into document_chunks table
        │
        ├── Commit entire transaction atomically
        │
        ├── Guard D: No extractable text found?         → 400 error
        │     (catches scanned/image-only PDFs)
        │
        └── Encode all chunk texts into 384-dim normalized vectors
              Load existing FAISS index or create fresh IndexFlatIP(384)
              Add new vectors to index
              Save index to vector_index.faiss
              Save chunk ID mapping to chunk_ids.pkl
```

### Search Flow (Ask)

```
User question
        │
        ├── Encode question into 384-dim normalized vector
        │
        ├── FAISS index.search(query_vector, 15)
        │     Returns 15 closest vector positions
        │
        ├── Translate positions → real database chunk IDs
        │     via chunk_ids.pkl mapping list
        │
        ├── SQL: fetch chunk text, page number, filename
        │     JOIN document_chunks + documents
        │     ORDER BY original FAISS relevance ranking
        │
        ├── Deduplicate by normalized text content
        │
        └── Return top 3 unique results with citations
```

### Delete Flow (FAISS Sync)

```
DELETE /documents/{id}
        │
        ├── Check document exists → 404 if not
        │
        ├── DELETE FROM documents WHERE id = ?
        │     ON DELETE CASCADE removes all chunks automatically
        │
        ├── conn.commit()
        │
        ├── Fetch all remaining chunks from database
        │
        ├── If chunks remain:
        │     Re-encode all remaining chunk texts (normalized)
        │     Rebuild fresh IndexFlatIP(384) from scratch
        │     Write new vector_index.faiss and chunk_ids.pkl
        │
        └── If no chunks remain:
              Delete vector_index.faiss and chunk_ids.pkl entirely
              Next search will correctly return "no index yet"
```

### Why Two Storage Systems?

PostgreSQL stores the actual text and metadata — good for exact lookups, relationships, and transactional safety. FAISS stores embedding vectors and answers "what is semantically similar to this?" — something SQL cannot do natively. They work together: FAISS finds the closest matches by meaning, PostgreSQL retrieves the actual content.

### Chunking Strategy

Text is split using NLTK's sentence tokenizer rather than naive character splitting, so sentences are never cut in half. Each chunk targets 500 characters with a 1-sentence overlap between consecutive chunks to preserve context at boundaries.

### Similarity Metric

Uses **cosine similarity** via `IndexFlatIP` with `normalize_embeddings=True`. All vectors are normalized to unit length before being stored, so the inner product operation becomes mathematically identical to cosine similarity. Cosine similarity measures the angle between two vectors — semantically similar text points in the same direction in vector space regardless of magnitude, making it the correct metric for `all-MiniLM-L6-v2` which was trained and evaluated against cosine similarity.

### Secure Credential Handling

Database credentials are loaded from a `.env` file via `python-dotenv` and never hardcoded in source code. The `.env` file is listed in `.gitignore` and is never committed to version control.

---

## API Reference

| Method | Endpoint | Description | Success Code |
|--------|----------|-------------|--------------|
| GET | `/` | Health check | 200 |
| POST | `/documents` | Manually register a document record | 201 |
| GET | `/documents` | List all documents, newest first | 200 |
| GET | `/documents/{id}` | Get a single document by ID | 200 / 404 |
| DELETE | `/documents/{id}` | Delete document and rebuild FAISS index | 200 / 404 |
| POST | `/documents/upload` | Upload PDF, chunk, embed, and index | 201 / 400 |
| GET | `/documents/{id}/chunks` | View all text chunks for a document | 200 / 404 |
| POST | `/ask` | Semantic search, returns top 3 passages | 200 |

**Upload error cases:**

| Situation | Guard | Status | Message |
|-----------|-------|--------|---------|
| Non-PDF file | A | 400 | Only PDF files are supported! |
| Empty file (0 bytes) | B | 400 | The uploaded PDF file is empty and cannot be indexed. |
| PDF with 0 pages | C | 400 | The uploaded file contains no readable document layout pages. |
| Scanned/image-only PDF | D | 400 | The PDF contains no extractable text. It may be a scanned image-only PDF. |
| Unknown document ID | — | 404 | Document not found |

**Example search request:**
```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the data privacy guidelines?"}'
```

**Example search response:**
```json
{
  "top_3_chunks": [
    {
      "file_name": "company_policy.pdf",
      "page_number": 4,
      "text": "All personal data must be handled in accordance with..."
    },
    {
      "file_name": "company_policy.pdf",
      "page_number": 7,
      "text": "Employees are prohibited from sharing customer data..."
    },
    {
      "file_name": "onboarding_guide.pdf",
      "page_number": 2,
      "text": "Data privacy is a core principle of our operations..."
    }
  ]
}
```

---

## Running the Tests

**Integration tests — requires the FastAPI server to be running:**
```bash
python test_app.py
```
Covers: non-PDF file rejection, empty file rejection, semantic search response shape, unknown document ID lookup.

**Unit tests — no server needed:**
```bash
pytest test_utils.py -v
```
Covers: chunk size limits, sentence-level overlap behavior.

---

## Limitations and Next Steps

**Current limitations:**

- No authentication — any user can upload or delete documents with no access control
- No pagination — `GET /documents` returns all records with no limit
- Scanned PDFs not supported — image-only PDFs have no embedded text; OCR integration would be needed
- FAISS rebuild on delete re-encodes all remaining chunks — acceptable at small scale, slow at large scale
- No connection pooling — a new database connection opens per request; `psycopg_pool` would be needed under real concurrent load
- Concurrent uploads can cause a FAISS write conflict since both requests read and overwrite the same index file

**Possible next steps:**

- Add user authentication via FastAPI's OAuth2 / JWT support
- Replace FAISS files with **pgvector** to store embeddings inside PostgreSQL, eliminating the dual-storage consistency problem entirely
- Add OCR support (e.g., Tesseract) to handle scanned image-only PDFs
- Add pagination to the documents list endpoint (`LIMIT` / `OFFSET`)
- Add a connection pool using `psycopg_pool.ConnectionPool`
- Add `response_model` schemas to FastAPI endpoints for validated, documented output shapes
- Add a submit button to the Streamlit search box to avoid firing a search request on every keystroke
