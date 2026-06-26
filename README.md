# Document Search Assistant

A semantic document search assistant that lets you upload PDF files and search their contents using natural language questions. Built with FastAPI, PostgreSQL + pgvector, SentenceTransformers, Streamlit, and Docker — upload a PDF, ask a question in plain English, and get back the three most relevant passages with page citations. Optionally generate an AI-written answer using OpenAI.

---

## Table of Contents

1. [What It Does](#what-it-does)
2. [Tech Stack](#tech-stack)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Running the App](#running-the-app)
6. [Usage Example](#usage-example)
7. [How It Works](#how-it-works)
8. [Database Schema](#database-schema)
9. [API Reference](#api-reference)
10. [Running the Tests](#running-the-tests)
11. [Limitations and Next Steps](#limitations-and-next-steps)

---

## What It Does

Upload any PDF and the system extracts its text, splits it into overlapping chunks, converts each chunk into a semantic embedding vector using `all-MiniLM-L6-v2`, and stores everything directly in PostgreSQL using the **pgvector** extension. When you ask a question, it encodes your question the same way and finds the chunks whose meaning is closest using **cosine similarity** — returning the top 3 matching passages with their source filename and page number.

Optionally, tick the **"Generate AI written answer"** checkbox to send those passages to OpenAI and get a clean, readable answer written from your documents.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11+ | Core language |
| FastAPI | Backend web API |
| PostgreSQL + pgvector | Database and vector similarity search |
| psycopg3 | Python to PostgreSQL connector |
| SentenceTransformers | Local embedding generation (`all-MiniLM-L6-v2`) |
| PyMuPDF | PDF text extraction |
| NLTK | Sentence-aware text chunking |
| httpx | Async HTTP calls to OpenAI API |
| Streamlit | Frontend UI |
| Docker & Docker Compose | One-command deployment |
| OpenAI API | AI-written answers (optional) |

---

## Prerequisites

- **Docker** and **Docker Compose** installed
- An **OpenAI API key** (optional — only needed for AI-written answers)

That's it. Everything else runs inside Docker.

> **No Docker?** The app also runs locally with Python and PostgreSQL installed. Set up your `.env` file, install dependencies with `pip install -r requirements.txt`, then run `uvicorn main:app --reload` and `streamlit run app.py` in two separate terminals. You will also need to install pgvector into your PostgreSQL instance — follow the official guide at https://github.com/pgvector/pgvector — and enable it once with `CREATE EXTENSION vector;` in your database.

> **No OpenAI API key?** No problem. The AI answer checkbox will show a friendly "unavailable" message. Regular semantic search works perfectly without it — nothing breaks.

---

## Installation

**Step 1 — Clone the repository:**
```bash
git clone https://github.com/your-username/document-search-assistant.git
cd document-search-assistant
```

**Step 2 — Create a `.env` file** in the project root with your credentials:
```
DB_NAME=document_db
DB_USER=postgres
DB_PASSWORD=your_password_here
DB_HOST=localhost
DB_PORT=5432
OPENAI_API_KEY=sk-your-openai-key-here
```

> **Note:** When running via Docker Compose, `DB_HOST` is automatically overridden to `db` by `docker-compose.yml` — no manual change needed. Use `localhost` only when running without Docker.

> `OPENAI_API_KEY` is optional. If not set, the AI answer feature will show a friendly unavailable message — everything else works normally.

---

## Running the App

One command starts everything:

```bash
docker-compose up --build
```

> `--build` rebuilds images if code has changed. First time setup always needs `--build`. Subsequent runs can use just `docker-compose up`.

This starts three containers:
- **PostgreSQL** database with pgvector
- **FastAPI** backend at `http://localhost:8000`
- **FastAPI interactive docs** at `http://localhost:8000/docs`
- **Streamlit** frontend at `http://localhost:8501`

To stop:
```bash
docker-compose down
```

---

## Usage Example

**Uploading a PDF:**

1. Open `http://localhost:8501` in your browser
2. Under "Upload Documents", click "Browse files" and select one or more PDF files
3. Click "Upload All Files"
4. Wait for the green success message — the PDF has been parsed, chunked, embedded, and stored in pgvector
5. Use the **"Clear Uploader"** button to reset the file picker if needed

**Searching:**

1. Under "Search Documents", type a natural language question
   - Example: `What are the main responsibilities of the board of directors?`
   - Example: `What does the policy say about data privacy?`
2. Optionally tick **"Generate AI written answer"** for a clean AI response
3. Top 3 matching passages appear with source filename and page number

**Resetting everything:**

Click "Reset Entire Vector System" in the left sidebar to delete all document records and embeddings. Use this to start fresh.

---

## How It Works

### Ingest Flow (Upload)

```
PDF file uploaded
        │
        ├── Guard A: Not a .pdf extension?         → 400 error
        ├── Guard B: File is 0 bytes?               → 400 error
        ├── Guard C: PDF has 0 pages?               → 400 error
        │
        ├── Insert row into documents table
        │
        ├── For each page:
        │     Extract raw text with PyMuPDF
        │     Split into sentence-aware chunks (NLTK, ~500 chars, 1 sentence overlap)
        │     Generate embedding with SentenceTransformer
        │     Insert chunk + embedding into document_chunks (pgvector)
        │
        ├── Guard D: No extractable text found?    → 400 error
        │     (catches scanned/image-only PDFs)
        │
        ├── All uploads, searches, and errors logged to app_system.log
        │
        └── Return document_id, total_pages, total_chunks_saved
```

### Search Flow (/ask)

```
User question
        │
        ├── Encode question into normalized 384-dim vector
        │
        ├── pgvector cosine similarity search
        │     ORDER BY embedding <=> query_vector (top 15)
        │
        ├── Deduplicate by normalized text content
        │
        └── Return top 3 unique results with file name, page number, similarity score
```

### AI Answer Flow (/ask-ai)

```
User question (checkbox ticked)
        │
        ├── Same pgvector search as /ask → top 3 passages
        │
        ├── Build context string from passages
        │
        ├── Call OpenAI gpt-3.5-turbo with:
        │     "Answer using ONLY the passages provided"
        │
        ├── On network error / timeout → graceful error message
        │
        └── Return ai_answer + sources_used (file_name, page_number per source)
```

### Why pgvector instead of FAISS?

pgvector stores embeddings directly inside PostgreSQL — no separate index files, no dual-storage sync issues, no FAISS rebuild on delete. A single SQL query handles both the vector search and the metadata retrieval. This makes the system simpler, safer, and more reliable.

### Chunking Strategy

Text is split using **NLTK's sentence tokenizer** so sentences are never cut in half. Each chunk targets 500 characters with a 1-sentence overlap between consecutive chunks to preserve context at boundaries.

### Similarity Metric

Uses **cosine similarity** via pgvector's `<=>` operator with `normalize_embeddings=True`. All vectors are normalized to unit length before being stored, making cosine similarity the correct metric for `all-MiniLM-L6-v2`.

---

## Database Schema

Two tables store all document data:

**`documents`** — one row per uploaded PDF
```sql
id          SERIAL PRIMARY KEY
filename    TEXT NOT NULL
upload_date TIMESTAMP DEFAULT now()
```

**`document_chunks`** — one row per text chunk with its embedding
```sql
id          SERIAL PRIMARY KEY
document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE
chunk_text  TEXT NOT NULL
chunk_order INTEGER NOT NULL
page_number INTEGER NOT NULL
embedding   VECTOR(384)
```

`ON DELETE CASCADE` means deleting a document automatically deletes all its chunks and embeddings in one atomic operation — no orphaned data is ever left behind.

---

## API Reference

| Method | Endpoint | Description | Success Code |
|--------|----------|-------------|--------------|
| GET | `/` | Health check | 200 |
| POST | `/documents` | Manually register a document record | 201 |
| GET | `/documents` | List all documents, newest first | 200 |
| GET | `/documents/{id}` | Get a single document by ID | 200 / 404 |
| DELETE | `/documents/{id}` | Delete document and all its chunks | 200 / 404 |
| POST | `/documents/upload` | Upload PDF, chunk, embed, and store | 201 / 400 |
| GET | `/documents/{id}/chunks` | View all text chunks for a document | 200 / 404 |
| POST | `/ask` | Semantic search, returns top 3 passages | 200 |
| POST | `/ask-ai` | AI-written answer from top passages | 200 / 500 / 502 / 503 / 504 |

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
      "text": "All personal data must be handled in accordance with...",
      "similarity_score": 0.91
    },
    {
      "file_name": "company_policy.pdf",
      "page_number": 7,
      "text": "Employees are prohibited from sharing customer data...",
      "similarity_score": 0.87
    },
    {
      "file_name": "onboarding_guide.pdf",
      "page_number": 2,
      "text": "Data privacy is a core principle of our operations...",
      "similarity_score": 0.84
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
Covers: non-PDF file rejection, empty file rejection, semantic search response shape and similarity scores, empty/whitespace question rejection, unknown document ID/chunks/delete lookups, AI endpoint structure validation, health check, and document list verification.

**Unit tests — no server needed:**
```bash
pytest test_utils.py -v
```
Covers: chunk size limits, overlap behavior, overlap disabled behavior.

**OpenAI connection test — requires OPENAI_API_KEY in `.env`:**
```bash
python test_openai.py
```
Covers: verifies OpenAI API key is valid and network connection to OpenAI is available.

---

## Limitations and Next Steps

**Current limitations:**

- No authentication — any user can upload or delete documents with no access control
- No pagination — `GET /documents` returns all records with no limit
- Scanned PDFs not supported — image-only PDFs have no embedded text; OCR integration would be needed
- AI answers require an active internet connection and a valid OpenAI API key
- No connection pooling — `psycopg_pool` would be needed under real concurrent load

**Possible next steps:**

- Add user authentication via FastAPI's OAuth2 / JWT support
- Add OCR support (e.g., Tesseract) to handle scanned image-only PDFs
- Add pagination to the documents list endpoint (`LIMIT` / `OFFSET`)
- Add a connection pool using `psycopg_pool.ConnectionPool`
- Add `response_model` schemas to FastAPI endpoints for validated, documented output shapes
- Support more file types (Word, plain text)