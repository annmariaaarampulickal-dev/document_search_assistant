import unittest
import requests

BASE_URL = "http://127.0.0.1:8000"


class TestDocumentSearchPipeline(unittest.TestCase):

    # ─────────────────────────────────────────────
    # UPLOAD VALIDATION TESTS
    # ─────────────────────────────────────────────

    def test_01_invalid_file_rejection(self):
        """Verify that uploading a text file instead of a PDF blocks processing with a 400 Error"""
        bad_payload = {"file": ("unsupported_notes.txt", b"Plain text file data...", "text/plain")}
        response = requests.post(f"{BASE_URL}/documents/upload", files=bad_payload)
        self.assertEqual(response.status_code, 400)
        # Verify the error message is meaningful
        self.assertIn("detail", response.json())

    def test_02_empty_file_rejection(self):
        """Verify that uploading a zero-byte document blocks processing with a 400 Error"""
        empty_payload = {"file": ("empty_layout.pdf", b"", "application/pdf")}
        response = requests.post(f"{BASE_URL}/documents/upload", files=empty_payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("detail", response.json())

    # ─────────────────────────────────────────────
    # SEARCH TESTS
    # ─────────────────────────────────────────────

    def test_03_semantic_search_execution(self):
        """Verify that /ask returns correct structural payload with top_3_chunks and similarity scores"""
        search_payload = {"question": "What are the core corporate operational guidelines?"}
        response = requests.post(f"{BASE_URL}/ask", json=search_payload)

        self.assertEqual(response.status_code, 200)

        data = response.json()

        # Verify top_3_chunks key exists and is a list
        self.assertIn("top_3_chunks", data)
        self.assertIsInstance(data["top_3_chunks"], list)

        # If results exist, verify each result has correct fields
        if data["top_3_chunks"]:
            first_result = data["top_3_chunks"][0]
            self.assertIn("file_name", first_result)
            self.assertIn("page_number", first_result)
            self.assertIn("text", first_result)
            self.assertIn("similarity_score", first_result)

            # Similarity score should be between 0 and 1
            self.assertGreaterEqual(first_result["similarity_score"], 0.0)
            self.assertLessEqual(first_result["similarity_score"], 1.0)

    def test_04_empty_question_rejection(self):
        """Verify that sending an empty question returns a 400 error"""
        response = requests.post(f"{BASE_URL}/ask", json={"question": ""})
        self.assertEqual(response.status_code, 400)
        self.assertIn("detail", response.json())

    def test_05_whitespace_question_rejection(self):
        """Verify that sending a whitespace-only question returns a 400 error"""
        response = requests.post(f"{BASE_URL}/ask", json={"question": "   "})
        self.assertEqual(response.status_code, 400)

    # ─────────────────────────────────────────────
    # DOCUMENT LOOKUP TESTS
    # ─────────────────────────────────────────────

    def test_06_unknown_document_id_lookup(self):
        """Verify that looking up a non-existent document ID triggers a clean 404 error"""
        dummy_id = 888888
        response = requests.get(f"{BASE_URL}/documents/{dummy_id}")
        self.assertEqual(response.status_code, 404)
        self.assertIn("detail", response.json())

    def test_07_unknown_document_chunks_lookup(self):
        """Verify that getting chunks for a non-existent document returns 404"""
        dummy_id = 888888
        response = requests.get(f"{BASE_URL}/documents/{dummy_id}/chunks")
        self.assertEqual(response.status_code, 404)

    def test_08_unknown_document_delete(self):
        """Verify that deleting a non-existent document returns 404"""
        dummy_id = 888888
        response = requests.delete(f"{BASE_URL}/documents/{dummy_id}")
        self.assertEqual(response.status_code, 404)

    # ─────────────────────────────────────────────
    # AI ENDPOINT TESTS
    # ─────────────────────────────────────────────

    def test_09_ask_ai_empty_question_rejection(self):
        """Verify that /ask-ai rejects empty questions with 400"""
        response = requests.post(f"{BASE_URL}/ask-ai", json={"question": ""})
        self.assertEqual(response.status_code, 400)

    def test_10_ask_ai_response_structure(self):
        """
        Verify /ask-ai returns correct structure.
        On restricted networks it returns 503 — both outcomes are valid.
        The important thing is the response is structured, not a crash.
        """
        response = requests.post(
            f"{BASE_URL}/ask-ai",
            json={"question": "What are the corporate guidelines?"},
            timeout=40
        )

        # Valid outcomes:
        # 200 — AI answer generated successfully
        # 503 — Network restricted (OpenAI blocked)
        # 500 — OPENAI_API_KEY not configured
        # 502 — OpenAI returned an error
        # 504 — OpenAI timed out
        self.assertIn(response.status_code, [200, 500, 502, 503, 504])

        # Either way, response should be valid JSON with a detail or answer field
        data = response.json()
        self.assertIsInstance(data, dict)

        if response.status_code == 200:
            # Success — verify correct fields
            self.assertIn("ai_answer", data)
            self.assertIn("sources_used", data)
            self.assertIsInstance(data["sources_used"], list)
            self.assertIsInstance(data["ai_answer"], str)
            self.assertGreater(len(data["ai_answer"]), 0)
        else:
            # Error — verify detail field exists
            self.assertIn("detail", data)

    # ─────────────────────────────────────────────
    # HEALTH CHECK TEST
    # ─────────────────────────────────────────────

    def test_11_health_check(self):
        """Verify the root endpoint is reachable and returns welcome message"""
        response = requests.get(f"{BASE_URL}/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("message", response.json())

    # ─────────────────────────────────────────────
    # DOCUMENT LIST TEST
    # ─────────────────────────────────────────────

    def test_12_list_documents(self):
        """Verify GET /documents returns a list"""
        response = requests.get(f"{BASE_URL}/documents")
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), list)


if __name__ == "__main__":
    print("🚀 Initiating Automated Pipeline Verification Test Routines...")
    unittest.main()
 