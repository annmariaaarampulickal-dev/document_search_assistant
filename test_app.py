import unittest
import requests
 
BASE_URL = "http://127.0.0.1:8000"
 
class TestDocumentSearchPipeline(unittest.TestCase):
 
    def test_01_invalid_file_rejection(self):
        """Verify that uploading a text file instead of a PDF blocks processing with a 400 Error"""
        bad_payload = {"file": ("unsupported_notes.txt", b"Plain text file data...", "text/plain")}
        response = requests.post(f"{BASE_URL}/documents/upload", files=bad_payload)
        self.assertEqual(response.status_code, 400)
 
    def test_02_empty_file_rejection(self):
        """Verify that uploading a zero-byte document blocks processing with a 400 Error"""
        empty_payload = {"file": ("empty_layout.pdf", b"", "application/pdf")}
        response = requests.post(f"{BASE_URL}/documents/upload", files=empty_payload)
        self.assertEqual(response.status_code, 400)
 
    def test_03_semantic_search_execution(self):
        """Verify that sending queries to the /ask endpoint triggers a proper structural payload response"""
        search_payload = {"question": "What are the core corporate operational guidelines?"}
        response = requests.post(f"{BASE_URL}/ask", json=search_payload)
        
        self.assertEqual(response.status_code, 200)
        self.assertIn("top_3_chunks", response.json())
        self.assertIsInstance(response.json()["top_3_chunks"], list)
 
    def test_04_unknown_document_id_lookup(self):
        """Verify that looking up a non-existent document tracking ID triggers a clean 404 error response"""
        dummy_id = 888888  # Evaluates an ID that doesn't exist in PostgreSQL records
        response = requests.get(f"{BASE_URL}/documents/{dummy_id}")
        self.assertEqual(response.status_code, 404)
 
if __name__ == "__main__":
    print("🚀 Initiating Automated Pipeline Verification Test Routines...")
    unittest.main()
 