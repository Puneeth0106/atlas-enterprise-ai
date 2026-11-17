import io
import unittest
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import app
from app.demo_context import Principal
from worker.ingest import process_s3_key


class UploadCompletionTests(unittest.TestCase):
    def setUp(self):
        self.http = TestClient(app)
        self.document_id = uuid4()
        self.payload = {"document_id": str(self.document_id)}

    @patch("app.main.connection")
    @patch("app.main.process_s3_key")
    def test_completion_rejects_missing_or_other_owner_document(self, process, connection):
        database = connection.return_value.__enter__.return_value
        database.execute.return_value.fetchone.return_value = None

        response = self.http.post("/api/documents/complete", json=self.payload)

        self.assertEqual(response.status_code, 404)
        query, params = database.execute.call_args.args
        self.assertIn("owner_sub=%s", query)
        principal = Principal()
        self.assertEqual(params, (self.document_id, principal.tenant, principal.subject))
        process.assert_not_called()

    @patch("app.main.connection")
    @patch("app.main.process_s3_key")
    def test_completion_uses_status_from_locked_worker(self, process, connection):
        database = connection.return_value.__enter__.return_value
        database.execute.return_value.fetchone.return_value = {"s3_key": "sample.txt"}

        for status, code in (("ready", 200), ("processing", 409), ("retired", 409), (None, 404)):
            with self.subTest(status=status):
                process.return_value = status
                response = self.http.post("/api/documents/complete", json=self.payload)
                self.assertEqual(response.status_code, code)
                process.assert_called_with("sample.txt")
                if status == "ready":
                    self.assertEqual(response.json(), {"status": "ready"})


class IngestionTests(unittest.TestCase):
    @patch("worker.ingest.connection")
    @patch("worker.ingest.client")
    def test_worker_skips_an_upload_claimed_by_another_request(self, s3, connection):
        database = connection.return_value.__enter__.return_value
        database.execute.return_value.fetchone.return_value = {"status": "processing"}

        self.assertEqual(process_s3_key("sample.txt"), "processing")

        database.execute.assert_called_once()
        self.assertIn("FOR UPDATE", database.execute.call_args.args[0])
        s3.assert_not_called()

    @patch("worker.ingest.connection")
    @patch("worker.ingest.client")
    @patch("worker.ingest.embed", return_value=[0.5])
    @patch("worker.ingest.upsert")
    def test_worker_retries_failed_upload_and_marks_it_ready(self, upsert, embed, s3, connection):
        database = connection.return_value.__enter__.return_value
        document_id = uuid4()
        database.execute.return_value.fetchone.return_value = {
            "status": "failed",
            "id": document_id,
            "current_version": 1,
            "title": "Sample",
            "tenant_id": "demo",
            "owner_sub": "reader",
            "allowed_groups": ["workspace"],
        }
        body = io.BytesIO(b"Example document")
        s3.return_value.get_object.return_value = {"ContentLength": 16, "Body": body}

        self.assertEqual(process_s3_key("sample.txt"), "ready")

        self.assertTrue(body.closed)
        embed.assert_called_once_with("Example document")
        upsert.assert_called_once_with("demo", [{
            "id": f"{document_id}:1:0",
            "values": [0.5],
            "metadata": {
                "document_id": str(document_id),
                "title": "Sample",
                "page": 1,
                "text": "Example document",
                "owner_sub": "reader",
                "allowed_groups": ["workspace"],
            },
        }])
        statements = [call.args[0] for call in database.execute.call_args_list]
        self.assertIn("status='processing'", statements[1])
        self.assertIn("status='ready'", statements[2])


if __name__ == "__main__":
    unittest.main()
