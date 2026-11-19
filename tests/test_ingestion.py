"""Credential-free extraction and ingestion recovery regressions."""

import io
import unittest
from types import SimpleNamespace
from unittest.mock import call, patch
from uuid import UUID

from docx import Document
from pypdf import PdfWriter

from worker.ingest import chunks, parse, process_s3_key


class ExtractionTests(unittest.TestCase):
    def test_text_preserves_utf8_content(self):
        self.assertEqual(parse("Café handbook".encode(), ".txt"), [(1, "Café handbook")])

    def test_invalid_utf8_is_rejected(self):
        with self.assertRaises(UnicodeDecodeError):
            parse(b"\xff", ".txt")

    def test_docx_extracts_paragraphs_in_order(self):
        document = Document()
        document.add_paragraph("Team handbook")
        document.add_paragraph("Leave policy")
        blob = io.BytesIO()
        document.save(blob)
        self.assertEqual(parse(blob.getvalue(), ".docx"), [(1, "Team handbook\nLeave policy")])

    def test_blank_pdf_preserves_page_numbers_without_inventing_text(self):
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        writer.add_blank_page(width=100, height=100)
        blob = io.BytesIO()
        writer.write(blob)
        self.assertEqual(parse(blob.getvalue(), ".pdf"), [(1, ""), (2, "")])

    def test_chunks_normalize_whitespace_and_keep_overlap(self):
        self.assertEqual(list(chunks("  abcde\n fghij\tklmno  ", size=8, overlap=2)),
                         ["abcde fg", "fghij kl", "klmno"])

    def test_empty_text_produces_no_chunks(self):
        self.assertEqual(list(chunks(" \n\t")), [])


class IngestionRecoveryTests(unittest.TestCase):
    def setUp(self):
        def mocked(name, **kwargs):
            patcher = patch(f"worker.ingest.{name}", **kwargs)
            mock = patcher.start()
            self.addCleanup(patcher.stop)
            return mock

        self.connection = mocked("connection")
        self.database = self.connection.return_value.__enter__.return_value
        self.document = {
            "id": UUID("00000000-0000-0000-0000-000000000001"),
            "status": "pending", "current_version": 3, "title": "Handbook",
            "tenant_id": "tenant-a", "owner_sub": "reader-a",
            "allowed_groups": ["team-a"],
        }
        self.database.execute.return_value.fetchone.return_value = self.document
        self.s3 = mocked("client")
        self.embed = mocked("embed", return_value=[0.5])
        self.upsert = mocked("upsert")
        self.audit = mocked("audit")
        self.settings = mocked("settings", return_value=SimpleNamespace(
            s3_bucket="test-documents", max_upload_bytes=32))

    def upload(self, blob, declared_size=None):
        body = io.BytesIO(blob)
        self.s3.return_value.get_object.return_value = {
            "ContentLength": len(blob) if declared_size is None else declared_size,
            "Body": body,
        }
        self.addCleanup(body.close)
        return body

    def assert_failed(self, error, message, key="handbook.txt"):
        with self.assertLogs("atlas.worker", level="ERROR"):
            with self.assertRaisesRegex(error, message):
                process_s3_key(key)
        self.assertEqual(self.database.execute.call_args_list, [
            call("SELECT * FROM documents WHERE s3_key=%s FOR UPDATE", (key,)),
            call("UPDATE documents SET status='processing' WHERE id=%s", (self.document["id"],)),
            call("UPDATE documents SET status='failed' WHERE id=%s", (self.document["id"],)),
        ])
        self.audit.assert_not_called()

    def test_missing_document_does_not_read_storage(self):
        self.database.execute.return_value.fetchone.return_value = None
        self.assertIsNone(process_s3_key("missing.txt"))
        self.database.execute.assert_called_once()
        self.s3.assert_not_called()

    def test_ready_document_is_not_reindexed(self):
        self.document["status"] = "ready"
        self.assertEqual(process_s3_key("handbook.txt"), "ready")
        self.database.execute.assert_called_once()
        self.s3.assert_not_called()

    def test_declared_oversize_closes_body_before_extraction(self):
        body = self.upload(b"small body", declared_size=33)
        self.assert_failed(ValueError, "Upload too large")
        self.assertTrue(body.closed)
        self.embed.assert_not_called()
        self.upsert.assert_not_called()

    def test_actual_oversize_is_rejected_even_if_declared_size_is_small(self):
        body = self.upload(b"x" * 33, declared_size=1)
        self.assert_failed(ValueError, "Upload too large")
        self.assertTrue(body.closed)
        self.embed.assert_not_called()
        self.upsert.assert_not_called()

    def test_exact_size_limit_is_accepted(self):
        body = self.upload(b"x" * 32)
        self.assertEqual(process_s3_key("handbook.txt"), "ready")
        self.assertTrue(body.closed)
        self.s3.assert_called_once_with("s3")
        self.s3.return_value.get_object.assert_called_once_with(
            Bucket="test-documents", Key="handbook.txt")
        self.embed.assert_called_once_with("x" * 32)
        self.audit.assert_called_once_with(
            self.database, "reader-a", "tenant-a", "ingest_ready",
            {"document_id": str(self.document["id"]), "chunks": 1})

    def test_invalid_pdf_signature_is_rejected_with_uppercase_suffix(self):
        body = self.upload(b"not a pdf")
        self.assert_failed(ValueError, "Invalid PDF", key="handbook.PDF")
        self.assertTrue(body.closed)
        self.embed.assert_not_called()
        self.upsert.assert_not_called()

    def test_empty_extraction_is_failed_without_indexing(self):
        body = self.upload(b" \n\t")
        self.assert_failed(ValueError, "No extractable text")
        self.assertTrue(body.closed)
        self.embed.assert_not_called()
        self.upsert.assert_not_called()

    def test_storage_failure_is_reraised_and_marks_document_failed(self):
        self.s3.return_value.get_object.side_effect = RuntimeError("storage unavailable")
        self.assert_failed(RuntimeError, "storage unavailable")
        self.embed.assert_not_called()
        self.upsert.assert_not_called()

    def test_extraction_failure_closes_body_and_marks_document_failed(self):
        body = self.upload(b"\xff")
        self.assert_failed(UnicodeDecodeError, "utf-8")
        self.assertTrue(body.closed)
        self.embed.assert_not_called()
        self.upsert.assert_not_called()

    def test_embedding_failure_does_not_write_vectors(self):
        body = self.upload(b"Team handbook")
        self.embed.side_effect = RuntimeError("embedding unavailable")
        self.assert_failed(RuntimeError, "embedding unavailable")
        self.assertTrue(body.closed)
        self.upsert.assert_not_called()

    @patch("worker.ingest.parse")
    def test_partial_vector_failure_retries_same_ids_in_tenant_batches(self, parse_pages):
        # 51 short pages cross the 50-record batch boundary without large fixtures.
        parse_pages.return_value = [(page, f"Page {page}") for page in range(1, 52)]
        self.upload(b"placeholder")
        self.upsert.side_effect = [None, RuntimeError("vector unavailable")]
        self.assert_failed(RuntimeError, "vector unavailable")
        failed_batches = self.upsert.call_args_list[:]
        self.assertEqual([len(c.args[1]) for c in failed_batches], [50, 1])

        self.document["status"] = "failed"
        self.upload(b"placeholder")
        self.upsert.reset_mock(side_effect=True)
        self.assertEqual(process_s3_key("handbook.txt"), "ready")
        self.assertEqual(self.upsert.call_args_list, failed_batches)
        self.database.execute.assert_called_with(
            "UPDATE documents SET status='ready' WHERE id=%s", (self.document["id"],))
        records = [record for batch in failed_batches for record in batch.args[1]]
        self.assertEqual(len({record["id"] for record in records}), 51)
        for i, record in enumerate(records):
            self.assertEqual(record, {
                "id": f'{self.document["id"]}:3:{i}', "values": [0.5],
                "metadata": {
                    "document_id": str(self.document["id"]), "title": "Handbook",
                    "page": i + 1, "text": f"Page {i + 1}",
                    "owner_sub": "reader-a", "allowed_groups": ["team-a"],
                },
            })
        self.assertTrue(all(batch.args[0] == "tenant-a" for batch in failed_batches))
        self.audit.assert_called_once_with(
            self.database, "reader-a", "tenant-a", "ingest_ready",
            {"document_id": str(self.document["id"]), "chunks": 51})


if __name__ == "__main__":
    unittest.main()
