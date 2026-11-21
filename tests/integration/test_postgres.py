"""Real transactions in a disposable schema; cloud operations stay mocked."""
import io
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo

from app.db import audit, connection
from worker.ingest import process_s3_key


class PostgresTransactions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Deliberately do not fall back to the application's DATABASE_URL or .env.
        cls.base_url = os.environ.get('ATLAS_TEST_DATABASE_URL')
        if not cls.base_url:
            raise RuntimeError('ATLAS_TEST_DATABASE_URL must name a disposable test database')
        cls.schema = 'atlas_test_' + uuid4().hex
        with psycopg.connect(cls.base_url, autocommit=True) as conn:
            conn.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(cls.schema)))
        cls.addClassCleanup(cls.drop_schema)
        cls.url = make_conninfo(cls.base_url, options=f'-c search_path={cls.schema},public -c statement_timeout=5000')
        cls.settings_patch = patch('app.db.settings', return_value=SimpleNamespace(database_url=cls.url))
        cls.settings_patch.start()
        cls.addClassCleanup(cls.settings_patch.stop)
        with connection() as conn:
            conn.execute(Path('infra/schema.sql').read_text(encoding='utf-8'))

    @classmethod
    def drop_schema(cls):
        with psycopg.connect(cls.base_url, autocommit=True) as conn:
            conn.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(cls.schema)))

    def setUp(self):
        with connection() as conn:
            conn.execute('TRUNCATE documents, audit_events RESTART IDENTITY')
        self.document_id = uuid4()
        self.key = f'{self.document_id}.txt'
        with connection() as conn:
            conn.execute('''INSERT INTO documents
                (id, tenant_id, title, owner_sub, allowed_groups, s3_key, status)
                VALUES (%s, 'demo', 'Example', 'reader', ARRAY['workspace'], %s, 'pending')''',
                (self.document_id, self.key))
        self.s3 = self.enterContext(patch('worker.ingest.client'))
        self.s3.return_value.get_object.return_value = {
            'ContentLength': 16, 'Body': io.BytesIO(b'Example document')}
        self.embed = self.enterContext(patch('worker.ingest.embed', return_value=[0.5]))
        self.upsert = self.enterContext(patch('worker.ingest.upsert'))

    def state(self):
        with connection() as conn:
            status = conn.execute('SELECT status FROM documents WHERE id=%s', (self.document_id,)).fetchone()['status']
            events = conn.execute('SELECT actor, tenant_id, action, detail FROM audit_events ORDER BY id').fetchall()
        return status, events

    def test_connection_commits_status_and_json_audit(self):
        with connection() as conn:
            conn.execute("UPDATE documents SET status='ready' WHERE id=%s", (self.document_id,))
            audit(conn, 'reader', 'demo', 'test_commit', {'document_id': str(self.document_id)})
        self.assertEqual(self.state(), ('ready', [{
            'actor': 'reader', 'tenant_id': 'demo', 'action': 'test_commit',
            'detail': {'document_id': str(self.document_id)},
        }]))

    def test_exception_rolls_back_status_and_audit(self):
        with self.assertRaisesRegex(RuntimeError, 'abort transaction'):
            with connection() as conn:
                conn.execute("UPDATE documents SET status='ready' WHERE id=%s", (self.document_id,))
                audit(conn, 'reader', 'demo', 'test_rollback', {})
                raise RuntimeError('abort transaction')
        self.assertEqual(self.state(), ('pending', []))

    def test_worker_commits_ready_and_skips_duplicate(self):
        self.assertEqual(process_s3_key(self.key), 'ready')
        self.assertEqual(process_s3_key(self.key), 'ready')
        status, events = self.state()
        self.assertEqual(status, 'ready')
        self.assertEqual(events, [{
            'actor': 'reader', 'tenant_id': 'demo', 'action': 'ingest_ready',
            'detail': {'document_id': str(self.document_id), 'chunks': 1},
        }])
        self.s3.assert_called_once_with('s3')
        self.upsert.assert_called_once()

    def test_failed_vector_write_is_committed_and_can_retry(self):
        self.upsert.side_effect = RuntimeError('vector service unavailable')
        with self.assertRaisesRegex(RuntimeError, 'vector service unavailable'):
            process_s3_key(self.key)
        self.assertEqual(self.state(), ('failed', []))
        self.upsert.side_effect = None
        self.s3.return_value.get_object.return_value['Body'] = io.BytesIO(b'Example document')
        self.assertEqual(process_s3_key(self.key), 'ready')
        self.assertEqual(self.state()[0], 'ready')
        self.assertEqual(len(self.state()[1]), 1)
        self.assertEqual(self.upsert.call_args_list[0], self.upsert.call_args_list[1])

    def test_audit_constraint_failure_rolls_back_ready_before_marking_failed(self):
        # A real SQL error after UPDATE must roll the ready transaction back.
        with patch('worker.ingest.audit', side_effect=lambda conn, *args:
                   audit(conn, None, 'demo', 'ingest_ready', {})):
            with self.assertRaises(psycopg.errors.NotNullViolation):
                process_s3_key(self.key)
        self.assertEqual(self.state(), ('failed', []))

    def test_concurrent_request_observes_committed_processing_claim(self):
        entered, release = Event(), Event()
        body = self.s3.return_value.get_object.return_value

        def blocked_download(**kwargs):
            entered.set()
            if not release.wait(5):
                raise TimeoutError('test did not release download')
            return body

        self.s3.return_value.get_object.side_effect = blocked_download
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(process_s3_key, self.key)
            try:
                self.assertTrue(entered.wait(5))
                self.assertEqual(self.state(), ('processing', []))
                self.assertEqual(process_s3_key(self.key), 'processing')
                self.s3.assert_called_once_with('s3')
            finally:
                release.set()
            self.assertEqual(first.result(timeout=5), 'ready')
        self.assertEqual(len(self.state()[1]), 1)


if __name__ == '__main__':
    unittest.main()
