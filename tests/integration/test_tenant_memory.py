"""Tenant SQL and HTTP behavior with real PostgreSQL and injected demo identities."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

import psycopg
from fastapi.testclient import TestClient

from postgres_support import PostgresCase
from app.db import connection
from app.demo_context import Principal, principal
from app.main import app
from app.memory import history, list_messages, save_turn


class TenantMemoryTests(PostgresCase):
    def setUp(self):
        with connection() as conn:
            conn.execute('TRUNCATE documents, chat_messages, audit_events RESTART IDENTITY')
        self.reader = Principal(subject='reader', tenant='alpha', admin=False)
        previous_overrides = app.dependency_overrides.copy()
        self.addCleanup(setattr, app, 'dependency_overrides', previous_overrides)
        app.dependency_overrides[principal] = lambda: self.reader
        self.http = self.enterContext(TestClient(app))
        self.conversation = uuid4()

    def document(self, tenant, owner, groups):
        document_id = uuid4()
        with connection() as conn:
            conn.execute('''INSERT INTO documents
                (id, tenant_id, title, owner_sub, allowed_groups, s3_key, status)
                VALUES (%s, %s, 'Example', %s, %s, %s, 'pending')''',
                (document_id, tenant, owner, groups, str(document_id)))
        return str(document_id)

    def test_document_listing_requires_tenant_and_group_or_owner(self):
        owned = self.document('alpha', 'reader', [])
        shared = self.document('alpha', 'other', ['workspace'])
        self.document('alpha', 'other', ['private'])
        self.document('beta', 'reader', ['workspace'])
        response = self.http.get('/api/documents')
        self.assertEqual(response.status_code, 200)
        self.assertEqual({row['id'] for row in response.json()['documents']}, {owned, shared})

    def test_completion_hides_foreign_and_non_owned_documents(self):
        foreign = self.document('beta', 'reader', ['workspace'])
        shared = self.document('alpha', 'other', ['workspace'])
        with patch('app.main.process_s3_key') as ingest:
            for document_id in (foreign, shared, str(uuid4())):
                with self.subTest(document_id=document_id):
                    response = self.http.post('/api/documents/complete', json={'document_id': document_id})
                    self.assertEqual(response.status_code, 404)
                    self.assertEqual(response.json(), {'detail': 'Document not found'})
            ingest.assert_not_called()

    def test_owned_completion_passes_stored_key_to_ingestion(self):
        owned = self.document('alpha', 'reader', [])
        with patch('app.main.process_s3_key', return_value='ready') as ingest:
            response = self.http.post('/api/documents/complete', json={'document_id': owned})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'status': 'ready'})
        ingest.assert_called_once_with(owned)

    def test_message_route_isolates_same_conversation_id_between_tenants(self):
        save_turn('alpha', self.conversation, 'Alpha question', 'Alpha answer')
        save_turn('beta', self.conversation, 'Private question', 'Private answer')
        response = self.http.get(f'/api/conversations/{self.conversation}/messages')
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body['conversation_id'], str(self.conversation))
        self.assertEqual([(row['role'], row['content']) for row in body['messages']],
                         [('user', 'Alpha question'), ('assistant', 'Alpha answer')])
        self.assertTrue(all(row['created_at'] for row in body['messages']))
        self.reader = Principal(tenant='gamma')
        self.assertEqual(self.http.get(f'/api/conversations/{self.conversation}/messages').json()['messages'], [])

    def test_history_returns_recent_complete_turns_in_insert_order(self):
        # One transaction gives every row the same timestamp; ordering must use IDs.
        with connection() as conn:
            for i in range(4):
                conn.execute('''INSERT INTO chat_messages(tenant_id, conversation_id, role, content)
                    VALUES ('alpha', %s, 'user', %s), ('alpha', %s, 'assistant', %s)''',
                    (self.conversation, f'q{i}', self.conversation, f'a{i}'))
        save_turn('beta', self.conversation, 'Private question', 'Private answer')
        save_turn('alpha', uuid4(), 'Other conversation', 'Other answer')
        with patch('app.memory.settings', return_value=SimpleNamespace(memory_turns=2)):
            self.assertEqual(history('alpha', self.conversation), [
                {'role': 'user', 'content': 'q2'}, {'role': 'assistant', 'content': 'a2'},
                {'role': 'user', 'content': 'q3'}, {'role': 'assistant', 'content': 'a3'}])

    def test_failed_assistant_insert_does_not_leave_half_a_turn(self):
        save_turn('alpha', self.conversation, 'Existing question', 'Existing answer')
        with self.assertRaises(psycopg.errors.NotNullViolation):
            save_turn('alpha', self.conversation, 'New question', None)
        self.assertEqual([row['content'] for row in list_messages('alpha', self.conversation)],
                         ['Existing question', 'Existing answer'])

    def test_chat_uses_only_tenant_history_and_persists_response(self):
        save_turn('alpha', self.conversation, 'Earlier question', 'Earlier answer')
        save_turn('beta', self.conversation, 'Private question', 'Private answer')
        with patch('app.main.chat', return_value={'answer': 'New answer', 'sources': []}) as chat:
            response = self.http.post('/api/chat', json={
                'question': 'What about the next step?', 'conversation_id': str(self.conversation)})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['conversation_id'], str(self.conversation))
        chat.assert_called_once_with(self.reader, 'What about the next step?', [
            {'role': 'user', 'content': 'Earlier question'},
            {'role': 'assistant', 'content': 'Earlier answer'}])
        self.assertEqual([row['content'] for row in list_messages('alpha', self.conversation)],
                         ['Earlier question', 'Earlier answer', 'What about the next step?', 'New answer'])
        self.assertEqual([row['content'] for row in list_messages('beta', self.conversation)],
                         ['Private question', 'Private answer'])

    def test_chat_failure_does_not_persist_a_partial_turn(self):
        with patch('app.main.chat', side_effect=RuntimeError('model unavailable')):
            with self.assertRaisesRegex(RuntimeError, 'model unavailable'):
                self.http.post('/api/chat', json={
                    'question': 'What is the next step?', 'conversation_id': str(self.conversation)})
        self.assertEqual(list_messages('alpha', self.conversation), [])


if __name__ == '__main__':
    unittest.main()
