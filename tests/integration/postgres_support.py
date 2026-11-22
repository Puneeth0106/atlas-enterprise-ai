"""Disposable PostgreSQL schema shared by integration test cases."""
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo

from app.db import connection


class PostgresCase(unittest.TestCase):
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
