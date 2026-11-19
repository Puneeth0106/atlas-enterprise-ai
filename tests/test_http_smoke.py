"""Exercise the web surface without cloud services or a database."""
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app


class HttpSmokeTests(unittest.TestCase):
    def setUp(self):
        self.http = TestClient(app)
        self.addCleanup(self.http.close)

    @patch("app.main.connection")
    def test_liveness_does_not_require_a_database(self, connection):
        response = self.http.get("/health/live")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "alive"})
        connection.assert_not_called()

    def test_home_serves_the_template_and_browser_assets(self):
        response = self.http.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn('id="chatForm"', response.text)
        for asset in ("/static/js/app.js", "/static/css/styles.css"):
            with self.subTest(asset=asset):
                self.assertIn(asset, response.text)
                result = self.http.get(asset)
                self.assertEqual(result.status_code, 200)
                self.assertTrue(result.content)

    @patch("app.main.connection")
    def test_readiness_queries_the_database(self, connection):
        database = connection.return_value.__enter__.return_value

        response = self.http.get("/health/ready")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ready"})
        database.execute.assert_called_once_with("SELECT 1")

    @patch("app.main.connection", side_effect=RuntimeError("private connection details"))
    def test_readiness_returns_unavailable_without_leaking_connection_details(self, connection):
        response = self.http.get("/health/ready")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"detail": "Database unavailable"})
        self.assertNotIn("private connection details", response.text)


if __name__ == "__main__":
    unittest.main()
