"""Check a running, unconfigured Atlas server using only the standard library."""
import argparse
import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


def fetch(base_url, path):
    try:
        with urlopen(base_url + path, timeout=2) as response:
            return response.status, response.headers, response.read()
    except HTTPError as response:
        with response:
            return response.code, response.headers, response.read()


def check(base_url, startup_timeout=60):
    deadline = time.monotonic() + startup_timeout
    while True:
        try:
            status, _, body = fetch(base_url, "/health/live")
            if status == 200 and json.loads(body) == {"status": "alive"}:
                break
        except (URLError, TimeoutError, ConnectionError, ValueError):
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError("Server did not become live before the startup deadline")
        time.sleep(0.5)

    status, headers, body = fetch(base_url, "/")
    if status != 200 or "text/html" not in headers.get("Content-Type", ""):
        raise RuntimeError("Home page did not return HTML")
    if b'id="chatForm"' not in body:
        raise RuntimeError("Home page is missing the chat form")
    for path in ("/static/js/app.js", "/static/css/styles.css"):
        if path.encode() not in body:
            raise RuntimeError(f"Home page is missing asset reference: {path}")
        status, _, asset = fetch(base_url, path)
        if status != 200 or not asset:
            raise RuntimeError(f"Asset was not served: {path}")

    status, _, body = fetch(base_url, "/health/ready")
    if status != 503 or json.loads(body) != {"detail": "Database unavailable"}:
        raise RuntimeError("Unconfigured database did not produce the expected readiness failure")
    print("HTTP smoke passed: liveness, home, assets, and unconfigured readiness")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url", help="URL of an Atlas server without DATABASE_URL")
    args = parser.parse_args()
    check(args.base_url.rstrip("/"))
