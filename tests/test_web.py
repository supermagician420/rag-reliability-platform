import http.client
import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

from raglab.web.server import RateLimiter, create_server


class WebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.server = create_server("127.0.0.1", 0, os.path.join(cls.temp_dir.name, "rag.db"), "mock")
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = "http://127.0.0.1:%d" % cls.server.server_port

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        cls.temp_dir.cleanup()

    def request(self, method, path, payload=None, content_type="application/json", origin=None):
        body = None
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(self.base + path, data=body, method=method)
        if body is not None:
            request.add_header("Content-Type", content_type)
        if origin:
            request.add_header("Origin", origin)
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()

    def test_health(self):
        status, body = self.request("GET", "/api/health")
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["documents"], 5)

    def test_documents(self):
        status, body = self.request("GET", "/api/documents")
        self.assertEqual(status, 200)
        self.assertEqual(len(json.loads(body)["documents"]), 5)

    def test_index_and_static(self):
        status, body = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b"RAG Reliability Lab", body)
        status, _ = self.request("GET", "/static/styles.css")
        self.assertEqual(status, 200)

    def test_static_traversal_is_blocked(self):
        status, _ = self.request("GET", "/static/%2e%2e/%2e%2e/Windows/win.ini")
        self.assertEqual(status, 404)

    def test_ask(self):
        status, body = self.request("POST", "/api/ask", {"question": "病假需要提前几天申请？"})
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertEqual(payload["status"], "answered")
        self.assertEqual(payload["citations"], ["[S1]"])

    def test_empty_question_is_400(self):
        status, _ = self.request("POST", "/api/ask", {"question": ""})
        self.assertEqual(status, 400)

    def test_wrong_content_type_is_415(self):
        status, _ = self.request("POST", "/api/ask", {"question": "病假"}, content_type="text/plain")
        self.assertEqual(status, 415)

    def test_cross_origin_is_403(self):
        status, _ = self.request("POST", "/api/ask", {"question": "病假"}, origin="https://evil.example")
        self.assertEqual(status, 403)

    def test_rate_limit(self):
        original = self.server.RequestHandlerClass.limiter
        self.server.RequestHandlerClass.limiter = RateLimiter(limit=2, window=60)
        try:
            codes = [
                self.request("POST", "/api/ask", {"question": "病假需要提前几天申请？" })[0]
                for _ in range(3)
            ]
            self.assertEqual(codes[-1], 429)
        finally:
            self.server.RequestHandlerClass.limiter = original

    def test_body_too_large(self):
        body = b'{"question":"' + b"x" * 17000 + b'"}'
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        connection.request("POST", "/api/ask", body=body, headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        self.assertEqual(response.status, 413)
        connection.close()


if __name__ == "__main__":
    unittest.main()
