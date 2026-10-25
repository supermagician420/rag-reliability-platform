"""Standard-library HTTP UI with JSON-only, same-origin requests."""

import json
import mimetypes
import os
import threading
import time
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Deque, Dict
from urllib.parse import unquote, urlsplit

from ..answer import Answerer, make_mock_answerer
from ..llm import LLMError, OpenAICompatibleLLM
from ..seed import seeded_store


STATIC_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
MAX_BODY = 16 * 1024
RATE_LIMIT = 30
RATE_WINDOW = 60.0


class RateLimiter:
    def __init__(self, limit: int = RATE_LIMIT, window: float = RATE_WINDOW):
        self.limit = limit
        self.window = window
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] > self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return False
            hits.append(now)
            return True


class RagHandler(BaseHTTPRequestHandler):
    store: Any = None
    answerer: Any = None
    limiter: Any = None
    llm_mode = "mock"
    server_version = "RagLab/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        # Do not log full URLs (they can contain questions in other deployments).
        print("[%s] %s" % (self.client_address[0], fmt % args))

    def _common_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")

    def _send(self, status: int, body: bytes, content_type: str, cache: str = "no-store") -> None:
        self.send_response(status)
        self._common_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _html(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self._common_headers()
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; base-uri 'none'; frame-ancestors 'none'",
        )
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/api/health":
            self._json(
                200,
                {
                    "status": "ok",
                    "llm": self.llm_mode,
                    "documents": self.store.count_documents(),
                    "chunks": self.store.count_chunks(),
                },
            )
            return
        if path == "/api/documents":
            self._json(200, {"documents": self.store.list_documents()})
            return
        if path in ("/", "/index.html"):
            with open(os.path.join(STATIC_ROOT, "index.html"), "rb") as handle:
                self._html(200, handle.read())
            return
        if path.startswith("/static/"):
            relative = unquote(path[len("/static/"):])
            if not relative or "\x00" in relative:
                self._json(404, {"error": "not found"})
                return
            candidate = os.path.abspath(os.path.join(STATIC_ROOT, relative))
            root = os.path.normcase(STATIC_ROOT + os.sep)
            if not os.path.normcase(candidate).startswith(root) or not os.path.isfile(candidate):
                self._json(404, {"error": "not found"})
                return
            content_type = mimetypes.guess_type(candidate)[0] or "application/octet-stream"
            with open(candidate, "rb") as handle:
                self._send(200, handle.read(), content_type + ("; charset=utf-8" if content_type.startswith("text/") else ""))
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        if path != "/api/ask":
            self._json(404, {"error": "not found"})
            return

        # Same-origin only: browsers send Origin for cross-origin POSTs.
        origin = self.headers.get("Origin", "")
        if origin:
            host = self.headers.get("Host", "")
            if urlsplit(origin).netloc != host:
                self._json(403, {"error": "cross-origin request rejected"})
                return

        if not self.limiter.allow(self.client_address[0]):
            self._json(429, {"error": "请求过于频繁，请稍后再试。"})
            return

        content_type = self.headers.get("Content-Type", "")
        if "application/json" not in content_type.lower():
            self._json(415, {"error": "Content-Type 必须是 application/json"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._json(400, {"error": "无效的 Content-Length"})
            return
        if length <= 0:
            self._json(400, {"error": "请求体不能为空"})
            return
        if length > MAX_BODY:
            self._json(413, {"error": "请求体过大"})
            return

        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            self._json(400, {"error": "请求体不是合法 JSON"})
            return
        if not isinstance(payload, dict):
            self._json(400, {"error": "请求体必须是 JSON 对象"})
            return
        question = payload.get("question", "")
        if not isinstance(question, str):
            self._json(400, {"error": "question 必须是字符串"})
            return

        try:
            result = self.answerer.answer(question)
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
            return
        except LLMError as exc:
            self._json(503, {"error": "LLM 服务暂时不可用。", "detail": str(exc)})
            return
        except Exception:
            self._json(500, {"error": "服务内部错误。"})
            return

        payload = result.to_dict()
        payload["sources"] = [
            {
                "title": source.get("title", ""),
                "snippet": source.get("snippet", ""),
                "score": source.get("score", 0.0),
                "coverage": source.get("coverage", 0.0),
            }
            for source in result.sources
        ]
        self._json(503 if result.status == "error" else 200, payload)


def create_server(host: str, port: int, db_path: str, llm_mode: str = "mock") -> ThreadingHTTPServer:
    store = seeded_store(db_path)
    if llm_mode == "openai":
        answerer = Answerer(store, OpenAICompatibleLLM())
    else:
        answerer = make_mock_answerer(store)

    class BoundHandler(RagHandler):
        pass

    BoundHandler.store = store
    BoundHandler.answerer = answerer
    BoundHandler.limiter = RateLimiter()
    BoundHandler.llm_mode = llm_mode
    server = ThreadingHTTPServer((host, port), BoundHandler)
    server.daemon_threads = True
    return server
