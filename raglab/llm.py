"""LLM adapters with readable failures and an offline mock."""

import json
import os
import urllib.error
import urllib.request
from typing import Optional


class LLMError(RuntimeError):
    """Base class for failures that should be shown to the user."""


class LLMConfigError(LLMError):
    """Raised when the adapter is not configured."""


class LLMResponseError(LLMError):
    """Raised when the provider returns an unusable response."""


class MockLLM:
    """Deterministic adapter used by tests and the offline demo."""

    def __init__(self, answer: str = "根据资料，我无法回答这个问题。", error: Optional[Exception] = None):
        self.answer = answer
        self.error = error
        self.calls = 0
        self.last_system = ""
        self.last_prompt = ""

    def complete(self, system: str, prompt: str) -> str:
        self.calls += 1
        self.last_system = system
        self.last_prompt = prompt
        if self.error is not None:
            raise self.error
        return self.answer


class OpenAICompatibleLLM:
    """Minimal ``/chat/completions`` client using only the standard library."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.api_key = api_key if api_key is not None else (
            os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        )
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
        self.timeout = timeout

    @property
    def endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return self.base_url + "/chat/completions"

    def complete(self, system: str, prompt: str) -> str:
        if not self.api_key:
            raise LLMConfigError(
                "未配置 LLM_API_KEY（或 OPENAI_API_KEY）。离线演示请使用 --llm mock。"
            )

        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + self.api_key,
                "Content-Type": "application/json",
                "User-Agent": "rag-reliability-lab/0.1",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:400]
            except Exception:
                pass
            if exc.code == 401:
                message = "LLM 认证失败（401）。请检查 API Key。"
            elif exc.code == 429:
                message = "LLM 限流（429）。请稍后重试或切换模型。"
            elif 500 <= exc.code < 600:
                message = "LLM 服务暂时不可用（HTTP %s）。" % exc.code
            else:
                message = "LLM 请求失败（HTTP %s）。" % exc.code
            if detail:
                message += " 详情：" + detail
            raise LLMError(message) from exc
        except urllib.error.URLError as exc:
            raise LLMError("无法连接 LLM 服务：%s" % exc.reason) from exc
        except TimeoutError as exc:
            raise LLMError("LLM 请求超时，请稍后重试。") from exc

        try:
            data = json.loads(raw)
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError("LLM 返回了无法解析的响应。") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMResponseError("LLM 返回了空答案。")
        return content.strip()
