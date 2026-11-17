"""OpenAI-compatible chat client used by the research model.

Every research stage receives a callable ``llm(messages, temperature=..., max_tokens=...) -> str``.
``ChatModel`` is that callable for any endpoint that implements ``/chat/completions``.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

#: Answers that are JSON or a few kilobytes of prose.
DEFAULT_MAX_TOKENS = 32_000

_TRANSIENT_HTTP = (408, 409, 429, 500, 502, 503, 504)
_CONTINUE = ("Continue the previous response exactly where it stopped. Do not repeat text that "
             "was already written; output only the remaining continuation.")


@dataclass
class ChatModel:
    """A chat model behind an OpenAI-compatible endpoint.

    A reply cut off at the token limit is continued in further rounds and concatenated, so long
    outputs are never silently truncated. When a reasoning model spends the whole limit on hidden
    reasoning and returns no text, the limit is raised instead.
    """

    model: str
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    reasoning_effort: str | None = None
    temperature: float | None = None
    max_tokens: int = DEFAULT_MAX_TOKENS
    max_tokens_ceiling: int = 128_000
    token_parameter: str = "max_completion_tokens"
    timeout: float = 1800.0
    retries: int = 6
    max_continuations: int = 6
    extra_body: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, value: dict[str, Any]) -> "ChatModel":
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(value) - known)
        if unknown:
            raise ValueError(f"unknown model setting(s): {', '.join(unknown)}")
        return cls(**value)

    # ------------------------------------------------------------------ transport
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.api_key_env or "")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _request(self, messages: list[dict[str, Any]], *, temperature: float | None,
                 max_tokens: int) -> tuple[str, str]:
        payload: dict[str, Any] = {"model": self.model, "messages": messages,
                                   self.token_parameter: int(max_tokens)}
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        else:
            chosen = self.temperature if temperature is None else temperature
            if chosen is not None:
                payload["temperature"] = chosen
        payload.update(self.extra_body)
        data = json.dumps(payload).encode("utf-8")
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        last_error: Exception | None = None
        for attempt in range(max(1, self.retries)):
            request = urllib.request.Request(url, data=data, headers=self._headers(),
                                             method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                choice = (body.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                return str(message.get("content") or ""), str(choice.get("finish_reason") or "")
            except urllib.error.HTTPError as error:
                if error.code not in _TRANSIENT_HTTP:
                    detail = error.read().decode("utf-8", "replace")[:500]
                    raise RuntimeError(f"chat request failed with HTTP {error.code}: {detail}") \
                        from error
                last_error = error
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
                last_error = error
            time.sleep(min(60.0, 4.0 * (2 ** attempt)))
        raise RuntimeError(f"chat request failed after {self.retries} attempts: {last_error}")

    # ------------------------------------------------------------------ public call
    def __call__(self, messages: list[dict[str, Any]], temperature: float | None = None,
                 max_tokens: int | None = None) -> str:
        conversation = list(messages)
        budget = max(1, int(max_tokens or self.max_tokens))
        parts: list[str] = []
        for _ in range(max(1, self.max_continuations)):
            content, finish = self._request(conversation, temperature=temperature,
                                            max_tokens=budget)
            if finish == "length" and not content.strip():
                if budget >= self.max_tokens_ceiling:
                    break
                budget = min(budget * 4, self.max_tokens_ceiling)
                continue
            parts.append(content)
            if finish != "length":
                break
            conversation = conversation + [{"role": "assistant", "content": content},
                                           {"role": "user", "content": _CONTINUE}]
        return "".join(parts).strip()
