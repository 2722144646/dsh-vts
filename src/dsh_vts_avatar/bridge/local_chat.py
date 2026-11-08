"""Local dialogue path: the avatar answers with an Ollama model on this machine.

Used for conversations private enough that they should not be relayed through
the host agent. The reply text is generated here and spoken directly; callers
only ever see that she spoke, not what she said.
"""

from __future__ import annotations

import asyncio
import json
import urllib.request
from typing import Any

from ..config import LLMConfig


class LocalChat:
    """Thin synchronous Ollama client wrapped for asyncio use."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = self.config.endpoint.rstrip("/") + "/api/chat"
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.config.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    async def reply(
        self, user_text: str, history: list[dict[str, str]], extra_system: str = ""
    ) -> str:
        """Return one in-character reply from the local model."""
        system = (self.config.persona or "").strip()
        if extra_system:
            system = (system + "\n\n" + extra_system).strip()
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        limit = max(2, int(self.config.history_turns) * 2)
        messages.extend(
            {"role": m["role"], "content": m["content"]}
            for m in history[-limit:]
            if m.get("role") in ("user", "assistant") and m.get("content")
        )
        messages.append({"role": "user", "content": user_text})
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"temperature": 0.8, "num_predict": 140},
        }
        data = await asyncio.to_thread(self._post, payload)
        message = data.get("message") or {}
        return str(message.get("content") or "").strip()
