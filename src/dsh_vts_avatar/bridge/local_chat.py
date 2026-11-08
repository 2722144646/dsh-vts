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


class LocalDirector:
    """Asks the local model to pick the next idle motion.

    Falls back to ``None`` on any failure (timeout, bad JSON, unknown kind);
    the rule-based pool then decides, so a dead model never freezes the avatar.
    """

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
        with urllib.request.urlopen(request, timeout=self.config.director_timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    async def suggest(
        self, kinds: list[str], context: dict[str, Any]
    ) -> tuple[str, float] | None:
        prompt = (
            "你是 Live2D 角色的待机导演。根据下面的状态，从候选动作里挑一个最自然的，"
            "只输出 JSON，不要解释：{\"kind\":\"候选之一\",\"intensity\":0.1-1.2}\n"
            "尽量避开 recent 里刚做过的动作，除非完全没有别的选择。\n"
            f"状态：{json.dumps(context, ensure_ascii=False)}\n"
            f"候选：{', '.join(kinds)}"
        )
        payload = {
            "model": self.config.director_model,
            "messages": [
                {"role": "system", "content": "你只输出一行 JSON。"},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "think": False,
            "options": {"temperature": 0.7, "num_predict": 100},
        }
        try:
            data = await asyncio.to_thread(self._post, payload)
        except Exception:
            return None
        text = str((data.get("message") or {}).get("content") or "").strip()
        if text.startswith("```"):
            text = text.strip("`").strip()
            if text.startswith("json"):
                text = text[4:].strip()
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return None
        kind = str(obj.get("kind") or obj.get("action") or obj.get("name") or "")
        if kind not in kinds:
            return None
        try:
            intensity = float(obj.get("intensity", 0.5))
        except (TypeError, ValueError):
            intensity = 0.5
        return kind, round(min(1.2, max(0.1, intensity)), 3)


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
