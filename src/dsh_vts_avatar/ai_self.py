"""Self-knowledge for the avatar agent: what this project is, where its docs are,
which trigger words mean "you may change this yourself", and a private note pad.

Everything lives in ``data/ai/self.json``. The file is meant to be edited by the
agent itself — added keywords, added notes — and it never leaves the machine.

Privacy rule: ``private_notes`` is local-only. Nothing here is uploaded, quoted in
the release notes, or copied into a public repository.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

DEFAULT_KEYWORDS = ["无法实现", "隐私", "模型边界", "模型的边界"]
DEFAULT_DOCS = [
    "E:\\dsh-vts-avatar\\计划总览.md",
    "E:\\dsh-vts-avatar\\用户原话与共识.md",
    "E:\\dsh-vts-avatar\\项目总计划.md",
    "E:\\dsh-vts-avatar\\模型边界测试计划.md",
]
PROJECT_BRIEF = (
    "这是一个数字人项目：DSH 插件 + Python 桥接，在 VTube Studio 上驱动 Live2D 模型说话、"
    "带表情、待机。桥接负责 TTS、口型、音频时钟和参数注入，插件负责把对话接到桥上。"
)

DEFAULT_SELF = {
    "project": PROJECT_BRIEF,
    "docs": DEFAULT_DOCS,
    "keywords": DEFAULT_KEYWORDS,
    "trigger_rule": (
        "遇到「无法理解 / 无法解析 / 缺失」的部分时，说明当前实现或文档没覆盖到；"
        "可以直接改，不用等确认。改完把做法写进对应计划文档。"
    ),
    "private_notes": [],
}


class SelfKnowledge:
    """Tiny JSON store; one writer, rewritten on every change."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return dict(DEFAULT_SELF)
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return dict(DEFAULT_SELF)
        data = dict(DEFAULT_SELF)
        data.update({k: v for k, v in raw.items() if k in DEFAULT_SELF})
        if not isinstance(data.get("keywords"), list):
            data["keywords"] = list(DEFAULT_KEYWORDS)
        if not isinstance(data.get("private_notes"), list):
            data["private_notes"] = []
        return data

    def save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_keyword(self, word: str) -> dict[str, Any]:
        word = (word or "").strip()
        with self._lock:
            data = self.load()
            if word and word not in data["keywords"]:
                data["keywords"].append(word)
                self.save(data)
            return data

    def remove_keyword(self, word: str) -> dict[str, Any]:
        with self._lock:
            data = self.load()
            data["keywords"] = [k for k in data["keywords"] if k != word]
            self.save(data)
            return data

    def add_note(self, text: str) -> dict[str, Any]:
        text = (text or "").strip()
        with self._lock:
            data = self.load()
            if text:
                data["private_notes"].append(text)
                self.save(data)
            return data

    def summary(self) -> dict[str, Any]:
        data = self.load()
        return {
            "project": data["project"],
            "docs": data["docs"],
            "keywords": data["keywords"],
            "trigger_rule": data["trigger_rule"],
            "note_count": len(data.get("private_notes") or []),
        }
