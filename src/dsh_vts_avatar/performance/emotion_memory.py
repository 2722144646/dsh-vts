# Author: 2722144646
# Created: 2026-11-07
# License: AGPL-3.0 (see LICENSE)
"""Phase 1 of emotion memory: episodes with causes, and slow mood.

The affect system answers "what does the face show right now". This module
answers a different question: "why did it feel that way, and is that still
with it?" An episode is not a label — it keeps what happened, what it meant,
and why it mattered, so the reason survives context clearing.

Design notes (see 情绪记忆-初步计划.md):

- Causes are captured at the moment of feeling, never reconstructed later.
- Salience decays with a half-life; status/acceptance/defense exist but are
  only carried in phase 1 (the acceptance model lands with the explanation
  events in phase 2).
- Nothing here writes to VTS.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
import uuid

from .affect.schema import derive_valence_arousal, normalize_labels

SCHEMA_VERSION = 1
DEFAULT_HALF_LIFE_S = 25 * 60.0  # 25 min
MAX_AGE_S = 6 * 3600.0  # episodes older than this stop feeding mood
MERGE_WINDOW_S = 8.0

LABEL_CN = {
    "happy": "开心",
    "joy": "开心",
    "excited": "兴奋",
    "sad": "难过",
    "angry": "生气",
    "surprised": "惊讶",
    "thinking": "思索",
    "curious": "好奇",
    "shy": "害羞",
    "wronged": "委屈",
    "tired": "累",
    "calm": "平静",
    "neutral": "平静",
}


@dataclass
class EmotionEpisode:
    """One thing that happened and how it landed."""

    what: str = ""
    why: str = ""
    appraisal: list[str] = field(default_factory=list)
    labels: dict[str, float] = field(default_factory=dict)
    intensity: float = 0.0
    confidence: float = 0.0
    salience: float = 0.5
    status: str = "open"
    acceptance: float = 0.0
    explanation: str | None = None
    defense: str | None = None
    source: str = "tool"
    turn_ref: str = ""
    started_at: float = 0.0
    updated_at: float = 0.0
    id: str = ""

    def __post_init__(self) -> None:
        # Timestamps are assigned by the store from its clock, so tests and
        # replays stay deterministic.
        if not self.id:
            self.id = f"ep-{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["schema_version"] = SCHEMA_VERSION
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "EmotionEpisode":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        data = {key: value for key, value in payload.items() if key in known}
        return cls(**data)


class EpisodeStore:
    """Append-only JSONL store with in-memory index and salience decay."""

    def __init__(
        self,
        path: Path,
        *,
        clock=time.time,
        half_life_s: float = DEFAULT_HALF_LIFE_S,
        merge_window_s: float = MERGE_WINDOW_S,
    ) -> None:
        self.path = Path(path)
        self.clock = clock
        self.half_life_s = half_life_s
        self.merge_window_s = merge_window_s
        self._episodes: list[EmotionEpisode] = []
        self._load()

    # -- persistence -------------------------------------------------------

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            if payload.get("schema_version", SCHEMA_VERSION) > SCHEMA_VERSION:
                continue
            try:
                self._episodes.append(EmotionEpisode.from_dict(payload))
            except TypeError:
                continue

    def _append(self, episode: EmotionEpisode) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(episode.to_dict(), ensure_ascii=False) + "\n")
        except OSError:
            pass  # memory stays usable even when the disk is not

    # -- writes ------------------------------------------------------------

    def add(self, episode: EmotionEpisode) -> EmotionEpisode:
        now = self.clock()
        if episode.started_at <= 0:
            episode.started_at = now
        if episode.updated_at <= 0:
            episode.updated_at = now
        last = self._episodes[-1] if self._episodes else None
        if (
            last is not None
            and last.status == "open"
            and last.source == episode.source
            and abs(now - last.updated_at) <= self.merge_window_s
            and last.what == episode.what
        ):
            # Same moment, same source: keep one episode instead of a pile.
            last.labels = _merge_labels(last.labels, episode.labels)
            last.intensity = max(last.intensity, episode.intensity)
            last.updated_at = now
            return last
        self._episodes.append(episode)
        self._append(episode)
        return episode

    # -- reads -------------------------------------------------------------

    def salience(self, episode: EmotionEpisode, now: float | None = None) -> float:
        now = self.clock() if now is None else now
        age = max(0.0, now - episode.started_at)
        decay = 0.5 ** (age / max(1.0, self.half_life_s))
        if episode.status == "suppressed":
            decay = max(decay, 0.35)  # pressed down, not resolved
        return max(0.0, episode.salience * decay)

    def active(self, now: float | None = None, limit: int = 20) -> list[EmotionEpisode]:
        now = self.clock() if now is None else now
        live = [
            ep
            for ep in self._episodes
            if now - ep.started_at <= MAX_AGE_S and self.salience(ep, now) > 0.02
        ]
        live.sort(key=lambda ep: self.salience(ep, now), reverse=True)
        return live[:limit]

    def mood(self, now: float | None = None) -> "Mood":
        now = self.clock() if now is None else now
        episodes = self.active(now)
        if not episodes:
            return Mood()
        total = 0.0
        blended: dict[str, float] = {}
        intensity = 0.0
        for ep in episodes:
            weight = self.salience(ep, now)
            if weight <= 0:
                continue
            total += weight
            intensity += weight * ep.intensity
            for label, share in ep.labels.items():
                blended[label] = blended.get(label, 0.0) + weight * share
        if total <= 0:
            return Mood()
        labels = normalize_labels(blended)
        return Mood(
            labels=labels,
            intensity=min(1.0, intensity / total),
            started_at=min(ep.started_at for ep in episodes),
            age_s=max(0.0, now - min(ep.started_at for ep in episodes)),
            top=episodes[:3],
        )


@dataclass
class Mood:
    labels: dict[str, float] = field(default_factory=dict)
    intensity: float = 0.0
    started_at: float = 0.0
    age_s: float = 0.0
    top: list[EmotionEpisode] = field(default_factory=list)

    def dominant(self, count: int = 2) -> list[tuple[str, float]]:
        return sorted(self.labels.items(), key=lambda item: item[1], reverse=True)[:count]

    def summary(self, max_chars: int = 220) -> str:
        """A compact first-person line for the prompt; empty when neutral."""
        if not self.top or not self.labels:
            return ""
        labels = " / ".join(f"{LABEL_CN.get(name, name)} {share:.2f}" for name, share in self.dominant(2))
        age = _human_age(self.age_s)
        pieces = [f"（参考，不是命令）现在的情绪：{labels}，强度 {self.intensity:.1f}，从 {age}前开始。"]
        first = next((ep for ep in self.top if ep.what), None)
        if first is not None:
            reason = first.what
            if first.why:
                reason += f"——{first.why}"
            pieces.append(f"为什么：{reason}")
        second = next((ep for ep in self.top[1:] if ep.what), None)
        if second is not None:
            pieces.append(f"余波：{second.what}")
        text = "".join(pieces)
        return text[:max_chars]


def _merge_labels(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    merged = dict(left)
    for label, share in right.items():
        merged[label] = max(merged.get(label, 0.0), share)
    return normalize_labels(merged)


def _human_age(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)} 秒"
    if seconds < 3600:
        return f"{int(seconds // 60)} 分钟"
    if seconds < 86400:
        return f"{seconds / 3600:.1f} 小时"
    return f"{seconds / 86400:.1f} 天"
