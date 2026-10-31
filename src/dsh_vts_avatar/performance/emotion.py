"""Continuous emotion state with attack/release and decay."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
import time


@dataclass
class EmotionState:
    valence: float = 0.0  # -1..1
    arousal: float = 0.0  # 0..1
    intensity: float = 0.0  # 0..1
    target_valence: float = 0.0
    target_arousal: float = 0.0
    target_intensity: float = 0.0
    last_update: float = 0.0
    hold_until: float = 0.0

    def __post_init__(self) -> None:
        if not self.last_update:
            self.last_update = time.perf_counter()

    def set(self, name: str, intensity: float = 0.7) -> None:
        name = name.lower().strip()
        intensity = max(0.0, min(1.0, float(intensity)))
        presets = {
            "happy": (0.75, 0.55),
            "joy": (0.8, 0.6),
            "excited": (0.7, 0.9),
            "sad": (-0.7, 0.2),
            "angry": (-0.65, 0.85),
            "surprised": (0.15, 0.9),
            "surprise": (0.15, 0.9),
            "thinking": (-0.05, 0.35),
            "curious": (0.25, 0.5),
            "shy": (0.2, 0.35),
            "neutral": (0.0, 0.15),
            "calm": (0.15, 0.15),
        }
        v, a = presets.get(name, (0.0, 0.25))
        self.target_valence = v * intensity
        self.target_arousal = a * intensity
        self.target_intensity = intensity
        # Do not jump the current value; let update() ease into the target.
        # A tiny immediate nudge keeps the face from feeling one beat late.
        self.valence += (self.target_valence - self.valence) * 0.12
        self.arousal += (self.target_arousal - self.arousal) * 0.12
        self.intensity += (self.target_intensity - self.intensity) * 0.12
        self.hold_until = time.perf_counter() + 1.8

    def update(self, dt: float) -> None:
        dt = max(0.0, min(0.2, dt))
        # Slow attack/release: expressions should bloom, hold, then fade.
        attack = 1.0 - math.exp(-dt / 0.38)
        release = 1.0 - math.exp(-dt / 2.2)
        self.valence += (self.target_valence - self.valence) * (attack if abs(self.target_valence) > abs(self.valence) else release)
        self.arousal += (self.target_arousal - self.arousal) * (attack if self.target_arousal > self.arousal else release)
        self.intensity += (self.target_intensity - self.intensity) * release
        # Hold the expression, then drift back toward a calm baseline.
        if time.perf_counter() >= self.hold_until:
            decay = 1.0 - math.exp(-dt / 4.0)
            self.target_valence *= 1.0 - decay * 0.30
            self.target_arousal *= 1.0 - decay * 0.22
            self.target_intensity *= 1.0 - decay * 0.18
        self.last_update = time.perf_counter()


_LEXICON: dict[str, tuple[str, float]] = {
    "太好了": ("happy", 0.8),
    "太棒": ("happy", 0.85),
    "好耶": ("happy", 0.8),
    "真好": ("happy", 0.6),
    "开心": ("happy", 0.8),
    "高兴": ("happy", 0.7),
    "哈哈": ("happy", 0.7),
    "喜欢": ("happy", 0.6),
    "可爱": ("happy", 0.6),
    "有趣": ("happy", 0.55),
    "厉害": ("happy", 0.6),
    "不错": ("happy", 0.5),
    "棒": ("happy", 0.55),
    "期待": ("happy", 0.55),
    "放心": ("happy", 0.45),
    "没问题": ("happy", 0.45),
    "当然": ("happy", 0.4),
    "谢谢": ("happy", 0.5),
    "好的": ("happy", 0.35),
    "收到": ("happy", 0.35),
    "抱歉": ("sad", 0.6),
    "对不起": ("sad", 0.6),
    "难过": ("sad", 0.8),
    "可惜": ("sad", 0.6),
    "唉": ("sad", 0.5),
    "累": ("sad", 0.5),
    "困": ("sad", 0.4),
    "生气": ("angry", 0.8),
    "讨厌": ("angry", 0.7),
    "烦": ("angry", 0.6),
    "离谱": ("angry", 0.6),
    "居然": ("surprised", 0.7),
    "竟然": ("surprised", 0.7),
    "真的吗": ("surprised", 0.8),
    "不会吧": ("surprised", 0.8),
    "天哪": ("surprised", 0.8),
    "哇": ("surprised", 0.8),
    "诶": ("surprised", 0.55),
    "咦": ("surprised", 0.55),
    "让我想想": ("thinking", 0.6),
    "也许": ("thinking", 0.4),
    "可能": ("thinking", 0.35),
    "嗯": ("thinking", 0.3),
    "great": ("happy", 0.8),
    "happy": ("happy", 0.8),
    "love": ("happy", 0.7),
    "nice": ("happy", 0.6),
    "sorry": ("sad", 0.7),
    "sad": ("sad", 0.7),
    "angry": ("angry", 0.8),
    "wow": ("surprised", 0.8),
    "really": ("surprised", 0.6),
    "maybe": ("thinking", 0.4),
}

_TAG_RE = re.compile(r"\[(emotion|情绪)\s*:\s*([a-zA-Z_\u4e00-\u9fff]+)(?:\s+intensity\s*[:=]\s*([0-9.]+))?\]", re.I)


def emotion_from_text(text: str) -> tuple[str, float] | None:
    """Cheap, local affect heuristic; no LLM round-trip and no visible tags."""
    tag = _TAG_RE.search(text)
    if tag:
        try:
            intensity = float(tag.group(3)) if tag.group(3) else 0.8
        except ValueError:
            intensity = 0.8
        return tag.group(2).lower(), max(0.0, min(1.0, intensity))
    lowered = text.lower()
    for token, value in _LEXICON.items():
        if token in lowered:
            return value
    if "?" in text or "？" in text:
        return ("curious", 0.45)
    if "!" in text or "！" in text:
        return ("excited", 0.55)
    if "~" in text or "～" in text:
        return ("happy", 0.45)
    if "..." in text or "……" in text:
        return ("thinking", 0.4)
    return None


def strip_emotion_tags(text: str) -> str:
    return _TAG_RE.sub("", text)
