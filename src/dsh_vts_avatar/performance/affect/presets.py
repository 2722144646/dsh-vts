"""Affect anchors: the vocabulary shared by evidence sources and the composer.

An anchor is the *coordinate* of a label, not an expression. Expressions are
built in the composer (Phase 2) from a blend of labels; preset screenshots stay
as reference points for calibration and blind tests.

Values mirror ``performance.emotion.EmotionState.set`` so that the legacy path
and the affect path agree on what "happy" or "sad" means at intensity 1.0.
"""

from __future__ import annotations

# label -> (valence -1..1, arousal 0..1)
ANCHORS: dict[str, tuple[float, float]] = {
    "happy": (0.75, 0.55),
    "joy": (0.80, 0.60),
    "excited": (0.70, 0.90),
    "sad": (-0.70, 0.20),
    "tired": (-0.30, 0.10),
    "angry": (-0.65, 0.85),
    "surprised": (0.15, 0.90),
    "thinking": (-0.05, 0.35),
    "wronged": (-0.35, 0.35),
    "curious": (0.25, 0.50),
    "shy": (0.20, 0.35),
    "neutral": (0.00, 0.15),
    "calm": (0.15, 0.15),
}

# Unknown labels must not silently become a strong emotion. They fall back to
# the neutral anchor, which only nudges the state.
UNKNOWN_ANCHOR = (0.0, 0.25)

# Labels the legacy keyword/tag path can emit. Kept for the compatibility
# layer that turns legacy presets into evidence instead of hard overrides.
LEGACY_LABELS = tuple(ANCHORS)


def anchor(label: str) -> tuple[float, float]:
    return ANCHORS.get(label, UNKNOWN_ANCHOR)


def is_known(label: str) -> bool:
    return label in ANCHORS
