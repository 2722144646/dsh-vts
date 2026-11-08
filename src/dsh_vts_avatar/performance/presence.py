"""Presence director: what she does when nobody is talking.

Idle behaviour is not random animation — it is chosen from how she feels
(mood), how close the relationship is, and what the diary says. The director
re-plans only when the inputs cross a threshold, so a quiet evening does not
flicker between "bright" and "low" every few seconds.

The caller (engine tick) passes a small ``inputs`` dict; the director returns
an idle cue kind (from ``actions.ACTION_KINDS``) or ``None``.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field


# Kinds whose animation touches mouth parameters. They are allowed at idle but
# must yield immediately when speech starts.
MOUTH_KINDS = {"idle_sigh", "idle_hum", "idle_yawn"}

# Weighted pools per profile. Keys must exist in actions.py.
POOLS: dict[str, dict[str, float]] = {
    "calm": {"idle_glance": 3.0, "idle_tilt": 2.0, "idle_nod": 1.0, "idle_breathe": 2.0},
    "cozy": {"idle_look_user": 3.0, "idle_tilt": 2.0, "idle_sigh": 1.0, "idle_hum": 1.0},
    "bright": {"idle_glance": 2.0, "idle_tilt": 2.5, "idle_nod": 1.5, "idle_hum": 1.5},
    "low": {"idle_look_down": 3.0, "idle_sigh": 2.0, "idle_tilt": 1.0, "idle_breathe": 2.0},
    "tense": {"idle_look_away": 3.0, "idle_breathe": 2.0, "idle_tilt": 1.0, "idle_nod": 0.5},
}

COOLDOWN_S = 3.0
MIN_GAP = 6.0
MAX_GAP = 12.0


@dataclass
class PresenceProfile:
    key: str = "calm"
    energy: float = 0.55
    warmth: float = 0.3
    tension: float = 0.0
    mouth: float = 0.2
    note: str = ""


@dataclass
class PresenceDirector:
    """Chooses idle cues from mood + relationship + diary."""

    base_energy: float = 0.55
    rng: random.Random = field(default_factory=random.Random)
    profile: PresenceProfile = field(default_factory=PresenceProfile)
    _next_at: float = 0.0
    _last: list[str] = field(default_factory=list)

    # -- inputs -----------------------------------------------------------

    def _bucket(self, inputs: dict) -> PresenceProfile:
        labels = {
            str(k).lower(): float(v)
            for k, v in (inputs.get("mood_labels") or {}).items()
        }
        sad = max(labels.get("sad", 0.0), labels.get("suppressed", 0.0),
                  labels.get("wronged", 0.0), labels.get("guilty", 0.0))
        doubt = labels.get("doubtful", 0.0)
        bright = max(labels.get("happy", 0.0), labels.get("relieved", 0.0),
                     labels.get("joy", 0.0), labels.get("excited", 0.0))
        intensity = float(inputs.get("mood_intensity") or 0.0)
        trust = float(inputs.get("trust") or 0.5)
        closeness = float(inputs.get("closeness") or 0.2)
        friction = float(inputs.get("friction") or 0.0)

        tension = max(friction, 0.0) * 0.7 + sad * 0.35 + doubt * 0.15
        warmth = closeness * 0.5 + trust * 0.35 + bright * 0.25 - tension * 0.3
        energy = self.base_energy * (0.75 + 0.5 * max(intensity, 0.15))
        mouth = 0.15 + bright * 0.4 + max(sad - 0.4, 0.0) * 0.3

        if tension >= 0.45:
            key = "tense"
            mouth *= 0.2
        elif sad >= 0.35 and bright < 0.3:
            key = "low"
        elif bright >= 0.45:
            key = "bright"
        elif warmth >= 0.55:
            key = "cozy"
        else:
            key = "calm"
        note = (
            f"tension={tension:.2f} warmth={warmth:.2f} "
            f"sad={sad:.2f} bright={bright:.2f}"
        )
        return PresenceProfile(
            key=key,
            energy=min(1.0, max(0.2, energy)),
            warmth=min(1.0, max(0.0, warmth)),
            tension=min(1.0, max(0.0, tension)),
            mouth=min(1.0, max(0.0, mouth)),
            note=note,
        )

    # -- scheduling -------------------------------------------------------

    def observe(self, inputs: dict) -> PresenceProfile:
        """Update the profile; the pool switches only when the bucket does."""
        self.profile = self._bucket(inputs)
        return self.profile

    def _pick(self, now: float) -> tuple[str, float] | None:
        pool = POOLS.get(self.profile.key, POOLS["calm"])
        candidates = {
            kind: weight
            for kind, weight in pool.items()
            if kind in MOUTH_KINDS and self.profile.mouth < 0.2
        }
        # Mouth kinds are gated by the mouth appetite; if that leaves nothing,
        # fall back to the full pool minus mouth kinds.
        if not candidates:
            candidates = {
                kind: weight
                for kind, weight in pool.items()
                if kind not in MOUTH_KINDS
            }
        if not candidates:
            return None
        # Avoid the last two choices so it does not look like a loop.
        for kind in self._last[-2:]:
            candidates.pop(kind, None)
        if not candidates:
            return None
        kinds = list(candidates)
        weights = [candidates[kind] for kind in kinds]
        kind = self.rng.choices(kinds, weights=weights, k=1)[0]
        self._last = (self._last + [kind])[-4:]
        intensity = 0.45 + 0.35 * min(1.0, self.profile.energy)
        return kind, round(intensity, 3)

    def next_cue(self, now: float, *, speaking: bool) -> tuple[str, float] | None:
        """Return (kind, intensity) when it is time for an idle cue."""
        if speaking:
            return None
        gap = self.rng.uniform(MIN_GAP, MAX_GAP)
        if now < self._next_at:
            return None
        self._next_at = now + gap
        return self._pick(now)


def idle_mouth_allowed(state: str, timeline_active: bool) -> bool:
    """Idle mouth cues are only safe when nothing else is using the mouth."""
    return state != "speaking" and not timeline_active


def silence_seconds(now: float, last_voice: float) -> float:
    return max(0.0, now - last_voice) if last_voice else 0.0


def quiet_bucket(seconds: float) -> str:
    """After a long silence she settles down instead of fidgeting."""
    if seconds >= 300:
        return "very_quiet"
    if seconds >= 90:
        return "quiet"
    return "normal"


def next_gap(seconds: float, rng: random.Random) -> float:
    bucket = quiet_bucket(seconds)
    if bucket == "very_quiet":
        return rng.uniform(MAX_GAP * 1.5, MAX_GAP * 2.5)
    if bucket == "quiet":
        return rng.uniform(MAX_GAP, MAX_GAP * 1.6)
    return rng.uniform(MIN_GAP, MAX_GAP)
