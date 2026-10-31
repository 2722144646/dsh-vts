"""Short, procedural action cues (nods, shakes, surprise)."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random


def _ease_in_out(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def _ease_out(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return 1.0 - (1.0 - x) ** 3


def _minjerk(x: float) -> float:
    """5th-order minimum-jerk curve: zero velocity/acceleration at both ends."""
    x = max(0.0, min(1.0, x))
    return 10.0 * x**3 - 15.0 * x**4 + 6.0 * x**5


def _bezier(x: float, *, x1: float = 0.42, y1: float = 0.0, x2: float = 0.58, y2: float = 1.0) -> float:
    """Cubic Bezier ease with numerically solved time parameter."""
    x = max(0.0, min(1.0, x))
    lo, hi = 0.0, 1.0
    for _ in range(14):
        mid = (lo + hi) * 0.5
        bx = 3.0 * (1.0 - mid) ** 2 * mid * x1 + 3.0 * (1.0 - mid) * mid**2 * x2 + mid**3
        if bx < x:
            lo = mid
        else:
            hi = mid
    t = (lo + hi) * 0.5
    return 3.0 * (1.0 - t) ** 2 * t * y1 + 3.0 * (1.0 - t) * t**2 * y2 + t**3


@dataclass
class ActionCue:
    kind: str
    start: int
    duration: int
    intensity: float = 0.7
    seed: int = 0
    curve: str = "minjerk"
    finished: bool = False

    def sample(self, sample: int) -> dict[str, float]:
        if sample < self.start or sample >= self.start + self.duration:
            return {}
        t = (sample - self.start) / max(1, self.duration)
        amp = max(0.0, min(1.4, self.intensity))
        if self.kind == "demo_nod":
            curve = _bezier if self.curve == "bezier" else _minjerk
            if t < 0.45:
                pulse = curve(t / 0.45)
            else:
                pulse = 1.0 - curve((t - 0.45) / 0.55)
            return {"pitch": 8.0 * self.intensity * pulse}
        if self.kind in ("nod", "nod_strong"):
            # Anticipation -> down -> hold -> slow return.
            if t < 0.12:
                pitch = -1.5 * _minjerk(t / 0.12)
            elif t < 0.32:
                pitch = -1.5 + 9.5 * _minjerk((t - 0.12) / 0.20)
            elif t < 0.44:
                pitch = 8.0
            else:
                pitch = 8.0 * (1.0 - _minjerk((t - 0.44) / 0.56))
            return {"pitch": pitch * amp}
        if self.kind in ("shake", "shake_head"):
            if t < 0.1:
                return {"yaw": 0.0}
            yaw = math.sin((t - 0.1) * math.pi * 5.0) * 8.0 * (1.0 - t)
            return {"yaw": yaw * amp}
        if self.kind in ("surprise", "startle"):
            if t < 0.18:
                pitch = -10.0 * _minjerk(t / 0.18)
            else:
                pitch = -10.0 * (1.0 - _minjerk((t - 0.18) / 0.82))
            return {"pitch": pitch * amp, "eye_wide": amp}
        if self.kind in ("tilt", "thinking_tilt"):
            tilt = _minjerk(min(1.0, t * 2.0)) * (1.0 - _minjerk(max(0.0, (t - 0.5) * 2.0))) * 4.0
            return {"roll": tilt * amp}
        if self.kind == "emphasis":
            pitch = math.sin(t * math.pi) * 2.5
            return {"pitch": pitch * amp}
        return {}

    @classmethod
    def create(
        cls,
        kind: str,
        *,
        start: int,
        sample_rate: int,
        intensity: float = 0.7,
        duration_ms: int | None = None,
        curve: str = "minjerk",
    ) -> "ActionCue":
        durations = {
            "nod": 680,
            "nod_strong": 820,
            "shake": 700,
            "shake_head": 700,
            "surprise": 760,
            "startle": 760,
            "tilt": 900,
            "thinking_tilt": 1000,
            "emphasis": 340,
        }
        duration = duration_ms or durations.get(kind, 420)
        return cls(
            kind=kind,
            start=start,
            duration=max(1, int(sample_rate * duration / 1000)),
            intensity=max(0.0, min(1.4, intensity)),
            seed=random.randint(0, 1_000_000),
            curve=curve,
        )
