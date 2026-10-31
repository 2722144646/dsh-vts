"""Application-layer performance mixer.

VTS is a renderer here, not a mix bus. Every parameter has one owner and a
composition rule; only the final values are sent in one `set` request.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

import numpy as np

from ..config import AvatarConfig
from ..tts.timeline import Timeline, VisemeSegment
from ..vts.profile import ModelProfile
from .actions import ActionCue
from .emotion import EmotionState


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def smoothstep(x: float) -> float:
    x = clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _smooth(current: float, target: float, dt: float, attack: float, release: float) -> float:
    tau = attack if abs(target) > abs(current) else release
    alpha = 1.0 - math.exp(-max(0.0, dt) / max(0.001, tau))
    return current + (target - current) * alpha


class BlinkController:
    """Self-owned blink. VTS auto-blink should be disabled for this to be clean."""

    def __init__(self, per_minute: float = 14.0, seed: int | None = None):
        self.per_minute = max(2.0, min(60.0, per_minute))
        self._rng = random.Random(seed)
        self._next = time.perf_counter() + self._interval()
        self._started: float | None = None

    def _interval(self) -> float:
        base = 60.0 / self.per_minute
        return base * self._rng.uniform(0.65, 1.45)

    def update(self, now: float) -> float:
        if self._started is None and now >= self._next:
            self._started = now
            self._next = now + self._interval()
        if self._started is None:
            return 1.0
        t = (now - self._started) / 0.16
        if t >= 1.0:
            self._started = None
            return 1.0
        if t < 0.42:
            return 1.0 - smoothstep(t / 0.42)
        return smoothstep((t - 0.42) / 0.58)

    def reset(self) -> None:
        self._started = None
        self._next = time.perf_counter() + self._interval()


@dataclass
class PerformanceMixer:
    avatar: AvatarConfig
    sample_rate: int
    profile: ModelProfile | None = None
    backend: str = "continuous"
    emotion: EmotionState = field(default_factory=EmotionState)
    state: str = "idle"
    actions: list[ActionCue] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.backend = self.backend or "continuous"
        self._last_time = time.perf_counter()
        self._started = self._last_time
        self._last_sample = 0
        self._blink = BlinkController(self.avatar.blink_per_min)
        self._values: dict[str, float] = {}
        self._vel: dict[str, float] = {}
        self._state_bias = {"x": 0.0, "y": 0.0, "z": 0.0}
        self._speech_blend = 0.0
        self._speech_target = 0.0
        self._mouth_backend = self._decide_backend()

    def _decide_backend(self) -> str:
        requested = (self.avatar.mouth_backend or "auto").lower()
        if requested != "auto":
            return requested
        if self.profile is not None:
            return self.profile.choose_mouth_backend(self.avatar, "auto")
        return "continuous"

    def set_state(self, state: str) -> None:
        state = (state or "idle").lower()
        if state not in {"idle", "thinking", "speaking", "listening", "error"}:
            state = "idle"
        self.state = state
        self._speech_target = 1.0 if state == "speaking" else 0.0

    def set_profile(self, profile: ModelProfile | None, backend: str | None = None) -> None:
        self.profile = profile
        if backend is not None:
            self.backend = backend
        self._mouth_backend = backend or self._decide_backend()

    def set_emotion(self, name: str, intensity: float = 0.7) -> None:
        self.emotion.set(name, intensity)

    def add_action(
        self,
        kind: str,
        intensity: float = 0.7,
        *,
        at_sample: int | None = None,
        duration_ms: int | None = None,
        curve: str = "minjerk",
    ) -> ActionCue:
        cue = ActionCue.create(
            kind,
            start=int(self._last_sample if at_sample is None else at_sample),
            sample_rate=self.sample_rate,
            intensity=intensity,
            duration_ms=duration_ms,
            curve=curve,
        )
        self.actions.append(cue)
        if len(self.actions) > 24:
            self.actions = self.actions[-24:]
        return cue

    def reset(self) -> None:
        self.actions.clear()
        self._blink.reset()
        self._last_time = time.perf_counter()

    def compute(self, sample: int, timeline: Timeline | None, now: float | None = None) -> dict[str, float]:
        now = time.perf_counter() if now is None else now
        self._last_sample = int(sample)
        dt = clamp(now - self._last_time, 0.001, 0.15)
        self._last_time = now
        self.emotion.update(dt)
        self._update_state_bias(dt)
        self._speech_blend = _smooth(
            self._speech_blend, self._speech_target, dt, 0.22, 0.45
        )

        segment = timeline.at(sample) if timeline is not None else _silence()
        rms = timeline.rms_at(sample) if timeline is not None else 0.0
        env = _energy(rms)

        speech = self._speech(segment, env)
        emotional = self._emotion_targets()
        heads = self._head_targets(now)
        actions = self._action_targets(int(sample))
        blink = self._blink.update(now)

        smile = clamp(emotional["smile"] + speech["form"] * 0.20, -1.0, 1.0)
        mouth_open = max(speech["open"], emotional["open"])
        if self.state == "thinking":
            mouth_open *= 0.25
        voice = speech["voice"]

        eye_base = clamp(emotional["eye"] * blink, 0.0, 1.0)
        brow = emotional["brow"]
        angry = emotional["angry"]

        head_x = heads["x"] + actions.get("yaw", 0.0) * self.avatar.head_gain * self._speech_blend
        head_y = heads["y"] + actions.get("pitch", 0.0) * self.avatar.head_gain * self._speech_blend
        head_z = heads["z"] + actions.get("roll", 0.0) * self.avatar.head_gain * self._speech_blend
        if "eye_wide" in actions:
            eye_base = max(eye_base, clamp(actions["eye_wide"], 0.0, 1.0))

        body_x = math.sin((now - self._started) * 2 * math.pi * 0.06 + 1.4) * 0.9
        body_y = math.sin((now - self._started) * 2 * math.pi * 0.04 + 0.2) * 0.4
        body_z = math.sin((now - self._started) * 2 * math.pi * 0.05 + 2.7) * 1.2

        params: dict[str, float] = {}
        a = self.avatar
        jaw = 0.0
        funnel = 0.0
        pucker = 0.0
        if segment.kind == "vowel":
            if segment.viseme == "O":
                funnel = 0.45
                pucker = 0.32
            elif segment.viseme == "U":
                funnel = 0.22
                pucker = 0.52
            jaw = 0.65 * mouth_open
        elif segment.kind == "consonant":
            # Bilabials close the jaw; other consonants keep a small opening.
            jaw = 0.0 if segment.open <= 0.06 else 0.45 * mouth_open
        if self._mouth_backend in ("continuous", "hybrid"):
            params[a.mouth_open] = self._sv(a.mouth_open, mouth_open * a.mouth_gain, dt, 0.025, 0.080)
        if self._mouth_backend in ("vowel", "hybrid"):
            for name, value in voice.items():
                target = value * a.mouth_gain
                params[name] = self._sv(name, target, dt, 0.020, 0.070)
            # The vowel blendshape defines the shape; MouthOpen supplies the
            # visible jaw opening. Keep it substantial, not a token 0.18.
            params[a.mouth_open] = self._sv(
                a.mouth_open, mouth_open * 0.85 * a.mouth_gain, dt, 0.025, 0.080
            )
        if self.profile is not None:
            if self.profile.has(a.jaw_open):
                params[a.jaw_open] = self._sv(a.jaw_open, jaw * a.mouth_gain, dt, 0.025, 0.085)
            if self.profile.has(a.mouth_funnel):
                params[a.mouth_funnel] = self._sv(a.mouth_funnel, funnel * a.mouth_gain, dt, 0.030, 0.090)
            if self.profile.has(a.mouth_pucker):
                params[a.mouth_pucker] = self._sv(a.mouth_pucker, pucker * a.mouth_gain, dt, 0.030, 0.090)
            if self.profile.has(a.voice_silence):
                # VTS's automatic lip-sync setup gates the mouth layer with
                # ParamSilence. Keep it "not silent" for the whole spoken
                # utterance so the layer does not flicker between words.
                gate = 0.0 if self.state == "speaking" else 1.0
                params[a.voice_silence] = self._sv(a.voice_silence, gate, dt, 0.18, 0.45)
        params[a.mouth_smile] = self._sv(a.mouth_smile, smile, dt, 0.18, 0.55)
        params[a.eye_left] = self._sv(a.eye_left, eye_base, dt, 0.07, 0.18)
        params[a.eye_right] = self._sv(a.eye_right, eye_base, dt, 0.07, 0.18)
        if (self.avatar.head_motion or "minjerk").lower() == "spring":
            head_smooth = lambda name, target: self._spring(
                name,
                target,
                dt,
                freq=self.avatar.head_spring_hz,
                zeta=self.avatar.head_spring_zeta,
            )
        elif (self.avatar.head_motion or "").lower() == "bezier":
            # The action curves themselves are cubic Bezier; the idle sway is
            # a smooth sine. A light one-pole only removes state/emotion steps,
            # it does not add a second spring or visible lag.
            head_smooth = lambda name, target: self._sv(name, target, dt, 0.08, 0.16)
        else:
            # Minimum-jerk keyframes plus a light asymmetric low-pass: no
            # spring bounce, no visible snap at the start of a nod.
            head_smooth = lambda name, target: self._sv(name, target, dt, 0.13, 0.30)
        params[a.head_x] = self._head_value(a.head_x, clamp(head_x, -30, 30), dt, head_smooth)
        params[a.head_y] = self._head_value(a.head_y, clamp(head_y, -30, 30), dt, head_smooth)
        params[a.head_z] = self._head_value(a.head_z, clamp(head_z, -30, 30), dt, head_smooth)
        # Body is a custom input in VTS; only send it when the contract proves
        # that it exists (or the user explicitly configured a custom contract).
        if self.profile is not None and self.profile.has(a.body_x):
            params[a.body_x] = self._spring(a.body_x, clamp(body_x, -30, 30), dt, freq=0.8)
            params[a.body_y] = self._spring(a.body_y, clamp(body_y, -30, 30), dt, freq=0.8)
            params[a.body_z] = self._spring(a.body_z, clamp(body_z, -30, 30), dt, freq=0.8)
        # Optional emotion parameters; harmless if absent because profile filters.
        params["Brows"] = self._sv("Brows", clamp(brow, -1, 1), dt, 0.18, 0.50)
        params["FaceAngry"] = self._sv("FaceAngry", clamp(angry, 0, 1), dt, 0.20, 0.60)
        if self.profile is not None and self.profile.has(a.brow_inner):
            params[a.brow_inner] = self._sv(
                a.brow_inner, clamp(emotional["inner_brow"], 0, 1), dt, 0.18, 0.55
            )

        if self.profile is not None:
            params = self.profile.filter(params)
        return {key: float(value) for key, value in params.items() if value == value}

    # -- speech -----------------------------------------------------------

    def _speech(self, segment: VisemeSegment, env: float) -> dict[str, object]:
        if segment.kind == "silence":
            return {"open": 0.0, "form": 0.0, "voice": {}, "viseme": ""}
        if segment.kind == "consonant":
            open_value = segment.open
            # Bilabials must override RMS completely.
            if segment.open <= 0.06:
                open_value = 0.035
            voice = {}
            if segment.viseme:
                voice = {self._voice_param(segment.viseme): segment.vowel_gain * 0.35}
            return {"open": open_value, "form": 0.0, "voice": voice, "viseme": segment.viseme}
        # Vowel: RMS modulates amplitude with a floor so /i/ and /u/ survive.
        amp = 0.35 + 0.65 * env
        open_value = segment.open * amp
        vowel = segment.vowel_gain * (0.55 + 0.45 * env)
        voice = {self._voice_param(segment.viseme): vowel}
        return {"open": open_value, "form": segment.form, "voice": voice, "viseme": segment.viseme}

    def _voice_param(self, viseme: str) -> str:
        a = self.avatar
        return {
            "A": a.voice_a,
            "I": a.voice_i,
            "U": a.voice_u,
            "E": a.voice_e,
            "O": a.voice_o,
        }.get(viseme, a.voice_a)

    # -- emotion ----------------------------------------------------------

    def _emotion_targets(self) -> dict[str, float]:
        v = self.emotion.valence
        a = self.emotion.arousal
        i = self.emotion.intensity
        smile = 0.20 + 0.82 * v * (0.6 + 0.4 * i)
        # Surprise widens; sadness lowers; anger narrows and lowers brow.
        eye = 0.92 + 0.10 * max(0.0, a - 0.55) - 0.12 * max(0.0, -v) - 0.04 * max(0.0, v)
        brow = 0.10 + 0.55 * a - 0.45 * max(0.0, -v)
        angry = 0.0
        if v < -0.15 and a > 0.55:
            angry = min(1.0, (-v) * (a - 0.4) * 1.5)
        inner_brow = max(0.0, -v) * 0.9
        open_emotion = 0.35 * max(0.0, a - 0.65) if v >= 0 else 0.0
        return {
            "smile": smile,
            "eye": eye,
            "brow": brow,
            "angry": angry,
            "inner_brow": inner_brow,
            "open": open_emotion,
        }

    # -- head -------------------------------------------------------------

    def _head_targets(self, now: float) -> dict[str, float]:
        t = now - self._started
        gain = clamp(self.avatar.head_gain, 0.5, 2.5)
        x = (
            1.80 * math.sin(2 * math.pi * 0.071 * t)
            + 0.60 * math.sin(2 * math.pi * 0.193 * t + 1.3)
        )
        y = 1.05 * math.sin(2 * math.pi * 0.113 * t + 0.7)
        z = (
            2.00 * math.sin(2 * math.pi * 0.053 * t + 2.0)
            + 0.50 * math.sin(2 * math.pi * 0.227 * t + 0.4)
        )
        v = self.emotion.valence
        a = self.emotion.arousal
        # A small emotional bias; never a big pose change.
        y += (-2.8 * max(0.0, -v) + 1.0 * max(0.0, v)) * self._speech_blend
        z += (3.2 * max(0.0, -v) - 1.0 * max(0.0, v)) * self._speech_blend
        x += self._state_bias["x"] * self._speech_blend
        y += self._state_bias["y"] * self._speech_blend
        z += self._state_bias["z"] * self._speech_blend
        return {"x": x * gain, "y": y * gain, "z": z * gain}

    def _update_state_bias(self, dt: float) -> None:
        """Ease the thinking/listening pose instead of switching it instantly."""
        target = {"x": 0.0, "y": 0.0, "z": 0.0}
        if self.state == "thinking":
            target = {"x": 1.4, "y": -2.0, "z": 5.0}
        elif self.state == "listening":
            target = {"x": 0.5, "y": 0.8, "z": 1.6}
        for key in ("x", "y", "z"):
            self._state_bias[key] = _smooth(
                self._state_bias[key], target[key], dt, 0.28, 0.55
            )

    def _action_targets(self, sample: int) -> dict[str, float]:
        out: dict[str, float] = {}
        active: list[ActionCue] = []
        for cue in self.actions:
            if sample >= cue.start + cue.duration:
                cue.finished = True
                continue
            if sample < cue.start:
                active.append(cue)
                continue
            values = cue.sample(sample)
            for key, value in values.items():
                out[key] = out.get(key, 0.0) + value
            active.append(cue)
        self.actions = active
        return out

    # -- smoothing --------------------------------------------------------

    def _sv(self, name: str, target: float, dt: float, attack: float, release: float) -> float:
        current = self._values.get(name, target)
        value = _smooth(current, target, dt, attack, release)
        self._values[name] = value
        return value

    def _spring(self, name: str, target: float, dt: float, *, freq: float = 1.25, zeta: float = 0.9) -> float:
        """Critically damped-ish spring for head/body motion.

        VTS head parameters are visible immediately; an EMA on a large nod
        target looks like a sudden snap. The spring carries velocity through
        the motion and settles without overshooting much.
        """
        current = self._values.get(name, target)
        velocity = self._vel.get(name, 0.0)
        omega = 2.0 * math.pi * freq
        accel = omega * omega * (target - current) - 2.0 * zeta * omega * velocity
        velocity += accel * dt
        current += velocity * dt
        self._values[name] = current
        self._vel[name] = velocity
        return current

    def _head_value(self, name: str, target: float, dt: float, smooth) -> float:
        """Rate-limit the head before any easing.

        This is the safety net: no state change, action cue, or model event can
        ask the neck to move faster than ``head_max_speed`` degrees/second.
        """
        current = self._values.get(name, target)
        if self.avatar.head_max_speed <= 0:
            return smooth(name, target)
        max_step = float(self.avatar.head_max_speed) * dt
        limited = current + clamp(target - current, -max_step, max_step)
        return smooth(name, limited)


def _silence() -> VisemeSegment:
    return VisemeSegment(0, 1, "_", "", "silence", 0.0, 0.0, 0.0)


def _energy(rms: float) -> float:
    db = 20.0 * math.log10(max(1e-6, float(rms)))
    floor_db = -45.0
    ceil_db = -12.0
    return clamp((db - floor_db) / (ceil_db - floor_db), 0.0, 1.0) ** 0.7


def emotion_bias_from_text(text: str) -> tuple[str, float] | None:
    from .emotion import emotion_from_text

    return emotion_from_text(text)
