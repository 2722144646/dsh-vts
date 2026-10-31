"""Audio playback and the audible clock.

The mixer never trusts wall-clock time while speech is playing. It asks this
engine where the *audible* sample is, then looks up the corresponding phoneme.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Protocol

import numpy as np

from ..config import AudioConfig

log = logging.getLogger(__name__)

try:  # Optional high-quality resampler.
    import soxr  # type: ignore
except Exception:  # pragma: no cover - fallback path
    soxr = None


class Clock(Protocol):
    sample_rate: int

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def clear(self) -> None: ...
    def append(self, audio: np.ndarray) -> None: ...
    def now_sample(self) -> int: ...
    def duration_samples(self) -> int: ...
    @property
    def is_running(self) -> bool: ...


def _resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if src_rate == dst_rate or audio.size == 0:
        return audio
    if soxr is not None:
        return np.asarray(soxr.resample(audio, src_rate, dst_rate), dtype=np.float32)
    # Linear fallback; adequate for a first run, not for final quality.
    duration = audio.size / src_rate
    out_size = max(1, int(round(duration * dst_rate)))
    src_x = np.linspace(0.0, duration, audio.size, endpoint=False)
    dst_x = np.linspace(0.0, duration, out_size, endpoint=False)
    return np.interp(dst_x, src_x, audio).astype(np.float32)


class AudioEngine:
    """PortAudio output stream plus an audible sample counter."""

    def __init__(self, config: AudioConfig, source_rate: int, *, logger: logging.Logger | None = None):
        import sounddevice as sd

        self._sd = sd
        self.config = config
        self.source_rate = int(source_rate)
        self.sample_rate = int(config.sample_rate)
        self.log = logger or log
        self._lock = threading.Lock()
        self._audio = np.zeros(0, dtype=np.float32)
        self._cursor = 0
        self._frames_consumed = 0
        self._stream = None
        self._running = False
        self._latency_samples = 0
        self._last_underrun_log = 0.0

    def _pick_device(self) -> int | None:
        if self.config.device is not None:
            return self.config.device
        try:
            default = self._sd.default.device
            out = default[1] if isinstance(default, (tuple, list)) else default
            if out is not None and int(out) >= 0:
                name = str(self._sd.query_devices(out).get("name", ""))
                if any(token in name.lower() for token in ("bluetooth", "wh-1000", "hands-free")):
                    self.log.warning(
                        "default output looks like Bluetooth (%s); use wired audio for low latency",
                        name,
                    )
                return int(out)
        except Exception:
            pass
        return None

    def start(self) -> None:
        if self._running:
            return
        device = self._pick_device()
        self._stream = self._sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=256,
            device=device,
            latency="low",
            callback=self._callback,
        )
        self._stream.start()
        self._running = True
        try:
            latency_s = float(self._stream.latency)
        except Exception:
            latency_s = 0.02
        self._latency_samples = int(latency_s * self.sample_rate) + int(
            self.config.av_offset_ms * self.sample_rate / 1000.0
        )
        self.log.info(
            "audio started: device=%s rate=%d latency=%.1fms av_offset=%dms",
            device,
            self.sample_rate,
            latency_s * 1000,
            self.config.av_offset_ms,
        )

    def _callback(self, outdata, frames, _time_info, status) -> None:
        if status:
            # Avoid log spam during underruns.
            now = time.perf_counter()
            if now - self._last_underrun_log > 1.0:
                self._last_underrun_log = now
                self.log.warning("audio callback status: %s", status)
        with self._lock:
            end = self._cursor + frames
            chunk = self._audio[self._cursor:end]
            if chunk.size < frames:
                outdata[: chunk.size, 0] = chunk
                outdata[chunk.size :, 0] = 0.0
            else:
                outdata[:, 0] = chunk
            self._cursor = min(end, self._audio.size)
            self._frames_consumed += frames

    def append(self, audio: np.ndarray) -> None:
        data = _resample(audio, self.source_rate, self.sample_rate)
        with self._lock:
            # A little silence between chunks avoids clicks at boundaries.
            self._audio = np.concatenate([self._audio, data])
            self._audio = self._audio.astype(np.float32, copy=False)

    def clear(self) -> None:
        with self._lock:
            self._audio = np.zeros(0, dtype=np.float32)
            self._cursor = 0
            self._frames_consumed = 0

    def stop(self) -> None:
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def now_sample(self) -> int:
        with self._lock:
            played = max(0, self._frames_consumed - self._latency_samples)
        # Map audible output samples back to the source timeline.
        if self.sample_rate == self.source_rate:
            return int(played)
        return int(played * self.source_rate / self.sample_rate)

    def duration_samples(self) -> int:
        with self._lock:
            return int(self._audio.size * self.source_rate / self.sample_rate)

    @property
    def is_running(self) -> bool:
        return self._running


class NullAudioEngine:
    """Silent clock for VTS-only testing.

    Appending audio advances the virtual timeline by its duration; playback
    then follows monotonic time. This exercises the exact same mixer path
    without requiring a working output device.
    """

    def __init__(self, config: AudioConfig, source_rate: int):
        self.config = config
        self.source_rate = int(source_rate)
        self.sample_rate = int(config.sample_rate)
        self._total = np.zeros(0, dtype=np.float32)
        self._started_at: float | None = None
        self._offset = 0

    def start(self) -> None:
        if self._started_at is None:
            self._started_at = time.perf_counter()

    def stop(self) -> None:
        self._started_at = None
        self._total = np.zeros(0, dtype=np.float32)
        self._offset = 0

    def clear(self) -> None:
        self._total = np.zeros(0, dtype=np.float32)
        self._offset = 0
        self._started_at = time.perf_counter()

    def append(self, audio: np.ndarray) -> None:
        data = _resample(audio, self.source_rate, self.sample_rate)
        self._total = np.concatenate([self._total, data])

    def now_sample(self) -> int:
        if self._started_at is None:
            return 0
        elapsed = time.perf_counter() - self._started_at
        played = int(elapsed * self.sample_rate) + self._offset
        played = min(played, int(self._total.size))
        if self.sample_rate == self.source_rate:
            return played
        return int(played * self.source_rate / self.sample_rate)

    def duration_samples(self) -> int:
        return int(self._total.size * self.source_rate / self.sample_rate)

    @property
    def is_running(self) -> bool:
        return self._started_at is not None


def make_clock(config: AudioConfig, source_rate: int) -> Clock:
    if config.enabled:
        return AudioEngine(config, source_rate)
    return NullAudioEngine(config, source_rate)
