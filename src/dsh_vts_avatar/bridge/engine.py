"""The local avatar engine: TTS -> timeline -> audio clock -> mixer -> VTS."""

from __future__ import annotations

import asyncio
import logging
import math
import random
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

import numpy as np

from ..audio.engine import Clock, make_clock
from ..config import Config
from ..ndi.capture import NDICapture
from ..performance import PerformanceMixer, emotion_from_text, strip_emotion_tags
from ..performance.affect import AffectEvidence, AffectLog, AffectState
from ..performance.mouth_monitor import MouthMonitor
from ..performance.recorder import record_expression
from ..tts import PiperTTS, SyntheticTTS
from ..tts.distill import DistilledPiperTTS
from ..tts.rvc import RVCConverter
from ..tts.timeline import Timeline
from ..vts.client import VTSClient, VTSError, find_vts_port
from ..vts.profile import ModelProfile, load_for_model, profile_path
from .segmentation import Segmenter
from .text_filter import SpeechTextFilter

log = logging.getLogger(__name__)

EventSink = Callable[[dict[str, Any]], Awaitable[None]]


class AvatarEngine:
    def __init__(self, config: Config, *, on_event: EventSink | None = None):
        self.config = config
        self.on_event = on_event
        self.profile = ModelProfile.load(config.contract_path)
        self.tts: PiperTTS | SyntheticTTS | None = None
        self.clock: Clock | None = None
        self.mixer: PerformanceMixer | None = None
        self.vts = VTSClient(config.vts)
        self.segmenter = Segmenter()
        self.speech_filter = SpeechTextFilter()
        self.segmenter_gesture_rate = config.avatar.gesture_rate
        self.timeline: Timeline | None = None
        self._parts: list[Timeline] = []
        self._pending_gap_s = 0.0
        self._queue: asyncio.Queue[str] | None = None
        self._worker: asyncio.Task[None] | None = None
        self._ticker: asyncio.Task[None] | None = None
        self._watcher: asyncio.Task[None] | None = None
        self._closing = False
        self._utterance_open = False
        self._last_emit = 0.0
        self._last_sample = 0
        self._started = False
        self.utterance_id = ""
        self._last_tts_ms = 0.0
        self._last_error = ""
        self.ndi: NDICapture | None = None
        self._last_ndi = None
        self._affect_buffer = ""
        self._model_id = self.profile.model_id
        self._last_model_check = 0.0
        self._model_refreshing = False
        self._recording = False
        self._last_params: dict[str, float] = {}
        self._speech_generation = 0
        self.mouth_monitor = MouthMonitor()
        self.rvc = RVCConverter(config.rvc)
        # Phase 1/3: append-only record of every evidence record and the state
        # it produced, so a session can be replayed and audited later.
        self.affect_log = AffectLog(config.root / "data" / "affect")

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._load_tts()
        await self._warmup_tts()
        self.clock = make_clock(self.config.audio, self.tts.sample_rate)
        self.mixer = PerformanceMixer(
            self.config.avatar,
            self.tts.sample_rate,
            profile=self.profile,
            backend=self.profile.choose_mouth_backend(self.config.avatar, "auto"),
            use_affect=self.config.emotion.is_affect,
        )
        self.mixer.set_state("idle")
        self._queue = asyncio.Queue(maxsize=16)
        self._worker = asyncio.create_task(self._synth_worker(), name="avatar-tts")
        await self._connect_vts()
        self._watcher = asyncio.create_task(self._vts_watchdog(), name="avatar-vts-startup-check")
        if self.config.audio.auto_calibrate and hasattr(self.clock, "calibrate_output_latency"):
            await asyncio.to_thread(self.clock.calibrate_output_latency)
        self._open_ndi()
        self._ticker = asyncio.create_task(self._tick_loop(), name="avatar-tick")
        await self._emit(
            {
                "type": "ready",
                "vts_connected": self.vts.connected,
                "profile": self.profile.summary(),
                "tts": type(self.tts).__name__,
                "sample_rate": self.tts.sample_rate,
                "backend": self.mixer.backend if self.mixer else "none",
            }
        )

    def _load_tts(self) -> None:
        if self.config.tts.distill_model is not None:
            distilled = DistilledPiperTTS(self.config.tts)
            if distilled.available:
                try:
                    distilled.load()
                    self.tts = distilled
                    log.info("using distilled Piper TTS at %d Hz", distilled.sample_rate)
                    return
                except Exception as exc:
                    self._last_error = f"Distilled Piper unavailable: {exc}"
                    log.warning("%s; falling back to reference Piper", self._last_error)
        piper = PiperTTS(self.config.tts)
        if piper.available:
            try:
                piper.load()
                self.tts = piper
                log.info("using Piper TTS at %d Hz", piper.sample_rate)
                return
            except Exception as exc:
                self._last_error = f"Piper unavailable: {exc}"
                log.warning("%s; falling back to silent synthetic TTS", self._last_error)
        else:
            self._last_error = "Piper voice/config not found; using silent synthetic TTS"
            log.warning(self._last_error)
        self.tts = SyntheticTTS(language=self.config.tts.language, sample_rate=48000)

    async def _warmup_tts(self) -> None:
        """Pay the first ONNX inference cost before the first real reply."""
        if self.tts is None:
            return
        try:
            started = time.perf_counter()
            await asyncio.to_thread(self.tts.synthesize, "你好。")
            log.info("TTS warmup finished in %.0f ms", (time.perf_counter() - started) * 1000)
        except Exception as exc:
            log.warning("TTS warmup failed: %s", exc)

    async def _connect_vts(self) -> bool:
        try:
            await self.vts.connect()
        except Exception as exc:
            # A VTS restart can re-bind its API on a neighbouring port. Probe
            # before falling back to the error path; otherwise every restart
            # silently disconnects the avatar until the bridge is restarted.
            port = await find_vts_port(self.config.vts.host, exclude=self.config.vts.port)
            if port is None:
                return await self._vts_connect_failed(exc)
            log.info("VTS API port moved %s -> %s", self.config.vts.port, port)
            self.config.vts.port = port
            try:
                await self.vts.connect()
            except Exception as retry_exc:
                return await self._vts_connect_failed(retry_exc)
        try:
            self.profile = await self._query_profile()
        except Exception as exc:
            return await self._vts_connect_failed(exc)
        self._model_id = self.profile.model_id
        if self.mixer is not None:
            self.mixer.set_profile(
                self.profile,
                self.profile.choose_mouth_backend(self.config.avatar, "auto"),
            )
        log.info("VTS ready: %s", self.profile.summary())
        return True

    async def _vts_connect_failed(self, exc: Exception) -> bool:
        self._last_error = str(exc)
        log.warning("VTS connection failed: %s", exc)
        await self._emit(
            {
                "type": "error",
                "code": "vts_connect_failed",
                "message": str(exc),
                "retryable": True,
            }
        )
        return False

    async def _vts_watchdog(self) -> None:
        """One startup check, then leave the connection alone.

        Same policy as the audio calibration: probe once when the bridge
        starts (including a VTS API port that moved), and do not poll. A later
        VTS restart is re-attached by the explicit ``reconnect`` message.
        """
        await asyncio.sleep(3.0)
        if self._closing or self._recording:
            return
        if self.vts.connected and self.vts.authenticated:
            return
        log.info("VTS startup check: connection lost, reconnecting once")
        try:
            await self.vts.close()
        except Exception:
            pass
        try:
            if await self._connect_vts():
                self._last_error = ""
                log.info("VTS startup check: reconnected")
                await self._emit({"type": "vts.reconnected", "profile": self.profile.summary()})
        except Exception as exc:
            self._last_error = f"VTS reconnect failed: {exc}"
            log.warning(self._last_error)

    async def _query_profile(self) -> ModelProfile:
        """Build a model profile from the live VTS parameter lists.

        A contract probe is model-specific. If the saved contract belongs to a
        different model (the user switched avatars), do not apply its probe
        results; build a fresh capability profile instead.
        """
        model = await self.vts.current_model()
        model_id = str(model.get("modelID", ""))
        model_name = str(model.get("modelName", ""))
        inputs = await self.vts.list_input_params()
        live2d = await self.vts.list_live2d_params()
        saved = load_for_model(self.config.profile_dir, model_id)
        if not saved.input_params:
            # Migration fallback: the old single contract.json may hold this
            # model's tuning/probes. Only use it if the model ID matches.
            legacy = ModelProfile.load(self.config.contract_path)
            if legacy.model_id and legacy.model_id == model_id:
                saved = legacy
        if saved.input_params:
            saved.model_name = model_name or saved.model_name
            saved.mouth_backend = saved.choose_mouth_backend(self.config.avatar, self.config.avatar.mouth_backend)
            return saved
        profile = ModelProfile(model_name=model_name, model_id=model_id)
        from ..vts.profile import ParameterInfo

        for raw in inputs.get("defaultParameters", []):
            info = ParameterInfo.from_vts(raw)
            if info.name:
                profile.input_params[info.name] = info
        for raw in inputs.get("customParameters", []):
            info = ParameterInfo.from_vts(raw)
            if info.name:
                profile.input_params[info.name] = info
        for raw in live2d.get("parameters", []):
            info = ParameterInfo.from_vts(raw)
            if info.name:
                profile.live2d_params[info.name] = info
        profile.mouth_backend = profile.choose_mouth_backend(self.config.avatar, self.config.avatar.mouth_backend)
        return profile

    async def _refresh_model(self) -> None:
        if self._model_refreshing or not self.vts.authenticated:
            return
        self._model_refreshing = True
        try:
            model = await self.vts.current_model()
            model_id = str(model.get("modelID", ""))
            if not model_id or model_id == self._model_id:
                return
            log.info("VTS model changed: %s -> %s", self._model_id, model_id)
            await self.cancel()
            profile = await self._query_profile()
            self.profile = profile
            self._model_id = model_id
            if self.mixer is not None:
                self.mixer.set_profile(
                    profile,
                    profile.choose_mouth_backend(self.config.avatar, "auto"),
                )
            await self._emit(
                {
                    "type": "model.changed",
                    "model": profile.summary(),
                    "backend": profile.mouth_backend,
                }
            )
        except Exception as exc:
            log.debug("model refresh failed: %s", exc)
        finally:
            self._model_refreshing = False

    async def reconnect(self) -> bool:
        if self.vts.authenticated:
            return True
        return await self._connect_vts()

    def _open_ndi(self) -> None:
        try:
            capture = NDICapture(source_hint=self.config.ndi.source, snapshot_dir=self.config.snapshot_dir)
            if capture.open(wait_ms=1200):
                self.ndi = capture
                log.info("NDI self-view connected")
            else:
                capture.close()
                log.info("NDI self-view not found (enable NDI output in VTS)")
        except Exception as exc:
            log.info("NDI self-view unavailable: %s", exc)

    async def snapshot(self, frames: int = 1, interval_ms: int = 180) -> dict[str, Any] | None:
        """Capture one frame or a short NDI sequence for self-inspection."""
        if self.ndi is None:
            self._open_ndi()
        if self.ndi is None:
            return None
        try:
            if frames > 1:
                metrics_list, sheet = await asyncio.to_thread(
                    self.ndi.capture_sequence, frames, interval_ms
                )
                if not metrics_list or sheet is None:
                    return None
                last = metrics_list[-1]
                self._last_ndi = last
                return {
                    "path": sheet,
                    "sequence": sheet,
                    "paths": [item.snapshot for item in metrics_list],
                    "frames": len(metrics_list),
                    "width": last.width,
                    "height": last.height,
                    "brightness": round(last.brightness, 2),
                    "motion": round(max(item.motion for item in metrics_list), 3),
                    "mean_motion": round(
                        sum(item.motion for item in metrics_list) / len(metrics_list), 3
                    ),
                    "mouth_motion": round(max(item.mouth_motion for item in metrics_list), 3),
                    "mean_mouth_motion": round(
                        sum(item.mouth_motion for item in metrics_list) / len(metrics_list), 3
                    ),
                    "head_motion": round(max(item.head_motion for item in metrics_list), 3),
                    "mean_head_motion": round(
                        sum(item.head_motion for item in metrics_list) / len(metrics_list), 3
                    ),
                    "source": last.source,
                    "timestamp": last.timestamp,
                }
            metrics = await asyncio.to_thread(self.ndi.snapshot)
            if metrics is None:
                return None
            self._last_ndi = metrics
            return {
                "path": metrics.snapshot,
                "width": metrics.width,
                "height": metrics.height,
                "brightness": round(metrics.brightness, 2),
                "motion": round(metrics.motion, 3),
                "mouth_motion": round(metrics.mouth_motion, 3),
                "head_motion": round(metrics.head_motion, 3),
                "source": metrics.source,
                "timestamp": metrics.timestamp,
            }
        except Exception as exc:
            log.warning("NDI snapshot failed: %s", exc)
            return None

    async def close(self) -> None:
        self._closing = True
        # Give the model a short release ramp before dropping the VTS
        # connection; abruptly stopping injection makes VTS snap back to idle.
        if self.vts.authenticated and self.mixer is not None and self.clock is not None:
            try:
                self.mixer.set_state("idle")
                for _ in range(10):
                    params = self.mixer.compute(self.clock.now_sample(), self.timeline)
                    if params:
                        await self.vts.inject(params)
                    await asyncio.sleep(1.0 / 30.0)
            except Exception:
                pass
        for task in (self._ticker, self._worker, self._watcher):
            if task is not None:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        if self.clock is not None:
            self.clock.stop()
        if self.ndi is not None:
            self.ndi.close()
            self.ndi = None
        await self.vts.close()

    # -- public speech API -------------------------------------------------

    async def begin_utterance(self, utterance_id: str | None = None) -> None:
        await self.cancel()
        self.utterance_id = utterance_id or f"utt-{int(time.time() * 1000)}"
        self.segmenter = Segmenter()
        self.speech_filter.reset()
        self._affect_buffer = ""
        self._segment_count = 0
        self._utterance_open = True
        if self.mixer is not None:
            self.mixer.set_state("speaking")

    async def feed_text(self, text: str) -> None:
        if not text:
            return
        if self.mixer is not None:
            # Keywords can be split across deltas ("开" + "心"); evaluate a
            # short rolling window instead of each fragment in isolation.
            self._affect_buffer = (self._affect_buffer + text)[-32:]
            affect = emotion_from_text(self._affect_buffer)
            if affect:
                self.mixer.set_emotion(*affect)
                if self.config.emotion.is_affect:
                    self._push_affect(affect[0], affect[1])
        clean = strip_emotion_tags(text)
        clean = self.speech_filter.feed(clean)
        if not clean:
            return
        segments = self.segmenter.feed(clean)
        for segment in segments:
            self._enqueue(segment)

    async def end_utterance(self) -> None:
        self._utterance_open = False
        tail = self.speech_filter.flush()
        if tail:
            for segment in self.segmenter.feed(tail):
                self._enqueue(segment)
        for segment in self.segmenter.flush():
            self._enqueue(segment)
        if self._queue is not None and self._queue.empty() and self.timeline is None:
            if self.mixer is not None:
                self.mixer.set_state("idle")

    def _enqueue(self, segment: str) -> None:
        segment = segment.strip()
        if not segment or self._queue is None:
            return
        # Do not let a stray brace, fence or punctuation-only fragment reach
        # Piper; it produces no audio and used to poison the speech queue.
        if not any(ch.isalnum() or "\u4e00" <= ch <= "\u9fff" for ch in segment):
            return
        try:
            self._queue.put_nowait(segment)
        except asyncio.QueueFull:
            # Drop the oldest pending phrase rather than blocking text flow.
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except Exception:
                pass
            try:
                self._queue.put_nowait(segment)
            except Exception:
                pass

    async def cancel(self) -> None:
        self._speech_generation += 1
        if self._queue is not None:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                except Exception:
                    break
        self._parts.clear()
        self._pending_gap_s = 0.0
        self.timeline = None
        self._last_sample = 0
        self.mouth_monitor.reset()
        if self.clock is not None:
            self.clock.clear()
        if self.mixer is not None:
            self.mixer.set_state("idle")
            self.mixer.reset()
        self._utterance_open = False
        await self._emit({"type": "cancelled", "utterance_id": self.utterance_id})

    def set_state(self, state: str) -> None:
        if self.mixer is not None:
            self.mixer.set_state(state)

    def add_action(self, kind: str, intensity: float = 0.7) -> None:
        if self.mixer is not None:
            self.mixer.add_action(kind, intensity)

    def set_emotion(self, name: str, intensity: float = 0.7) -> None:
        if self.mixer is not None:
            self.mixer.set_emotion(name, intensity)
            if self.config.emotion.is_affect:
                # A manual instruction is the one source allowed to hard
                # override; keep it at full confidence.
                self._push_affect(name, intensity, source="manual", confidence=1.0)
            # Thinking/shy have no fixed standard take yet, but they must
            # still be triggerable. Improvise them from the generic preset
            # plus a combined head/eye/brow action.
            gesture = {
                "thinking": "think_side",
                "思考": "think_side",
                "shy": "shy_avoid",
                "害羞": "shy_avoid",
            }.get(name)
            if gesture and not any(cue.kind == gesture and not cue.finished for cue in self.mixer.actions):
                self.mixer.add_action(gesture, intensity)

    def _push_affect(
        self,
        label: str,
        intensity: float,
        *,
        source: str | None = None,
        confidence: float | None = None,
    ) -> None:
        """Turn one legacy reading into one affect evidence record.

        Phase 2 stays single-source on purpose; Phase 4 is where several
        evidence records get weighed against each other.
        """
        if self.mixer is None or not self.config.emotion.is_affect:
            return
        if source is None:
            markers = ("[emotion", "[情绪", "【情绪")
            source = "tag" if any(marker in self._affect_buffer for marker in markers) else "keyword"
        if confidence is None:
            confidence = 0.9 if source == "tag" else 0.6
        evidence = AffectEvidence(
            source=source,
            labels={label: 1.0},
            intensity=intensity,
            confidence=confidence,
        )
        state = AffectState.from_evidence(evidence)
        self.affect_log.evidence(evidence)
        self.affect_log.state(state, applied=True)
        self.mixer.set_affect(state)

    def feel(
        self,
        labels: dict[str, float] | None = None,
        intensity: float = 0.7,
        confidence: float = 0.8,
        note: str = "",
        turn_id: str = "",
    ) -> dict[str, Any]:
        """Describe the current emotional state (Phase 3).

        ``labels`` is a weighted vocabulary (``{"happy": 0.6, "tired": 0.4}``),
        not a preset name. In ``affect`` mode it becomes one evidence record and
        the composer blends it; in ``legacy`` mode it degrades to the dominant
        preset so old behavior and old logs stay comparable.
        """
        evidence = AffectEvidence(
            source="tool",
            labels=dict(labels or {}),
            intensity=intensity,
            confidence=confidence,
            note=note,
            turn_id=turn_id,
        )
        state = AffectState.from_evidence(evidence)
        self.affect_log.evidence(evidence)
        if self.mixer is None:
            self.affect_log.state(state, applied=False)
            return {"ok": False, "error": "mixer not ready", "labels": state.labels}

        if self.config.emotion.is_affect:
            # A low-confidence reading must not slam the face to full anchor.
            # Scale the display strength instead of dropping the evidence, so
            # Phase 4 fusion still sees what was said.
            state.intensity = evidence.intensity * (0.5 + 0.5 * evidence.confidence)
            self.affect_log.state(state, applied=True)
            self.mixer.set_affect(state)
            dominant, weight = state.dominant()[0]
            if weight >= 0.5:
                # Keep the learned-expression/profile path alive; the face
                # targets themselves come from the composer.
                self.mixer.set_emotion(dominant, state.intensity)
        else:
            dominant, _weight = state.dominant()[0]
            self.affect_log.state(state, applied=True, mode="legacy")
            self.mixer.set_emotion(dominant, state.intensity)

        return {
            "ok": True,
            "mode": self.config.emotion.mode,
            "labels": state.labels,
            "dominant": state.dominant(2),
            "intensity": round(state.intensity, 4),
            "confidence": state.confidence,
            "valence": round(state.valence, 4),
            "arousal": round(state.arousal, 4),
        }

    async def record_expression(self, name: str, seconds: float = 4.0) -> dict[str, Any]:
        """Read-only capture of a user-performed expression.

        The 30 Hz writer is paused for the whole capture; the user can move
        VTS with the mouse, face tracking, or VTS actions without our writes
        fighting them. The resulting delta profile is saved on this model.
        """
        if not self.vts.authenticated:
            raise RuntimeError("VTS is not connected")
        self._recording = True
        try:
            summary = await record_expression(self.vts, name, duration_s=seconds)
            if self.profile is not None:
                self.profile.learned_expressions[name] = summary
                direct = {
                    key: float(info["peak"])
                    for key, info in summary.get("input_deltas", {}).items()
                    if abs(float(info.get("delta", 0.0))) >= 0.02
                }
                self.profile.expression_map[name] = direct
                self.profile.save(profile_path(self.config.profile_dir, self.profile.model_id))
                if self.mixer is not None:
                    self.mixer.set_profile(self.profile)
            return summary
        finally:
            self._recording = False

    def demo_head(self, *, curve: str = "minjerk", intensity: float = 0.5, duration_ms: int = 900) -> bool:
        if self.clock is None or self.mixer is None or self.tts is None:
            return False
        at = self.clock.now_sample() + int(0.10 * self.tts.sample_rate)
        self.mixer.add_action(
            "demo_nod",
            intensity=max(0.05, min(1.4, intensity)),
            at_sample=at,
            duration_ms=duration_ms,
            curve="bezier" if curve == "bezier" else "minjerk",
        )
        return True

    # -- synthesis ---------------------------------------------------------

    async def _synth_worker(self) -> None:
        assert self._queue is not None
        while not self._closing:
            text = await self._queue.get()
            generation = self._speech_generation
            try:
                started = time.perf_counter()
                assert self.tts is not None
                timeline = await asyncio.to_thread(self.tts.synthesize, text)
                if self.rvc.enabled:
                    timeline = await asyncio.to_thread(self.rvc.convert, timeline, text)
                if generation != self._speech_generation:
                    # The user cancelled/began another utterance while this
                    # synthesis was in flight. Never append stale speech.
                    log.info("discarding stale TTS result")
                    continue
                self._last_tts_ms = (time.perf_counter() - started) * 1000.0
                self._append_timeline(timeline, text)
            except Exception as exc:
                self._last_error = f"TTS failed: {exc}"
                log.exception("TTS failed for %r", text)
                await self._emit(
                    {
                        "type": "error",
                        "code": "tts_failed",
                        "message": str(exc),
                        "retryable": True,
                    }
                )
            finally:
                self._queue.task_done()

    def _append_timeline(self, timeline: Timeline, text: str) -> None:
        if self.clock is None or self.mixer is None:
            return
        if self._parts and self._pending_gap_s > 0:
            sample_rate = int(timeline.sample_rate)
            gap_audio = np.zeros(int(sample_rate * self._pending_gap_s), dtype=np.float32)
            gap = Timeline(
                sample_rate=sample_rate,
                audio=gap_audio,
                spans=[],
                segments=[],
                rms=np.zeros(max(1, gap_audio.size // max(1, int(sample_rate * 0.01))), dtype=np.float32),
                rms_hop=max(1, int(sample_rate * 0.01)),
            )
            self.clock.append(gap.audio)
            self._parts.append(gap)
            self._pending_gap_s = 0.0
        start_sample = sum(part.duration_samples for part in self._parts)
        self._parts.append(timeline)
        self.timeline = Timeline.concat(self._parts)
        self.clock.append(timeline.audio)
        if not self.clock.is_running:
            self.clock.start()
        self.mixer.set_state("speaking")
        self._pending_gap_s = 0.0
        self._segment_count += 1
        # A small speech emphasis / gesture on selected phrase starts.
        if text and self.segmenter_gesture_rate > 0:
            sample_rate = self.tts.sample_rate if self.tts else 48000
            # If TTS fell behind playback, the scheduled start is already in
            # the past. Starting a cue mid-curve is exactly what looks like a
            # sudden head pop; clamp the start to the next tick instead.
            now_sample = self.clock.now_sample()
            start = max(
                start_sample + int(0.04 * sample_rate),
                now_sample + int(0.05 * sample_rate),
            )
            if "！" in text or "!" in text or "？" in text or "?" in text:
                self.mixer.add_action("emphasis", 0.50, at_sample=start)
            elif self._segment_count % 3 == 0:
                self.mixer.add_action("nod", 0.35, at_sample=start)
            elif self._segment_count % 4 == 0:
                # A slow look to the side at a phrase boundary, with the eyes
                # leading the head. This is a small cue, not a constant scan.
                side = "turn_look_left" if (self._segment_count // 4) % 2 else "turn_look_right"
                self.mixer.add_action(side, 0.28, at_sample=start)

    # -- control loop ------------------------------------------------------

    async def _tick_loop(self) -> None:
        period = 1.0 / 30.0
        next_at = asyncio.get_running_loop().time()
        while not self._closing:
            now = time.perf_counter()
            if self._recording:
                # Record-only mode: never write while the user is performing.
                await asyncio.sleep(period)
                continue
            if self.vts.authenticated and now - self._last_model_check >= 2.0:
                self._last_model_check = now
                asyncio.create_task(self._refresh_model(), name="avatar-model-refresh")
            if self.clock is not None and self.mixer is not None:
                sample = self.clock.now_sample()
                self._last_sample = sample
                params = self.mixer.compute(sample, self.timeline, now)
                self._last_params = params
                if self.timeline is not None:
                    mouth_segment = self.timeline.at(sample)
                    audible = self.timeline.rms_at(sample) > 0.01 or (
                        mouth_segment.kind != "silence" and mouth_segment.open > 0.02
                    )
                else:
                    audible = False
                self.mouth_monitor.observe(
                    now,
                    params,
                    speaking=self.mixer.state == "speaking",
                    audible=audible,
                )
                if self.vts.authenticated and params:
                    try:
                        await self.vts.inject(params)
                    except Exception as exc:
                        if not self._last_error:
                            self._last_error = str(exc)
                        log.debug("inject failed: %s", exc)
                for err in self.vts.drain_inject_errors():
                    self._last_error = f"VTS {err.error_id}: {err.vts_message}"
                    log.warning("VTS injection error: %s", err)
                if self._should_idle(sample):
                    self.mixer.set_state("idle")
                if now - self._last_emit >= 0.10:
                    self._last_emit = now
                    await self._emit(
                        {
                            "type": "playback",
                            "utterance_id": self.utterance_id,
                            "sample": sample,
                            "duration": self.timeline.duration_samples if self.timeline else 0,
                            "state": self.mixer.state,
                            "vts_connected": self.vts.connected,
                            "last_error": self._last_error,
                        }
                    )
            next_at += period
            delay = next_at - asyncio.get_running_loop().time()
            if delay < 0:
                # Do not accumulate a burst of catch-up frames.
                next_at = asyncio.get_running_loop().time()
                delay = 0
            await asyncio.sleep(delay)

    def _should_idle(self, sample: int) -> bool:
        if self._utterance_open:
            return False
        if self._queue is not None and not self._queue.empty():
            return False
        if self.timeline is None:
            return True
        return sample >= self.timeline.duration_samples

    # -- events ------------------------------------------------------------

    async def _emit(self, payload: dict[str, Any]) -> None:
        if self.on_event is None:
            return
        try:
            await self.on_event(payload)
        except Exception:
            log.debug("event sink failed", exc_info=True)

    def parameters(self) -> dict[str, Any]:
        segment = self.timeline.at(self._last_sample) if self.timeline is not None else None
        return {
            "sample": self._last_sample,
            "state": self.mixer.state if self.mixer else "idle",
            "params": dict(self._last_params),
            "phoneme": segment.phoneme if segment else None,
            "viseme": segment.viseme if segment else None,
            "mouth_stuck": self.mouth_monitor.stuck,
            "mouth_activity": round(self.mouth_monitor.activity, 4),
        }

    def set_head_pose(self, yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0, seconds: float = 8.0) -> bool:
        if self.mixer is None:
            return False
        self.mixer.set_head_pose(yaw, pitch, roll, seconds)
        return True

    def status(self) -> dict[str, Any]:
        return {
            "vts_connected": self.vts.connected,
            "vts_authenticated": self.vts.authenticated,
            "tts": type(self.tts).__name__ if self.tts else None,
            "backend": self.mixer.backend if self.mixer else None,
            "model": self.profile.model_name,
            "state": self.mixer.state if self.mixer else "idle",
            "sample": self._last_sample,
            "last_error": self._last_error,
            "mouth_stuck": self.mouth_monitor.stuck,
            "mouth_activity": round(self.mouth_monitor.activity, 4),
            "ndi_connected": self.ndi is not None,
            "last_ndi": self._last_ndi.__dict__ if self._last_ndi is not None else None,
            "contract": str(self.config.contract_path),
        }
