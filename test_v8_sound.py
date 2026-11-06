"""Synthesize and play the v8 candidate through the connected headphones."""

from dataclasses import replace
from pathlib import Path
import wave

import numpy as np
import sounddevice as sd

from dsh_vts_avatar.config import load_config
from dsh_vts_avatar.tts.distill import DistilledPiperTTS

TEXT = "你好，我是肥鱼。今天天气不错，我们开始吧。"
OUT = Path("dsh-vts-avatar/data/audit/v8_sound")


def write(path: Path, audio: np.ndarray, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


cfg = load_config("dsh-vts-avatar/config.toml")
cfg.tts = replace(
    cfg.tts,
    length_scale=1.15,
    postprocess=False,
    distill_model=Path("dsh-vts-avatar/models/feiyu_distill_v8_final.onnx"),
    distill_voice_config=Path("dsh-vts-avatar/models/feiyu_distill_v8_final.onnx.json"),
    distill_length_scale=0.61,
)
tts = DistilledPiperTTS(cfg.tts)
tts.load()
timeline = tts.synthesize(TEXT)
path = OUT / "v8_final.wav"
write(path, timeline.audio, timeline.sample_rate)
print(
    "v8",
    "duration", round(timeline.duration_s, 3),
    "rate", timeline.sample_rate,
    "spans", len(timeline.spans),
    "rms", round(float(np.sqrt(np.mean(timeline.audio ** 2))), 4),
    "peak", round(float(np.max(np.abs(timeline.audio))), 4),
    path,
)
print("outputs", [(i, d["name"]) for i, d in enumerate(sd.query_devices()) if d["max_output_channels"] > 0])
device = 5
try:
    sd.check_output_settings(device=device, samplerate=timeline.sample_rate, channels=1)
except Exception:
    device = 20
    sd.check_output_settings(device=device, samplerate=timeline.sample_rate, channels=1)
print("playing on", device, sd.query_devices(device)["name"])
sd.play(timeline.audio, timeline.sample_rate, device=device)
sd.wait()
print("played")
