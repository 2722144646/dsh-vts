from dataclasses import replace
from pathlib import Path
import wave

import numpy as np

from dsh_vts_avatar.config import load_config
from dsh_vts_avatar.tts.piper_tts import PiperTTS

TEXT = "你好，我是肥鱼。今天天气不错，我们开始吧。"
OUT = Path("dsh-vts-avatar/data/audit/voice_ab")


def write(path: Path, audio: np.ndarray, rate: int) -> None:
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


cfg = load_config("dsh-vts-avatar/config.toml")
base = cfg.tts
for name, model, voice_config in [
    ("huayan_03", base.voice_model, base.voice_config),
    ("feiyu_distill_v1_03", Path("dsh-vts-avatar/models/feiyu_distill_v1.onnx"), Path("dsh-vts-avatar/models/feiyu_distill_v1.onnx.json")),
]:
    conf = replace(base, voice_model=model, voice_config=voice_config, postprocess=True, length_scale=1.15)
    tts = PiperTTS(conf)
    timeline = tts.synthesize(TEXT)
    path = OUT / f"{name}.wav"
    write(path, timeline.audio, timeline.sample_rate)
    print(name, round(timeline.duration_s, 3), path)
