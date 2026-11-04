from dataclasses import replace
from pathlib import Path
import wave

import numpy as np

from dsh_vts_avatar.config import load_config
from dsh_vts_avatar.tts.piper_tts import PiperTTS
from dsh_vts_avatar.tts.distill import DistilledPiperTTS

TEXT = "你好，我是肥鱼。今天天气不错，我们开始吧。"
OUT = Path("dsh-vts-avatar/data/audit/distill_integration/compare_v5.wav")


def read(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as fh:
        rate = fh.getframerate()
        return np.frombuffer(fh.readframes(fh.getnframes()), dtype="<i2").astype(np.float32) / 32768.0, rate


cfg = load_config("dsh-vts-avatar/config.toml")
base = replace(cfg.tts, length_scale=1.15, postprocess=False)
reference = PiperTTS(base)
ref = reference.synthesize(TEXT)

distill_cfg = replace(
    base,
    distill_model=Path("dsh-vts-avatar/models/feiyu_distill_v5.onnx"),
    distill_voice_config=Path("dsh-vts-avatar/models/feiyu_distill_v5.onnx.json"),
    distill_length_scale=0.61,
)
distilled = DistilledPiperTTS(distill_cfg)
distilled.load()
cand = distilled.synthesize(TEXT)

rate = ref.sample_rate
gap = np.zeros(int(rate * 0.6), dtype=np.float32)
audio = np.concatenate([ref.audio, gap, cand.audio])
with wave.open(str(OUT), "wb") as fh:
    fh.setnchannels(1)
    fh.setsampwidth(2)
    fh.setframerate(rate)
    fh.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
print("reference", round(ref.duration_s, 3), "distilled", round(cand.duration_s, 3), "out", OUT)
