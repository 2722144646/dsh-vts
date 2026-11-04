from dataclasses import replace
from pathlib import Path
import wave

import numpy as np

from dsh_vts_avatar.config import load_config
from dsh_vts_avatar.tts.distill import DistilledPiperTTS

TEXTS = [
    "你好，我是肥鱼。今天天气不错，我们开始吧。",
    "早上好，昨晚睡得怎么样？",
    "我刚泡了一杯热茶，你要不要也来一杯？",
    "这个周末你有什么安排吗？",
    "别着急，慢慢说，我在听。",
]
OUT = Path("dsh-vts-avatar/data/audit/distill_integration")


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
    distill_model=Path("dsh-vts-avatar/models/feiyu_distill_v4.onnx"),
    distill_voice_config=Path("dsh-vts-avatar/models/feiyu_distill_v4.onnx.json"),
    distill_length_scale=0.61,
)
tts = DistilledPiperTTS(cfg.tts)
tts.load()
for index, text in enumerate(TEXTS, 1):
    timeline = tts.synthesize(text)
    path = OUT / f"{index:02d}.wav"
    write(path, timeline.audio, timeline.sample_rate)
    print(index, round(timeline.duration_s, 3), len(timeline.spans), path)
