"""Compare duration stability of the distilled voice across several sentences."""

from dataclasses import replace
from pathlib import Path
import wave

import numpy as np

from dsh_vts_avatar.config import load_config
from dsh_vts_avatar.tts.piper_tts import PiperTTS

TEXTS = [
    "你好，我是肥鱼。今天天气不错，我们开始吧。",
    "早上好，昨晚睡得怎么样？",
    "我刚泡了一杯热茶，你要不要也来一杯？",
    "这个周末你有什么安排吗？",
    "别着急，慢慢说，我在听。",
]
OUT = Path("dsh-vts-avatar/data/audit/voice_eval_multi")


def write(path: Path, audio: np.ndarray, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


def main() -> None:
    cfg = load_config("dsh-vts-avatar/config.toml")
    base = replace(cfg.tts, postprocess=False, length_scale=1.15)
    candidate = replace(
        base,
        voice_model=Path("dsh-vts-avatar/models/feiyu_distill_v4.onnx"),
        voice_config=Path("dsh-vts-avatar/models/feiyu_distill_v4.onnx.json"),
        length_scale=0.61,
    )
    base_tts = PiperTTS(base)
    cand_tts = PiperTTS(candidate)
    ratios = []
    for index, text in enumerate(TEXTS, 1):
        a = base_tts.synthesize(text)
        b = cand_tts.synthesize(text)
        ratio = b.duration_s / a.duration_s if a.duration_s else 0.0
        ratios.append(ratio)
        print(
            index,
            "base", round(a.duration_s, 3),
            "candidate", round(b.duration_s, 3),
            "ratio", round(ratio, 3),
        )
        write(OUT / f"{index:02d}_huayan.wav", a.audio, a.sample_rate)
        write(OUT / f"{index:02d}_candidate.wav", b.audio, b.sample_rate)
    print("mean_ratio", round(float(np.mean(ratios)), 3), "max_dev", round(max(abs(r - 1) for r in ratios), 3))


if __name__ == "__main__":
    main()
