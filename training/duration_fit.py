"""Fit the distilled Piper voice to a reference duration per sentence.

The candidate is re-synthesized with an adjusted length_scale, not time-stretched.
That keeps its own phoneme spans aligned with its own audio, so mouth sync and
duration stay consistent.
"""

from __future__ import annotations

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
OUT = Path("dsh-vts-avatar/data/audit/voice_fitted")
CANDIDATE = Path("dsh-vts-avatar/models/feiyu_distill_v4.onnx")


def write(path: Path, audio: np.ndarray, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


def main() -> None:
    cfg = load_config("dsh-vts-avatar/config.toml")
    reference_cfg = replace(cfg.tts, postprocess=False, length_scale=1.15)
    candidate_cfg = replace(
        reference_cfg,
        voice_model=CANDIDATE,
        voice_config=CANDIDATE.with_suffix(".onnx.json"),
        length_scale=1.15,
    )
    reference = PiperTTS(reference_cfg)
    candidate = PiperTTS(candidate_cfg)
    ratios = []
    for index, text in enumerate(TEXTS, 1):
        target = reference.synthesize(text)
        scale = 1.15
        candidate.config.length_scale = scale
        current = candidate.synthesize(text)
        for _ in range(3):
            ratio = target.duration_s / max(0.001, current.duration_s)
            if abs(ratio - 1.0) < 0.004:
                break
            scale = max(0.40, min(2.5, scale * ratio))
            candidate.config.length_scale = scale
            current = candidate.synthesize(text)
        final_ratio = current.duration_s / target.duration_s
        ratios.append(final_ratio)
        print(
            index,
            "target", round(target.duration_s, 3),
            "fitted", round(current.duration_s, 3),
            "ratio", round(final_ratio, 4),
            "scale", round(scale, 3),
            "spans", len(current.spans),
        )
        write(OUT / f"{index:02d}_target.wav", target.audio, target.sample_rate)
        write(OUT / f"{index:02d}_fitted.wav", current.audio, current.sample_rate)
    print("max_dev", round(max(abs(r - 1.0) for r in ratios), 4))


if __name__ == "__main__":
    main()
