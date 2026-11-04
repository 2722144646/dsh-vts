"""Compare a candidate Piper voice against the current huayan voice."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import wave

import numpy as np

from dsh_vts_avatar.config import load_config
from dsh_vts_avatar.tts.piper_tts import PiperTTS

TEXT = "你好，我是肥鱼。今天天气不错，我们开始吧。"


def write(path: Path, audio: np.ndarray, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--length-scale", type=float, default=1.15)
    parser.add_argument("--output-dir", type=Path, default=Path("dsh-vts-avatar/data/audit/voice_eval"))
    args = parser.parse_args()

    cfg = load_config("dsh-vts-avatar/config.toml")
    base_cfg = replace(cfg.tts, postprocess=False, length_scale=1.15)
    candidate_cfg = replace(
        base_cfg,
        voice_model=args.candidate,
        voice_config=args.candidate.with_suffix(args.candidate.suffix + ".json"),
        length_scale=args.length_scale,
    )
    for name, config in (("huayan", base_cfg), ("candidate", candidate_cfg)):
        tts = PiperTTS(config)
        timeline = tts.synthesize(TEXT)
        path = args.output_dir / f"{name}.wav"
        write(path, timeline.audio, timeline.sample_rate)
        print(
            name,
            "duration", round(timeline.duration_s, 3),
            "rate", timeline.sample_rate,
            "spans", len(timeline.spans),
            "samples", timeline.audio.size,
            path,
        )


if __name__ == "__main__":
    main()
