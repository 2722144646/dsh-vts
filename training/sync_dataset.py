"""Sync accepted CosyVoice takes into the Piper dataset layout."""

from __future__ import annotations

import shutil
from pathlib import Path

SOURCE = Path("dsh-vts-avatar/data/voice_training/cosyvoice")
TARGET = Path("dsh-vts-avatar/training/dataset")


def main() -> None:
    source_wav = SOURCE / "wav"
    target_wav = TARGET / "wav"
    target_wav.mkdir(parents=True, exist_ok=True)
    accepted = [
        line.strip()
        for line in (SOURCE / "metadata.csv").read_text(encoding="utf-8").splitlines()
        if "|" in line
    ]
    ids = {line.split("|", 1)[0] for line in accepted}
    for wav in target_wav.glob("*.wav"):
        if wav.stem not in ids:
            wav.unlink()
    copied = 0
    for identifier in ids:
        source = source_wav / f"{identifier}.wav"
        target = target_wav / source.name
        if source.exists() and (not target.exists() or target.stat().st_size != source.stat().st_size):
            shutil.copyfile(source, target)
            copied += 1
    (TARGET / "metadata.csv").write_text("\n".join(accepted) + "\n", encoding="utf-8")
    print("entries", len(accepted), "copied", copied, "wavs", len(list(target_wav.glob("*.wav"))))


if __name__ == "__main__":
    main()
