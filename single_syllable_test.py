"""Isolate the mouth shape for single syllables without live TTS timing.

The bridge is stopped for this probe so it is the only writer to VTS.
"""

import asyncio
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from dsh_vts_avatar.config import load_config
from dsh_vts_avatar.ndi.capture import NDICapture
from dsh_vts_avatar.performance.mixer import PerformanceMixer
from dsh_vts_avatar.tts.piper_tts import PiperTTS
from dsh_vts_avatar.vts.client import VTSClient
from dsh_vts_avatar.vts.profile import ModelProfile, profile_path

OUT = Path("dsh-vts-avatar/data/audit/syllables")
TEXT = "妈，爸，婆，逼，呜，啊。"


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config("dsh-vts-avatar/config.toml")
    tts = PiperTTS(cfg.tts)
    timeline = tts.synthesize(TEXT)
    profile = ModelProfile.load(
        profile_path(cfg.profile_dir, "8cb985f5b9514021a5c5d941b482e57b")
    )
    mixer = PerformanceMixer(cfg.avatar, timeline.sample_rate, profile=profile)
    mixer.set_state("speaking")

    # Find the first vowel of each comma-separated syllable group.
    targets = []
    for index, segment in enumerate(timeline.segments):
        if segment.kind != "vowel":
            continue
        targets.append((segment.start, segment.end, segment.viseme, segment.phoneme))
    print("targets", targets)
    if len(targets) < 6:
        print("WARNING: expected 6 syllables")

    # Precompute the mixer at 30Hz and keep the parameter snapshot at each target.
    selected = {}
    target_mid = [start + (end - start) // 2 for start, end, _, _ in targets]
    for sample in range(0, timeline.duration_samples, timeline.sample_rate // 30):
        out = mixer.compute(sample, timeline, time.perf_counter() + sample / timeline.sample_rate)
        for i, mid in enumerate(target_mid):
            if abs(sample - mid) < timeline.sample_rate // 60:
                selected[i] = out

    client = VTSClient(cfg.vts)
    await client.connect()
    ndi = NDICapture(source_hint=cfg.ndi.source, snapshot_dir=OUT)
    if not ndi.open(wait_ms=4000):
        raise RuntimeError("no NDI")
    labels = ["ma(A)", "ba(A)", "po(O)", "bi(I)", "wu(U)", "a(A)"]
    target_indices = list(range(min(len(labels), len(targets))))
    selected = {i: selected[target_indices[i]] for i in range(len(target_indices)) if target_indices[i] in selected}
    met = {}
    try:
        for i, params in selected.items():
            await client.inject(params)
            await asyncio.sleep(0.45)
            snap = await asyncio.to_thread(ndi.snapshot)
            if snap and snap.snapshot:
                dst = OUT / f"{i}_{labels[i]}.jpg"
                shutil.copyfile(snap.snapshot, dst)
                met[labels[i]] = {
                    "file": str(dst),
                    "MouthOpen": round(float(params.get("MouthOpen", 0)), 3),
                    "VoiceA": round(float(params.get("VoiceA", 0)), 3),
                    "VoiceI": round(float(params.get("VoiceI", 0)), 3),
                    "VoiceU": round(float(params.get("VoiceU", 0)), 3),
                    "VoiceE": round(float(params.get("VoiceE", 0)), 3),
                    "VoiceO": round(float(params.get("VoiceO", 0)), 3),
                }
                print(labels[i], met[labels[i]])
    finally:
        ndi.close()
        await client.close()

    tiles = []
    for label, row in met.items():
        im = cv2.imdecode(np.frombuffer(Path(row["file"]).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        im = cv2.resize(im[80:1000, 450:1500], (320, 280))
        cv2.putText(im, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        tiles.append(im)
    rows = []
    for i in range(0, len(tiles), 3):
        row = tiles[i:i + 3]
        while len(row) < 3:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    if rows:
        sheet = np.vstack(rows)
        ok, buf = cv2.imencode(".jpg", sheet)
        (OUT / "syllables.jpg").write_bytes(buf.tobytes())
    (OUT / "result.json").write_text(json.dumps(met, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", OUT / "syllables.jpg")


asyncio.run(main())
