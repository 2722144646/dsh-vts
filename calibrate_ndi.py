"""Measure audio-to-NDI mouth delay during a real bridge utterance."""

import asyncio
import json
import shutil
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import soundcard as sc
import websockets

OUT = Path("dsh-vts-avatar/data/audit/av_ndi")
TEXT = "爸爸，妈妈，怕。八百标兵奔北坡。"
SECONDS = 6.0
SR = 48000


async def snap(ws, index: int) -> dict | None:
    request_id = f"av-{index}"
    await ws.send(json.dumps({"type": "snapshot", "request_id": request_id}))
    while True:
        message = json.loads(await ws.recv())
        if message.get("type") == "snapshot.result" and message.get("request_id") == request_id:
            return message.get("metrics")


async def capture() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    speaker = sc.default_speaker()
    mic = sc.get_microphone(speaker.name, include_loopback=True)
    recorded: dict[str, object] = {}

    def record() -> None:
        with mic.recorder(samplerate=SR, channels=1) as recorder:
            recorded["start"] = time.time()
            recorded["audio"] = np.asarray(recorder.record(numframes=int(SR * SECONDS))).reshape(-1)

    thread = threading.Thread(target=record)
    async with websockets.connect("ws://127.0.0.1:8765") as ws:
        await ws.recv()
        await ws.send(json.dumps({"type": "head_pose", "pitch": 12, "seconds": 20}))
        await asyncio.sleep(0.8)
        thread.start()
        await asyncio.sleep(0.35)
        await ws.send(json.dumps({"type": "utterance.start", "utterance_id": "av-ndi"}))
        await ws.send(json.dumps({"type": "text.delta", "text": TEXT}, ensure_ascii=False))
        await ws.send(json.dumps({"type": "utterance.end", "utterance_id": "av-ndi"}))
        frames = []
        for index in range(32):
            await asyncio.sleep(0.07)
            metrics = await snap(ws, index)
            if not metrics:
                continue
            source = Path(metrics["path"])
            target = OUT / f"{index:02d}.jpg"
            shutil.copyfile(source, target)
            frames.append((index, float(metrics["timestamp"]), target))
    thread.join()
    audio = np.asarray(recorded["audio"], dtype=np.float32)
    start = float(recorded["start"])
    np.save(OUT / "audio.npy", audio)
    win = int(SR * 0.02)
    rms = np.array([float(np.sqrt(np.mean(audio[i:i + win] ** 2))) for i in range(0, len(audio) - win, win)])
    threshold = max(0.002, float(np.median(rms[:10]) * 6))
    audio_index = next((i for i, value in enumerate(rms) if value > threshold), 0)
    audio_onset = start + audio_index * win / SR
    print("audio_onset", audio_onset, "threshold", round(threshold, 5))
    # Mouth ROI: lower center of the full NDI frame.
    samples = []
    for index, timestamp, path in frames:
        image = cv2.imread(str(path))
        height, width = image.shape[:2]
        crop = image[int(height * 0.42):int(height * 0.68), int(width * 0.40):int(width * 0.60)]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        dark = float(np.mean(gray < 60))
        samples.append((timestamp, dark, path))
    baseline = float(np.median([item[1] for item in samples[:3]])) if samples else 0.0
    mouth_index = next((i for i, item in enumerate(samples) if item[1] > baseline + 0.015), None)
    if mouth_index is None:
        print("mouth onset not detected; dark values", [round(item[1], 4) for item in samples])
        return
    mouth_onset = samples[mouth_index][0]
    print("mouth_onset", mouth_onset)
    print("mouth_minus_audio_ms", round((mouth_onset - audio_onset) * 1000, 1))
    print("baseline_dark", round(baseline, 4), "onset_dark", round(samples[mouth_index][1], 4))


asyncio.run(capture())
