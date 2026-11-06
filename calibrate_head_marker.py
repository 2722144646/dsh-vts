"""Visual marker calibration: a head pulse plus an audible click."""

import asyncio
import json
import shutil
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import soundcard as sc
import sounddevice as sd
import websockets

OUT = Path("dsh-vts-avatar/data/audit/head_marker")
SR = 48000


async def snapshot(ws, index: int):
    request_id = f"hm-{index}"
    await ws.send(json.dumps({"type": "snapshot", "request_id": request_id}))
    while True:
        message = json.loads(await ws.recv())
        if message.get("type") == "snapshot.result" and message.get("request_id") == request_id:
            return message.get("metrics")


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    speaker = sc.default_speaker()
    mic = sc.get_microphone(speaker.name, include_loopback=True)
    recorder_data: dict[str, object] = {}

    def record() -> None:
        with mic.recorder(samplerate=SR, channels=1) as recorder:
            recorder_data["start"] = time.time()
            recorder_data["audio"] = np.asarray(recorder.record(numframes=SR * 3)).reshape(-1)

    record_thread = threading.Thread(target=record)
    async with websockets.connect("ws://127.0.0.1:8765") as ws:
        await ws.recv()
        await ws.send(json.dumps({"type": "head_pose", "pitch": 0, "seconds": 0.5}))
        await asyncio.sleep(0.8)
        record_thread.start()
        await asyncio.sleep(0.2)
        frames = []
        for index in range(35):
            metrics = await snapshot(ws, index)
            if metrics:
                source = Path(metrics["path"])
                target = OUT / f"{index:02d}.jpg"
                shutil.copyfile(source, target)
                frames.append((float(metrics["timestamp"]), target))
            if index == 3:
                await ws.send(json.dumps({"type": "head_pose", "pitch": 15, "seconds": 0.6}))
                click = np.zeros(int(SR * 0.3), dtype=np.float32)
                click[:480] = np.hanning(480).astype(np.float32) * 0.8
                sd.play(click, SR, device=19)
                time.sleep(0.05)
            await asyncio.sleep(0.045)
    record_thread.join()
    audio = np.asarray(recorder_data["audio"], dtype=np.float32)
    start = float(recorder_data["start"])
    np.save(OUT / "audio.npy", audio)
    win = int(SR * 0.02)
    rms = np.array([float(np.sqrt(np.mean(audio[i:i + win] ** 2))) for i in range(0, len(audio) - win, win)])
    threshold = max(0.002, float(np.median(rms[:10]) * 6))
    audio_index = next((i for i, value in enumerate(rms) if value > threshold), 0)
    audio_onset = start + audio_index * win / SR
    # Head ROI; head motion should be much larger than the static background.
    base_gray = None
    head_onset = None
    for timestamp, path in frames:
        image = cv2.imread(str(path))
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image[int(height * 0.08):int(height * 0.58), int(width * 0.20):int(width * 0.80)], cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 180))
        if base_gray is None:
            base_gray = gray.astype(np.float32)
            continue
        difference = float(np.mean(cv2.absdiff(base_gray, gray.astype(np.float32))))
        if difference > 2.0:
            head_onset = timestamp
            print("head difference", round(difference, 2), "timestamp", timestamp)
            break
    print("audio_onset", audio_onset)
    print("head_onset", head_onset)
    if head_onset is not None:
        print("visual_minus_audio_ms", round((head_onset - audio_onset) * 1000, 1))


asyncio.run(main())
