"""P1 redo: capture synchronized NDI frames, mouth parameters and loopback audio."""

import asyncio
import json
import shutil
import threading
import time
import wave
from pathlib import Path

import cv2
import numpy as np
import soundcard as sc
import websockets

OUT = Path("dsh-vts-avatar/data/audit/p1_redo")
TEXT = "爸爸，妈妈，怕。阿，伊，乌。八百标兵奔北坡。"
SR = 48000
SECONDS = 7.0


async def snapshot(ws, index: int):
    request_id = f"p1r-{index}"
    await ws.send(json.dumps({"type": "snapshot", "request_id": request_id}))
    while True:
        message = json.loads(await ws.recv())
        if message.get("type") == "snapshot.result" and message.get("request_id") == request_id:
            return message.get("metrics")


async def parameters(ws, index: int):
    request_id = f"p1p-{index}"
    await ws.send(json.dumps({"type": "parameters", "request_id": request_id}))
    while True:
        message = json.loads(await ws.recv())
        if message.get("type") == "parameters.result" and message.get("request_id") == request_id:
            return message.get("data")


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    speaker = sc.default_speaker()
    mic = sc.get_microphone(speaker.name, include_loopback=True)
    recorded: dict[str, object] = {}

    def record() -> None:
        with mic.recorder(samplerate=SR, channels=1) as recorder:
            recorded["start"] = time.time()
            recorded["audio"] = np.asarray(recorder.record(numframes=SR * SECONDS)).reshape(-1)

    thread = threading.Thread(target=record)
    async with websockets.connect("ws://127.0.0.1:8765") as ws:
        await ws.recv()
        await ws.send(json.dumps({"type": "head_pose", "pitch": 12, "seconds": 30}))
        await asyncio.sleep(0.6)
        thread.start()
        await asyncio.sleep(0.25)
        await ws.send(json.dumps({"type": "utterance.start", "utterance_id": "p1-redo"}))
        await ws.send(json.dumps({"type": "text.delta", "text": TEXT}, ensure_ascii=False))
        await ws.send(json.dumps({"type": "utterance.end", "utterance_id": "p1-redo"}))
        rows = []
        for index in range(48):
            await asyncio.sleep(0.08)
            metrics = await snapshot(ws, index)
            data = await parameters(ws, index)
            if metrics:
                source = Path(metrics["path"])
                target = OUT / f"{index:02d}.jpg"
                shutil.copyfile(source, target)
                rows.append(
                    {
                        "i": index,
                        "timestamp": metrics["timestamp"],
                        "path": str(target),
                        "state": data.get("state"),
                        "phoneme": data.get("phoneme"),
                        "viseme": data.get("viseme"),
                        "mouth_open": data.get("params", {}).get("MouthOpen"),
                        "voice_a": data.get("params", {}).get("VoiceA"),
                        "voice_i": data.get("params", {}).get("VoiceI"),
                        "voice_u": data.get("params", {}).get("VoiceU"),
                    }
                )
    thread.join()
    audio = np.asarray(recorded["audio"], dtype=np.float32)
    start = float(recorded["start"])
    (OUT / "meta.json").write_text(json.dumps({"audio_start": start}, ensure_ascii=False), encoding="utf-8")
    with wave.open(str(OUT / "audio.wav"), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(SR)
        fh.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    (OUT / "params.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    # Contact sheet with timestamps and phonemes.
    tiles = []
    for row in rows:
        image = cv2.imread(row["path"])
        image = image[380:1020, 560:1280]
        image = cv2.resize(image, (360, 320))
        label = f'{row["timestamp"] - start:.2f}s {row["phoneme"] or "-"}'
        cv2.putText(image, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        tiles.append(image)
    if tiles:
        columns = 8
        rows_img = []
        for index in range(0, len(tiles), columns):
            part = tiles[index:index + columns]
            while len(part) < columns:
                part.append(np.zeros_like(tiles[0]))
            rows_img.append(np.hstack(part))
        cv2.imwrite(str(OUT / "sheet.jpg"), np.vstack(rows_img))
    print("frames", len(rows), "audio", OUT / "audio.wav", "sheet", OUT / "sheet.jpg")
    # Audio envelope vs mouth trace.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    win = int(SR * 0.02)
    rms = np.array([float(np.sqrt(np.mean(audio[i:i + win] ** 2))) for i in range(0, len(audio) - win, win)])
    times = np.arange(rms.size) * win / SR
    plt.figure(figsize=(14, 5))
    plt.plot(times, rms / max(1e-9, float(rms.max())), label="audio RMS (normalized)", color="#4a90d9")
    if rows:
        mt = np.array([row["timestamp"] - start for row in rows])
        mo = np.array([row["mouth_open"] or 0.0 for row in rows])
        plt.plot(mt, mo, label="MouthOpen", color="#d94a4a", marker="o", markersize=2)
        for row in rows:
            if row["phoneme"] and row["phoneme"] != "_":
                plt.text(row["timestamp"] - start, 1.03, row["phoneme"], fontsize=7, ha="center")
    plt.ylim(0, 1.12)
    plt.xlabel("seconds from loopback start")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT / "trace.png", dpi=130)
    print("trace", OUT / "trace.png")


asyncio.run(main())
