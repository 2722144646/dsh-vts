"""Clean global clock audit: sustained vowel, loopback audio + mouth parameters."""

import asyncio
import json
import threading
import time
import wave
from pathlib import Path

import numpy as np
import soundcard as sc
import websockets

OUT = Path("dsh-vts-avatar/data/audit/global_sync")
TEXT = "啊" * 18
SR = 48000
SECONDS = 9.0


async def parameters(ws, index: int):
    request_id = f"gs-{index}"
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
            recorded["audio"] = np.asarray(recorder.record(numframes=int(SR * SECONDS))).reshape(-1)

    thread = threading.Thread(target=record)
    async with websockets.connect("ws://127.0.0.1:8765") as ws:
        await ws.recv()
        thread.start()
        await asyncio.sleep(0.2)
        await ws.send(json.dumps({"type": "utterance.start", "utterance_id": "global-sync"}))
        await ws.send(json.dumps({"type": "text.delta", "text": TEXT}, ensure_ascii=False))
        await ws.send(json.dumps({"type": "utterance.end", "utterance_id": "global-sync"}))
        rows = []
        for index in range(110):
            await asyncio.sleep(0.05)
            data = await parameters(ws, index)
            rows.append(
                {
                    "timestamp": time.time(),
                    "state": data.get("state"),
                    "phoneme": data.get("phoneme"),
                    "mouth_open": data.get("params", {}).get("MouthOpen"),
                }
            )
            if data.get("state") == "idle" and index > 20:
                break
    thread.join()
    audio = np.asarray(recorded["audio"], dtype=np.float32)
    start = float(recorded["start"])
    (OUT / "meta.json").write_text(json.dumps({"audio_start": start}, ensure_ascii=False), encoding="utf-8")
    (OUT / "params.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    with wave.open(str(OUT / "audio.wav"), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(SR)
        fh.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    print("captured", len(rows), "frames", OUT / "audio.wav")


asyncio.run(main())
