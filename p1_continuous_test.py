"""P1 acceptance: drive one minute of continuous speech and capture NDI frames."""

import asyncio
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import websockets

OUT = Path("dsh-vts-avatar/data/audit/p1_continuous")
TEXT = "今天天气很好，我们一起出去走走吧。" * 14


async def snapshot(ws, index: int) -> Path | None:
    request_id = f"p1-{index}"
    await ws.send(json.dumps({"type": "snapshot", "request_id": request_id}))
    while True:
        message = json.loads(await ws.recv())
        if message.get("type") == "snapshot.result" and message.get("request_id") == request_id:
            path = (message.get("metrics") or {}).get("path")
            if path:
                target = OUT / f"{index:02d}.jpg"
                shutil.copyfile(path, target)
                return target
    return None


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    async with websockets.connect("ws://127.0.0.1:8765") as ws:
        await ws.recv()
        await ws.send(json.dumps({"type": "utterance.start", "utterance_id": "p1-continuous"}))
        await ws.send(json.dumps({"type": "text.delta", "text": TEXT}, ensure_ascii=False))
        await ws.send(json.dumps({"type": "utterance.end", "utterance_id": "p1-continuous"}))
        frames = []
        for index in range(20):
            await asyncio.sleep(3.0)
            path = await snapshot(ws, index)
            if path:
                frames.append((index * 3, path))
                print("captured", index * 3, path)
    tiles = []
    for seconds, path in frames:
        image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        image = cv2.resize(image, (320, 180))
        cv2.putText(image, f"{seconds}s", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        tiles.append(image)
    rows = []
    for start in range(0, len(tiles), 4):
        row = tiles[start:start + 4]
        while len(row) < 4:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    if rows:
        ok, buffer = cv2.imencode(".jpg", np.vstack(rows))
        (OUT / "sheet.jpg").write_bytes(buffer.tobytes())
        print("saved", OUT / "sheet.jpg")


asyncio.run(main())
