import asyncio
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import websockets

OUT = Path("dsh-vts-avatar/data/audit/distill_live")


async def snapshot(ws, label):
    rid = f"distill-{label}"
    await ws.send(json.dumps({"type": "snapshot", "request_id": rid}))
    while True:
        raw = json.loads(await ws.recv())
        if raw.get("type") == "snapshot.result" and raw.get("request_id") == rid:
            path = (raw.get("metrics") or {}).get("path")
            if path:
                target = OUT / f"{label}.jpg"
                shutil.copyfile(path, target)
                return target
    return None


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    async with websockets.connect("ws://127.0.0.1:8765") as ws:
        await ws.recv()
        await ws.send(json.dumps({"type": "utterance.start", "utterance_id": "distill-live"}))
        await ws.send(json.dumps({"type": "text.delta", "text": "你好，我是肥鱼。今天天气不错，我们开始吧。"}, ensure_ascii=False))
        await ws.send(json.dumps({"type": "utterance.end", "utterance_id": "distill-live"}))
        for i in range(12):
            await asyncio.sleep(0.35)
            await snapshot(ws, f"{i:02d}")
    tiles = []
    for p in sorted(OUT.glob("[0-9][0-9].jpg")):
        im = cv2.imdecode(np.frombuffer(p.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        im = cv2.resize(im, (320, 180))
        cv2.putText(im, p.stem, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        tiles.append(im)
    rows = []
    for i in range(0, len(tiles), 4):
        row = tiles[i:i + 4]
        while len(row) < 4:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    ok, buf = cv2.imencode(".jpg", np.vstack(rows))
    (OUT / "live.jpg").write_bytes(buf.tobytes())
    print("saved", OUT / "live.jpg")


asyncio.run(main())
