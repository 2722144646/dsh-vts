import asyncio
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import websockets

OUT = Path("dsh-vts-avatar/data/audit/lip_test")


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    async with websockets.connect("ws://127.0.0.1:8765") as ws:
        await ws.recv()
        await ws.send(json.dumps({"type": "utterance.start", "utterance_id": "lip-test"}))
        await ws.send(json.dumps({"type": "text.delta", "text": "妈妈爸爸婆婆，抱抱宝宝，今天天气真好。"}, ensure_ascii=False))
        await ws.send(json.dumps({"type": "utterance.end", "utterance_id": "lip-test"}))
        for i in range(20):
            await asyncio.sleep(0.20)
            rid = f"lip-{i}"
            await ws.send(json.dumps({"type": "snapshot", "request_id": rid}))
            while True:
                raw = json.loads(await ws.recv())
                if raw.get("type") == "snapshot.result" and raw.get("request_id") == rid:
                    path = (raw.get("metrics") or {}).get("path")
                    if path:
                        shutil.copyfile(path, OUT / f"{i:02d}.jpg")
                    break
    tiles = []
    for p in sorted(OUT.glob("[0-9][0-9].jpg")):
        im = cv2.imdecode(np.frombuffer(p.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        im = cv2.resize(im, (320, 180))
        cv2.putText(im, p.stem, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        tiles.append(im)
    rows = []
    for i in range(0, len(tiles), 5):
        row = tiles[i:i + 5]
        while len(row) < 5:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)
    ok, buf = cv2.imencode(".jpg", sheet)
    (OUT / "lip_test.jpg").write_bytes(buf.tobytes())
    print("saved", OUT / "lip_test.jpg")


asyncio.run(main())
