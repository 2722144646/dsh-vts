import asyncio
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import websockets

OUT = Path("dsh-vts-avatar/data/audit/speech")


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    async with websockets.connect("ws://127.0.0.1:8765") as ws:
        await ws.recv()
        await ws.send(json.dumps({"type": "utterance.start", "utterance_id": "audit-speech"}))
        await ws.send(json.dumps({"type": "text.delta", "text": "妈妈爸爸婆婆，抱抱宝宝，今天天气真好。"}, ensure_ascii=False))
        await ws.send(json.dumps({"type": "utterance.end", "utterance_id": "audit-speech"}))
        for i in range(18):
            await asyncio.sleep(0.18)
            rid = f"speech-{i}"
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
        im = cv2.resize(im[40:760, 500:1300], (300, 270))
        cv2.putText(im, p.stem, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        tiles.append(im)
    rows = []
    for i in range(0, len(tiles), 6):
        row = tiles[i:i + 6]
        while len(row) < 6:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)
    ok, buf = cv2.imencode(".jpg", sheet)
    (OUT / "speech.jpg").write_bytes(buf.tobytes())
    print("saved", OUT / "speech.jpg")


asyncio.run(main())
