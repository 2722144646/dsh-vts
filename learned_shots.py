import asyncio
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import websockets

NAMES = ["委屈", "开心", "难过", "惊讶", "思考", "害羞"]
LABELS = ["Q", "H", "S", "SP", "T", "SH"]
OUT = Path("dsh-vts-avatar/data/demos/learned_shots")


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    async with websockets.connect("ws://127.0.0.1:8765") as ws:
        await ws.recv()
        for i, name in enumerate(NAMES):
            await ws.send(json.dumps({"type": "emotion", "name": name, "intensity": 1.0}, ensure_ascii=False))
            await asyncio.sleep(1.2)
            rid = f"learned-{i}"
            await ws.send(json.dumps({"type": "snapshot", "request_id": rid}))
            while True:
                raw = json.loads(await ws.recv())
                if raw.get("type") == "snapshot.result" and raw.get("request_id") == rid:
                    path = (raw.get("metrics") or {}).get("path")
                    if path:
                        unique = OUT / f"{i}_{name}.jpg"
                        shutil.copyfile(path, unique)
                        frames.append((LABELS[i], str(unique)))
                    break
        await ws.send(json.dumps({"type": "emotion", "name": "neutral", "intensity": 0.5}))
    tiles = []
    for label, path in frames:
        im = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        im = im[40:760, 500:1300]
        im = cv2.resize(im, (360, 324))
        cv2.putText(im, label, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        tiles.append(im)
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    sheet = np.vstack(rows)
    ok, buf = cv2.imencode(".jpg", sheet)
    (OUT / "learned.jpg").write_bytes(buf.tobytes())
    print("saved", OUT / "learned.jpg")


asyncio.run(main())
