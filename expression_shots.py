import asyncio
import json
from pathlib import Path
import shutil
import sys

import cv2
import websockets

PRESETS = ["happy", "sad", "angry", "surprised", "thinking", "shy", "wronged"]
# Optional output directory, e.g. data/expression_shots_affect for an A/B run.
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "data" / "expression_shots"


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    async with websockets.connect("ws://127.0.0.1:8765") as ws:
        await ws.recv()
        for i, name in enumerate(PRESETS):
            await ws.send(json.dumps({"type": "emotion", "name": name, "intensity": 0.85}))
            await asyncio.sleep(1.1)
            rid = f"expr-{i}"
            await ws.send(json.dumps({"type": "snapshot", "request_id": rid}))
            while True:
                raw = json.loads(await ws.recv())
                if raw.get("type") == "snapshot.result" and raw.get("request_id") == rid:
                    metrics = raw.get("metrics") or {}
                    path = metrics.get("path")
                    if path:
                        unique = OUT / f"{name}.jpg"
                        shutil.copyfile(path, unique)
                        frames.append((name, str(unique)))
                    break
            print(name, metrics.get("path"))
        await ws.send(json.dumps({"type": "emotion", "name": "neutral", "intensity": 0.5}))
    tiles = []
    for name, path in frames:
        im = cv2.imread(path)
        if im is None:
            continue
        crop = im[40:760, 500:1300]
        crop = cv2.resize(crop, (400, 360))
        cv2.putText(crop, name, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        tiles.append(crop)
    rows = []
    for r in range(0, len(tiles), 3):
        row = tiles[r:r+3]
        while len(row) < 3:
            row.append(tiles[0] * 0)
        rows.append(cv2.hconcat(row))
    cv2.imwrite(str(OUT / "expressions.jpg"), cv2.vconcat(rows))
    print("saved", OUT / "expressions.jpg")


asyncio.run(main())
