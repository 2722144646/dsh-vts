import asyncio
import json
import random
import shutil
from pathlib import Path

import cv2
import websockets

PRESETS = ["happy", "sad", "angry", "surprised", "thinking", "shy"]
OUT = Path("dsh-vts-avatar/data/expression_shots")


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    order = PRESETS[:]
    random.Random(7).shuffle(order)
    frames = []
    key = {}
    async with websockets.connect("ws://127.0.0.1:8765") as ws:
        await ws.recv()
        for i, name in enumerate(order):
            label = chr(ord("A") + i)
            await ws.send(json.dumps({"type": "emotion", "name": name, "intensity": 0.9}))
            await asyncio.sleep(1.15)
            rid = f"blind-{i}"
            await ws.send(json.dumps({"type": "snapshot", "request_id": rid}))
            while True:
                raw = json.loads(await ws.recv())
                if raw.get("type") == "snapshot.result" and raw.get("request_id") == rid:
                    path = (raw.get("metrics") or {}).get("path")
                    if path:
                        unique = OUT / f"blind_{label}.jpg"
                        shutil.copyfile(path, unique)
                        frames.append((label, str(unique)))
                        key[label] = name
                    break
        await ws.send(json.dumps({"type": "emotion", "name": "neutral", "intensity": 0.5}))

    tiles = []
    for label, path in frames:
        im = cv2.imread(path)
        if im is None:
            continue
        crop = im[40:760, 500:1300]
        crop = cv2.resize(crop, (360, 324))
        cv2.putText(crop, label, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        tiles.append(crop)
    rows = []
    for r in range(0, len(tiles), 3):
        row = tiles[r:r + 3]
        while len(row) < 3:
            row.append(tiles[0] * 0)
        rows.append(cv2.hconcat(row))
    cv2.imwrite(str(OUT / "blind_test.jpg"), cv2.vconcat(rows))
    (OUT / "blind_key.json").write_text(
        json.dumps({"order": order, "key": key}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("saved", OUT / "blind_test.jpg")
    print("key", key)


asyncio.run(main())
