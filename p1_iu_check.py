"""Focused visual check: isolated I vs U, head tilted up, mouth crop."""

import asyncio
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import websockets

OUT = Path("dsh-vts-avatar/data/audit/p1_iu_check")
OUT.mkdir(parents=True, exist_ok=True)


async def snap(ws, name: str) -> Path:
    rid = f"iu-{name}"
    await ws.send(json.dumps({"type": "snapshot", "request_id": rid}))
    while True:
        message = json.loads(await ws.recv())
        if message.get("type") == "snapshot.result" and message.get("request_id") == rid:
            path = (message.get("metrics") or {}).get("path")
            target = OUT / f"{name}.jpg"
            shutil.copyfile(path, target)
            return target


async def say(ws, prefix: str, text: str) -> None:
    await ws.send(json.dumps({"type": "utterance.start", "utterance_id": prefix}))
    await ws.send(json.dumps({"type": "text.delta", "text": text}, ensure_ascii=False))
    await ws.send(json.dumps({"type": "utterance.end", "utterance_id": prefix}))
    for index in range(10):
        await asyncio.sleep(0.20)
        await snap(ws, f"{prefix}_{index:02d}")
    await asyncio.sleep(0.7)


async def main() -> None:
    async with websockets.connect("ws://127.0.0.1:8765") as ws:
        await ws.recv()
        await ws.send(json.dumps({"type": "head_pose", "pitch": 12, "seconds": 25}))
        await asyncio.sleep(0.8)
        await say(ws, "I", "伊伊伊伊")
        await say(ws, "U", "乌乌乌乌")
    # Full frames
    for prefix in ("I", "U"):
        images = []
        for index in range(10):
            image = cv2.imread(str(OUT / f"{prefix}_{index:02d}.jpg"))
            image = cv2.resize(image, (320, 180))
            cv2.putText(image, f"{prefix}{index}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            images.append(image)
        cv2.imwrite(str(OUT / f"{prefix}_sheet.jpg"), np.vstack([np.hstack(images[:5]), np.hstack(images[5:])]))
    # Mouth crops (face sits near the center of the 1920x1080 frame)
    crops = []
    for prefix in ("I", "U"):
        row = []
        for index in range(10):
            image = cv2.imread(str(OUT / f"{prefix}_{index:02d}.jpg"))
            crop = image[760:1080, 620:1300]
            crop = cv2.resize(crop, (680, 400))
            cv2.putText(crop, f"{prefix}{index}", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            row.append(crop)
        crops.append(np.hstack(row))
    cv2.imwrite(str(OUT / "crops.jpg"), np.vstack(crops))
    print(OUT / "crops.jpg")


asyncio.run(main())
