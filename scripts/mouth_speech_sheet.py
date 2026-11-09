"""Speak a vowel sequence and return a zoomed mouth montage (no detection)."""

import asyncio
import json
import pathlib

import cv2
import numpy as np
import websockets

TEXT = "阿——伊——乌——诶——哦"
BOX = (635, 705, 925, 1020)  # top, bottom, left, right (full-res, eye-checked)
OUT = pathlib.Path(r"E:\dsh-vts-avatar\mouth_speech.jpg")


async def main() -> None:
    async with websockets.connect("ws://127.0.0.1:8765", max_size=None) as ws:
        await ws.send(json.dumps({"type": "hello", "client": "ms", "protocol": 1}))
        seq = 0

        async def call(msg, expect):
            nonlocal seq
            seq += 1
            msg = dict(msg)
            msg["request_id"] = f"m{seq}"
            await ws.send(json.dumps(msg))
            while True:
                reply = json.loads(await asyncio.wait_for(ws.recv(), 60))
                if reply.get("type") == expect and reply.get("request_id") == msg["request_id"]:
                    return reply

        await ws.send(json.dumps({"type": "utterance.start", "utterance_id": "ms1"}))
        await ws.send(json.dumps({"type": "text.delta", "text": TEXT}))
        await ws.send(json.dumps({"type": "utterance.end"}))

        shots = []
        for i in range(6):
            await asyncio.sleep(0.28)
            reply = await call({"type": "snapshot", "frames": 1}, "snapshot.result")
            path = (reply.get("metrics") or {}).get("path")
            if not path:
                continue
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            top, bottom, left, right = BOX
            crop = img[top:bottom, left:right]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (380, 280), interpolation=cv2.INTER_CUBIC)
            cv2.putText(crop, str(i), (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 80, 255), 2)
            shots.append(crop)

    if shots:
        sheet = np.hstack(shots)
        cv2.imencode(".jpg", sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tofile(str(OUT))
        print("frames:", len(shots), "saved:", OUT.exists())


asyncio.run(main())
