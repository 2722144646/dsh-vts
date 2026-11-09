"""Speak the vowel line with the current backend and return a mouth-sequence sheet."""

import asyncio
import json
import pathlib

import cv2
import numpy as np
import websockets

TEXT = "阿——伊——乌——诶——哦"
CROP = (820, 520, 1120, 780)
OUT = pathlib.Path(r"E:\dsh-vts-avatar\shape_live.jpg")


async def main() -> None:
    async with websockets.connect("ws://127.0.0.1:8765", max_size=None) as ws:
        await ws.send(json.dumps({"type": "hello", "client": "sl", "protocol": 1}))
        seq = 0

        async def call(msg, expect):
            nonlocal seq
            seq += 1
            msg = dict(msg)
            msg["request_id"] = f"s{seq}"
            await ws.send(json.dumps(msg))
            while True:
                reply = json.loads(await asyncio.wait_for(ws.recv(), 60))
                if reply.get("type") == expect and reply.get("request_id") == msg["request_id"]:
                    return reply

        await ws.send(json.dumps({"type": "utterance.start", "utterance_id": "sl1"}))
        await ws.send(json.dumps({"type": "text.delta", "text": TEXT}))
        await ws.send(json.dumps({"type": "utterance.end"}))

        panels = []
        last = None
        for _ in range(8):
            await asyncio.sleep(0.22)
            reply = await call({"type": "snapshot", "frames": 1}, "snapshot.result")
            path = (reply.get("metrics") or {}).get("path")
            if not path:
                continue
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            if last is not None and float(np.mean(cv2.absdiff(img, last))) < 0.01:
                continue
            last = img
            x0, y0, x1, y1 = CROP
            crop = img[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            panels.append(cv2.resize(crop, (360, 312), interpolation=cv2.INTER_CUBIC))

    if panels:
        sheet = np.hstack(panels)
        cv2.imencode(".jpg", sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tofile(str(OUT))
        print("frames:", len(panels), "saved:", OUT.exists())
    else:
        print("no frames")


asyncio.run(main())
