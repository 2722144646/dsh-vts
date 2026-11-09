"""Speak one line and return a contact sheet of the frames in the mouth window."""

import asyncio
import json
import pathlib
import shutil

import websockets

TEXT = "阿伊乌诶哦，阿伊乌诶哦"
OUT = pathlib.Path(r"E:\dsh-vts-avatar\live_sheet.jpg")


async def main() -> None:
    async with websockets.connect("ws://127.0.0.1:8765", max_size=None) as ws:
        await ws.send(json.dumps({"type": "hello", "client": "live", "protocol": 1}))
        seq = 0

        async def call(msg, expect):
            nonlocal seq
            seq += 1
            msg = dict(msg)
            msg["request_id"] = f"l{seq}"
            await ws.send(json.dumps(msg))
            while True:
                reply = json.loads(await asyncio.wait_for(ws.recv(), 60))
                if reply.get("type") == expect and reply.get("request_id") == msg["request_id"]:
                    return reply

        await ws.send(json.dumps({"type": "utterance.start", "utterance_id": "live1"}))
        await ws.send(json.dumps({"type": "text.delta", "text": TEXT}))
        await ws.send(json.dumps({"type": "utterance.end"}))
        await asyncio.sleep(1.2)
        reply = await call({"type": "snapshot", "frames": 10, "interval_ms": 220}, "snapshot.result")
        metrics = reply.get("metrics") or {}
        sheet = metrics.get("sequence") or metrics.get("path")
        print("sheet:", sheet)
        if sheet and pathlib.Path(sheet).exists():
            shutil.copyfile(sheet, OUT)
            print("saved:", OUT, OUT.exists())
        print("mouth_motion:", metrics.get("mouth_motion"), "frames:", metrics.get("frames"))


asyncio.run(main())
