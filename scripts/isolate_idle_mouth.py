"""Isolate what moves the mouth while the bridge is not injecting.

Pauses bridge injection, samples NDI frames, prints per-frame mouth motion, and
resumes. Run it once with music playing and once with music paused to tell a
microphone lip-sync apart from the model's own idle physics.
"""

import asyncio
import json

import websockets

N = 10


async def main() -> None:
    async with websockets.connect("ws://127.0.0.1:8765", max_size=None) as ws:
        await ws.send(json.dumps({"type": "hello", "client": "isolate", "protocol": 1}))
        seq = 0

        async def call(msg, expect):
            nonlocal seq
            seq += 1
            msg = dict(msg)
            msg["request_id"] = f"i{seq}"
            await ws.send(json.dumps(msg))
            while True:
                reply = json.loads(await asyncio.wait_for(ws.recv(), 40))
                if reply.get("type") == expect and reply.get("request_id") == msg["request_id"]:
                    return reply

        await call({"type": "inject.pause", "paused": True}, "inject.pause.result")
        await asyncio.sleep(2.0)
        try:
            reply = await call({"type": "snapshot", "frames": N, "interval_ms": 220}, "snapshot.result")
            m = reply.get("metrics") or {}
            per = m.get("mouth_motion_frames") or m.get("mouth_motions") or None
            print("frames   :", m.get("frames"))
            print("mouth max:", m.get("mouth_motion"))
            print("mouth mean:", m.get("mean_mouth_motion"))
            print("head max :", m.get("head_motion"))
            print("motion   :", m.get("motion"))
            print("per-frame:", per)
        finally:
            await call({"type": "inject.pause", "paused": False}, "inject.pause.result")
            print("resumed")


asyncio.run(main())
