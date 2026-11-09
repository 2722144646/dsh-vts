import asyncio
import json

import websockets


async def main() -> None:
    async with websockets.connect("ws://127.0.0.1:8765", max_size=None) as ws:
        await ws.send(json.dumps({"type": "hello", "client": "chk", "protocol": 1}))
        seq = 0

        async def call(msg, expect):
            nonlocal seq
            seq += 1
            msg = dict(msg)
            msg["request_id"] = f"c{seq}"
            await ws.send(json.dumps(msg))
            while True:
                reply = json.loads(await asyncio.wait_for(ws.recv(), 30))
                if reply.get("type") == expect and reply.get("request_id") == msg["request_id"]:
                    return reply

        await ws.send(json.dumps({"type": "utterance.start", "utterance_id": "chk"}))
        await ws.send(json.dumps({"type": "text.delta", "text": "阿伊乌诶哦"}))
        await ws.send(json.dumps({"type": "utterance.end"}))
        rows = []
        for _ in range(12):
            await asyncio.sleep(0.12)
            reply = await call({"type": "parameters"}, "parameters.result")
            data = reply.get("data") or {}
            params = data.get("params") or {}
            rows.append(
                {
                    "viseme": data.get("viseme"),
                    "open": round(float(params.get("MouthOpen", 0)), 2),
                    "smile": round(float(params.get("MouthSmile", 0)), 2),
                    "x": round(float(params.get("MouthX", 0)), 2),
                    "voiceA": round(float(params.get("VoiceA", 0)), 2),
                    "voiceI": round(float(params.get("VoiceI", 0)), 2),
                }
            )
        for row in rows:
            print(row)


asyncio.run(main())
