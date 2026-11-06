import asyncio
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import websockets

OUT = Path("dsh-vts-avatar/data/audit")
EXPRESSIONS = [
    ("neutral", "neutral"),
    ("happy", "happy"),
    ("sad", "sad"),
    ("angry", "angry"),
    ("surprised", "surprised"),
    ("wronged", "wronged"),
    ("thinking", "thinking"),
    ("shy", "shy"),
]


async def snap(ws, label):
    rid = f"audit-{label}-{asyncio.get_running_loop().time():.3f}"
    await ws.send(json.dumps({"type": "snapshot", "request_id": rid}))
    while True:
        raw = json.loads(await ws.recv())
        if raw.get("type") == "snapshot.result" and raw.get("request_id") == rid:
            path = (raw.get("metrics") or {}).get("path")
            if path:
                dst = OUT / f"{label}.jpg"
                shutil.copyfile(path, dst)
                return dst
    return None


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    async with websockets.connect("ws://127.0.0.1:8765") as ws:
        await ws.recv()
        for name, label in EXPRESSIONS:
            await ws.send(json.dumps({"type": "emotion", "name": name, "intensity": 0.85}))
            await asyncio.sleep(0.9)
            await snap(ws, label)
        await ws.send(json.dumps({"type": "emotion", "name": "neutral", "intensity": 0.5}))
        await asyncio.sleep(0.6)
        # Capture the combined turn action at four phases.
        await ws.send(json.dumps({"type": "action", "kind": "turn_look_left", "intensity": 0.7}))
        for i, delay in enumerate([0.10, 0.20, 0.30, 0.35]):
            await asyncio.sleep(delay)
            await snap(ws, f"turn_{i}")
        await ws.send(json.dumps({"type": "emotion", "name": "neutral", "intensity": 0.5}))

    tiles = []
    for name, label in EXPRESSIONS:
        path = OUT / f"{label}.jpg"
        if not path.exists():
            continue
        im = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        im = cv2.resize(im[40:760, 500:1300], (360, 324))
        cv2.putText(im, label, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        tiles.append(im)
    rows = []
    for i in range(0, len(tiles), 4):
        row = tiles[i:i + 4]
        while len(row) < 4:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)
    ok, buf = cv2.imencode(".jpg", sheet)
    (OUT / "expressions.jpg").write_bytes(buf.tobytes())

    turn = []
    for i in range(4):
        p = OUT / f"turn_{i}.jpg"
        if p.exists():
            im = cv2.imdecode(np.frombuffer(p.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            im = cv2.resize(im[40:760, 500:1300], (360, 324))
            cv2.putText(im, f"turn {i}", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            turn.append(im)
    if turn:
        ok, buf = cv2.imencode(".jpg", np.hstack(turn))
        (OUT / "turn.jpg").write_bytes(buf.tobytes())
    print("saved", OUT / "expressions.jpg", OUT / "turn.jpg")


asyncio.run(main())
