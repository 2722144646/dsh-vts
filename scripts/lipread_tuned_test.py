"""Vowel base + small same-domain tweaks (pucker/funnel/smile offsets).

    python scripts/lipread_tuned_test.py --out E:\\dsh-vts-avatar\\大肥鱼\\lipread_tuned
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import pathlib
import shutil

import cv2
import numpy as np
import websockets

VOWELS = "AIUEO"


def tuned_params(vowel: str, voice: float = 1.0) -> dict[str, float]:
    out = {
        "MouthOpen": 0.55,
        "FaceAngleX": 0.0,
        "FaceAngleY": 0.0,
        "FaceAngleZ": 0.0,
        f"Voice{vowel}": voice,
    }
    if vowel in "OU":
        out["MouthPucker"] = 0.2
        out["MouthFunnel"] = 0.15
    if vowel == "E":
        out["MouthSmile"] = 0.1
    return out


class Bridge:
    def __init__(self, url: str) -> None:
        self.url = url
        self.ws = None
        self.seq = 0

    async def __aenter__(self):
        self.ws = await websockets.connect(self.url, max_size=None)
        await self.ws.send(json.dumps({"type": "hello", "client": "tuned", "protocol": 1}))
        return self

    async def __aexit__(self, *exc):
        await self.ws.close()

    async def call(self, msg, expect):
        self.seq += 1
        msg = dict(msg)
        msg["request_id"] = f"t{self.seq}"
        await self.ws.send(json.dumps(msg))
        while True:
            reply = json.loads(await asyncio.wait_for(self.ws.recv(), 30))
            if reply.get("type") == expect and reply.get("request_id") == msg["request_id"]:
                return reply

    async def send(self, msg):
        self.seq += 1
        msg = dict(msg)
        msg["request_id"] = f"s{self.seq}"
        await self.ws.send(json.dumps(msg))

    async def hold(self, params, seconds=0.6):
        async def pump():
            while True:
                await self.send({"type": "params.inject", "params": params})
                await asyncio.sleep(0.03)

        task = asyncio.create_task(pump())
        try:
            await asyncio.sleep(seconds)
            reply = await self.call({"type": "snapshot", "frames": 1}, "snapshot.result")
            path = (reply.get("metrics") or {}).get("path")
            return pathlib.Path(path) if path else None
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


def mouth_crop(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where((gray < 235).astype(np.uint8) > 0)
    if len(xs) == 0:
        h, w = img.shape[:2]
        crop = img[h // 3 : h // 2, w // 3 : 2 * w // 3]
    else:
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        w, h, cx = x1 - x0, y1 - y0, (x0 + x1) // 2
        crop = img[y0 + int(h * 0.12) : y0 + int(h * 0.30), cx - int(w * 0.16) : cx + int(w * 0.16)]
    return cv2.resize(crop, (420, 300))


async def run(out: pathlib.Path, url: str, voice: float) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    frames = out / "frames"
    frames.mkdir(exist_ok=True)
    crops = {}
    async with Bridge(url) as bridge:
        await bridge.call({"type": "inject.pause", "paused": True}, "inject.pause.result")
        try:
            for vowel in VOWELS:
                src = await bridge.hold(tuned_params(vowel, voice))
                dst = frames / f"tuned_{vowel}.jpg"
                if src and src.exists():
                    shutil.copyfile(src, dst)
                    crops[vowel] = mouth_crop(cv2.imdecode(np.fromfile(str(dst), dtype=np.uint8), cv2.IMREAD_COLOR))
        finally:
            await bridge.call({"type": "inject.pause", "paused": False}, "inject.pause.result")
    diffs = {}
    for a, b in itertools.combinations(VOWELS, 2):
        if a in crops and b in crops:
            ga = cv2.cvtColor(crops[a], cv2.COLOR_BGR2GRAY)
            gb = cv2.cvtColor(crops[b], cv2.COLOR_BGR2GRAY)
            diffs[f"{a}-{b}"] = round(float(np.mean(cv2.absdiff(ga, gb))), 2)
    values = list(diffs.values()) or [0.0]
    report = {
        "voice": voice,
        "pairwise": diffs,
        "mean": round(sum(values) / len(values), 2),
        "min_pair": min(diffs.items(), key=lambda kv: kv[1]),
        "unreadable": [k for k, v in diffs.items() if v < 10],
    }
    (out / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/lipread_tuned")
    parser.add_argument("--url", default="ws://127.0.0.1:8765")
    parser.add_argument("--voice", type=float, default=1.0)
    args = parser.parse_args()
    report = asyncio.run(run(pathlib.Path(args.out), args.url, args.voice))
    print(json.dumps({k: report[k] for k in ("mean", "min_pair", "unreadable")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
