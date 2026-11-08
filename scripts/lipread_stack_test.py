"""Stacked-channel lip-read test at a chosen vowel amplitude.

    python scripts/lipread_stack_test.py --voice 0.9 --out E:\\dsh-vts-avatar\\冰糖\\lipread_09

Schemes: vowel90, vowel90+smile, vowel90+vb, all90. Head zeroed, injection
paused, MouthOpen fixed; nothing here changes the normal speech path.
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
SHAPE = {
    "A": (0.10, 0.10),
    "I": (0.60, 0.00),
    "E": (0.30, 0.05),
    "O": (-0.30, 0.65),
    "U": (-0.50, 0.85),
}


def params_for(scheme: str, vowel: str, voice: float) -> dict[str, float]:
    width, roundness = SHAPE[vowel]
    out = {"MouthOpen": 0.55, "FaceAngleX": 0.0, "FaceAngleY": 0.0, "FaceAngleZ": 0.0}
    if "vowel" in scheme:
        out[f"Voice{vowel}"] = voice
    if "smile" in scheme:
        out["MouthSmile"] = max(0.0, min(1.0, 0.45 + 0.45 * width))
    if "vb" in scheme:
        out["JawOpen"] = 0.25 + 0.25 * roundness
        out["MouthFunnel"] = roundness * 0.8
        out["MouthPucker"] = roundness * 0.7
    return out


class Bridge:
    def __init__(self, url: str) -> None:
        self.url = url
        self.ws = None
        self.seq = 0

    async def __aenter__(self):
        self.ws = await websockets.connect(self.url, max_size=None)
        await self.ws.send(json.dumps({"type": "hello", "client": "lipstack", "protocol": 1}))
        return self

    async def __aexit__(self, *exc):
        await self.ws.close()

    async def call(self, msg, expect):
        self.seq += 1
        msg = dict(msg)
        msg["request_id"] = f"c{self.seq}"
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
    schemes = ["vowel", "vowel+smile", "vowel+vb", "vowel+smile+vb"]
    report = {"voice": voice, "channels": {}}
    async with Bridge(url) as bridge:
        await bridge.call({"type": "inject.pause", "paused": True}, "inject.pause.result")
        try:
            for scheme in schemes:
                crops = {}
                for vowel in VOWELS:
                    src = await bridge.hold(params_for(scheme, vowel, voice))
                    dst = frames / f"{scheme.replace('+', '_')}_{vowel}.jpg"
                    if src and src.exists():
                        shutil.copyfile(src, dst)
                        crops[vowel] = mouth_crop(cv2.imdecode(np.fromfile(str(dst), dtype=np.uint8), cv2.IMREAD_COLOR))
                diffs = {}
                for a, b in itertools.combinations(VOWELS, 2):
                    if a in crops and b in crops:
                        ga = cv2.cvtColor(crops[a], cv2.COLOR_BGR2GRAY)
                        gb = cv2.cvtColor(crops[b], cv2.COLOR_BGR2GRAY)
                        diffs[f"{a}-{b}"] = round(float(np.mean(cv2.absdiff(ga, gb))), 2)
                values = list(diffs.values()) or [0.0]
                report["channels"][scheme] = {
                    "pairwise": diffs,
                    "mean": round(sum(values) / len(values), 2),
                    "min_pair": min(diffs.items(), key=lambda kv: kv[1]),
                    "unreadable": [k for k, v in diffs.items() if v < 10],
                }
        finally:
            await bridge.call({"type": "inject.pause", "paused": False}, "inject.pause.result")
    (out / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/lipread_stack")
    parser.add_argument("--url", default="ws://127.0.0.1:8765")
    parser.add_argument("--voice", type=float, default=0.9)
    args = parser.parse_args()
    report = asyncio.run(run(pathlib.Path(args.out), args.url, args.voice))
    print(json.dumps({k: (v["mean"], v["min_pair"], v["unreadable"]) for k, v in report["channels"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
