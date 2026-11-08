"""Compare mouth channels for lip-readability on the current VTS model.

Channels tested separately (head zeroed, bridge injection paused):
    vowel  VoiceA/I/U/E/O
    smile  MouthSmile width
    vb     JawOpen / MouthFunnel / MouthPucker / MouthShrug
    all    every channel together (try to compensate weak pairs)

For each channel the five vowels are rendered with a fixed MouthOpen, the mouth
ROI is cropped automatically, and pairwise ROI differences are reported:
    >= 25 clear, 10-25 close, < 10 unreadable.

    python scripts/lipread_test.py --out E:\\dsh-vts-avatar\\冰糖\\lipread
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import shutil
from typing import Any

import cv2
import numpy as np
import websockets

VOWELS = "AIUEO"
# vowel -> (voice width form -1..1, roundness 0..1, openness extra 0..1)
SHAPE = {
    "A": (0.10, 0.10, 0.10),
    "I": (0.60, 0.00, 0.00),
    "E": (0.30, 0.05, 0.05),
    "O": (-0.30, 0.65, 0.05),
    "U": (-0.50, 0.85, 0.00),
}


def scheme_params(scheme: str, vowel: str) -> dict[str, float]:
    width, roundness, extra = SHAPE[vowel]
    base: dict[str, float] = {"MouthOpen": 0.55, "FaceAngleX": 0.0, "FaceAngleY": 0.0, "FaceAngleZ": 0.0}
    out = dict(base)
    if scheme in ("vowel", "all"):
        out[f"Voice{vowel}"] = 1.0
    if scheme in ("smile", "all"):
        out["MouthSmile"] = max(0.0, min(1.0, 0.45 + 0.45 * width))
    if scheme in ("vb", "all"):
        out["JawOpen"] = 0.15 + 0.25 * extra + 0.2 * roundness
        out["MouthFunnel"] = roundness * 0.8
        out["MouthPucker"] = roundness * 0.7
        out["MouthShrug"] = width * 0.25
    return out


class Bridge:
    def __init__(self, url: str) -> None:
        self.url = url
        self.ws = None
        self.seq = 0

    async def __aenter__(self):
        self.ws = await websockets.connect(self.url, max_size=None)
        await self.ws.send(json.dumps({"type": "hello", "client": "lipread", "protocol": 1}))
        return self

    async def __aexit__(self, *exc):
        await self.ws.close()

    async def call(self, msg: dict[str, Any], expect: str) -> dict[str, Any]:
        self.seq += 1
        msg = dict(msg)
        msg["request_id"] = f"l{self.seq}"
        await self.ws.send(json.dumps(msg))
        while True:
            reply = json.loads(await asyncio.wait_for(self.ws.recv(), 30))
            if reply.get("type") == expect and reply.get("request_id") == msg["request_id"]:
                return reply

    async def send(self, msg: dict[str, Any]) -> None:
        self.seq += 1
        msg = dict(msg)
        msg["request_id"] = f"s{self.seq}"
        await self.ws.send(json.dumps(msg))

    async def hold(self, params: dict[str, float], seconds: float = 0.55) -> pathlib.Path | None:
        async def pump() -> None:
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
        return img[h // 3 : h // 2, w // 3 : 2 * w // 3]
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    w, h, cx = x1 - x0, y1 - y0, (x0 + x1) // 2
    return img[y0 + int(h * 0.12) : y0 + int(h * 0.30), cx - int(w * 0.16) : cx + int(w * 0.16)]


def pairwise(diffs: dict[str, float]) -> dict[str, str]:
    out = {}
    for key, value in diffs.items():
        out[key] = "clear" if value >= 25 else ("close" if value >= 10 else "unreadable")
    return out


async def run(out: pathlib.Path, url: str) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    frames = out / "frames"
    frames.mkdir(exist_ok=True)
    report: dict[str, Any] = {"channels": {}}
    async with Bridge(url) as bridge:
        await bridge.call({"type": "inject.pause", "paused": True}, "inject.pause.result")
        try:
            for scheme in ("vowel", "smile", "vb", "all"):
                crops = {}
                for vowel in VOWELS:
                    src = await bridge.hold(scheme_params(scheme, vowel))
                    dst = frames / f"{scheme}_{vowel}.jpg"
                    if src and src.exists():
                        shutil.copyfile(src, dst)
                        img = cv2.imdecode(np.fromfile(str(dst), dtype=np.uint8), cv2.IMREAD_COLOR)
                        crops[vowel] = mouth_crop(img)
                diffs = {}
                for i in range(len(VOWELS)):
                    for j in range(i + 1, len(VOWELS)):
                        a, b = VOWELS[i], VOWELS[j]
                        if a in crops and b in crops:
                            ga = cv2.cvtColor(crops[a], cv2.COLOR_BGR2GRAY)
                            gb = cv2.cvtColor(crops[b], cv2.COLOR_BGR2GRAY)
                            if ga.shape == gb.shape:
                                diffs[f"{a}-{b}"] = round(float(np.mean(cv2.absdiff(ga, gb))), 2)
                values = list(diffs.values()) or [0.0]
                report["channels"][scheme] = {
                    "pairwise": diffs,
                    "verdict": pairwise(diffs),
                    "mean": round(sum(values) / len(values), 2),
                    "min_pair": min(diffs.items(), key=lambda kv: kv[1]) if diffs else None,
                    "unreadable_pairs": [k for k, v in pairwise(diffs).items() if v == "unreadable"],
                }
        finally:
            await bridge.call({"type": "inject.pause", "paused": False}, "inject.pause.result")
    (out / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 唇读通道对比", ""]
    for scheme, data in report["channels"].items():
        lines.append(f"## {scheme}")
        lines.append(f"- 平均差异 {data['mean']}，最弱对 {data['min_pair']}")
        for pair, value in data["pairwise"].items():
            lines.append(f"- {pair}: {value} ({data['verdict'][pair]})")
        lines.append("")
    (out / "报告.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/lipread")
    parser.add_argument("--url", default="ws://127.0.0.1:8765")
    args = parser.parse_args()
    report = asyncio.run(run(pathlib.Path(args.out), args.url))
    print(json.dumps({k: (v["mean"], v["unreadable_pairs"]) for k, v in report["channels"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
