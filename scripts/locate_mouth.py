"""Locate the mouth by diffing two poses instead of guessing from eyes.

Inject a neutral pose, then a pose with the mouth wide open; the region that
changes is by definition the mouth. Works on any model, no eye/headgear
assumptions.

    python scripts/locate_mouth.py --out E:\\dsh-vts-avatar\\mouth_probe
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import shutil
import sys

import cv2
import numpy as np
import websockets

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lipread_test import Bridge  # noqa: E402

NEUTRAL = {
    "MouthOpen": 0.0,
    "MouthSmile": 0.0,
    "MouthX": 0.0,
    "JawOpen": 0.0,
    "MouthFunnel": 0.0,
    "MouthPucker": 0.0,
    "FaceAngleX": 0.0,
    "FaceAngleY": 0.0,
    "FaceAngleZ": 0.0,
}
OPEN = {**NEUTRAL, "MouthOpen": 1.0, "JawOpen": 1.0, "MouthSmile": 1.0}


async def run(out: pathlib.Path, url: str) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    async with Bridge(url) as bridge:
        await bridge.call({"type": "inject.pause", "paused": True}, "inject.pause.result")
        try:
            a_src = await bridge.hold(NEUTRAL, seconds=0.6)
            b_src = await bridge.hold(OPEN, seconds=0.6)
        finally:
            await bridge.call({"type": "inject.pause", "paused": False}, "inject.pause.result")
    a_path = out / "neutral.jpg"
    b_path = out / "open.jpg"
    for src, dst in ((a_src, a_path), (b_src, b_path)):
        if src and src.exists():
            shutil.copyfile(src, dst)
    a = cv2.imdecode(np.fromfile(str(a_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    b = cv2.imdecode(np.fromfile(str(b_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if a is None or b is None or a.shape != b.shape:
        return {"error": "capture failed"}
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.int16)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.int16)
    diff = np.abs(ga - gb).astype(np.uint8)
    diff = cv2.GaussianBlur(diff, (7, 7), 0)
    _t, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, _lab, stats, _cent = cv2.connectedComponentsWithStats(mask, 8)
    best = None
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 50 or w < 6 or h < 4:
            continue
        score = area
        if best is None or score > best[0]:
            best = (score, x, y, w, h)
    if best is None:
        return {"error": "no changed region found"}
    _score, x, y, w, h = best
    pad_x, pad_y = max(6, w // 3), max(6, h // 3)
    box = [int(max(0, y - pad_y)), int(min(a.shape[0], y + h + pad_y)),
           int(max(0, x - pad_x)), int(min(a.shape[1], x + w + pad_x))]
    crop_a = a[box[0]:box[1], box[2]:box[3]]
    crop_b = b[box[0]:box[1], box[2]:box[3]]
    cv2.imwrite(str(out / "mouth_neutral.png"), crop_a)
    cv2.imwrite(str(out / "mouth_open.png"), crop_b)
    result = {
        "box": box,
        "changed_area": int(w * h),
        "frame_size": [int(a.shape[1]), int(a.shape[0])],
        "mouth_center": [int((box[2] + box[3]) / 2), int((box[0] + box[1]) / 2)],
    }
    (out / "box.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/mouth_probe")
    parser.add_argument("--url", default="ws://127.0.0.1:8765")
    args = parser.parse_args()
    result = asyncio.run(run(pathlib.Path(args.out), args.url))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
