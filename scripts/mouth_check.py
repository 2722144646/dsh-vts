"""Per-channel vowel frames + montage, for eyeballing lip readability.

    python scripts/mouth_check.py capture --out DIR
    python scripts/mouth_check.py montage --out DIR --crop 900,600,1020,700

capture: for each channel (vowel/smile/shape/hybrid) inject each vowel and save
a fresh NDI frame. montage: crop + label them into one image per channel.
"""

from __future__ import annotations

import argparse
import asyncio
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lipread_test import Bridge, scheme_params  # noqa: E402

VOWELS = "AIUEO"
CHANNELS = ("vowel", "smile", "shape", "hybrid")


async def capture(out: pathlib.Path) -> None:
    frames = out / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    async with Bridge("ws://127.0.0.1:8765") as bridge:
        await bridge.call({"type": "inject.pause", "paused": True}, "inject.pause.result")
        try:
            for channel in CHANNELS:
                last = None
                for vowel in VOWELS:
                    img = None
                    for _attempt in range(6):
                        src = await bridge.hold(scheme_params(channel, vowel), seconds=0.8)
                        new = cv2.imdecode(np.fromfile(str(src), dtype=np.uint8), cv2.IMREAD_COLOR)
                        if new is None:
                            continue
                        # NDI can answer with the previous frame; require a change.
                        if last is not None and float(np.mean(cv2.absdiff(new, last))) < 0.05:
                            continue
                        img = new
                        last = new
                        break
                    if img is None:
                        print(channel, vowel, "no fresh frame")
                        continue
                    path = frames / f"{channel}_{vowel}.jpg"
                    cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tofile(str(path))
                    print(channel, vowel, "ok")
        finally:
            await bridge.call({"type": "inject.pause", "paused": False}, "inject.pause.result")


def montage(out: pathlib.Path, crop: tuple[int, int, int, int] | None, scale: float) -> None:
    frames = out / "frames"
    for channel in CHANNELS:
        panels = []
        for vowel in VOWELS:
            path = frames / f"{channel}_{vowel}.jpg"
            if not path.exists():
                continue
            img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if crop:
                x0, y0, x1, y1 = crop
                img = img[y0:y1, x0:x1]
            target_w = int(img.shape[1] * scale)
            target_h = int(img.shape[0] * scale)
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            cv2.putText(img, vowel, (8, int(target_h * 0.18)), cv2.FONT_HERSHEY_SIMPLEX,
                        max(0.8, target_h / 220), (0, 80, 255), 3)
            panels.append(img)
        if not panels:
            continue
        sheet = np.hstack(panels)
        path = out / f"{channel}.jpg"
        cv2.imencode(".jpg", sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tofile(str(path))
        print(channel, "->", path.name, sheet.shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["capture", "montage", "all"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--crop", default="", help="x0,y0,x1,y1 in pixels")
    parser.add_argument("--scale", type=float, default=0.32)
    args = parser.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    crop = tuple(int(v) for v in args.crop.split(",")) if args.crop else None
    if args.mode in ("capture", "all"):
        asyncio.run(capture(out))
    if args.mode in ("montage", "all"):
        montage(out, crop, args.scale)


if __name__ == "__main__":
    main()
