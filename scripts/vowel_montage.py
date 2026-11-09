"""Capture one frame per vowel and build a mouth montage for visual judgement.

Mouth box comes from a hand-checked coordinate (already verified on screen),
then follows the head with template matching. Produces montage.jpg that a
human/model can simply look at.
"""

from __future__ import annotations

import asyncio
import pathlib
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lipread_test import Bridge, scheme_params  # noqa: E402

VOWELS = "AIUEO"
BOX = (645, 695, 935, 1005)  # top, bottom, left, right (hand-verified)


def crop(img, box):
    top, bottom, left, right = box
    return img[top:bottom, left:right]


def track(prev_crop, img, box):
    """Follow the mouth: match the previous crop inside a padded window."""
    top, bottom, left, right = box
    pad_y, pad_x = 60, 80
    h, w = img.shape[:2]
    y0, y1 = max(0, top - pad_y), min(h, bottom + pad_y)
    x0, x1 = max(0, left - pad_x), min(w, right + pad_x)
    window = img[y0:y1, x0:x1]
    if window.shape[0] >= prev_crop.shape[0] and window.shape[1] >= prev_crop.shape[1]:
        res = cv2.matchTemplate(window, prev_crop, cv2.TM_CCOEFF_NORMED)
        _mn, score, _ml, loc = cv2.minMaxLoc(res)
        if score >= 0.5:
            dy, dx = loc[1] - pad_y, loc[0] - pad_x
            return (top + dy, bottom + dy, left + dx, right + dx)
    return box


async def main() -> None:
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else r"E:\dsh-vts-avatar\蒂亚圣诞树\vowel_montage")
    out.mkdir(parents=True, exist_ok=True)
    box = BOX
    crops = {}
    async with Bridge("ws://127.0.0.1:8765") as bridge:
        await bridge.call({"type": "inject.pause", "paused": True}, "inject.pause.result")
        try:
            prev = None
            last_full = None
            for vowel in VOWELS:
                scheme = sys.argv[2] if len(sys.argv) > 2 else "vowel"
                img = None
                for attempt in range(6):
                    src = await bridge.hold(scheme_params(scheme, vowel), seconds=0.7)
                    cand = cv2.imdecode(np.fromfile(str(src), dtype=np.uint8), cv2.IMREAD_COLOR)
                    if cand is None:
                        continue
                    # A timed-out NDI grab returns the previous frame unchanged.
                    if prev is None or float(np.mean(cv2.absdiff(cand, last_full))) > 0.05:
                        img = cand
                        last_full = cand
                        break
                    print(vowel, "stale frame, retry", attempt)
                if img is None:
                    continue
                if prev is not None:
                    box = track(prev, img, box)
                crop_img = crop(img, box)
                prev = crop_img
                up = cv2.resize(crop_img, (420, 300), interpolation=cv2.INTER_CUBIC)
                cv2.putText(up, vowel, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 80, 255), 3)
                crops[vowel] = up
                shutil.copyfile(src, out / f"{vowel}.jpg")
        finally:
            await bridge.call({"type": "inject.pause", "paused": False}, "inject.pause.result")
    if crops:
        sheet = np.hstack([crops[v] for v in VOWELS if v in crops])
        cv2.imencode(".jpg", sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tofile(str(out / "montage.jpg"))
        print("box:", box, "saved:", (out / "montage.jpg").exists())


asyncio.run(main())
