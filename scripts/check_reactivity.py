"""Decisive check: does the model react to injection at all?

Compares neutral vs face-turn vs mouth-open. A face-turn MUST change the image;
if the mouth does not while the face does, the mouth input is not wired.
"""

import asyncio
import pathlib
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lipread_test import Bridge  # noqa: E402

OUT = pathlib.Path(r"E:\dsh-vts-avatar\蒂亚圣诞树\reactivity")

CASES = {
    "neutral": {"MouthOpen": 0.0, "MouthSmile": 0.0, "FaceAngleX": 0.0, "FaceAngleY": 0.0},
    "face30": {"MouthOpen": 0.0, "MouthSmile": 0.0, "FaceAngleX": 30.0, "FaceAngleY": 0.0},
    "open1": {"MouthOpen": 1.0, "MouthSmile": 0.0, "FaceAngleX": 0.0, "FaceAngleY": 0.0},
    "smile1": {"MouthOpen": 0.0, "MouthSmile": 1.0, "FaceAngleX": 0.0, "FaceAngleY": 0.0},
}


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    shots = {}
    async with Bridge("ws://127.0.0.1:8765") as bridge:
        await bridge.call({"type": "inject.pause", "paused": True}, "inject.pause.result")
        try:
            for name, params in CASES.items():
                src = await bridge.hold(params, seconds=1.0)
                dst = OUT / f"{name}.jpg"
                if src and src.exists():
                    shutil.copyfile(src, dst)
                    shots[name] = cv2.imdecode(np.fromfile(str(dst), dtype=np.uint8), cv2.IMREAD_COLOR)
        finally:
            await bridge.call({"type": "inject.pause", "paused": False}, "inject.pause.result")

    base = cv2.cvtColor(shots["neutral"], cv2.COLOR_BGR2GRAY).astype(np.int16)
    for name, img in shots.items():
        if name == "neutral":
            continue
        diff = np.abs(base - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int16))
        print(f"{name:8} mean={diff.mean():.3f} max={int(diff.max())} px>25={int((diff > 25).sum())}")


asyncio.run(main())
