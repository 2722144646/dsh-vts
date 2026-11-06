"""Render three direct parameter-mapping variants for 蒂亚.

This bypasses the generic expression template and writes the model's input
parameters directly, then captures NDI frames for comparison.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import cv2

from dsh_vts_avatar.config import load_config
from dsh_vts_avatar.ndi.capture import NDICapture
from dsh_vts_avatar.vts.client import VTSClient

OUT = Path("dsh-vts-avatar/data/expression_shots")
EMOTIONS = ["happy", "sad", "angry", "surprised", "thinking", "shy"]

BASE = {
    "MouthOpen": 0.0, "MouthSmile": 0.10, "JawOpen": 0.0,
    "MouthFunnel": 0.0, "MouthPucker": 0.0, "MouthShrug": 0.0,
    "MouthPressLipOpen": 0.0, "MouthX": 0.0, "CheekPuff": 0.0,
    "TongueOut": 0.0, "VoiceA": 0.0, "VoiceI": 0.0, "VoiceU": 0.0,
    "VoiceE": 0.0, "VoiceO": 0.0, "VoiceSilence": 1.0,
    "FaceAngleX": 0.0, "FaceAngleY": 0.0, "FaceAngleZ": 0.0,
    "EyeOpenLeft": 1.0, "EyeOpenRight": 1.0,
    "EyeLeftX": 0.0, "EyeLeftY": 0.0, "EyeRightX": 0.0, "EyeRightY": 0.0,
    "Brows": 0.10, "BrowInnerUp": 0.0, "BrowLeftY": 0.0, "BrowRightY": 0.0,
    "FaceAngry": 0.0,
}

# A = restrained, B = clear, C = stage. Values are direct VTS input values.
VARIANTS = {
    "A": {
        "happy": {"MouthSmile": 0.85, "EyeOpenLeft": 0.92, "EyeOpenRight": 0.92, "Brows": 0.30, "FaceAngleZ": 1.0},
        "sad": {"MouthSmile": 0.0, "EyeOpenLeft": 0.60, "EyeOpenRight": 0.60, "BrowInnerUp": 0.90, "BrowLeftY": 0.15, "BrowRightY": 0.15, "FaceAngleY": -3.0, "FaceAngleZ": -1.5, "EyeLeftY": -0.35, "EyeRightY": -0.35},
        "angry": {"MouthSmile": 0.0, "EyeOpenLeft": 0.60, "EyeOpenRight": 0.60, "FaceAngry": 0.75, "BrowInnerUp": 0.25, "Brows": 0.05, "BrowLeftY": 0.40, "BrowRightY": 0.40, "FaceAngleY": 1.0, "FaceAngleZ": -2.0},
        "surprised": {"MouthSmile": 0.15, "EyeOpenLeft": 0.98, "EyeOpenRight": 0.98, "Brows": 0.90, "BrowInnerUp": 0.10, "MouthOpen": 0.35, "FaceAngleY": -2.0},
        "thinking": {"MouthSmile": 0.15, "EyeOpenLeft": 0.88, "EyeOpenRight": 0.88, "Brows": 0.45, "BrowLeftY": 0.70, "BrowRightY": 0.20, "EyeLeftX": 0.40, "EyeRightX": 0.40, "EyeLeftY": 0.10, "EyeRightY": 0.10, "FaceAngleY": 2.0, "FaceAngleZ": 3.0},
        "shy": {"MouthSmile": 0.50, "EyeOpenLeft": 0.70, "EyeOpenRight": 0.70, "BrowInnerUp": 0.30, "BrowLeftY": 0.24, "BrowRightY": 0.24, "EyeLeftX": -0.25, "EyeRightX": -0.25, "EyeLeftY": 0.30, "EyeRightY": 0.30, "FaceAngleY": -1.0, "FaceAngleZ": 2.0},
    },
    "B": {
        "happy": {"MouthSmile": 0.98, "EyeOpenLeft": 0.85, "EyeOpenRight": 0.85, "Brows": 0.45, "FaceAngleZ": 2.0},
        "sad": {"MouthSmile": 0.0, "EyeOpenLeft": 0.40, "EyeOpenRight": 0.40, "BrowInnerUp": 1.0, "BrowLeftY": 0.20, "BrowRightY": 0.20, "FaceAngleY": -6.0, "FaceAngleZ": -3.0, "EyeLeftY": -0.55, "EyeRightY": -0.55},
        "angry": {"MouthSmile": 0.0, "EyeOpenLeft": 0.40, "EyeOpenRight": 0.40, "FaceAngry": 0.95, "BrowInnerUp": 0.30, "Brows": 0.02, "BrowLeftY": 0.65, "BrowRightY": 0.65, "FaceAngleY": 2.0, "FaceAngleZ": -4.0},
        "surprised": {"MouthSmile": 0.18, "EyeOpenLeft": 1.0, "EyeOpenRight": 1.0, "Brows": 1.0, "BrowInnerUp": 0.10, "MouthOpen": 0.60, "FaceAngleY": -4.0},
        "thinking": {"MouthSmile": 0.15, "EyeOpenLeft": 0.85, "EyeOpenRight": 0.85, "Brows": 0.45, "BrowLeftY": 0.90, "BrowRightY": 0.10, "EyeLeftX": 0.70, "EyeRightX": 0.70, "EyeLeftY": 0.20, "EyeRightY": 0.20, "FaceAngleY": 3.0, "FaceAngleZ": 6.0},
        "shy": {"MouthSmile": 0.65, "EyeOpenLeft": 0.55, "EyeOpenRight": 0.55, "BrowInnerUp": 0.35, "BrowLeftY": 0.25, "BrowRightY": 0.25, "EyeLeftX": -0.45, "EyeRightX": -0.45, "EyeLeftY": 0.45, "EyeRightY": 0.45, "FaceAngleY": -2.0, "FaceAngleZ": 4.0},
    },
    "C": {
        "happy": {"MouthSmile": 1.0, "EyeOpenLeft": 0.80, "EyeOpenRight": 0.80, "Brows": 0.55, "FaceAngleZ": 2.5},
        "sad": {"MouthSmile": 0.0, "EyeOpenLeft": 0.25, "EyeOpenRight": 0.25, "BrowInnerUp": 1.0, "BrowLeftY": 0.25, "BrowRightY": 0.25, "FaceAngleY": -9.0, "FaceAngleZ": -4.0, "EyeLeftY": -0.65, "EyeRightY": -0.65},
        "angry": {"MouthSmile": 0.0, "EyeOpenLeft": 0.25, "EyeOpenRight": 0.25, "FaceAngry": 1.0, "BrowInnerUp": 0.35, "Brows": 0.0, "BrowLeftY": 0.85, "BrowRightY": 0.85, "FaceAngleY": 3.0, "FaceAngleZ": -6.0},
        "surprised": {"MouthSmile": 0.20, "EyeOpenLeft": 1.0, "EyeOpenRight": 1.0, "Brows": 1.0, "BrowInnerUp": 0.10, "MouthOpen": 0.80, "FaceAngleY": -6.0},
        "thinking": {"MouthSmile": 0.10, "EyeOpenLeft": 0.80, "EyeOpenRight": 0.80, "Brows": 0.45, "BrowLeftY": 1.0, "BrowRightY": 0.0, "EyeLeftX": 0.90, "EyeRightX": 0.90, "EyeLeftY": 0.30, "EyeRightY": 0.30, "FaceAngleY": 4.0, "FaceAngleZ": 9.0},
        "shy": {"MouthSmile": 0.75, "EyeOpenLeft": 0.40, "EyeOpenRight": 0.40, "BrowInnerUp": 0.40, "BrowLeftY": 0.24, "BrowRightY": 0.24, "EyeLeftX": -0.60, "EyeRightX": -0.60, "EyeLeftY": 0.60, "EyeRightY": 0.60, "FaceAngleY": -3.0, "FaceAngleZ": 6.0},
    },
}


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config("dsh-vts-avatar/config.toml")
    client = VTSClient(cfg.vts)
    await client.connect()
    ndi = NDICapture(source_hint=cfg.ndi.source, snapshot_dir=OUT)
    if not ndi.open(wait_ms=4000):
        raise RuntimeError("NDI source not found")
    cells = {}
    try:
        for variant, emotions in VARIANTS.items():
            for emotion in EMOTIONS:
                await client.inject(BASE)
                await asyncio.sleep(0.30)
                payload = dict(BASE)
                payload.update(emotions[emotion])
                await client.inject(payload)
                await asyncio.sleep(0.75)
                metrics = await asyncio.to_thread(ndi.snapshot)
                if metrics is None or not metrics.snapshot:
                    raise RuntimeError(f"snapshot failed {variant}/{emotion}")
                path = OUT / f"map_{variant}_{emotion}.jpg"
                shutil.copyfile(metrics.snapshot, path)
                cells[(variant, emotion)] = str(path)
                print(variant, emotion, path)
    finally:
        ndi.close()
        await client.close()

    tiles = []
    for emotion in EMOTIONS:
        row = []
        for variant in ("A", "B", "C"):
            im = cv2.imread(cells[(variant, emotion)])
            if im is None:
                continue
            crop = im[40:760, 500:1300]
            crop = cv2.resize(crop, (300, 270))
            cv2.putText(crop, f"{variant} {emotion}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 255), 2)
            row.append(crop)
        tiles.append(cv2.hconcat(row))
    cv2.imwrite(str(OUT / "mapping_options.jpg"), cv2.vconcat(tiles))
    (OUT / "mapping_options.json").write_text(
        json.dumps(VARIANTS, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("saved", OUT / "mapping_options.jpg")


asyncio.run(main())
