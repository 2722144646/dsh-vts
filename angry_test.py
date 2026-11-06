import asyncio
import shutil
from pathlib import Path

import cv2

from dsh_vts_avatar.config import load_config
from dsh_vts_avatar.ndi.capture import NDICapture
from dsh_vts_avatar.vts.client import VTSClient

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
    "FaceAngry": 0.0, "EyeSquintL": 0.0, "EyeSquintR": 0.0,
}

VARIANTS = {
    "A": {"MouthSmile": 0.0, "EyeOpenLeft": 0.25, "EyeOpenRight": 0.25, "FaceAngry": 1.0,
          "BrowInnerUp": 0.35, "Brows": 0.0, "BrowLeftY": 0.85, "BrowRightY": 0.85,
          "CheekPuff": 1.0, "FaceAngleY": 3.0, "FaceAngleZ": -6.0},
    "B": {"MouthSmile": 0.0, "EyeOpenLeft": 0.10, "EyeOpenRight": 0.10, "FaceAngry": 1.0,
          "BrowInnerUp": 0.35, "Brows": 1.0, "BrowLeftY": 0.0, "BrowRightY": 0.0,
          "CheekPuff": 1.0, "MouthShrug": 1.0, "MouthPressLipOpen": 0.8,
          "FaceAngleY": 4.0, "FaceAngleZ": -8.0},
    "C": {"MouthSmile": 0.0, "EyeOpenLeft": 0.10, "EyeOpenRight": 0.10, "FaceAngry": 1.0,
          "BrowInnerUp": 0.35, "Brows": 0.0, "BrowLeftY": 1.0, "BrowRightY": 1.0,
          "CheekPuff": 1.0, "MouthShrug": 1.0, "MouthPressLipOpen": 0.8,
          "FaceAngleY": 4.0, "FaceAngleZ": -8.0},
    "D": {"MouthSmile": 0.0, "EyeOpenLeft": 0.75, "EyeOpenRight": 0.75, "EyeSquintL": 0.85,
          "EyeSquintR": 0.85, "FaceAngry": 1.0, "BrowInnerUp": 0.35, "Brows": 0.0,
          "BrowLeftY": 0.0, "BrowRightY": 0.0, "CheekPuff": 1.0,
          "FaceAngleY": 3.0, "FaceAngleZ": -6.0},
    "E": {"MouthSmile": 0.0, "EyeOpenLeft": 0.55, "EyeOpenRight": 0.55, "EyeSquintL": 1.0,
          "EyeSquintR": 1.0, "FaceAngry": 1.0, "BrowInnerUp": 0.35, "Brows": 0.8,
          "BrowLeftY": 0.0, "BrowRightY": 0.0, "CheekPuff": 1.0,
          "FaceAngleY": 4.0, "FaceAngleZ": -6.0},
    "F": {"MouthSmile": 0.0, "EyeOpenLeft": 0.55, "EyeOpenRight": 0.55, "EyeLeftY": 0.55,
          "EyeRightY": 0.55, "FaceAngry": 1.0, "BrowInnerUp": 0.35, "Brows": 0.0,
          "BrowLeftY": 0.0, "BrowRightY": 0.0, "CheekPuff": 1.0,
          "FaceAngleY": 4.0, "FaceAngleZ": -4.0},
}


async def main():
    out = Path("dsh-vts-avatar/data/expression_shots")
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config("dsh-vts-avatar/config.toml")
    client = VTSClient(cfg.vts)
    await client.connect()
    ndi = NDICapture(source_hint=cfg.ndi.source, snapshot_dir=out)
    if not ndi.open(wait_ms=4000):
        raise RuntimeError("no ndi")
    paths = {}
    try:
        for name, values in VARIANTS.items():
            await client.inject(BASE)
            await asyncio.sleep(0.35)
            payload = dict(BASE)
            payload.update(values)
            await client.inject(payload)
            await asyncio.sleep(0.8)
            m = await asyncio.to_thread(ndi.snapshot)
            if m and m.snapshot:
                p = out / f"angry_{name}.jpg"
                shutil.copyfile(m.snapshot, p)
                paths[name] = str(p)
                print(name, p)
    finally:
        ndi.close()
        await client.close()
    tiles = []
    for name, path in paths.items():
        im = cv2.imread(path)
        crop = cv2.resize(im[40:760, 500:1300], (360, 324))
        cv2.putText(crop, name, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        tiles.append(crop)
    cv2.imwrite(str(out / "angry_options.jpg"), cv2.hconcat(tiles))
    print("saved", out / "angry_options.jpg")


asyncio.run(main())
