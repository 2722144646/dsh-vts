import asyncio
import shutil
from pathlib import Path

import cv2
import numpy as np

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
    "FaceAngry": 0.0,
}

STATES = {
    "neutral": {},
    "pbm_press_1": {"MouthOpen": 0.02, "MouthPressLipOpen": 1.0},
    "pbm_press_04": {"MouthOpen": 0.02, "MouthPressLipOpen": 0.4},
    "fv_press_035": {"MouthOpen": 0.10, "JawOpen": 0.10, "MouthPressLipOpen": 0.35},
    "shrug_08": {"MouthOpen": 0.13, "MouthShrug": 0.8},
    "u": {"VoiceU": 1.0, "VoiceSilence": 0.0, "MouthOpen": 0.28, "JawOpen": 0.25, "MouthFunnel": 0.22, "MouthPucker": 0.52},
    "o": {"VoiceO": 1.0, "VoiceSilence": 0.0, "MouthOpen": 0.55, "JawOpen": 0.45, "MouthFunnel": 0.45, "MouthPucker": 0.32},
}


async def main():
    out = Path("dsh-vts-avatar/data/vb_audit")
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config("dsh-vts-avatar/config.toml")
    client = VTSClient(cfg.vts)
    await client.connect()
    ndi = NDICapture(source_hint=cfg.ndi.source, snapshot_dir=out)
    if not ndi.open(wait_ms=4000):
        raise RuntimeError("no ndi")
    paths = {}
    try:
        for name, extra in STATES.items():
            payload = dict(BASE)
            payload.update(extra)
            await client.inject(payload)
            await asyncio.sleep(0.8)
            m = await asyncio.to_thread(ndi.snapshot)
            if m and m.snapshot:
                p = out / f"{name}.jpg"
                shutil.copyfile(m.snapshot, p)
                paths[name] = str(p)
                print(name, p)
    finally:
        ndi.close()
        await client.close()
    tiles = []
    for name, path in paths.items():
        im = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        im = cv2.resize(im[40:760, 500:1300], (360, 324))
        cv2.putText(im, name, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        tiles.append(im)
    rows = []
    for i in range(0, len(tiles), 3):
        row = tiles[i:i + 3]
        while len(row) < 3:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)
    ok, buf = cv2.imencode(".jpg", sheet)
    (out / "vb_audit.jpg").write_bytes(buf.tobytes())
    print("saved", out / "vb_audit.jpg")


asyncio.run(main())
