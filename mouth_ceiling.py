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
    "Brows": 0.10, "BrowInnerUp": 0.0, "BrowLeftY": 0.0, "BrowRightY": 0.0,
    "FaceAngry": 0.0,
}

STATES = {
    "neutral": {},
    "m1": {"MouthOpen": 1.0},
    "m1_j1": {"MouthOpen": 1.0, "JawOpen": 1.0},
    "m1_j1_f1": {"MouthOpen": 1.0, "JawOpen": 1.0, "MouthFunnel": 1.0},
    "m1_j1_p1": {"MouthOpen": 1.0, "JawOpen": 1.0, "MouthPucker": 1.0},
    "m1_j1_pm1": {"MouthOpen": 1.0, "JawOpen": 1.0, "MouthPucker": -1.0},
    "m1_j1_a1": {"MouthOpen": 1.0, "JawOpen": 1.0, "VoiceA": 1.0, "VoiceSilence": 0.0},
    "m1_j1_s1": {"MouthOpen": 1.0, "JawOpen": 1.0, "MouthSmile": 1.0},
    "m12": {"MouthOpen": 1.2},
    "m15": {"MouthOpen": 1.5},
    "m2": {"MouthOpen": 2.0},
}


async def main():
    out = Path("dsh-vts-avatar/data/vb_audit/ceiling")
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config("dsh-vts-avatar/config.toml")
    client = VTSClient(cfg.vts)
    await client.connect()
    ndi = NDICapture(source_hint=cfg.ndi.source, snapshot_dir=out)
    if not ndi.open(wait_ms=4000):
        raise RuntimeError("no ndi")
    rows = []
    try:
        for name, extra in STATES.items():
            payload = dict(BASE)
            payload.update(extra)
            error = None
            try:
                await client.inject(payload)
            except Exception as exc:
                error = str(exc)
            await asyncio.sleep(0.75)
            live = await client.list_live2d_params()
            vals = {p["name"]: float(p["value"]) for p in live.get("parameters", [])}
            m = await asyncio.to_thread(ndi.snapshot)
            path = None
            if m and m.snapshot:
                path = out / f"{name}.jpg"
                shutil.copyfile(m.snapshot, path)
            rows.append((name, vals.get("ParamMouthOpenY"), vals.get("ParamMouthForm"), vals.get("ParamMouthX"), error, path))
            print(name, "openY", vals.get("ParamMouthOpenY"), "form", vals.get("ParamMouthForm"), "err", error)
    finally:
        ndi.close()
        await client.close()

    tiles = []
    for name, open_y, form, mx, error, path in rows:
        if path is None:
            continue
        im = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        crop = cv2.resize(im[430:650, 820:1120], (360, 264))
        label = f"{name} y={open_y:.2f}"
        cv2.putText(crop, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        tiles.append(crop)
    rows_img = []
    for i in range(0, len(tiles), 3):
        row = tiles[i:i + 3]
        while len(row) < 3:
            row.append(np.zeros_like(tiles[0]))
        rows_img.append(np.hstack(row))
    ok, buf = cv2.imencode(".jpg", np.vstack(rows_img))
    (out / "ceiling.jpg").write_bytes(buf.tobytes())
    print("saved", out / "ceiling.jpg")


asyncio.run(main())
