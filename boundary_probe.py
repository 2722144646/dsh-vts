import asyncio
import json
import time
from pathlib import Path

import cv2

from dsh_vts_avatar.config import load_config
from dsh_vts_avatar.ndi.capture import NDICapture
from dsh_vts_avatar.vts.client import VTSClient

BASE = {
    "MouthOpen": 0.0,
    "MouthSmile": 0.0,
    "JawOpen": 0.0,
    "MouthFunnel": 0.0,
    "MouthPucker": 0.0,
    "MouthShrug": 0.0,
    "MouthPressLipOpen": 0.0,
    "MouthX": 0.0,
    "CheekPuff": 0.0,
    "TongueOut": 0.0,
    "VoiceA": 0.0,
    "VoiceI": 0.0,
    "VoiceU": 0.0,
    "VoiceE": 0.0,
    "VoiceO": 0.0,
    "VoiceSilence": 1.0,
    "FaceAngleX": 0.0,
    "FaceAngleY": 0.0,
    "FaceAngleZ": 0.0,
    "EyeOpenLeft": 1.0,
    "EyeOpenRight": 1.0,
    "EyeLeftX": 0.0,
    "EyeLeftY": 0.0,
    "EyeRightX": 0.0,
    "EyeRightY": 0.0,
    "Brows": 0.0,
    "BrowInnerUp": 0.0,
    "FaceAngry": 0.0,
}

TESTS = [
    ("head_x_min", {"FaceAngleX": -30.0}),
    ("head_x_max", {"FaceAngleX": 30.0}),
    ("head_y_min", {"FaceAngleY": -30.0}),
    ("head_y_max", {"FaceAngleY": 30.0}),
    ("head_z_min", {"FaceAngleZ": -90.0}),
    ("head_z_max", {"FaceAngleZ": 90.0}),
    ("mouth_open_max", {"MouthOpen": 1.0}),
    ("jaw_open_max", {"JawOpen": 1.0}),
    ("funnel_max", {"MouthFunnel": 1.0}),
    ("pucker_max", {"MouthPucker": 1.0}),
    ("pucker_min", {"MouthPucker": -1.0}),
    ("press_max", {"MouthPressLipOpen": 1.3}),
    ("press_min", {"MouthPressLipOpen": -1.3}),
    ("shrug_max", {"MouthShrug": 1.0}),
    ("mouth_x_max", {"MouthX": 1.0}),
    ("mouth_x_min", {"MouthX": -1.0}),
    ("cheek_max", {"CheekPuff": 1.0}),
    ("tongue_max", {"TongueOut": 1.0}),
    ("smile_max", {"MouthSmile": 1.0}),
    ("angry_max", {"FaceAngry": 1.0}),
    ("brows_max", {"Brows": 1.0}),
    ("inner_brow_max", {"BrowInnerUp": 1.0}),
    ("eye_l_closed", {"EyeOpenLeft": 0.0}),
    ("eye_l_open", {"EyeOpenLeft": 1.0}),
    ("eye_lx_min", {"EyeLeftX": -1.0}),
    ("eye_lx_max", {"EyeLeftX": 1.0}),
    ("eye_ly_min", {"EyeLeftY": -1.0}),
    ("eye_ly_max", {"EyeLeftY": 1.0}),
    ("voice_a", {"VoiceA": 1.0, "VoiceSilence": 0.0}),
    ("voice_i", {"VoiceI": 1.0, "VoiceSilence": 0.0}),
    ("voice_u", {"VoiceU": 1.0, "VoiceSilence": 0.0}),
    ("voice_e", {"VoiceE": 1.0, "VoiceSilence": 0.0}),
    ("voice_o", {"VoiceO": 1.0, "VoiceSilence": 0.0}),
]


async def main():
    cfg = load_config("dsh-vts-avatar/config.toml")
    out_dir = cfg.snapshot_dir / "boundary"
    out_dir.mkdir(parents=True, exist_ok=True)
    client = VTSClient(cfg.vts)
    await client.connect()
    inputs = await client.list_input_params()
    params = (inputs.get("defaultParameters") or []) + (inputs.get("customParameters") or [])
    ranges = {
        p["name"]: {
            "min": p.get("min"),
            "max": p.get("max"),
            "default": p.get("defaultValue"),
            "addedBy": p.get("addedBy"),
        }
        for p in params
    }
    ndi = NDICapture(source_hint=cfg.ndi.source, snapshot_dir=out_dir)
    if not ndi.open(wait_ms=3000):
        print("no ndi")
        return
    report = {"model": await client.current_model(), "ranges": ranges, "tests": {}}
    try:
        await client.inject(BASE)
        time.sleep(0.6)
        base_frame, _ = ndi.capture(timeout_ms=2000)
        if base_frame is not None:
            cv2.imwrite(str(out_dir / "base.jpg"), base_frame)
        for name, values in TESTS:
            payload = dict(BASE)
            payload.update(values)
            try:
                await client.inject(payload)
                time.sleep(0.55)
                readback = await client.parameter_value(next(iter(values)))
                frame, _ = ndi.capture(timeout_ms=2000)
                path = out_dir / f"{name}.jpg"
                if frame is not None:
                    cv2.imwrite(str(path), frame)
                report["tests"][name] = {
                    "target": values,
                    "readback": readback.get("value"),
                    "image": str(path),
                    "status": "ok",
                }
                print(name, "target", values, "readback", readback.get("value"), flush=True)
            except Exception as exc:
                report["tests"][name] = {"target": values, "error": str(exc)}
                print(name, "ERR", exc, flush=True)
            await client.inject(BASE)
            time.sleep(0.15)
        (cfg.snapshot_dir / "boundary.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("saved", cfg.snapshot_dir / "boundary.json")
    finally:
        ndi.close()
        await client.close()


asyncio.run(main())
