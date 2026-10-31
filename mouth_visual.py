import asyncio
import time

import cv2

from dsh_vts_avatar.config import load_config
from dsh_vts_avatar.ndi.capture import NDICapture
from dsh_vts_avatar.vts.client import VTSClient

ALL = {
    "MouthOpen": 0.0,
    "MouthSmile": 0.0,
    "VoiceA": 0.0,
    "VoiceI": 0.0,
    "VoiceU": 0.0,
    "VoiceE": 0.0,
    "VoiceO": 0.0,
    "VoiceSilence": 1.0,
    "JawOpen": 0.0,
    "MouthFunnel": 0.0,
    "MouthPucker": 0.0,
    "MouthShrug": 0.0,
}


async def state(client, ndi, cfg, name, values):
    payload = dict(ALL)
    payload.update(values)
    await client.inject(payload)
    time.sleep(0.65)
    frame, _ = ndi.capture(timeout_ms=2000)
    if frame is not None:
        crop = frame[260:560, 650:1150]
        crop = cv2.resize(crop, None, fx=2, fy=2)
        cv2.imwrite(str(cfg.snapshot_dir / f"{name}.jpg"), crop)
        print("saved", name)


async def main():
    cfg = load_config("dsh-vts-avatar/config.toml")
    client = VTSClient(cfg.vts)
    await client.connect()
    ndi = NDICapture(source_hint=cfg.ndi.source, snapshot_dir=cfg.ndi.snapshot_dir)
    if not ndi.open(wait_ms=3000):
        print("no ndi")
        return
    try:
        await state(client, ndi, cfg, "m0_base", {})
        await state(client, ndi, cfg, "m1_open", {"MouthOpen": 1.0, "VoiceSilence": 0.0})
        await state(client, ndi, cfg, "m2_jaw", {"JawOpen": 1.0, "VoiceSilence": 0.0})
        await state(client, ndi, cfg, "m3_funnel", {"MouthFunnel": 1.0, "VoiceSilence": 0.0})
        await state(client, ndi, cfg, "m4_pucker", {"MouthPucker": 1.0, "VoiceSilence": 0.0})
        await state(client, ndi, cfg, "m5_voiceA", {"VoiceA": 1.0, "VoiceSilence": 0.0})
    finally:
        ndi.close()
        await client.close()


asyncio.run(main())
