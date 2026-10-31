import asyncio
import time

from dsh_vts_avatar.config import load_config
from dsh_vts_avatar.ndi.capture import NDICapture
from dsh_vts_avatar.vts.client import VTSClient


async def main():
    cfg = load_config("dsh-vts-avatar/config.toml")
    client = VTSClient(cfg.vts)
    await client.connect()
    ndi = NDICapture(source_hint=cfg.ndi.source, snapshot_dir=cfg.ndi.snapshot_dir)
    if not ndi.open(wait_ms=3000):
        print("no ndi")
        return
    try:
        await client.inject({"VoiceA": 0.0, "VoiceI": 0.0, "VoiceU": 0.0, "VoiceE": 0.0, "VoiceO": 0.0, "MouthOpen": 0.0})
        time.sleep(0.5)
        base, _ = ndi.capture(timeout_ms=2000)
        if base is not None:
            import cv2
            cv2.imwrite(str(cfg.snapshot_dir / "test_base.jpg"), base)

        await client.inject({"VoiceA": 0.95, "VoiceI": 0.0, "VoiceU": 0.0, "VoiceE": 0.0, "VoiceO": 0.0, "MouthOpen": 0.0})
        time.sleep(0.5)
        voice, _ = ndi.capture(timeout_ms=2000)
        if voice is not None:
            import cv2
            cv2.imwrite(str(cfg.snapshot_dir / "test_voice_a.jpg"), voice)

        await client.inject({"VoiceA": 0.0, "VoiceI": 0.0, "VoiceU": 0.0, "VoiceE": 0.0, "VoiceO": 0.0, "MouthOpen": 0.95})
        time.sleep(0.5)
        open_frame, _ = ndi.capture(timeout_ms=2000)
        if open_frame is not None:
            import cv2
            cv2.imwrite(str(cfg.snapshot_dir / "test_open.jpg"), open_frame)

        print("saved", cfg.snapshot_dir)
    finally:
        ndi.close()
        await client.close()


asyncio.run(main())
