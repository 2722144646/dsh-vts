import asyncio
import time

import cv2

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
        for name, values in [
            ("full_closed", {"MouthOpen": 0.0, "VoiceSilence": 0.0, "MouthSmile": 0.0}),
            ("full_open", {"MouthOpen": 1.0, "VoiceSilence": 0.0, "MouthSmile": 0.0}),
            ("full_jaw", {"MouthOpen": 0.0, "VoiceSilence": 0.0, "JawOpen": 1.0}),
        ]:
            await client.inject(values)
            time.sleep(1.5)
            frame, _ = ndi.capture(timeout_ms=2000)
            if frame is not None:
                cv2.imwrite(str(cfg.snapshot_dir / f"{name}.jpg"), frame)
                print("saved", name)
    finally:
        ndi.close()
        await client.close()


asyncio.run(main())
