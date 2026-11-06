import asyncio
import shutil
from pathlib import Path

import cv2

from dsh_vts_avatar.config import load_config
from dsh_vts_avatar.ndi.capture import NDICapture
from dsh_vts_avatar.vts.client import VTSClient


async def main():
    out = Path("dsh-vts-avatar/data/expression_shots")
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config("dsh-vts-avatar/config.toml")
    client = VTSClient(cfg.vts)
    await client.connect()
    ndi = NDICapture(source_hint=cfg.ndi.source, snapshot_dir=out)
    if not ndi.open(wait_ms=4000):
        raise RuntimeError("no ndi")
    try:
        for name, active, params in (
            ("haqi", True, {}),
            ("haqi_angry", True, {"FaceAngry": 1.0, "CheekPuff": 1.0, "EyeOpenLeft": 0.6, "EyeOpenRight": 0.6}),
        ):
            r = await client.request(
                "ExpressionActivationRequest",
                {"file": "哈气.exp3.json", "fadeTime": 0.4, "active": active},
                timeout=5,
            )
            print(name, r.get("messageType"))
            if params:
                await client.inject(params)
            await asyncio.sleep(1.0)
            m = await asyncio.to_thread(ndi.snapshot)
            if m and m.snapshot:
                p = out / f"{name}.jpg"
                shutil.copyfile(m.snapshot, p)
                print(p)
            await client.request(
                "ExpressionActivationRequest",
                {"file": "哈气.exp3.json", "fadeTime": 0.2, "active": False},
                timeout=5,
            )
            await asyncio.sleep(0.4)
    finally:
        ndi.close()
        await client.close()


asyncio.run(main())
