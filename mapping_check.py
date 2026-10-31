import asyncio
import time

from dsh_vts_avatar.config import load_config
from dsh_vts_avatar.vts.client import VTSClient

WATCH = ["ParamAngleX", "ParamAngleY", "ParamAngleZ", "ParamMouthOpenY", "ParamMouthForm", "ParamA", "ParamI", "ParamU", "ParamE", "ParamO"]


async def main():
    cfg = load_config("dsh-vts-avatar/config.toml")
    client = VTSClient(cfg.vts)
    await client.connect()

    async def show(label):
        raw = await client.list_live2d_params()
        vals = {p["name"]: p["value"] for p in raw["parameters"]}
        print(label, {k: round(float(vals.get(k, 0)), 3) for k in WATCH})

    try:
        await show("base")
        await client.inject({"FaceAngleY": 30.0, "MouthOpen": 1.0, "VoiceA": 1.0, "VoiceSilence": 0.0})
        await asyncio.sleep(0.5)
        await show("max")
        await asyncio.sleep(1.2)
        await show("after_1.2s")
        await client.inject({"FaceAngleY": 0.0, "MouthOpen": 0.0, "VoiceA": 0.0, "VoiceSilence": 1.0})
    finally:
        await client.close()


asyncio.run(main())
