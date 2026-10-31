import asyncio

from dsh_vts_avatar.config import load_config
from dsh_vts_avatar.vts.client import VTSClient


async def main():
    cfg = load_config("dsh-vts-avatar/config.toml")
    client = VTSClient(cfg.vts)
    await client.connect()
    try:
        raw = await client.list_live2d_params()
        params = {p["name"]: float(p["value"]) for p in raw["parameters"]}
        interesting = [
            n for n in params
            if any(k in n.lower() for k in ("mouth", "lip", "jaw", "silence", "pucker", "funnel"))
            or n in {"ParamA", "ParamI", "ParamU", "ParamE", "ParamO", "ParamMouthForm", "ParamMouthOpenY"}
        ]
        print("INTERESTING", interesting)
        for name in ["MouthOpen", "MouthSmile", "JawOpen", "MouthFunnel", "MouthPucker", "MouthShrug", "VoiceA", "VoiceI", "VoiceU", "VoiceE", "VoiceO", "VoiceSilence"]:
            for p in raw["parameters"]:
                params[p["name"]] = float(p["value"])
            before = dict(params)
            try:
                await client.inject({name: 0.9 if name not in ("VoiceSilence",) else 0.0})
                await asyncio.sleep(0.45)
                after_raw = await client.list_live2d_params()
                after = {p["name"]: float(p["value"]) for p in after_raw["parameters"]}
                diffs = sorted(
                    ((n, after.get(n, 0) - before.get(n, 0)) for n in interesting),
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )
                print("\n", name, [(n, round(d, 3)) for n, d in diffs[:12]])
            except Exception as exc:
                print(name, "ERR", exc)
    finally:
        await client.close()


asyncio.run(main())
