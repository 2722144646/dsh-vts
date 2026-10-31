import asyncio

from dsh_vts_avatar.config import load_config
from dsh_vts_avatar.vts.client import VTSClient


async def main():
    cfg = load_config("dsh-vts-avatar/config.toml")
    client = VTSClient(cfg.vts)
    await client.connect()
    try:
        base_raw = await client.list_live2d_params()
        base = {p["name"]: float(p["value"]) for p in base_raw["parameters"]}
        for name, value in [
            ("MouthSmile", 1.0),
            ("FaceAngry", 1.0),
            ("Brows", 1.0),
            ("FaceAngleY", 20.0),
            ("FaceAngleZ", 20.0),
            ("CheekPuff", 1.0),
        ]:
            try:
                await client.inject({name: value})
                await asyncio.sleep(0.6)
                raw = await client.list_live2d_params()
                cur = {p["name"]: float(p["value"]) for p in raw["parameters"]}
                diffs = sorted(
                    ((n, cur.get(n, 0) - base.get(n, 0)) for n in cur),
                    key=lambda item: abs(item[1]),
                    reverse=True,
                )
                print(name, [(n, round(d, 3)) for n, d in diffs[:12]])
                # reset all watched params to defaults before next probe
                await client.inject({name: 0.0})
                await asyncio.sleep(0.2)
                base = cur
            except Exception as exc:
                print(name, "ERR", exc)
    finally:
        await client.close()


asyncio.run(main())
