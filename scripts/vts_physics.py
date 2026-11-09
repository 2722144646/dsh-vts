"""Turn the model's physics off (or back to normal) through the VTS API.

    python scripts/vts_physics.py --strength 0     # freeze for a measurement
    python scripts/vts_physics.py --strength 1     # restore
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dsh_vts_avatar.config import load_config  # noqa: E402
from dsh_vts_avatar.vts.client import VTSClient  # noqa: E402


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strength", type=float, default=0.0)
    parser.add_argument("--config", default="config.toml")
    args = parser.parse_args()

    config = load_config(args.config)
    client = VTSClient(config.vts)
    await client.connect()
    try:
        try:
            reply = await client.request(
                "SetCurrentModelPhysicsRequest", {"strengthOverride": args.strength}
            )
            print("physics:", json.dumps(reply, ensure_ascii=False)[:300])
        except Exception as exc:  # noqa: BLE001 - report and continue
            print("physics request rejected:", type(exc).__name__, exc)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
