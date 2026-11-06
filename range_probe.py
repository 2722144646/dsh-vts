"""Probe min/max boundaries for the currently loaded VTS model and save them
into that model's profile. NDI is optional; this only measures parameter
readback and Live2D response.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from dsh_vts_avatar.config import load_config
from dsh_vts_avatar.vts.client import VTSClient
from dsh_vts_avatar.vts.profile import ModelProfile, load_for_model, profile_path

CORE = [
    "MouthOpen", "MouthSmile", "JawOpen", "MouthFunnel", "MouthPucker",
    "MouthShrug", "MouthPressLipOpen", "MouthX", "CheekPuff", "TongueOut",
    "VoiceA", "VoiceI", "VoiceU", "VoiceE", "VoiceO", "VoiceSilence",
    "FaceAngleX", "FaceAngleY", "FaceAngleZ",
    "EyeOpenLeft", "EyeOpenRight", "EyeLeftX", "EyeLeftY", "EyeRightX", "EyeRightY",
    "Brows", "BrowInnerUp", "BrowLeftY", "BrowRightY", "FaceAngry",
]


def diff(before: dict[str, float], after: dict[str, float], limit: int = 6) -> list[str]:
    out = []
    for name, value in after.items():
        old = before.get(name)
        if old is None:
            continue
        delta = value - old
        if abs(delta) >= 0.01:
            out.append((name, delta))
    out.sort(key=lambda item: abs(item[1]), reverse=True)
    return [f"{name} {delta:+.3f}" for name, delta in out[:limit]]


async def main() -> None:
    cfg = load_config("dsh-vts-avatar/config.toml")
    client = VTSClient(cfg.vts)
    await client.connect()
    try:
        model = await client.current_model()
        model_id = str(model.get("modelID", ""))
        profile = load_for_model(cfg.profile_dir, model_id)
        if not profile.input_params:
            raise RuntimeError("run `probe` first: profile has no input parameters")
        print("MODEL", model.get("modelName"), model_id)
        print("INPUTS", len(profile.input_params), "LIVE2D", len(profile.live2d_params))

        for key, default in (
            ("head_gain", cfg.avatar.head_gain),
            ("mouth_gain", cfg.avatar.mouth_gain),
            ("blink_per_min", cfg.avatar.blink_per_min),
        ):
            profile.tuning.setdefault(key, float(default))

        base = {
            name: info.default
            for name, info in profile.input_params.items()
            if name in CORE
        }
        base["VoiceSilence"] = 1.0

        async def snap() -> dict[str, float]:
            raw = await client.list_live2d_params()
            return {
                str(item.get("name")): float(item.get("value", 0.0))
                for item in raw.get("parameters", [])
                if item.get("name")
            }

        await client.inject(base)
        await asyncio.sleep(0.35)
        baseline = await snap()
        for name in CORE:
            info = profile.input_params.get(name)
            if info is None:
                continue
            entry: dict[str, Any] = {
                "min": info.minimum,
                "max": info.maximum,
                "default": info.default,
                "added_by": info.added_by,
            }
            for label, value in (("test_min", info.minimum), ("test_max", info.maximum)):
                payload = dict(base)
                payload[name] = value
                try:
                    await client.inject(payload)
                    await asyncio.sleep(0.30)
                    readback = await client.parameter_value(name)
                    after = await snap()
                    entry[label] = {
                        "target": value,
                        "readback": readback.get("value"),
                        "moved": diff(baseline, after),
                    }
                except Exception as exc:
                    entry[label] = {"target": value, "error": str(exc)}
                finally:
                    await client.inject(base)
                    await asyncio.sleep(0.08)
                    baseline = await snap()
            profile.range_tests[name] = entry
            print(name, json.dumps(entry, ensure_ascii=False))

        target = profile_path(cfg.profile_dir, model_id)
        profile.save(target)
        print("SAVED", target)
        print("TUNING", profile.tuning)
    finally:
        await client.close()


asyncio.run(main())
