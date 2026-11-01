"""Read-only expression recording.

The user performs an expression in VTube Studio while this recorder polls the
input and Live2D parameter lists. It never writes a parameter. The result is a
compact delta profile that can be stored on the model profile and used as a
custom expression hint.
"""

from __future__ import annotations

import asyncio
import math
import time
from datetime import datetime, timezone
from typing import Any

from ..vts.client import VTSClient


def _values(raw: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in raw.get("defaultParameters", []) + raw.get("customParameters", []):
        name = item.get("name")
        if name:
            out[str(name)] = float(item.get("value", 0.0))
    return out


def _live_values(raw: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in raw.get("parameters", []):
        name = item.get("name")
        if name:
            out[str(name)] = float(item.get("value", 0.0))
    return out


async def _sample(client: VTSClient) -> tuple[dict[str, float], dict[str, float]]:
    inputs = await client.list_input_params()
    live = await client.list_live2d_params()
    return _values(inputs), _live_values(live)


async def record_expression(
    client: VTSClient,
    name: str,
    *,
    duration_s: float = 4.0,
    baseline_s: float = 0.8,
    hz: float = 20.0,
) -> dict[str, Any]:
    duration_s = max(1.0, min(15.0, float(duration_s)))
    hz = max(5.0, min(30.0, float(hz)))
    model = await client.current_model()
    period = 1.0 / hz

    # Let the caller's last injected frame expire before reading the user.
    await asyncio.sleep(0.45)
    baseline_inputs: list[dict[str, float]] = []
    baseline_live: list[dict[str, float]] = []
    for _ in range(max(1, math.ceil(baseline_s * hz))):
        i, l = await _sample(client)
        baseline_inputs.append(i)
        baseline_live.append(l)
        await asyncio.sleep(period)

    def average(rows: list[dict[str, float]]) -> dict[str, float]:
        keys = set().union(*(row.keys() for row in rows)) if rows else set()
        return {
            key: sum(row.get(key, 0.0) for row in rows) / max(1, len(rows))
            for key in keys
        }

    base_in = average(baseline_inputs)
    base_live = average(baseline_live)
    samples_inputs: list[dict[str, float]] = []
    samples_live: list[dict[str, float]] = []
    started = time.perf_counter()
    while time.perf_counter() - started < duration_s:
        i, l = await _sample(client)
        samples_inputs.append(i)
        samples_live.append(l)
        await asyncio.sleep(period)

    def summarize(
        base: dict[str, float],
        samples: list[dict[str, float]],
        *,
        threshold: float = 0.015,
    ) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        keys = set(base) | set().union(*(row.keys() for row in samples)) if samples else set(base)
        for key in keys:
            values = [row.get(key, 0.0) for row in samples]
            if not values:
                continue
            peak = max(values, key=lambda value: abs(value - base.get(key, 0.0)))
            delta = peak - base.get(key, 0.0)
            if abs(delta) < threshold:
                continue
            out[key] = {
                "base": round(base.get(key, 0.0), 5),
                "peak": round(peak, 5),
                "delta": round(delta, 5),
                "min": round(min(values), 5),
                "max": round(max(values), 5),
            }
        return dict(sorted(out.items(), key=lambda kv: abs(kv[1]["delta"]), reverse=True))

    inputs = summarize(base_in, samples_inputs)
    live = summarize(base_live, samples_live, threshold=0.01)
    return {
        "name": name,
        "created": datetime.now(timezone.utc).isoformat(),
        "model_id": str(model.get("modelID", "")),
        "model_name": str(model.get("modelName", "")),
        "duration_s": round(duration_s, 2),
        "samples": len(samples_inputs),
        "baseline_inputs": {k: round(v, 5) for k, v in base_in.items()},
        "input_deltas": inputs,
        "live2d_deltas": live,
        "top_inputs": list(inputs)[:12],
        "top_live2d": list(live)[:12],
    }
