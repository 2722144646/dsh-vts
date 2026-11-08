"""Generic boundary probe for whatever model VTS currently has loaded.

Reads the current contract (every input parameter plus its min/max/default),
pauses the bridge's own injection, drives each parameter to its minimum and
maximum, captures an NDI frame for each extreme, and ranks parameters by how
much they actually move the model. No parameter list is hardcoded here.

    python scripts/model_boundary_test.py --out E:\\dsh-vts-avatar\\冰糖\\boundary
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import shutil
from typing import Any

import websockets


def _jpeg_path(path: pathlib.Path) -> bool:
    try:
        return path.exists() and path.stat().st_size > 0
    except OSError:
        return False


class Bridge:
    def __init__(self, url: str) -> None:
        self.url = url
        self.ws = None
        self.seq = 0

    async def __aenter__(self):
        self.ws = await websockets.connect(self.url, max_size=None)
        await self.ws.send(json.dumps({"type": "hello", "client": "boundary", "protocol": 1}))
        return self

    async def __aexit__(self, *exc):
        await self.ws.close()

    async def call(self, msg: dict[str, Any], expect: str) -> dict[str, Any]:
        self.seq += 1
        msg = dict(msg)
        msg["request_id"] = f"b{self.seq}"
        await self.ws.send(json.dumps(msg))
        while True:
            reply = json.loads(await asyncio.wait_for(self.ws.recv(), 30))
            if reply.get("type") == expect and reply.get("request_id") == msg["request_id"]:
                return reply

    async def inject(self, params: dict[str, float]) -> None:
        await self.call({"type": "params.inject", "params": params}, "params.inject.result")

    async def snapshot(self) -> pathlib.Path | None:
        reply = await self.call({"type": "snapshot", "frames": 1}, "snapshot.result")
        path = (reply.get("metrics") or {}).get("path")
        return pathlib.Path(path) if path else None


def diff(gray_a, gray_b) -> float:
    import cv2
    import numpy as np

    if gray_a is None or gray_b is None or gray_a.shape != gray_b.shape:
        return -1.0
    return round(float(np.mean(cv2.absdiff(gray_a, gray_b))), 2)


def load_gray(path: pathlib.Path):
    import cv2
    import numpy as np

    if not _jpeg_path(path):
        return None
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


async def run(contract_path: pathlib.Path, out: pathlib.Path, url: str) -> dict[str, Any]:
    import cv2
    import numpy as np

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    model = contract.get("model", {})
    params = contract.get("input_parameters", [])
    out.mkdir(parents=True, exist_ok=True)
    frames = out / "frames"
    frames.mkdir(exist_ok=True)

    async with Bridge(url) as bridge:
        await bridge.call({"type": "inject.pause", "paused": True}, "inject.pause.result")
        try:
            base = await bridge.snapshot()
            results: list[dict[str, Any]] = []
            for item in params:
                name = str(item.get("name") or "")
                try:
                    low = float(item.get("min", 0.0))
                    high = float(item.get("max", 1.0))
                    default = float(item.get("defaultValue", 0.0))
                except (TypeError, ValueError):
                    continue
                if not name or high == low:
                    continue
                await bridge.inject({name: high})
                await asyncio.sleep(0.15)
                hi = await bridge.snapshot()
                await bridge.inject({name: low})
                await asyncio.sleep(0.15)
                lo = await bridge.snapshot()
                await bridge.inject({name: default})
                f_hi = frames / f"{name}__max.jpg"
                f_lo = frames / f"{name}__min.jpg"
                for src, dst in ((hi, f_hi), (lo, f_lo)):
                    if src and _jpeg_path(src):
                        shutil.copyfile(src, dst)
                d = diff(load_gray(f_lo), load_gray(f_hi))
                results.append(
                    {
                        "name": name,
                        "min": low,
                        "max": high,
                        "default": default,
                        "diff_min_max": d,
                        "moved": d > 0.5,
                    }
                )
            results.sort(key=lambda r: r["diff_min_max"], reverse=True)
            report = {
                "model": model,
                "contract": str(contract_path),
                "parameter_count": len(results),
                "moved_count": sum(1 for r in results if r["moved"]),
                "parameters": results,
                "baseline": str(base) if base else None,
            }
            (out / "results.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            lines = [
                f"# 模型边界测试 · {model.get('modelName')}",
                "",
                f"- modelID：{model.get('modelID')}",
                f"- 参数总数：{report['parameter_count']}",
                f"- 真正会动的：{report['moved_count']}",
                "",
                "## 影响最大的参数",
                "",
            ]
            for row in results[:25]:
                lines.append(
                    f"- {row['name']}: {row['min']}..{row['max']} · 差异 {row['diff_min_max']}"
                )
            (out / "报告.md").write_text("\n".join(lines), encoding="utf-8")
            return report
        finally:
            await bridge.call({"type": "inject.pause", "paused": False}, "inject.pause.result")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", default="data/contract.json")
    parser.add_argument("--out", default="data/boundary")
    parser.add_argument("--url", default="ws://127.0.0.1:8765")
    args = parser.parse_args()
    report = asyncio.run(run(pathlib.Path(args.contract), pathlib.Path(args.out), args.url))
    print(json.dumps({k: report[k] for k in ("parameter_count", "moved_count")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
