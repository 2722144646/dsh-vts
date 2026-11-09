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

    async def send(self, msg: dict[str, Any]) -> None:
        self.seq += 1
        msg = dict(msg)
        msg["request_id"] = f"s{self.seq}"
        await self.ws.send(json.dumps(msg))

    async def inject(self, params: dict[str, float]) -> None:
        await self.call({"type": "params.inject", "params": params}, "params.inject.result")

    async def snapshot(self) -> pathlib.Path | None:
        reply = await self.call({"type": "snapshot", "frames": 1}, "snapshot.result")
        path = (reply.get("metrics") or {}).get("path")
        return pathlib.Path(path) if path else None

    async def hold_snapshot(self, params: dict[str, float], dst: pathlib.Path | None = None) -> pathlib.Path | None:
        """VTS keeps injected values only for the frame they arrive on, so keep
        re-sending while the NDI frame is grabbed."""
        async def pump() -> None:
            while True:
                await self.send({"type": "params.inject", "params": params})
                await asyncio.sleep(0.03)

        task = asyncio.create_task(pump())
        try:
            await asyncio.sleep(0.35)
            src = await self.snapshot()
            if src and dst is not None and _jpeg_path(src):
                shutil.copyfile(src, dst)
                return dst
            return src
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


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
                f_hi = frames / f"{name}__max.jpg"
                f_lo = frames / f"{name}__min.jpg"
                await bridge.hold_snapshot({name: high}, f_hi)
                await bridge.hold_snapshot({name: low}, f_lo)
                await bridge.inject({name: default})
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
            by_name = {r["name"]: r["diff_min_max"] for r in results}
            vowels = [by_name.get(n, 0.0) for n in ("VoiceA", "VoiceE", "VoiceI", "VoiceO", "VoiceU")]
            # Ranking now comes from lip-readability (lipread_test.py).
            # Boundary only proves a parameter is alive, never decides priority.
            vowel_score = round(sum(vowels) / max(1, len(vowels)), 2)
            smile_score = round(by_name.get("MouthSmile", 0.0), 2)
            open_score = round(by_name.get("MouthOpen", 0.0), 2)
            mouth_strategy = {
                "impact": {
                    "vowel": vowel_score,
                    "smile": smile_score,
                    "mouth_open": open_score,
                },
                "note": "impact only; mouth channel is chosen from lipread_test.py",
            }
            strategy = "see-lipread"
            report = {
                "model": model,
                "contract": str(contract_path),
                "parameter_count": len(results),
                "moved_count": sum(1 for r in results if r["moved"]),
                "parameters": results,
                "mouth_strategy": mouth_strategy,
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
                f"- 视觉影响（仅证明参数活着，不决定通道）：元音 {vowel_score} / smile {smile_score} / 开合 {open_score}",
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
