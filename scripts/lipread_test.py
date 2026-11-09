"""Compare mouth channels for lip-readability on the current VTS model.

Channels tested separately (head zeroed, bridge injection paused):
    vowel  VoiceA/I/U/E/O
    smile  MouthSmile width
    vb     JawOpen / MouthFunnel / MouthPucker / MouthShrug
    all    every channel together (try to compensate weak pairs)

For each channel the five vowels are rendered with a fixed MouthOpen, the mouth
ROI is cropped automatically, and pairwise ROI differences are reported:
    >= 25 clear, 10-25 close, < 10 unreadable.

    python scripts/lipread_test.py --out E:\\dsh-vts-avatar\\冰糖\\lipread
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import shutil
from typing import Any

import cv2
import numpy as np
import websockets

VOWELS = "AIUEO"
# vowel -> (voice width form -1..1, roundness 0..1, openness extra 0..1)
SHAPE_MAP = {
    "A": {"MouthOpen": 0.85, "MouthSmile": 0.10, "MouthX": 0.0},
    "I": {"MouthOpen": 0.10, "MouthSmile": 0.80, "MouthX": 0.15},
    "U": {"MouthOpen": 0.15, "MouthSmile": 0.05, "MouthX": -0.30},
    "E": {"MouthOpen": 0.45, "MouthSmile": 0.55, "MouthX": 0.0},
    "O": {"MouthOpen": 0.50, "MouthSmile": 0.10, "MouthX": 0.30},
}
SHAPE = {
    "A": (0.10, 0.10, 0.10),
    "I": (0.60, 0.00, 0.00),
    "E": (0.30, 0.05, 0.05),
    "O": (-0.30, 0.65, 0.05),
    "U": (-0.50, 0.85, 0.00),
}


def scheme_params(scheme: str, vowel: str) -> dict[str, float]:
    width, roundness, extra = SHAPE[vowel]
    base: dict[str, float] = {
        "MouthOpen": 0.55,
        "FaceAngleX": 0.0,
        "FaceAngleY": 0.0,
        "FaceAngleZ": 0.0,
        "EyeOpenLeft": 1.0,
        "EyeOpenRight": 1.0,
    }
    out = dict(base)
    if scheme in ("vowel", "hybrid", "all"):
        out[f"Voice{vowel}"] = 1.0
    if scheme in ("smile", "hybrid", "all"):
        out["MouthSmile"] = max(0.0, min(1.0, 0.45 + 0.45 * width))
    if scheme in ("shape", "all"):
        out.update(SHAPE_MAP[vowel])
    if scheme in ("vb", "all"):
        out["JawOpen"] = 0.15 + 0.25 * extra + 0.2 * roundness
        out["MouthFunnel"] = roundness * 0.8
        out["MouthPucker"] = roundness * 0.7
        out["MouthShrug"] = width * 0.25
    return out


class Bridge:
    def __init__(self, url: str) -> None:
        self.url = url
        self.ws = None
        self.seq = 0

    async def __aenter__(self):
        self.ws = await websockets.connect(self.url, max_size=None)
        await self.ws.send(json.dumps({"type": "hello", "client": "lipread", "protocol": 1}))
        return self

    async def __aexit__(self, *exc):
        await self.ws.close()

    async def call(self, msg: dict[str, Any], expect: str) -> dict[str, Any]:
        self.seq += 1
        msg = dict(msg)
        msg["request_id"] = f"l{self.seq}"
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

    async def hold(self, params: dict[str, float], seconds: float = 0.55) -> pathlib.Path | None:
        async def pump() -> None:
            while True:
                await self.send({"type": "params.inject", "params": params})
                await asyncio.sleep(0.03)

        task = asyncio.create_task(pump())
        try:
            await asyncio.sleep(seconds)
            reply = await self.call({"type": "snapshot", "frames": 1}, "snapshot.result")
            path = (reply.get("metrics") or {}).get("path")
            return pathlib.Path(path) if path else None
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


def find_mouth_box(img):
    """Return (top, bottom, left, right) for the mouth once, from one frame.

    The box must be computed ONCE per run and reused for every frame: re-running
    the eye detection per frame jitters by a pixel or two and that misalignment
    alone outshines the real mouth movement (measured: 12-49 vs 5-14).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark = (gray < 70).astype(np.uint8)
    n, _lab, stats, _cent = cv2.connectedComponentsWithStats(dark, 8)
    blobs = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area >= 40:
            blobs.append((area, x + bw // 2, y + bh // 2, bw, bh))
    blobs.sort(reverse=True)
    for i in range(min(len(blobs), 8)):
        _a1, cx1, cy1, w1, h1 = blobs[i]
        if not (8 <= w1 <= 90 and 8 <= h1 <= 90):
            continue
        for j in range(i + 1, min(len(blobs), 12)):
            _a2, cx2, cy2, w2, h2 = blobs[j]
            if not (8 <= w2 <= 90 and 8 <= h2 <= 90):
                continue
            if abs(cy1 - cy2) < 40 and 20 < abs(cx1 - cx2) < 300:
                cx, cy = (cx1 + cx2) // 2, (cy1 + cy2) // 2
                d = abs(cx1 - cx2)
                half = max(30, d // 2)
                return (cy + int(d * 0.22), cy + int(d * 0.72), max(0, cx - half), cx + half)
    h, w = gray.shape
    ys, xs = np.where((gray < 235).astype(np.uint8) > 0)
    if len(xs):
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        cw, ch, ccx = x1 - x0, y1 - y0, (x0 + x1) // 2
        return (
            y0 + int(ch * 0.12),
            y0 + int(ch * 0.30),
            ccx - int(cw * 0.16),
            ccx + int(cw * 0.16),
        )
    return (h // 3, h // 2, w // 3, 2 * w // 3)


def crop_with_box(img, box):
    top, bottom, left, right = box
    crop = img[top:bottom, left:right]
    if crop.size == 0:
        return cv2.resize(img[: min(60, img.shape[0]), : min(80, img.shape[1])], (420, 300))
    return cv2.resize(crop, (420, 300))


def mouth_crop(img):
    return crop_with_box(img, find_mouth_box(img))


def align_to(ref, img):
    """Remove whole-head translation before comparing two mouth patches.

    The model keeps swaying while we capture, and that sway alone moved the
    patch as much as a different vowel did (noise 28 vs signal 25). Phase
    correlation finds the shift, we undo it, and only the mouth shape is left.
    """
    a = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if a.shape != b.shape:
        return img
    (dx, dy), _response = cv2.phaseCorrelate(a, b)
    matrix = np.float32([[1, 0, -dx], [0, 1, -dy]])
    return cv2.warpAffine(
        img, matrix, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def pairwise(diffs: dict[str, float]) -> dict[str, str]:
    out = {}
    for key, value in diffs.items():
        out[key] = "clear" if value >= 25 else ("close" if value >= 10 else "unreadable")
    return out


async def run(out: pathlib.Path, url: str) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    frames = out / "frames"
    frames.mkdir(exist_ok=True)
    report: dict[str, Any] = {"channels": {}}
    box = None
    reference = None
    async with Bridge(url) as bridge:
        await bridge.call({"type": "inject.pause", "paused": True}, "inject.pause.result")
        try:
            for scheme in ("vowel", "smile", "vb", "shape", "hybrid", "all"):
                crops = {}
                noise = {}
                # Interleave two passes over the vowels and take the per-pixel
                # median: the model's own idle sway is low-frequency, so it
                # cancels instead of being attributed to the vowel.
                passes: dict[str, list] = {v: [] for v in VOWELS}
                for repeat in (1, 2):
                    for vowel in VOWELS:
                        src = await bridge.hold(scheme_params(scheme, vowel))
                        if not src or not src.exists():
                            continue
                        dst = frames / f"{scheme}_{vowel}_{repeat}.jpg"
                        shutil.copyfile(src, dst)
                        img = cv2.imdecode(np.fromfile(str(dst), dtype=np.uint8), cv2.IMREAD_COLOR)
                        if box is None:
                            box = find_mouth_box(img)
                            report["mouth_box"] = [int(v) for v in box]
                        passes[vowel].append(crop_with_box(img, box))
                for vowel, shots in passes.items():
                    if not shots:
                        continue
                    if not crops:
                        reference = shots[0]
                    aligned = [align_to(reference, shot) for shot in shots]
                    stack = np.stack(aligned, axis=0)
                    crops[vowel] = np.median(stack, axis=0).astype(np.uint8)
                    if len(aligned) > 1:
                        g1 = cv2.cvtColor(aligned[0], cv2.COLOR_BGR2GRAY)
                        g2 = cv2.cvtColor(aligned[1], cv2.COLOR_BGR2GRAY)
                        noise[vowel] = round(float(np.mean(cv2.absdiff(g1, g2))), 2)
                diffs = {}
                for i in range(len(VOWELS)):
                    for j in range(i + 1, len(VOWELS)):
                        a, b = VOWELS[i], VOWELS[j]
                        if a in crops and b in crops:
                            ga = cv2.cvtColor(crops[a], cv2.COLOR_BGR2GRAY)
                            gb = cv2.cvtColor(crops[b], cv2.COLOR_BGR2GRAY)
                            if ga.shape == gb.shape:
                                diffs[f"{a}-{b}"] = round(float(np.mean(cv2.absdiff(ga, gb))), 2)
                values = list(diffs.values()) or [0.0]
                noise_values = list(noise.values()) or [0.0]
                noise_mean = round(sum(noise_values) / len(noise_values), 2)
                report["channels"][scheme] = {
                    "pairwise": diffs,
                    "verdict": pairwise(diffs),
                    "mean": round(sum(values) / len(values), 2),
                    "min_pair": min(diffs.items(), key=lambda kv: kv[1]) if diffs else None,
                    "unreadable_pairs": [k for k, v in pairwise(diffs).items() if v == "unreadable"],
                    "noise": noise,
                    "noise_mean": noise_mean,
                    "signal_to_noise": round(round(sum(values) / len(values), 2) / noise_mean, 2)
                    if noise_mean
                    else None,
                }
        finally:
            await bridge.call({"type": "inject.pause", "paused": False}, "inject.pause.result")
    (out / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 唇读通道对比", ""]
    for scheme, data in report["channels"].items():
        lines.append(f"## {scheme}")
        lines.append(f"- 平均差异 {data['mean']}，最弱对 {data['min_pair']}")
        for pair, value in data["pairwise"].items():
            lines.append(f"- {pair}: {value} ({data['verdict'][pair]})")
        lines.append("")
    (out / "报告.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/lipread")
    parser.add_argument("--url", default="ws://127.0.0.1:8765")
    parser.add_argument("--model-id", default="", help="write this model's mouth strategy from readability")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--profile-dir", default="data/profiles")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    report = asyncio.run(run(pathlib.Path(args.out), args.url))
    print(json.dumps({k: (v["mean"], v["unreadable_pairs"]) for k, v in report["channels"].items()}, ensure_ascii=False))
    if args.model_id:
        from dsh_vts_avatar.performance.mouth_strategy import classify, save_strategy

        means = {
            name: report["channels"][name]["mean"]
            for name in ("shape", "vowel", "hybrid", "smile")
            if name in report["channels"]
        }
        data = classify(means)
        data["model"] = {"modelName": args.model_name, "modelID": args.model_id}
        save_strategy(pathlib.Path(args.profile_dir), pathlib.Path(args.root), args.model_id, data)
        print("strategy:", data["strategy"], data["scores"])


if __name__ == "__main__":
    main()
