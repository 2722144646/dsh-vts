"""Segment the freeform OBS performance by parameter activity and extract clips."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

VIDEO = Path(r"C:\Users\27221\Videos\2026-11-01 15-35-49.mp4")
SESSION = Path("dsh-vts-avatar/data/demos/20261101_153619_自由演示")
OUT = Path("dsh-vts-avatar/data/demos/freeform_aligned")
# 1000 Hz expression-start marker in OBS audio, matched to params cue_t.
OBS_CUE = 36.24
PARAM_CUE = 8.0466
OFFSET = OBS_CUE - PARAM_CUE
CROP = (1050, 40, 2450, 1050)
KEYS = [
    "FaceAngleX", "FaceAngleY", "FaceAngleZ",
    "EyeOpenLeft", "EyeOpenRight", "EyeLeftX", "EyeLeftY",
    "EyeRightX", "EyeRightY", "Brows", "BrowLeftY", "BrowRightY",
    "MouthOpen", "MouthSmile", "CheekPuff",
    "VoiceA", "VoiceI", "VoiceU", "VoiceE", "VoiceO",
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for line in (SESSION / "params.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("phase") == "expression" and "input" in row:
            rows.append(row)
    if len(rows) < 10:
        raise RuntimeError("not enough expression samples")
    times = np.array([row["t"] for row in rows], dtype=float)
    data = {key: np.array([float(row["input"].get(key, 0.0)) for row in rows]) for key in KEYS}

    # Normalize each parameter by its robust span, then measure parameter activity.
    activity = np.zeros(len(rows))
    for key, values in data.items():
        span = np.percentile(values, 95) - np.percentile(values, 5)
        if span < 1e-5:
            continue
        diff = np.abs(np.diff(values, prepend=values[0])) / span
        activity += np.clip(diff, 0, 2)
    # Smooth ~0.5 s.
    kernel = np.ones(5) / 5
    smooth = np.convolve(activity, kernel, mode="same")
    threshold = max(np.percentile(smooth, 55), smooth.mean() + 0.3 * smooth.std())
    mask = smooth >= threshold
    segments = []
    start = None
    for i, on in enumerate(mask):
        if on and start is None:
            start = i
        elif not on and start is not None:
            if times[i - 1] - times[start] >= 0.7:
                segments.append((start, i - 1))
            start = None
    if start is not None and times[-1] - times[start] >= 0.7:
        segments.append((start, len(rows) - 1))

    # Keep the strongest segments, in chronological order.
    ranked = sorted(
        segments,
        key=lambda ab: float(smooth[ab[0]:ab[1] + 1].max()),
        reverse=True,
    )[:8]
    ranked.sort(key=lambda ab: ab[0])

    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise RuntimeError("cannot open OBS video")
    tiles = []
    report = {
        "video": str(VIDEO),
        "session": str(SESSION),
        "offset_s": round(OFFSET, 3),
        "param_samples": len(rows),
        "segments": [],
    }
    for idx, (a, b) in enumerate(ranked, 1):
        peak = a + int(np.argmax(smooth[a:b + 1]))
        t_param = float(times[peak])
        t_obs = t_param + OFFSET
        changes = []
        for key in KEYS:
            before = float(data[key][a])
            after = float(data[key][peak])
            if abs(after - before) > 0.03:
                changes.append((key, round(after - before, 3)))
        changes.sort(key=lambda kv: abs(kv[1]), reverse=True)
        cap.set(cv2.CAP_PROP_POS_MSEC, t_obs * 1000)
        ok, frame = cap.read()
        path = None
        if ok:
            crop = frame[CROP[1]:CROP[3], CROP[0]:CROP[2]]
            crop = cv2.resize(crop, (400, 470))
            label = f"S{idx} {t_obs:.1f}s"
            cv2.putText(crop, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            tiles.append(crop)
            path = OUT / f"segment_{idx:02d}.jpg"
            ok2, buf = cv2.imencode(".jpg", crop)
            if ok2:
                path.write_bytes(buf.tobytes())
        report["segments"].append(
            {
                "id": idx,
                "param_t": round(t_param, 3),
                "obs_t": round(t_obs, 3),
                "duration_s": round(float(times[b] - times[a]), 2),
                "peak_activity": round(float(smooth[peak]), 3),
                "top_changes": changes[:8],
                "frame": str(path) if path else None,
            }
        )
    cap.release()
    rows_img = []
    for i in range(0, len(tiles), 4):
        row = tiles[i:i + 4]
        while len(row) < 4:
            row.append(np.zeros_like(tiles[0]))
        rows_img.append(np.hstack(row))
    if rows_img:
        sheet = np.vstack(rows_img)
        ok, buf = cv2.imencode(".jpg", sheet)
        if ok:
            (OUT / "segments.jpg").write_bytes(buf.tobytes())
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("WROTE", OUT / "segments.jpg")


if __name__ == "__main__":
    main()
