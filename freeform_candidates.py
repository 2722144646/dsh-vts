"""Create finer candidate segments from the freeform OBS session."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

VIDEO = Path(r"C:\Users\27221\Videos\2026-11-01 15-35-49.mp4")
SESSION = Path("dsh-vts-avatar/data/demos/20261101_153619_自由演示")
OUT = Path("dsh-vts-avatar/data/demos/freeform_aligned/candidates")
OFFSET = 28.193
CROP = (1050, 40, 2450, 1050)
FFMPEG = Path(r"D:\ruanjian\oopz\ffmpeg.exe")
KEYS = [
    "FaceAngleX", "FaceAngleY", "FaceAngleZ",
    "EyeOpenLeft", "EyeOpenRight", "EyeLeftX", "EyeLeftY",
    "EyeRightX", "EyeRightY", "Brows", "BrowLeftY", "BrowRightY",
    "MouthOpen", "MouthSmile", "CheekPuff",
    "VoiceA", "VoiceI", "VoiceU", "VoiceE", "VoiceO",
]


def category(changes: list[list]) -> str:
    names = [name for name, _ in changes]
    head = any(n.startswith("FaceAngle") for n in names)
    eyes = any(n.startswith("Eye") or n.startswith("Brow") for n in names)
    mouth = any(n.startswith("Mouth") or n.startswith("Voice") or n == "CheekPuff" for n in names)
    if head and mouth:
        return "head+mouth"
    if head and eyes:
        return "head+eyes"
    if mouth and eyes:
        return "expression+mouth"
    if head:
        return "head"
    if mouth:
        return "mouth"
    if eyes:
        return "eyes"
    return "motion"


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
    times = np.array([row["t"] for row in rows], dtype=float)
    data = {key: np.array([float(row["input"].get(key, 0.0)) for row in rows]) for key in KEYS}

    activity = np.zeros(len(rows))
    for key, values in data.items():
        span = np.percentile(values, 95) - np.percentile(values, 5)
        if span < 1e-5:
            continue
        activity += np.clip(np.abs(np.diff(values, prepend=values[0])) / span, 0, 2)
    smooth = np.convolve(activity, np.ones(4) / 4, mode="same")
    threshold = max(np.percentile(smooth, 35), smooth.mean() * 0.65)
    mask = smooth >= threshold

    raw = []
    start = None
    for i, on in enumerate(mask):
        if on and start is None:
            start = i
        elif not on and start is not None:
            if times[i - 1] - times[start] >= 0.45:
                raw.append((start, i - 1))
            start = None
    if start is not None:
        raw.append((start, len(rows) - 1))

    # Merge very close segments, then keep the strongest dozen.
    merged = []
    for a, b in raw:
        if merged and times[a] - times[merged[-1][1]] < 0.55:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    ranked = sorted(merged, key=lambda ab: float(smooth[ab[0]:ab[1] + 1].max()), reverse=True)[:14]
    ranked.sort(key=lambda ab: ab[0])

    cap = cv2.VideoCapture(str(VIDEO))
    tiles = []
    report = {"video": str(VIDEO), "session": str(SESSION), "offset_s": OFFSET, "segments": []}
    for idx, (a, b) in enumerate(ranked, 1):
        peak = a + int(np.argmax(smooth[a:b + 1]))
        t_param = float(times[peak])
        t_obs = t_param + OFFSET
        start_obs = float(times[a] + OFFSET)
        end_obs = float(times[b] + OFFSET)
        changes = []
        for key in KEYS:
            delta = float(data[key][peak] - data[key][a])
            if abs(delta) > 0.025:
                changes.append([key, round(delta, 3)])
        changes.sort(key=lambda kv: abs(kv[1]), reverse=True)
        cat = category(changes)
        cap.set(cv2.CAP_PROP_POS_MSEC, t_obs * 1000)
        ok, frame = cap.read()
        frame_path = None
        if ok:
            crop = frame[CROP[1]:CROP[3], CROP[0]:CROP[2]]
            crop = cv2.resize(crop, (360, 430))
            cv2.putText(crop, f"S{idx:02d} {cat}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            cv2.putText(crop, f"{t_obs:.1f}s", (8, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            tiles.append(crop)
            frame_path = OUT / f"S{idx:02d}.jpg"
            ok2, buf = cv2.imencode(".jpg", crop)
            if ok2:
                frame_path.write_bytes(buf.tobytes())

        clip_path = OUT / f"S{idx:02d}.mp4"
        subprocess.run(
            [
                str(FFMPEG), "-y", "-ss", f"{start_obs:.3f}", "-to", f"{end_obs:.3f}",
                "-i", str(VIDEO), "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-c:a", "aac", "-movflags", "+faststart", str(clip_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        report["segments"].append(
            {
                "id": idx,
                "category": cat,
                "param_start": round(float(times[a]), 3),
                "param_end": round(float(times[b]), 3),
                "obs_start": round(start_obs, 3),
                "obs_end": round(end_obs, 3),
                "duration_s": round(float(times[b] - times[a]), 2),
                "peak_activity": round(float(smooth[peak]), 3),
                "top_changes": changes[:8],
                "frame": str(frame_path) if frame_path else None,
                "clip": str(clip_path),
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
            (OUT / "candidates.jpg").write_bytes(buf.tobytes())
    (OUT / "segments.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2)[:8000])
    print("WROTE", OUT / "candidates.jpg")


if __name__ == "__main__":
    main()
