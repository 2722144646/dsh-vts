"""Align the OBS 委屈OBS2 recording with its parameter log.

Usage: python analyze_obs.py
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

VIDEO = Path(r"C:\Users\27221\Videos\2026-11-01 13-53-17.mp4")
SESSION = Path(r"dsh-vts-avatar/data/demos/20261101_135356_委屈OBS2")
OUT = Path(r"dsh-vts-avatar/data/demos/obs_aligned2")
# 1000 Hz cue: OBS 46.07s = params 7.739s
OBS_OFFSET = 46.07 - 7.739
CROP = (700, 40, 2400, 1050)  # x1, y1, x2, y2


def load_params() -> list[dict]:
    rows = []
    for line in (SESSION / "params.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def visual_metrics(video: Path) -> tuple[list[dict], dict[float, np.ndarray]]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError("cannot open OBS video")
    fps = cap.get(cv2.CAP_PROP_FPS)
    rows: list[dict] = []
    picks: dict[float, np.ndarray] = {}
    prev = None
    t = 45.5
    while t <= 56.2:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            t += 1 / 30
            continue
        crop = frame[CROP[1]:CROP[3], CROP[0]:CROP[2]]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = ((hsv[:, :, 1] > 35) & (hsv[:, :, 2] < 248)) | (hsv[:, :, 2] < 150)
        ys, xs = np.where(mask)
        area = float(len(xs))
        cx = float(xs.mean()) if len(xs) else 0.0
        cy = float(ys.mean()) if len(ys) else 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        motion = float(np.mean(cv2.absdiff(gray, prev))) if prev is not None and prev.shape == gray.shape else 0.0
        prev = gray
        rows.append({"obs_t": round(t, 3), "area": area, "cx": cx, "cy": cy, "motion": motion})
        for pick in (46.0, 46.5, 47.0, 48.0, 49.0, 50.0, 52.0, 54.0, 55.5):
            if abs(t - pick) < 1 / 60:
                picks[pick] = crop.copy()
        t += 1 / 30
    cap.release()
    return rows, picks


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    params = load_params()
    by_t = {float(row["t"]): row for row in params if "input" in row}
    visual, picks = visual_metrics(VIDEO)

    # Resample parameter curves to the visual timeline.
    times = np.array([row["obs_t"] for row in visual], dtype=float)
    keys = ["FaceAngleX", "FaceAngleY", "FaceAngleZ", "EyeOpenLeft", "EyeOpenRight", "MouthOpen", "MouthSmile"]
    curves: dict[str, np.ndarray] = {}
    for key in keys:
        vals = []
        for t_obs in times:
            t_param = t_obs - OBS_OFFSET
            nearest = min(by_t, key=lambda t: abs(t - t_param))
            vals.append(float(by_t[nearest]["input"].get(key, 0.0)))
        curves[key] = np.array(vals)

    def lag(x: np.ndarray, y: np.ndarray) -> tuple[int, float]:
        x = np.diff(x)
        y = np.diff(y)
        x = (x - x.mean()) / (x.std() + 1e-9)
        y = (y - y.mean()) / (y.std() + 1e-9)
        best = (0, -2.0)
        for k in range(-15, 16):
            if k < 0:
                a, b = x[:k], y[-k:]
            elif k > 0:
                a, b = x[k:], y[:-k]
            else:
                a, b = x, y
            score = float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else -2.0
            if score > best[1]:
                best = (k, score)
        return best

    cx = np.array([row["cx"] for row in visual])
    cy = np.array([row["cy"] for row in visual])
    motion = np.array([row["motion"] for row in visual])
    report = {
        "video": str(VIDEO),
        "session": str(SESSION),
        "offset_s": round(OBS_OFFSET, 3),
        "cue_obs_s": 46.07,
        "done_obs_s": 55.70,
        "visual_frames": len(visual),
        "param_samples": len([x for x in params if x.get("phase") == "expression"]),
        "lag": {
            "FaceAngleY_vs_cx": lag(curves["FaceAngleY"], cx),
            "FaceAngleX_vs_cy": lag(curves["FaceAngleX"], cy),
            "MouthOpen_vs_motion": lag(curves["MouthOpen"], motion),
        },
        "curves": {k: {"min": round(float(v.min()), 3), "max": round(float(v.max()), 3)} for k, v in curves.items()},
    }
    (OUT / "analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Compare this take with the earlier parameter-only 委屈.
    from dsh_vts_avatar.config import load_config
    from dsh_vts_avatar.vts.profile import ModelProfile, profile_path

    cfg = load_config("dsh-vts-avatar/config.toml")
    profile = ModelProfile.load(profile_path(cfg.profile_dir, "8cb985f5b9514021a5c5d941b482e57b"))
    old = profile.learned_expressions.get("委屈", {})
    new = profile.learned_expressions.get("委屈OBS2", {})
    comparison = {}
    for key in ("委屈", "委屈OBS2"):
        exp = profile.learned_expressions.get(key, {})
        comparison[key] = {
            "top_inputs": exp.get("top_inputs", [])[:8],
            "samples": exp.get("samples"),
            "cue_t": exp.get("cue_t"),
            "done_t": exp.get("done_t"),
            "top_deltas": {
                k: v.get("delta") for k, v in list(exp.get("input_deltas", {}).items())[:8]
            },
        }
    report["comparison"] = comparison
    (OUT / "analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    tiles = []
    for t in sorted(picks):
        crop = picks[t]
        tile = cv2.resize(crop, (400, 470))
        cv2.putText(tile, f"{t:.1f}s", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        tiles.append(tile)
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    sheet = np.vstack(rows)
    ok, buf = cv2.imencode(".jpg", sheet)
    if ok:
        (OUT / "aligned.jpg").write_bytes(buf.tobytes())

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("WROTE", OUT / "aligned.jpg")


if __name__ == "__main__":
    main()
