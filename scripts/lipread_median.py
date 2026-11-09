"""Run the lip-read test N times and pick the channel by median readability.

    python scripts/lipread_median.py --model-id <id> --model-name 冰糖 --out <dir> --runs 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import statistics
import subprocess
import sys

CHANNELS = ("shape", "vowel", "hybrid", "smile", "vb", "all")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/lipread")
    parser.add_argument("--url", default="ws://127.0.0.1:8765")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--model-id", default="")
    parser.add_argument("--model-name", default="")
    args = parser.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for i in range(1, args.runs + 1):
        sub = out / f"run{i}"
        cmd = [
            sys.executable,
            str(pathlib.Path(__file__).with_name("lipread_test.py")),
            "--out",
            str(sub),
            "--url",
            args.url,
        ]
        print(f"run {i}:", flush=True)
        subprocess.run(cmd, check=True)
        runs.append(json.loads((sub / "results.json").read_text(encoding="utf-8"))["channels"])

    med = {c: round(statistics.median([r[c]["mean"] for r in runs]), 2) for c in CHANNELS if c in runs[0]}
    print("median:", json.dumps(med, ensure_ascii=False))

    summary = {"model_id": args.model_id, "model_name": args.model_name, "runs": args.runs, "median": med}
    if args.model_id:
        from dsh_vts_avatar.performance.mouth_strategy import classify, save_strategy

        scores = {c: med[c] for c in ("shape", "vowel", "hybrid", "smile") if c in med}
        data = classify(scores)
        data["model"] = {"modelName": args.model_name, "modelID": args.model_id}
        data["median_runs"] = args.runs
        data["all_channels"] = med
        root = pathlib.Path(".").resolve()
        save_strategy(root / "data" / "profiles", root, args.model_id, data)
        summary["strategy"] = data["strategy"]
        print("strategy:", data["strategy"])
    (out / "median.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
