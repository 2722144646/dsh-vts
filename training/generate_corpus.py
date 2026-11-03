"""Resumable CosyVoice corpus generation with per-segment validation.

Every sentence is one job. A bad or interrupted take never enters metadata.csv
or the wav directory; the accepted/rejected logs are append-only and can be
used to resume after a crash.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
import urllib.request
import wave
import argparse

import numpy as np
import soxr

ROOT = Path("dsh-vts-avatar/data/voice_training/cosyvoice")
CORPUS = Path("dsh-vts-avatar/training/corpus.txt")
KEY_FILE = Path(r"E:\feiyu\service\.runtime\voice-service\api_key.txt")
BASE = "http://127.0.0.1:9880"
STYLE = "gentle"
MIN_SECONDS = 0.8
MAX_SECONDS = 15.0


def key() -> str:
    value = KEY_FILE.read_text(encoding="utf-8").strip()
    if not value:
        raise RuntimeError("empty Feiyu API key")
    return value


def request(path: str, *, payload: dict | None = None, raw: bool = False) -> bytes:
    headers = {"Authorization": "Bearer " + key()}
    data = None
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
    req = urllib.request.Request(BASE + path, data=data, headers=headers, method="POST" if payload is not None else "GET")
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.read()


def validate(path: Path, text: str) -> tuple[bool, str]:
    try:
        with wave.open(str(path), "rb") as fh:
            rate = fh.getframerate()
            channels = fh.getnchannels()
            width = fh.getsampwidth()
            frames = fh.getnframes()
            data = fh.readframes(frames)
    except Exception as exc:
        return False, f"bad_wav:{type(exc).__name__}"
    if not 16000 <= rate <= 48000:
        return False, f"bad_rate:{rate}"
    if channels != 1 or width != 2:
        return False, f"bad_format:{channels}ch/{width*8}bit"
    duration = frames / rate
    if not (MIN_SECONDS <= duration <= MAX_SECONDS):
        return False, f"bad_duration:{duration:.2f}"
    chars = max(1, len("".join(ch for ch in text if "\u4e00" <= ch <= "\u9fff")))
    per_char = duration / chars
    if not (0.12 <= per_char <= 1.00):
        return False, f"bad_rate_per_char:{per_char:.2f}"
    audio = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    if audio.size == 0:
        return False, "empty_audio"
    rms = float(np.sqrt(np.mean(audio * audio)))
    peak = float(np.max(np.abs(audio)))
    if rms < 0.004:
        return False, f"too_quiet:{rms:.4f}"
    if peak < 0.05:
        return False, f"low_peak:{peak:.3f}"
    clipped = float(np.mean(np.abs(audio) >= 0.999))
    if clipped > 0.01:
        return False, f"clipped:{clipped:.3f}"
    return True, f"ok:{duration:.2f}s/{rms:.3f}/{peak:.3f}"


def append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_metadata(accepted: list[dict]) -> None:
    lines = [f"{row['id']}|{row['text']}" for row in accepted]
    (ROOT / "metadata.csv").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def resample_to_piper(path: Path) -> float:
    with wave.open(str(path), "rb") as fh:
        rate = fh.getframerate()
        channels = fh.getnchannels()
        width = fh.getsampwidth()
        raw = fh.readframes(fh.getnframes())
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if rate != 22050:
        samples = soxr.resample(samples, rate, 22050).astype(np.float32)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(22050)
        fh.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return samples.size / 22050.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-minutes", type=float, default=60.0)
    parser.add_argument("--limit", type=int, default=0, help="only process N new sentences")
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "wav").mkdir(exist_ok=True)
    accepted_path = ROOT / "accepted.jsonl"
    rejected_path = ROOT / "rejected.jsonl"
    accepted: list[dict] = []
    done: set[str] = set()
    if accepted_path.exists():
        for line in accepted_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                accepted.append(row)
                done.add(row["id"])
            except Exception:
                pass
    total_seconds = sum(float(row.get("duration", 0)) for row in accepted)
    target_seconds = args.target_minutes * 60.0
    lines = [line.strip() for line in CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"accepted={len(accepted)} audio={total_seconds/60:.1f}min target={target_seconds/60:.0f}min", flush=True)

    consecutive_failures = 0
    processed = 0
    for text in lines:
        if total_seconds >= target_seconds:
            break
        if args.limit and processed >= args.limit:
            break
        processed += 1
        identifier = "cv_" + hashlib.sha256((STYLE + "\n" + text).encode("utf-8")).hexdigest()[:16]
        if identifier in done:
            continue
        wav = ROOT / "wav" / f"{identifier}.wav"
        part = ROOT / "wav" / f".{identifier}.part"
        part.unlink(missing_ok=True)
        last_error = ""
        for attempt in range(3):
            try:
                job = json.loads(request("/v1/speech", payload={"voice": "feiyu", "text": text, "style": STYLE}))
                job_id = str(job["id"])
                deadline = time.monotonic() + 300
                status = ""
                while time.monotonic() < deadline:
                    time.sleep(2.0)
                    state = json.loads(request(f"/v1/jobs/{job_id}"))
                    status = str(state.get("status", ""))
                    if status == "success":
                        data = request(f"/v1/jobs/{job_id}/audio")
                        part.write_bytes(data)
                        ok, detail = validate(part, text)
                        if not ok:
                            raise ValueError(detail)
                        duration = resample_to_piper(part)
                        part.replace(wav)
                        row = {"id": identifier, "text": text, "file": str(wav), "duration": round(duration, 3)}
                        append(accepted_path, row)
                        accepted.append(row)
                        done.add(identifier)
                        total_seconds += duration
                        write_metadata(accepted)
                        print(f"OK {identifier} {duration:.2f}s total={total_seconds/60:.1f}min {text[:16]}", flush=True)
                        consecutive_failures = 0
                        break
                    if status in {"failed", "interrupted"}:
                        raise RuntimeError(f"job_{status}:{state.get('error', '')}")
                else:
                    raise TimeoutError("job_timeout")
                if identifier in done:
                    break
            except Exception as exc:
                last_error = f"{type(exc).__name__}:{exc}"
                part.unlink(missing_ok=True)
                print(f"RETRY {identifier} {attempt+1}/3 {last_error}", flush=True)
                time.sleep(2.0)
        if identifier not in done:
            append(rejected_path, {"id": identifier, "text": text, "error": last_error})
            wav.unlink(missing_ok=True)
            consecutive_failures += 1
            print(f"REJECT {identifier} {last_error}", flush=True)
            if consecutive_failures >= 5:
                print("ABORT: five consecutive failures; accepted data is preserved", flush=True)
                break
    write_metadata(accepted)
    print(f"DONE accepted={len(accepted)} audio={total_seconds/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
