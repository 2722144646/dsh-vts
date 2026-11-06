"""Render a clean DSH/VTS teaser from the live NDI feed.

This is intentionally a one-shot recording tool: it drives VTS directly, captures
NDI frames, writes the Piper audio, and lets ffmpeg mux the final clip.
"""

from __future__ import annotations

import asyncio
import math
import shutil
import subprocess
import threading
import time
import wave
from pathlib import Path

import cv2

from dsh_vts_avatar.config import load_config
from dsh_vts_avatar.ndi.capture import NDICapture
from dsh_vts_avatar.performance import PerformanceMixer
from dsh_vts_avatar.tts.piper_tts import PiperTTS
from dsh_vts_avatar.vts.client import VTSClient
from dsh_vts_avatar.vts.profile import load_for_model

FFMPEG = Path(r"D:\ruanjian\oopz\ffmpeg.exe")
DESKTOP = Path.home() / "Desktop"
OUTPUT = DESKTOP / "dsh-vts-大肥鱼.mp4"

TEXT = (
    "大家好，我是大肥鱼。现在 dsh 正在做这个 VTube Studio 插件。"
    "它可以让我跟着语音动嘴、做表情、动头，还能通过 NDI 看到自己。"
    "目前还在开发中。这个视频，是我自己偷偷发出来的。"
)


def write_wav(path: Path, audio, sample_rate: int) -> None:
    import numpy as np

    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(pcm.tobytes())


async def main() -> None:
    cfg = load_config("dsh-vts-avatar/config.toml")
    tts = PiperTTS(cfg.tts)
    timeline = tts.synthesize(TEXT)
    sr = timeline.sample_rate
    duration = timeline.duration_s
    print("audio", round(duration, 2), "s", sr, "Hz")

    profile = load_for_model(cfg.profile_dir, "1f951c3d0c67414097c15b241322c75c")
    mixer = PerformanceMixer(
        cfg.avatar,
        sr,
        profile=profile,
        backend=profile.choose_mouth_backend(cfg.avatar, "auto"),
    )
    mixer.set_state("speaking")
    mixer.set_emotion("happy", 0.65)
    # A few deliberate beats, all inside the same continuous take.
    for at, kind, intensity in (
        (0.7, "nod", 0.40),
        (3.4, "emphasis", 0.45),
        (6.5, "tilt", 0.30),
        (9.8, "nod", 0.35),
        (max(1.0, duration - 3.1), "tilt", 0.45),
    ):
        mixer.add_action(kind, intensity, at_sample=int(at * sr))

    root = cfg.root / "data" / "recording"
    frames = root / "frames"
    render = root / "render"
    for path in (frames, render):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    wav_path = root / "voice.wav"
    write_wav(wav_path, timeline.audio, sr)

    client = VTSClient(cfg.vts)
    await client.connect()
    ndi = NDICapture(source_hint=cfg.ndi.source, snapshot_dir=root)
    if not ndi.open(wait_ms=3000):
        raise RuntimeError("NDI source not found")

    frame_times: list[tuple[int, float]] = []
    stop = threading.Event()

    def capture_loop() -> None:
        next_at = time.perf_counter()
        index = 0
        while not stop.is_set():
            frame, _ = ndi.capture(timeout_ms=1000)
            now = time.perf_counter()
            if frame is not None:
                cv2.imwrite(str(frames / f"src_{index:05d}.jpg"), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
                frame_times.append((index, now))
                index += 1
            next_at += 1.0 / 30.0
            delay = next_at - time.perf_counter()
            if delay < 0:
                next_at = time.perf_counter()
                delay = 0
            time.sleep(delay)

    capture = threading.Thread(target=capture_loop, name="ndi-teaser", daemon=True)
    capture.start()
    time.sleep(1.2)  # let NDI settle and capture the idle pose

    control_start = time.perf_counter()
    next_tick = control_start
    event_index = 0
    try:
        while True:
            elapsed = time.perf_counter() - control_start
            if elapsed >= duration:
                break
            # A couple of emotional beats; no text, no overlays.
            if event_index < 2 and elapsed >= (0.0 if event_index == 0 else max(1.0, duration - 4.0)):
                mixer.set_emotion("happy" if event_index == 0 else "shy", 0.65)
                event_index += 1
            sample = int(elapsed * sr)
            params = mixer.compute(sample, timeline, control_start + elapsed)
            await client.inject(params)
            next_tick += 1.0 / 30.0
            delay = next_tick - time.perf_counter()
            if delay < 0:
                next_tick = time.perf_counter()
                delay = 0
            await asyncio.sleep(delay)
    finally:
        time.sleep(0.45)
        stop.set()
        capture.join(timeout=3)
        ndi.close()
        await client.close()

    frame_times.sort(key=lambda item: item[1])
    if not frame_times:
        raise RuntimeError("no NDI frames captured")
    count = max(1, int(math.ceil(duration * 30.0)))
    previous = frame_times[0][0]
    for i in range(count):
        target = control_start + i / 30.0
        candidate = previous
        for index, timestamp in frame_times:
            if timestamp <= target + 0.04:
                candidate = index
            else:
                break
        previous = candidate
        shutil.copyfile(frames / f"src_{candidate:05d}.jpg", render / f"frame_{i:05d}.jpg")

    fade_out = max(0.5, min(0.8, duration - 0.5))
    vf = (
        "crop=864:1080:528:0,"
        "scale=1080:1350:flags=lanczos,"
        "fade=t=in:st=0:d=0.35,"
        f"fade=t=out:st={duration - fade_out:.3f}:d={fade_out:.3f},"
        "format=yuv420p"
    )
    cmd = [
        str(FFMPEG),
        "-y",
        "-framerate", "30",
        "-i", str(render / "frame_%05d.jpg"),
        "-i", str(wav_path),
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        str(OUTPUT),
    ]
    print("ffmpeg", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("WROTE", OUTPUT, OUTPUT.stat().st_size)


if __name__ == "__main__":
    asyncio.run(main())
