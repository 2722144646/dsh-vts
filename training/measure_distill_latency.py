"""Measure distilled voice latency, reference-fit factor, and exact duration."""

from dataclasses import replace
import time
from pathlib import Path

import numpy as np

from dsh_vts_avatar.config import load_config
from dsh_vts_avatar.tts.distill import DistilledPiperTTS

TEXTS = [
    "你好，我是肥鱼。",
    "今天天气很好，我们一起出去走走吧。",
    "早上好，昨晚睡得怎么样？",
    "我刚泡了一杯热茶，你要不要也来一杯？",
    "这个周末你有什么安排吗？",
    "我最近在学做菜，味道还不错。",
    "别着急，慢慢说，我在听。",
    "你先休息一下，剩下的交给我。",
    "这件事情我想听听你的想法。",
    "今天的工作终于做完了，感觉轻松多了。",
]

cfg = load_config("dsh-vts-avatar/config.toml")
tts = DistilledPiperTTS(cfg.tts)
tts.load()
latencies = []
factors = []
errors = []
for text in TEXTS:
    started = time.perf_counter()
    timeline = tts.synthesize(text)
    latency = (time.perf_counter() - started) * 1000
    latencies.append(latency)
    factors.append(tts.last_factor)
    errors.append(abs(timeline.duration_s - tts.last_reference_duration))
    print(
        round(latency, 1), "ms",
        "target", round(tts.last_reference_duration, 3),
        "out", round(timeline.duration_s, 3),
        "factor", round(tts.last_factor, 3),
        "spans", len(timeline.spans),
    )
print("latency mean/max ms", round(float(np.mean(latencies)), 1), round(max(latencies), 1))
print("factor min/max", round(min(factors), 3), round(max(factors), 3))
print("duration error max s", round(max(errors), 6))
