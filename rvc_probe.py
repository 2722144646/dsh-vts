import time

from dsh_vts_avatar.config import load_config
from dsh_vts_avatar.tts.piper_tts import PiperTTS
from dsh_vts_avatar.tts.rvc import RVCConverter

cfg = load_config("dsh-vts-avatar/config.toml")
cfg.rvc.enabled = True
tts = PiperTTS(cfg.tts)
timeline = tts.synthesize("你好，我是肥鱼。")
converter = RVCConverter(cfg.rvc)
started = time.perf_counter()
converted = converter.convert(timeline, "你好，我是肥鱼。")
print("seconds", round(time.perf_counter() - started, 2))
print("original", timeline.audio.size, timeline.duration_s)
print("converted", converted.audio.size, converted.duration_s)
print("same_length", converted.audio.size == timeline.audio.size)
