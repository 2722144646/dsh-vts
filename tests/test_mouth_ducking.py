"""The mouth belongs to speech: expression must not fight it while talking."""

import time

from dsh_vts_avatar.config import AvatarConfig
from dsh_vts_avatar.performance.mixer import (
    MOUTH_DUCK,
    NEUTRAL_SMILE,
    PerformanceMixer,
    duck_smile,
)
from dsh_vts_avatar.tts.timeline import Timeline

SILENT = {"open": 0.0, "form": 0.0, "voice": {}, "viseme": ""}


def _mixer(emotion: str | None = None) -> PerformanceMixer:
    mixer = PerformanceMixer(AvatarConfig(mouth_backend="vowel"), 48000)
    if emotion:
        mixer.set_emotion(emotion, 1.0)
    return mixer


def _run(mixer: PerformanceMixer, state: str, silent: bool = False, seconds: float = 3.0) -> dict:
    if silent:
        mixer._speech = lambda *a, **k: dict(SILENT)
        mixer._speech_coarticulated = lambda *a, **k: dict(SILENT)
    mixer.set_state(state)
    timeline = Timeline.synthetic("阿伊乌诶哦", sample_rate=48000)
    params = {}
    start = time.perf_counter()
    for i in range(int(seconds * 100)):
        params = mixer.compute(i * 480, timeline, start + i * 0.01)
    return params


def test_duck_smile_shrinks_around_neutral_not_zero():
    assert duck_smile(0.9, 0.25) == NEUTRAL_SMILE + (0.9 - NEUTRAL_SMILE) * 0.25
    assert duck_smile(0.2, 0.25) == NEUTRAL_SMILE + (0.2 - NEUTRAL_SMILE) * 0.25
    # At the neutral input nothing changes: the model needs 0.50 as its neutral.
    assert duck_smile(NEUTRAL_SMILE, 0.25) == NEUTRAL_SMILE


def test_duck_factor_follows_state_and_mouth():
    mixer = _mixer()
    mixer.set_state("idle")
    assert mixer._duck_factor(0.0) == 1.0
    assert mixer._duck_factor(0.4) == MOUTH_DUCK
    mixer.set_state("speaking")
    assert mixer._duck_factor(0.0) == MOUTH_DUCK


def test_smile_is_flattened_toward_neutral_while_speaking():
    params = _run(_mixer("excited"), "speaking")
    raw = 0.629  # excited @1.0 before ducking
    assert abs(params["MouthSmile"] - NEUTRAL_SMILE) < abs(raw - NEUTRAL_SMILE)


def test_expression_is_full_when_the_mouth_is_quiet():
    params = _run(_mixer("excited"), "idle", silent=True)
    assert params["MouthSmile"] > 0.55


def test_open_mouthed_expression_cannot_prop_the_mouth_open_while_speaking():
    # surprised @1.0 wants open=0.343. Mid-utterance (state=speaking) that must
    # be ducked, otherwise a surprised face keeps the mouth hanging open and
    # every vowel smears into the same wide shape.
    speaking = _run(_mixer("surprised"), "speaking", silent=True, seconds=0.4)
    idle = _run(_mixer("surprised"), "idle", silent=True, seconds=0.4)
    assert speaking["MouthOpen"] < idle["MouthOpen"] * 0.6
