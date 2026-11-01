import time

from dsh_vts_avatar.config import AvatarConfig
from dsh_vts_avatar.performance.emotion import EXPRESSION_TARGETS, EmotionState
from dsh_vts_avatar.performance.mixer import PerformanceMixer


def test_expression_presets_are_distinct():
    sad = EXPRESSION_TARGETS["sad"]
    happy = EXPRESSION_TARGETS["happy"]
    surprised = EXPRESSION_TARGETS["surprised"]
    assert sad["inner_brow"] > 0.8
    assert happy["smile"] > 0.8
    assert surprised["eye"] >= 1.0
    assert surprised["open"] > happy["open"]


def test_emotion_preset_blends_into_mixer_targets():
    mixer = PerformanceMixer(AvatarConfig(mouth_backend="continuous"), 48000)
    mixer.emotion.preset = "sad"
    mixer.emotion.preset_intensity = 0.9
    sad = mixer._emotion_targets()
    mixer.emotion.preset = "surprised"
    surprised = mixer._emotion_targets()
    assert sad["inner_brow"] > surprised["inner_brow"]
    assert surprised["eye"] > sad["eye"]
    assert surprised["open"] > sad["open"]


def test_emotion_set_keeps_preset_name():
    state = EmotionState()
    state.set("angry", 0.8)
    assert state.preset == "angry"
    assert state.target_preset_intensity == 0.8


def test_expression_gain_is_read_from_model_profile():
    from dsh_vts_avatar.vts.profile import ModelProfile

    profile = ModelProfile(model_id="t", tuning={"expression_gain": 0.5, "expression_head_gain": 0.5})
    mixer = PerformanceMixer(AvatarConfig(mouth_backend="continuous"), 48000, profile=profile)
    mixer.emotion.preset = "happy"
    mixer.emotion.preset_intensity = 1.0
    targets = mixer._emotion_targets()
    assert 0.5 < targets["smile"] < 0.9
