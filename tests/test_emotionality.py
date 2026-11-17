"""Rational / emotional dial: gains, timings, and the agent's temporary value."""

from pathlib import Path

import pytest

from dsh_vts_avatar.bridge.engine import AvatarEngine
from dsh_vts_avatar.config import AvatarConfig, Config
from dsh_vts_avatar.performance.mixer import PerformanceMixer


def _mixer(value: float) -> PerformanceMixer:
    return PerformanceMixer(AvatarConfig(emotionality=value), 48000)


def _engine(tmp_path: Path) -> AvatarEngine:
    engine = AvatarEngine(Config(root=tmp_path))
    engine.mixer = None
    state = engine.relationship.state
    state.trust = 0.0
    state.closeness = 0.0
    state.dependency = 0.0
    state.friction = 0.0
    return engine


def test_expression_gain_follows_the_dial():
    assert _mixer(0)._tone_gain("expression_gain", 1.0) == pytest.approx(0.6)
    assert _mixer(50)._tone_gain("expression_gain", 1.0) == pytest.approx(1.0)
    assert _mixer(100)._tone_gain("expression_gain", 1.0) == pytest.approx(1.4)


def test_affect_hold_and_release_stretch_with_the_dial():
    rational, emotional = _mixer(0), _mixer(100)
    assert rational.AFFECT_HOLD_S == pytest.approx(1.0)
    assert emotional.AFFECT_HOLD_S == pytest.approx(3.0)
    assert rational.AFFECT_RELEASE_S == pytest.approx(1.2)
    assert emotional.AFFECT_RELEASE_S == pytest.approx(3.2)


def test_mixer_uses_explicit_value_over_config():
    mixer = _mixer(50)
    mixer.set_emotionality(100)
    assert mixer._tone_gain("expression_gain", 1.0) == pytest.approx(1.4)
    mixer.set_emotionality(None)
    assert mixer._tone_gain("expression_gain", 1.0) == pytest.approx(1.0)


def test_bias_moves_with_relationship_and_stays_clamped(tmp_path):
    engine = _engine(tmp_path)
    assert engine.tone_bias() == pytest.approx(0.0)

    engine.relationship.state.friction = 1.0
    assert engine.tone_bias() == pytest.approx(10.0)  # careful, not emotional

    engine.relationship.state.friction = 0.0
    engine.relationship.state.closeness = 1.0
    engine.relationship.state.trust = 1.0
    engine.relationship.state.dependency = 1.0
    assert engine.tone_bias() == pytest.approx(-10.0)  # guard down, emotional


def test_agent_value_wins_then_user_clears_it(tmp_path):
    engine = _engine(tmp_path)
    assert engine.emotionality_effective() == pytest.approx(50.0)

    result = engine.set_tone(90, "在讲技术细节")
    assert result["ok"] is True
    assert engine.emotionality_effective() == pytest.approx(90.0)
    assert result["ttl_min"] >= 5.0

    cleared = engine.clear_tone()
    assert cleared["temporary"] is False
    assert engine.emotionality_effective() == pytest.approx(50.0)


def test_agent_value_expires_back_to_the_anchor(tmp_path):
    engine = _engine(tmp_path)
    engine.set_tone(10)
    engine._tone_until = 0.0  # pretend the period is over
    assert engine.emotionality_effective() == pytest.approx(50.0)


def test_ttl_follows_mood_and_dependency(tmp_path):
    engine = _engine(tmp_path)
    plain = engine.tone_ttl_minutes()

    engine.relationship.state.dependency = 1.0
    engine._mood_intensity = lambda: 1.0  # type: ignore[method-assign]
    emotional = engine.tone_ttl_minutes()

    assert 5.0 <= plain <= 60.0
    assert emotional > plain


def test_second_call_within_a_minute_is_refused(tmp_path):
    engine = _engine(tmp_path)
    assert engine.set_tone(20)["ok"] is True
    second = engine.set_tone(80)
    assert second["ok"] is False
    assert "60" in second["error"]


def test_settings_snapshot_exposes_the_dial(tmp_path):
    engine = _engine(tmp_path)
    snapshot = engine.settings_snapshot()
    assert snapshot["avatar"]["emotionality"] == pytest.approx(50.0)
    assert snapshot["tone"]["effective"] == pytest.approx(50.0)
    assert snapshot["tone"]["temporary"] is False
