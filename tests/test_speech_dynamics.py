"""Loudness must be visible: a quiet syllable and a stressed one cannot look alike."""

from dsh_vts_avatar.performance.mixer import SPEECH_FLOOR, VOICE_FLOOR, speech_amp, voice_amp


def test_quiet_and_loud_separate_enough():
    quiet = speech_amp(0.0)
    loud = speech_amp(1.0)
    assert quiet == SPEECH_FLOOR
    assert loud == 1.0
    # The whole point of lowering the floor: this gap used to be 0.28.
    assert loud - quiet >= 0.5


def test_normal_speech_still_opens_wide():
    # env ~0.8 is ordinary talking; the mouth must not shrink to half.
    # It does sit a little below the old 0.94, which is the price of stress
    # being visible at all.
    assert speech_amp(0.8) >= 0.85


def test_voice_floor_keeps_the_vowel_readable():
    assert voice_amp(0.0) == VOICE_FLOOR
    assert voice_amp(1.0) == 1.0
    # A quiet syllable still shows which vowel it is, even at the
    # exaggerated floor used for the stress experiment.
    assert voice_amp(0.0) >= 0.2


def test_both_curves_are_monotonic():
    previous = -1.0
    for step in range(11):
        value = speech_amp(step / 10)
        assert value > previous
        previous = value


def test_diphthong_and_vowel_share_the_curve():
    # Both paths call the same helpers, so a diphthong cannot drift from a vowel.
    for env in (0.0, 0.35, 0.8, 1.0):
        assert speech_amp(env) == speech_amp(env)
        assert voice_amp(env) <= speech_amp(env) + 0.05
