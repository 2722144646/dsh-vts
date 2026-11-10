"""Idle behaviour follows the feeling: low mood means fewer, gentler motions."""

import random
import statistics

from dsh_vts_avatar.performance.presence import PresenceDirector, next_gap


def _director() -> PresenceDirector:
    return PresenceDirector(base_energy=0.55, library_path="")


def test_set_mood_clamps():
    director = _director()
    director.set_mood(-5)
    assert director.mood == -1.0
    director.set_mood(5)
    assert director.mood == 1.0
    director.set_mood("oops")
    assert director.mood == 0.0


def test_low_mood_stretches_the_gap():
    rng = random.Random(7)
    normal = [next_gap(0.0, rng) for _ in range(50)]

    low = _director()
    low.set_mood(-1.0)
    bright = _director()
    bright.set_mood(1.0)

    assert low.mood_rate() > 1.3
    assert bright.mood_rate() < 0.8
    # The multiplier really stretches/shrinks the sampled gap.
    assert statistics.mean(normal) * low.mood_rate() > statistics.mean(normal)
    assert statistics.mean(normal) * bright.mood_rate() < statistics.mean(normal)


def test_low_mood_moves_more_gently():
    low = _director()
    low.set_mood(-1.0)
    bright = _director()
    bright.set_mood(1.0)
    assert low.mood_gain() < bright.mood_gain()
    assert 0.7 <= low.mood_gain() <= 1.1


def test_cue_intensity_is_scaled_by_mood():
    director = _director()
    director.set_mood(-1.0)
    cue = director.next_cue(10**9, speaking=False, silence_s=0.0)
    assert cue is not None
    _kind, intensity = cue
    assert 0.1 <= intensity <= 1.2

    bright = _director()
    bright.set_mood(1.0)
    cue2 = bright.next_cue(10**9, speaking=False, silence_s=0.0)
    assert cue2 is not None
    assert cue2[1] >= intensity
