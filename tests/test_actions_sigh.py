from dsh_vts_avatar.performance.actions import ActionCue


def test_sigh_opens_the_mouth_and_sinks_the_head():
    cue = ActionCue.create("sigh", start=0, sample_rate=1000, intensity=1.0)
    values = cue.sample(400)
    assert values["open"] > 0
    assert values["pitch"] > 0


def test_sigh_is_short_and_settles():
    cue = ActionCue.create("sigh", start=0, sample_rate=1000)
    assert cue.duration == 1800
    assert cue.sample(1799) != {}
    assert cue.sample(1800) == {}
