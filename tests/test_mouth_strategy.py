from dsh_vts_avatar.performance.mouth_strategy import classify


def test_both_channels_use_hybrid():
    r = classify(3.08, 40.31, 14.5)
    assert r["strategy"] == "hybrid"
    assert r["channels"] == ["vowel", "smile", "mouth_open"]


def test_only_vowel():
    assert classify(4.0, 1.0, 10.0)["strategy"] == "vowel"


def test_only_smile():
    assert classify(0.5, 20.0, 6.0)["strategy"] == "smile"


def test_only_open():
    assert classify(0.2, 1.0, 8.0)["strategy"] == "continuous"


def test_nothing():
    assert classify(0.1, 0.2, 0.3)["strategy"] == "none"
