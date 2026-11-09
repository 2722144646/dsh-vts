from dsh_vts_avatar.performance.mouth_strategy import READ_MIN, classify


def test_picks_highest_readability_channel():
    r = classify({"shape": 41.08, "vowel": 18.32, "hybrid": 16.81, "smile": 6.96})
    assert r["strategy"] == "shape"
    assert r["threshold"] == READ_MIN


def test_vowel_wins_when_higher():
    r = classify({"vowel": 20.0, "smile": 8.0})
    assert r["strategy"] == "vowel"


def test_hybrid_can_win():
    r = classify({"vowel": 11.0, "hybrid": 14.0, "smile": 9.0})
    assert r["strategy"] == "hybrid"


def test_below_threshold_falls_back_to_continuous():
    r = classify({"vowel": 9.9, "smile": 4.0}, open_ok=True)
    assert r["strategy"] == "continuous"
    assert r["usable"] == []


def test_nothing_usable_and_no_open():
    r = classify({}, open_ok=False)
    assert r["strategy"] == "none"
