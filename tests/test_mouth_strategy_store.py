from dsh_vts_avatar.performance.mouth_strategy import save_strategy, load_strategy


def test_per_model_strategy_is_isolated(tmp_path):
    root = tmp_path
    profiles = root / "profiles"
    save_strategy(profiles, root, "modelA", {"strategy": "vowel"})
    save_strategy(profiles, root, "modelB", {"strategy": "smile"})
    assert load_strategy(profiles, root, "modelA")["strategy"] == "vowel"
    assert load_strategy(profiles, root, "modelB")["strategy"] == "smile"
    assert load_strategy(profiles, root, "modelC") is None


def test_legacy_file_only_matches_its_model(tmp_path):
    root = tmp_path
    profiles = root / "profiles"
    (root / "data").mkdir(parents=True)
    (root / "data" / "mouth_strategy.json").write_text(
        '{"model": {"modelID": "modelA"}, "strategy": "hybrid"}', encoding="utf-8"
    )
    assert load_strategy(profiles, root, "modelA")["strategy"] == "hybrid"
    assert load_strategy(profiles, root, "modelB") is None
