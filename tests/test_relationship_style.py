from dsh_vts_avatar.performance.relationship import RelationshipStore


def _store(tmp_path):
    return RelationshipStore(tmp_path)


def test_neutral_relationship_has_no_style(tmp_path):
    rel = _store(tmp_path)
    assert rel.style() == ""


def test_low_trust_makes_speech_polite(tmp_path):
    rel = _store(tmp_path)
    rel.apply("boundary_crossed")
    rel.apply("boundary_crossed")
    style = rel.style()
    assert "客气" in style


def test_high_trust_and_closeness_relaxes_speech(tmp_path):
    rel = _store(tmp_path)
    for _ in range(12):
        rel.apply("vulnerability_shared")
        rel.apply("promise_kept")
    style = rel.style()
    assert "随意" in style or "放松" in style


def test_intimacy_closed_by_default_and_forced_costs_trust(tmp_path):
    rel = _store(tmp_path)
    assert rel.intimacy_allowed() is False
    before = rel.state.trust
    rel.apply("intimacy_forced")
    assert rel.state.trust < before
    assert rel.state.friction > 0


def test_intimacy_opens_with_close_relationship(tmp_path):
    rel = _store(tmp_path)
    for _ in range(20):
        rel.apply("vulnerability_shared")
        rel.apply("promise_kept")
    assert rel.intimacy_allowed() is True
