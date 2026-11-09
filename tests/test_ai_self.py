from dsh_vts_avatar.ai_self import DEFAULT_KEYWORDS, SelfKnowledge


def test_defaults_when_file_missing(tmp_path):
    store = SelfKnowledge(tmp_path / "ai" / "self.json")
    data = store.load()
    assert data["keywords"] == DEFAULT_KEYWORDS
    assert data["private_notes"] == []
    assert data["docs"]


def test_add_keyword_persists_and_dedupes(tmp_path):
    store = SelfKnowledge(tmp_path / "ai" / "self.json")
    store.add_keyword("情绪边界")
    store.add_keyword("情绪边界")
    data = store.load()
    assert data["keywords"].count("情绪边界") == 1
    assert "情绪边界" in data["keywords"]


def test_remove_keyword(tmp_path):
    store = SelfKnowledge(tmp_path / "ai" / "self.json")
    store.add_keyword("临时词")
    store.remove_keyword("临时词")
    assert "临时词" not in store.load()["keywords"]


def test_private_note_stays_local(tmp_path):
    store = SelfKnowledge(tmp_path / "ai" / "self.json")
    store.add_note("这件事我只想自己记着")
    data = store.load()
    assert data["private_notes"] == ["这件事我只想自己记着"]
    summary = store.summary()
    # The summary that reaches the prompt only carries a count, not the text.
    assert summary["note_count"] == 1
    assert "这件事我只想自己记着" not in str(summary)
