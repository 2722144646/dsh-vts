from dsh_vts_avatar.performance.diary import DiaryStore
from dsh_vts_avatar.performance.privacy import PrivacyStore, trust_required


def test_privacy_trust_threshold_and_share(tmp_path):
    store = PrivacyStore(tmp_path, "u1")
    item = store.add("feeling", "我其实很怕他哪天不来了", privacy=0.8)
    assert item.required == trust_required(0.8)
    assert not store.can_share(item, 0.3)
    assert store.can_share(item, 0.9)
    assert len(store.unshared()) == 1
    store.mark_shared(item.id)
    assert store.unshared() == []


def test_privacy_survives_reload(tmp_path):
    store = PrivacyStore(tmp_path, "u1")
    store.add("memory", "他说过会回来", privacy=0.4)
    reloaded = PrivacyStore(tmp_path, "u1")
    assert len(reloaded.unshared()) == 1


def test_diary_due_and_write(tmp_path):
    store = DiaryStore(tmp_path, "u1", max_bytes=10**6)
    assert store.due(now=1000.0) is True
    path = store.write("今天：聊了很久\n感受：平静", now=1000.0)
    assert path is not None and path.exists()
    assert store.due(now=1000.0 + 60) is False


def test_diary_cap_drops_oldest(tmp_path):
    store = DiaryStore(tmp_path, "u1", max_bytes=200)
    store.write("a" * 150, now=1000.0)
    store.write("b" * 150, now=1000.0 + 86400)
    files = list(store.root.glob("*.md"))
    assert len(files) == 1
    assert files[0].read_text(encoding="utf-8").startswith("b")
