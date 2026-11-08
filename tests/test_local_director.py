import asyncio

from dsh_vts_avatar.bridge.local_chat import LocalDirector
from dsh_vts_avatar.config import LLMConfig


def director_with(content):
    director = LocalDirector(LLMConfig(director_enabled=True, director_model="m"))
    director._post = lambda payload: {"message": {"content": content}}
    return director


def test_director_parses_json():
    director = director_with('{"kind": "idle_tilt", "intensity": 0.8}')
    assert asyncio.run(director.suggest(["idle_tilt", "idle_nod"], {})) == ("idle_tilt", 0.8)


def test_director_strips_code_fence():
    director = director_with('```json\n{"kind": "idle_nod", "intensity": 0.4}\n```')
    assert asyncio.run(director.suggest(["idle_nod"], {})) == ("idle_nod", 0.4)


def test_director_rejects_unknown_kind():
    director = director_with('{"kind": "idle_dance", "intensity": 0.5}')
    assert asyncio.run(director.suggest(["idle_tilt"], {})) is None


def test_director_rejects_bad_json():
    director = director_with("sorry, I cannot")
    assert asyncio.run(director.suggest(["idle_tilt"], {})) is None


def test_director_clamps_intensity():
    director = director_with('{"kind": "idle_tilt", "intensity": 9}')
    assert asyncio.run(director.suggest(["idle_tilt"], {})) == ("idle_tilt", 1.2)
