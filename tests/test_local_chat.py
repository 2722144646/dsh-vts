import asyncio

from dsh_vts_avatar.bridge.engine import AvatarEngine
from dsh_vts_avatar.config import Config, EmotionConfig, LLMConfig


class StubLocal:
    def __init__(self):
        self.seen = []

    async def reply(self, text, history, extra_system=""):
        self.seen.append((text, len(history)))
        return "嗯，我在听。"


def test_local_reply_speaks_and_keeps_history(tmp_path):
    config = Config(root=tmp_path)
    config.emotion = EmotionConfig(mode="affect")
    config.llm = LLMConfig(enabled=True, history_turns=2)
    engine = AvatarEngine(config)
    engine.local_chat = StubLocal()
    spoken = []

    async def begin():
        spoken.append(("begin",))

    async def feed(text):
        spoken.append(("feed", text))

    async def end():
        spoken.append(("end",))

    engine.begin_utterance = begin
    engine.feed_text = feed
    engine.end_utterance = end

    first = asyncio.run(engine.local_reply("今天想你了"))
    assert first["ok"] is True
    assert first["chars"] == len("嗯，我在听。")
    assert engine.local_chat.seen == [("今天想你了", 0)]
    assert [kind for kind, *_ in spoken] == ["begin", "feed", "end"]

    asyncio.run(engine.local_reply("嗯？"))
    assert engine.local_chat.seen[-1][1] == 2  # one full exchange retained


def test_local_reply_disabled(tmp_path):
    config = Config(root=tmp_path)
    config.emotion = EmotionConfig(mode="affect")
    config.llm = LLMConfig(enabled=False)
    engine = AvatarEngine(config)
    result = asyncio.run(engine.local_reply("hi"))
    assert result["ok"] is False
    assert "disabled" in result["error"]
