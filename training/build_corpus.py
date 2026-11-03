"""Build a Chinese corpus for CosyVoice -> Piper distillation."""

from __future__ import annotations

import re
from pathlib import Path

OUT = Path("dsh-vts-avatar/training/corpus.txt")

DOCS = [
    Path("dsh-vts-avatar/项目总计划.md"),
    Path("dsh-vts-avatar/README.md"),
    Path(r"E:\feiyu\00_INDEX.md"),
]

# Everyday sentences fill prosody/phoneme gaps that technical docs do not
# cover. Keep them short; the generator rejects bad takes independently.
EVERYDAY = [
    "今天天气很好，我们一起出去走走吧。",
    "早上好，昨晚睡得怎么样？",
    "我刚泡了一杯热茶，你要不要也来一杯？",
    "这个周末你有什么安排吗？",
    "我最近在学做菜，味道还不错。",
    "别着急，慢慢说，我在听。",
    "你先休息一下，剩下的交给我。",
    "这件事情我想听听你的想法。",
    "今天的工作终于做完了，感觉轻松多了。",
    "我们下午去公园散步，好吗？",
    "外面的风有点大，记得多穿一件衣服。",
    "我刚刚看到一条很有意思的消息。",
    "你喜欢晴天还是下雨天？",
    "这家店的面很好吃，下次带你去。",
    "我有点累了，想早点回家休息。",
    "你不用担心，一切都会好起来的。",
    "谢谢你一直陪着我。",
    "我们一起想办法，总会解决的。",
    "今天的月亮特别圆，你看到了吗？",
    "我在路上了，大概十分钟就到。",
    "先吃饭吧，工作可以等一会儿再做。",
    "你最近是不是没怎么休息好？",
    "我给你带了一点小礼物。",
    "别灰心，这次只是运气不太好。",
    "有什么不开心的事，可以跟我说。",
    "今天的咖啡有点苦，不过很香。",
    "我正在收拾房间，一会儿就弄完。",
    "晚上想吃什么？我来做。",
    "你说话的声音听起来有点疲惫。",
    "我们好久没有一起聊天了。",
    "外面开始下雨了，记得带伞。",
    "这个问题不急，明天再处理也可以。",
    "我刚刚没听清，你能再说一遍吗？",
    "你觉得这个颜色好看吗？",
    "我想把房间布置得温暖一点。",
    "今天路上有点堵，所以来晚了。",
    "你先忙，我在这里等你。",
    "这件事交给我，你放心吧。",
    "我准备去楼下买点水果。",
    "你笑起来的样子很好看。",
    "今天的晚饭我来准备。",
    "我们边吃边聊吧。",
    "你最近在追什么剧？",
    "这本书写得很有意思，推荐给你。",
    "我昨天梦到了一个很奇怪的地方。",
    "时间不早了，该休息了。",
    "明天还要早起，别熬夜。",
    "你能帮我拿一下那个杯子吗？",
    "谢谢你，今天辛苦你了。",
    "我们先从最简单的地方开始。",
    "别担心，我已经检查过了。",
    "这个方案还有几个地方要改。",
    "你方便的时候给我回个消息。",
    "我现在有点忙，晚点再聊。",
    "这件事我们明天再讨论。",
    "你今天看起来心情不错。",
    "我出去一下，很快就回来。",
    "记得按时吃饭，别总是忙工作。",
    "有你在，我就安心多了。",
    "今天的阳光照在身上很舒服。",
    "我们一起去看看新开的店吧。",
    "我想听听你的意见。",
    "这个问题没有你想的那么难。",
    "慢慢来，不用一次做完。",
    "你已经做得很好了。",
    "我们先确认一下时间。",
    "你到家以后告诉我一声。",
    "今天的会议开得有点久。",
    "我先去准备一下材料。",
    "你想喝点什么？我去拿。",
    "这个位置看起来不错。",
    "我们换个时间再约吧。",
    "我刚刚想到一个办法。",
    "你别急，我马上过来。",
    "今天的任务比想象中顺利。",
    "谢谢你提醒我。",
    "我一会儿给你打电话。",
    "你最近有空吗？",
    "我想请你帮个小忙。",
    "这件事情还需要再确认。",
    "我们先休息十分钟。",
    "你可以先试试看。",
    "不用客气，这是我应该做的。",
    "今天过得真快。",
    "我准备回家了。",
]


def clean(text: str) -> str | None:
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"`[^`]*`", "", text)
    text = re.sub(r"\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"[*#>_~]", "", text)
    text = re.sub(r"\s+", "", text)
    text = text.strip("，。！？；：、,.!?;:")
    if not (8 <= len(text) <= 40):
        return None
    if not re.search(r"[\u4e00-\u9fff]", text):
        return None
    if re.search(r"[A-Za-z0-9|：:+\-*=<>]", text):
        return None
    if text.count("，") + text.count("。") > 3:
        return None
    return text


def main() -> None:
    seen: set[str] = set()
    sentences: list[str] = []
    for item in EVERYDAY:
        cleaned = clean(item)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            sentences.append(cleaned)

    # Template sentences are not literature, but they provide broad initial /
    # final coverage and enough prosodic variety for a first voice model.
    times = ["今天", "明天", "昨天", "早上", "中午", "下午", "晚上", "周末"]
    people = ["我", "你", "他", "她", "我们", "大家", "朋友", "家人"]
    actions = [
        "想出去走走", "准备做一顿饭", "打算早点休息", "想去公园散步",
        "正在整理房间", "想喝一杯热茶", "要去看一场电影", "准备学习新东西",
        "觉得今天很充实", "想和你聊聊天", "需要休息一会儿", "打算买点水果",
        "正在等公交车", "想听听音乐", "准备明天的工作", "觉得天气很舒服",
        "想给家里打个电话", "打算周末出门", "正在收拾行李", "想试试新开的店",
        "想把房间打扫干净", "准备去超市买东西", "想早点回家", "打算看一会儿书",
        "正在学做一道新菜", "想出去晒晒太阳", "准备给朋友回消息", "觉得有点困了",
        "想安静地坐一会儿", "打算把衣服洗了",
    ]
    for when in times:
        for who in people:
            for action in actions:
                cleaned = clean(f"{when}，{who}{action}。")
                if cleaned and cleaned not in seen:
                    seen.add(cleaned)
                    sentences.append(cleaned)
                if len(sentences) >= 1500:
                    break
            if len(sentences) >= 1500:
                break
        if len(sentences) >= 1500:
            break

    # Enough candidates for a long run; the generator stops at target audio.
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(sentences) + "\n", encoding="utf-8")
    chars = sum(len(s) for s in sentences)
    print("sentences", len(sentences), "chars", chars, "estimated_minutes", round(chars / 4 / 60, 1))
    print("wrote", OUT)
    print("\n".join(sentences[:10]))


if __name__ == "__main__":
    main()
