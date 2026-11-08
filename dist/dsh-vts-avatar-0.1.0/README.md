# dsh VTS Avatar

让 dsh 通过 VTube Studio 的参数注入，驱动一个持续在线的 Live2D 角色：说话、嘴型、表情、
待机动作、情绪记忆，以及可选的本地模型。

> **没有一键安装器，也没有开机自启**（2026-11-08 的决定）。使用就是按下面的顺序手动启动几样东西。

## 快速开始

```powershell
cd E:\dsh-vts-avatar\project
powershell -ExecutionPolicy Bypass -File scripts\install.ps1   # 建 .venv、装依赖
powershell -ExecutionPolicy Bypass -File scripts\start.ps1     # 启动桥接
```

然后在 dsh 里启用 `dsh-vts-avatar` 插件即可。

打包给别人：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\package.ps1
# 产出 dist\dsh-vts-avatar-0.1.0.zip（不含 models / data / .venv）
```

---

## 1. 需要先装好

| 组件 | 用途 | 必须 |
|---|---|---|
| Windows 10/11 | 运行环境 | ✅ |
| Python 3.11+（本项目用 3.13） | 桥接 | ✅ |
| VTube Studio + Live2D 模型 | 显示角色、接收参数 | ✅ |
| Piper 中文语音（`models/` 下） | 让她出声 | ✅ |
| Node.js 20+ | 装 dsh 插件 | ✅ |
| Ollama（可选） | 待机导演 / 私有对话 | 可选 |

安装桥接：

```powershell
cd E:\dsh-vts-avatar\project
uv venv --python 3.13 .venv
uv pip install --python .venv\Scripts\python.exe -e .
```

## 2. 启动顺序

```text
① 打开 VTube Studio（加载模型，开启 API）
② 启动桥接：.venv\Scripts\python.exe -m dsh_vts_avatar --config config.toml serve
③ 在 dsh 里启用 dsh-vts-avatar 插件（它会自己连 ws://127.0.0.1:8765）
```

VTS 的 API 端口会自动探测（常见 8002 / 8003）；不需要手写端口。

### VTube Studio 里要设置的

1. 设置 → API：允许插件；允许 dsh 连接；
2. **关闭麦克风口型同步**（嘴型由程序写）；
3. **关闭自动眨眼**（眨眼由程序控制）；
4. 自动呼吸可以保留；
5. 想用「AI 看自己」就开 **NDI 输出**；
6. 确认模型有这些参数：`MouthOpen`、`MouthSmile`、`VoiceA/I/U/E/O`、`FaceAngleX/Y/Z`、`EyeOpenLeft/Right`、`BrowLeftY/BrowRightY`、`BodyAngleX/Y/Z`（后三者可选）。

## 3. 日常使用

插件连上后，dsh 的回复会自动让她说出来，不需要额外操作。额外工具：

| 工具 | 作用 |
|---|---|
| `vts_avatar_look` | 抓一张 NDI 画面，AI 看自己 |
| `vts_avatar_say` | 手动念一句测试文本 |
| `vts_avatar_feel` | 描述情绪（带权重、置信度、原因） |
| `vts_avatar_emote` | 强制一个表情预设 |
| `vts_avatar_explain` | 记录“解释有没有被接受” |
| `vts_avatar_private` / `recall` | 私密条目、回忆 |
| `vts_avatar_diary` | 写当天的日记（第一人称） |
| `vts_avatar_relationship` | 记录关系事件（12 种） |
| `vts_avatar_switch_user` | 切换用户（各自独立数据） |
| `vts_avatar_write_idle_actions` | 给某个待机档位写一批动作 |
| `vts_avatar_local_say` | 交给本地模型说（文本不进 dsh 上下文） |
| `vts_avatar_transcript_window` | 打开本地聊天记录窗口 |
| `vts_avatar_record` | 只读录制用户示范的表情 |

## 4. 本地模型（可选）

默认**不加载**任何模型；显存占用为 0。

- 模型目录：`E:\ollama\models`（别放 C 盘）
- 可用模型：`qwen3:8b`（导演候选）、`qwen2.5:7b`（回退）、
  `huihui_ai/qwen3-abliterated:8b`（私有对话）
- 待机导演：`config.toml` 里 `[llm] director_enabled = true` 打开，
  每 6–12 秒让 `qwen3:8b` 挑一个动作；关掉就退回规则池
- 私有对话：`vts_avatar_local_say`，文本只在本机生成并落盘到 `data/chat/<user>.jsonl`；
  置顶窗口可看原文（`vts_avatar_transcript_window`）
- 回复长度与人设在 `[llm] persona`；想更短就调小 `num_predict`（`local_chat.py`）

## 5. 音频

- 输出设备：默认跟随 **WASAPI 默认输出**（3.5mm 耳机也走这个端点，不会误出到扬声器）
- 延迟校准：启动时测一次（三下很轻的点击），结果缓存在 `data/audio_latency.json`；
  **缓存新鲜时不再重测**，所以正常重启是安静的
- 缓存失效（换设备/删了文件）会重新测；想强制重测就删掉那个 json 再启动
- 每段音频首尾有 4ms 淡入淡出，避免切块爆音

## 6. 数据与隐私

```text
data/emotion/<user>.jsonl     情绪事件
data/relationship/<user>/     关系状态与事件
data/diary/<user>/            日记（模型自己写的）
data/private/<user>.jsonl     私密条目
data/chat/<user>.jsonl        本地模型的对话原文
data/presence/idle_library.json  模型写的待机动作批次
data/audio_latency.json       音频延迟缓存
```

这些都在本机，不会上传。默认用户是 `default`。

## 7. 常见问题

| 现象 | 处理 |
|---|---|
| 插件说 “bridge is not connected” | 先确认 ② 的 `serve` 在跑；等 5–10 秒让插件重连 |
| VTS 连不上 / `WinError 1225` | VTube Studio 没开，或 API 没允许 |
| 声音出在扬声器 | 检查 Windows 默认输出是不是耳机；桥接按 WASAPI 默认走 |
| 嘴型慢半拍 | 删 `data/audio_latency.json` 重启，重新校准 |
| 听到三下轻点击 | 正常，是延迟校准；缓存存在时不会响 |
| 待机导演把显存占满 | `[llm] director_enabled = false`，或 `ollama stop` 手动卸载 |
| NDI 自看不到画面 | VTS 设置里开启 NDI 输出 |

## 8. 测试

```powershell
cd E:\dsh-vts-avatar\project
.venv\Scripts\python.exe -m pytest -q tests
```

当前 **128 passed**。

## 9. 仓库里不提交的东西

`data/`（token、聊天、日记、关系、截图）、`models/*.onnx`、`.venv/`、日志。
`config.toml` 里不含 token，可以提交；`data/vts_token.json` 不要提交。
