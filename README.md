# dsh VTS Avatar

让 dsh 通过 VTube Studio 的参数注入，驱动一个持续在线的 Live2D 角色：

- 中文 Piper TTS，音素时间轴驱动嘴型；
- `VoiceA/I/U/E/O` + `MouthOpen/JawOpen/MouthFunnel/MouthPucker`；
- 表情、眨眼、头部动作；
- 接管/离开淡入淡出；
- NDI 自看：AI 可以抓取当前画面看自己；
- dsh Host 插件；
- 预留本地 8B 待机导演接口。

完整路线图见 [项目总计划.md](项目总计划.md)。

## 目录

```text
src/dsh_vts_avatar/     Python 桥接服务
tests/                  Python 单元测试
index.js                dsh Host 插件
locale/                 插件中英文名称/描述
config.toml             本地配置（不含 token）
models/                 Piper 语音文件（本地，不提交）
data/                   token、contract、NDI 截图（本地，不提交）
```

## 环境

- Windows 10/11
- Python 3.11+
- Node.js 20+
- VTube Studio，启用 API（默认 `127.0.0.1:8001`）
- NDI 输出（可选，用于自看）
- Piper 中文语音：`zh_CN-huayan-medium.onnx` + `.onnx.json`

## 安装

```powershell
uv venv --python 3.13 .venv
uv pip install --python .venv\Scripts\python.exe -e .
```

下载语音：

```powershell
.venv\Scripts\python.exe -m piper.download_voices --download-dir dsh-vts-avatar\models zh_CN-huayan-medium
```

如果 HuggingFace 下载失败，也可以把模型和配置手动放到 `dsh-vts-avatar/models/`。

## VTS 设置

1. 打开 VTube Studio，加载模型；
2. 设置 → API：允许插件 API，端口 `8001`；
3. 关闭麦克风口型同步；
4. 关闭自动眨眼（当前由程序控制）；
5. 保留自动呼吸；
6. 如使用自看，开启 NDI 输出；
7. 确认模型映射：`MouthOpen`、`MouthSmile`、`VoiceA...`、`FaceAngle...`。

## 运行

```powershell
# 第一次：探测当前模型参数，生成 data/contract.json
.venv\Scripts\python.exe -m dsh_vts_avatar --config dsh-vts-avatar\config.toml probe

# 有声运行桥接服务
.venv\Scripts\python.exe -m dsh_vts_avatar --config dsh-vts-avatar\config.toml serve

# 直接测试一句
.venv\Scripts\python.exe -m dsh_vts_avatar --config dsh-vts-avatar\config.toml demo "你好，我是 dsh。"

# 只读录制一个用户自定义表情（先停桥接，避免写入抢参数）
.venv\Scripts\python.exe -m dsh_vts_avatar --config dsh-vts-avatar\config.toml record --name "委屈2" --seconds 5
```

同步采集参数 + NDI 帧（可选麦克风）：

```powershell
.venv\Scripts\python.exe -m dsh_vts_avatar --config dsh-vts-avatar\config.toml record --name "委屈2" --seconds 5 --ndi --audio
```

OBS 录制工作流：OBS 录 NDI 30/60fps + 桌面音频，Python 只跑参数录制（不加 `--ndi`）。  
开始时会先响一声 1kHz 提示音，再说“开始，表情名”，参数文件里会写入 `phase: "cue"` 和 `cue_t`；后期用 OBS 音频里的提示音对齐视频和参数。

`record` 会先发一帧中性参数，然后暂停所有写入，只读取 VTS 的输入参数和 Live2D 参数；用户听到语音提示后做表情。结束后把 session 保存到 `data/demos/<时间>_<名字>/`：`params.jsonl`、`frames/`、`frames.jsonl`、可选 `audio.wav`、`manifest.json`。

桥接 WebSocket：`ws://127.0.0.1:8765`。

## dsh 插件

插件包根目录就是本目录，`package.json` 已包含 `dsh.bundle.patch`。  
在 dsh 的插件管理里安装本目录后，插件会监听 `agent/assistant-stream`，把流式回复转发给桥接。

可用工具：

- `vts_avatar_look`：抓一张 NDI 当前画面，返回路径和运动指标；
- `vts_avatar_say`：让角色念一句测试文本；
- `vts_avatar_emote`：显式设置表情（预设或已录制的自定义表情名）；
- `vts_avatar_record`：只读录制用户做出的自定义表情，保存参数 delta 和范围。

## 不提交的内容

- `data/vts_token.json`：VTS 插件 token；
- `models/*.onnx`：语音模型，体积大；
- `data/ndi/`：本地截图；
- `data/contract.json`：旧版单模型探测结果（兼容用）；
- `data/profiles/`：**每个模型独立**的参数/校准档案，例如 `data/profiles/1f951c3d0c67414097c15b241322c75c.json`；
- `data/demos/`：用户示范的原始 session（参数、NDI 帧、音频），留给以后 8B 学习；不提交；
- `.venv/`、缓存、日志。

## 状态

当前是可用基础版：语音、嘴型、最小 jerk 头部、接管/离开淡入淡出、NDI 抓图、dsh 插件、6 个表情预设。  
自动语义表情、完整 VB 参数层、身体/眼神/动作队列、本地 8B 待机导演仍在计划中。
