# dsh × VTube Studio 交接摘要

生成时间：2026-11-04  
仓库：`C:\Users\27221\Documents\deepseek-harness\default-workspace\dsh-vts-avatar`  
当前提交：本文件所在提交（HEAD，尚未推送）；代码功能基线为已推送的 `1008f33`  
测试：29 个通过

## 1. 当前运行状态

- VTS API：`ws://127.0.0.1:8002`，模型 `蒂亚圣诞树树`，model_id `8cb985f5b9514021a5c5d941b482e57b`
- 桥接：`ws://127.0.0.1:8765`，当前加载 **v8 final 原生 Piper + 后处理**
- NDI：`BLOODPEACH (Live2D Camera NDI5)`
- Feiyu 服务：9880/9881 仍在监听
- 音频：Windows 默认输出是 **VB-Cable**，用户故意静音；不要自动切换到 Realtek
- 当前没有 active goal；声线目标仍未完成

## 2. 声线 / 蒸馏状态

### 语料

- 路径：`data/voice_training/cosyvoice/`
- 已接受：**637 条，39.8 分钟**
- 生成方式：CosyVoice3 `gentle` **纯 TTS**，没有 RVC
- 逐条校验：22050Hz、单声道、16bit、时长、每字时长、RMS、峰值、削波
- 坏样本不进入 `metadata.csv`；支持断点续跑
- 目标 1–2 小时未完成；后续需要继续补语料

### 模型

- 正式：`models/feiyu_distill_v8_final.onnx`，直接作为 Piper 声线
  - `length_scale = 1.20`
  - `postprocess = true`
  - `noise_scale = 0.50`
  - `noise_w_scale = 0.70`
  - 不再走 reference-timed atempo 包装
  - 这是用户选定的第 6 版调参
- 备份：`models/feiyu_distill_v7.onnx`
- v8 到达 `max_epochs=3469`，最后保存 checkpoint 为 epoch 3459
- v8 已切换为正式声线

### 参考时间轴方案

`src/dsh_vts_avatar/tts/distill.py`：

- huayan 提供目标时长
- 蒸馏模型提供音色
- 必要时用 ffmpeg `atempo` 拉伸蒸馏音频
- 同步缩放蒸馏模型自己的音素 spans
- v7 10 句：平均延迟 415ms，最大 490ms，时长误差最大约 11.6ms
- v8 final 10 句：平均 423ms，最大 521ms，误差最大约 35ms

## 3. 已完成的关键功能

### 基础

- VTS 连接、认证、token、模型识别
- Piper TTS、音素对齐、播放时钟
- 五元音 + MouthOpen 注入
- 头部最小 jerk、动作通道互斥、倾听注视、回正
- NDI 单张抓图
- dsh Host 插件基础版
- 29 个测试

### 表情 / 默认表情

- 6 个显式表情预设 + 关键词情绪
- 默认 neutral 已按模型映射修正：
  - `MouthSmile = 0.50`
  - `Brows = 0.575`
  - `EyeOpen = 0.55`
- 情绪衰减后回到 neutral 的阈值：`preset_intensity < 0.08`
- 修复了“说完话又变回不高兴”的问题

### 嘴型 P1

- 相邻元音：45ms 预备 / 30ms 延续 / 0.50 混合
- 双唇音 `/p/ /b/ /m/`：attack 8ms / release 15ms
- 参数回读显示双唇音 `MouthOpen` 最大值约 **0.036**（修复前 0.241）
- 60 秒连续说话（抬头视角）已抓 20 帧，嘴持续开合：
  - `data/audit/p1_continuous_headup/sheet.jpg`
- 双唇音/元音画面：
  - `data/audit/p1_bilabial_final/sheet.jpg`
- `/a/ /i/ /u/` 已单独抬头抓帧：
  - `data/audit/p1_vowels_isolated/sheet.jpg`
  - A 明显张大；I 较横向；U 较圆，但模型嘴部小，I/U 幅度偏弱

### 音频 / 设备

- 修复 `sounddevice.default.device` 是 `_InputOutputPair`、不是 tuple/list 导致的设备选择失败
- 修复 `query_devices(None)` 返回 `DeviceList`、旧代码却 `.get("name")` 导致的 TTS 崩溃
- 已撤回“自动切 Realtek”的逻辑，尊重用户设置的 VB-Cable 默认输出

### 调试 API

桥接支持：

```json
{"type": "parameters", "request_id": "..."}
```

返回当前 `params`、`phoneme`、`viseme`、`sample`。

```json
{"type": "head_pose", "pitch": 12, "seconds": 15}
```

临时抬头，用于检查被遮挡的嘴型；不会永久改变待机姿态。

## 4. 已知问题 / 未完成

1. 语料只有 39.8 分钟，目标 1–2 小时未达成
2. v8 已切换，但电音/沙哑/吞字的主观验收还没通过
3. 音色主观质量还没有用户确认
4. 没有 RVC；当前是 CosyVoice3 gentle 纯 TTS 蒸馏
5. P1 元音视觉已验收：A 清楚，I/U 可区分但幅度偏弱；下一步要么调 I/U，要么接受当前效果
6. P2 没有 LLM 结构化情绪，仍主要靠关键词
7. P3 没有身体轴、预测动作队列、运动条件校准
8. P4 本地 8B 待机导演未接入，当前没有确认可用的 8B 模型/运行时
9. P5 没有正式 A/V 偏移、barge-in、60 分钟稳定性指标
10. P6 只有单张 NDI 抓图，没有连续闭环/VLM 自检
11. P7 没有设置页、校准向导、一键启动、安装器
12. P8 全链路验收未做

## 5. 关键文件

- 计划：`项目总计划.md`
- 训练记录：`training/README.md`
- 蒸馏 TTS：`src/dsh_vts_avatar/tts/distill.py`
- 嘴型/表情/头部：`src/dsh_vts_avatar/performance/mixer.py`
- 音频时钟：`src/dsh_vts_avatar/audio/engine.py`
- 桥接协议：`src/dsh_vts_avatar/bridge/server.py`
- 桥接引擎：`src/dsh_vts_avatar/bridge/engine.py`
- 配置：`config.toml`
- 训练数据：`E:\dsh-vts-avatar-training`
- 语料：`data/voice_training/cosyvoice/`

## 6. 下一步建议

按大纲优先级：

1. P1 元音：A 已清楚；决定是否增强 I/U 后再收 P1
2. 再收 P2 的结构化情绪
3. 语音方向：用户确认音色后，继续补语料到 60 分钟，再训练 v9
4. P4 8B 需要先确认模型/运行时，否则继续阻塞
5. 最后做 P8 全链路验收

## 7. 运行命令

```powershell
cd C:\Users\27221\Documents\deepseek-harness\default-workspace
.\.venv\Scripts\python.exe -m dsh_vts_avatar --config dsh-vts-avatar/config.toml serve
```

VTS 必须先启动，API 端口为 8002。
