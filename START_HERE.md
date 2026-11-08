# dsh-vts-avatar 接手文档（START HERE）

- 位置：`E:\dsh-vts-avatar`（2026-11-06 从 C 盘迁出）
- 先读：`用户原话与共识.md`（最高优先级）→ `计划总览.md`（当前状态）
- 回滚标签：`v0.5.0-emotion-baseline`（情绪重构前）

---

## 0. 一分钟了解

```text
DSH 回复
  → dsh Host 插件（project/index.js）
  → Python 桥接（ws://127.0.0.1:8765）
  → Piper TTS + 音素时间轴
  → 播放时钟
  → VTS 参数注入
  → 嘴型 / 表情 / 头部 / 眼神 / NDI 自看
```

当前状态（2026-11-08）：S1–S5 全部落地；情绪 L1/L2/L3 收口；待机 5 分钟验收通过；
VTS 端设置逐项验收通过；测试 **129 passed**。验收记录见 `项目总计划.md` §11.7。
详细路线图见 `计划总览.md`。

---

## 1. 目录

```text
E:\dsh-vts-avatar\
  START_HERE.md                 <- 本文件
  计划总览.md                    <- 当前状态与下一步（权威）
  用户原话与共识.md               <- 用户原话 + 已定结论（最高优先级）
  项目总计划.md                  <- 技术方案、历史、边界(§10)、模型可插拔(§11)
  情绪系统重构计划.md             <- 情绪系统 Phase 0–4 实现记录
  情绪记忆-初步计划.md            <- 情绪记忆设计、已知问题、实施记录
  OTS存证-待办与说明.md          <- 时间戳操作
  著作权登记-材料清单与待办.md    <- 软著材料（暂缓）
  project\                      <- 代码仓库（git 完整历史）
    .venv\                      <- Python 3.13 虚拟环境
    src\dsh_vts_avatar\
    tests\
    models\ data\               <- 大文件，不入库
  backups\                      <- 历史备份
```

独立库：`E:\avatar_feel`（AGPL-3.0 + 商业授权），与本项目同源。

---

## 2. 环境与启动

- Windows，Python 3.13；虚拟环境 `E:\dsh-vts-avatar\project\.venv`
- VTS API 端口会漂移（最近一次 8003），启动时自动探测 8001–8010
- 桥接：`ws://127.0.0.1:8765`

```powershell
cd E:\dsh-vts-avatar\project
.\.venv\Scripts\python.exe -m dsh_vts_avatar --config config.toml serve
```

期望日志：`VTS ready` / `bridge listening on ws://127.0.0.1:8765`。

测试：

```powershell
cd E:\dsh-vts-avatar\project
.\.venv\Scripts\python.exe -m pytest -q tests
```

### 2.1 音频设备

**当前走 WASAPI 默认输出**（`device = "default"`，3.5mm 耳机也走这个端点）。
2026-11-07 实测：默认输出 = 耳机 (Realtek(R) Audio) [MME]，桥接日志 `device=4 … latency=96.0ms`。

默认设备会随 Windows 设置变化（换耳机、插拔设备都会切），这通常正是想要的；
如果要**钉死**某个设备，再跑枚举、把 `config.toml` 的 `device` 改成索引：

```powershell
.\.venv\Scripts\python.exe -c "import sounddevice as sd; [print(i,d['name'],d['hostapi'],d['max_output_channels']) for i,d in enumerate(sd.query_devices()) if d['max_output_channels']>0]"
```

延迟缓存 `data/audio_latency.json` 有效时启动不重测；换设备或删掉缓存后会重新校准。

---

## 3. 当前能力

| 能力 | 状态 | 证据 |
|---|---|---|
| 嘴型 / 音素时间轴 | ✅ | `data/audit` |
| 表情预设 + `vts_avatar_emote` + `[emotion:...]` | ✅ | `data/expression_shots` |
| 情绪融合（多来源、无硬覆盖、manual 0.5 票） | ✅ Phase 0–4 | `tests/test_affect_fusion.py` |
| `vts_avatar_feel`（labels/cause/why/appraisal） | ✅ | `tests/test_engine_feel.py` |
| 情绪记忆一期（事件 + 原因 + mood 回灌） | ✅ | `data/emotion/episodes.jsonl`、`tests/test_emotion_memory.py` |
| 模型可插拔 | 🟡 本地 Ollama 已接（私有对话 + 可选导演）；多后端抽象未做 | `项目总计划.md` §11 |
| 关系层 / 隐私 / 日记 / 解释采信 | ⬜ 二期 | `情绪记忆-初步计划.md` |
| OTS 存证 | ✅ | `avatar_feel/evidence/manifest.sha256.ots` |
| 软著登记 | ⏸ 暂缓 | `著作权登记-材料清单与待办.md` |

---

## 4. 已知缺口

- 蓝牙链路 A/V 同步没标定（别猜 `av_offset`，先跑校准）
- 蒂亚没有真正的 angry 绑定，angry 实际是“气鼓鼓/不满”
- NDI 只能慢速校验：当前模型嘴部被手挡住，不能靠 NDI 读唇
- 情绪记忆的 doubted/suppressed/accepted 状态迁移未做（等解释事件）
- 关系层、AI 隐私、日记未做

---

## 5. 绝对不要做的事

1. 不要重新训练 v8 或换声线，除非明确要求
2. 不要随手改 `av_offset` 猜同步
3. 不要重新引入 80ms 全局句间静音
4. 不要恢复 8ms/15ms 双唇快速过渡
5. 不要在没有备份时改 `_append_timeline`
6. 不要同时开两个桥接实例抢 8765
7. 不要把 `data/vts_token.json` 提交到 Git
8. 不要假设音频设备索引固定
9. 不要把 Realtek 的 461.9ms 直接套到蓝牙
10. 不要新建第二个情绪控制插件 / 第二个 VTS writer
11. 不要往仓库里加破甲提示词、成人内容或“解锁包”（见 `项目总计划.md` §10）

---

## 6. 回滚

```powershell
cd E:\dsh-vts-avatar\project
git checkout v0.5.0-emotion-baseline
```

只关 affect 路径（代码不回退）：

```toml
[emotion]
mode = "legacy"
```

备份恢复步骤见 `backups\dsh-vts-avatar_v0.5.0-emotion-baseline_20261106-134033\RESTORE.md`。
