# Feiyu → Piper distillation

目标：用 Feiyu CosyVoice `gentle` 生成中文语料，微调 Piper，得到实时、可对齐的肥鱼声线。

## 数据

- 生成器：`training/generate_corpus.py`
- 语料文本：`training/corpus.txt`
- 数据输出：`data/voice_training/cosyvoice/`
  - `wav/*.wav`：22050Hz、单声道、16bit
  - `metadata.csv`：`id|text`
  - `accepted.jsonl` / `rejected.jsonl`：校验结果，可断点续跑
- 当前第一轮：480 条，约 30 分钟
- 校验规则：采样率、声道、位深、时长、每字时长、RMS、峰值、削波率
- 失败或中断的样本不会进入 `metadata.csv`，也不会保留 WAV

继续生成：

```powershell
.venv\Scripts\python.exe dsh-vts-avatar\training\generate_corpus.py --target-minutes 60
```

## 训练环境（WSL Ubuntu 24.04）

- Python 3.10.22（uv 安装）
- piper-train：`/root/piper-train-src/src/python`
- venv：`/root/piper-train-src/src/python/.venv`
- torch 2.14.1 + cu130
- PyTorch Lightning 1.7.7
- torchmetrics 0.11.4
- NumPy 1.26.4
- setuptools 80.10.2
- espeak-ng + `cmn` 音素

安装过的基本命令：

```bash
uv venv --python 3.10.22 .venv
uv pip install --python .venv/bin/python "torch==1.13.1+cu117" ...
uv pip install --python .venv/bin/python "cython<1" "piper-phonemize~=1.1.0" \
  "librosa>=0.9.2,<1" "pytorch-lightning~=1.7.0" "onnxruntime>=1.11.0"
uv pip install --python .venv/bin/python "setuptools<81" "torchmetrics==0.11.4" "numpy==1.26.4"
bash build_monotonic_align.sh
```

实际最终 torch 被依赖解析为 2.14.1+cu130，可用。

## 必要补丁

1. 加载 checkpoint 时：

```bash
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

2. `piper_train/vits/lightning.py` 增加：

```python
def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
    scheduler.step()
```

3. `piper_train/export_onnx.py` 的 `torch.onnx.export` 增加：

```python
dynamo=False,
```

4. NumPy 必须 1.x，2.x 会触发 `np.Inf` 报错。

## 预处理

```bash
cd /root/piper-train-src/src/python
. .venv/bin/activate
python -m piper_train.preprocess \
  --language cmn \
  --input-dir /mnt/c/.../dsh-vts-avatar/training/dataset \
  --output-dir /mnt/c/.../dsh-vts-avatar/training/preprocessed \
  --dataset-format ljspeech \
  --single-speaker \
  --sample-rate 22050
```

数据集目录必须是：

```text
dataset/
  metadata.csv
  wav/*.wav
```

注意：Piper 要求目录名是 `wav`，不是 `wavs`。

## 微调

基线 checkpoint：

```text
training/checkpoints/huayan-medium/epoch=3269-step=2460540.ckpt
```

命令：

```bash
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
python -m piper_train \
  --dataset-dir /mnt/c/.../training/preprocessed \
  --accelerator gpu --devices 1 \
  --batch-size 4 --precision 32 \
  --validation-split 0 --num-test-examples 0 \
  --max-epochs 3325 --checkpoint-epochs 5 \
  --max-phoneme-ids 400 --limit-train-batches 120 \
  --resume_from_checkpoint /mnt/c/.../checkpoints/huayan-medium/epoch=3269-step=2460540.ckpt \
  --default-root-dir /mnt/c/.../training/run2
```

## 导出

```bash
python -m piper_train.export_onnx \
  /path/epoch=XXXX.ckpt \
  /path/feiyu_distill_v2.onnx
cp /path/training/preprocessed/config.json /path/feiyu_distill_v2.onnx.json
```

## 验收

- 同一句时长相对原 huayan 偏差 ≤10%
- ONNX 可被现有 `PiperTTS` 加载
- 音素 span 数正常
- 试听没有明显吞字/重复/噪声
- 通过后才替换 `config.toml` 的 `voice_model`

## 已测结果

- v1：30 epoch / 300 step，`loss_gen_all≈38.37`，导出成功
- v1 同一句：原 huayan 3.994s，蒸馏版 6.478s
- 结论：训练量不足，时长预测没收敛，不接入正式链路
- v6：从 v5 继续到 epoch 3349，导出 `models/feiyu_distill_v6.onnx`
- v6 混合时间轴：原 huayan 4.087s，蒸馏输出 4.087s，比例 1.000
- `DistilledPiperTTS` 已接入 bridge，配置指向 v6
- 仍需：语料扩到 60 分钟、人工试听、VTS 在线验收
