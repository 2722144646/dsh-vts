# OTS 存证：待办与说明（之后做）

- 记录时间：2026-11-06
- 状态：**已完成**。2026-11-07 09:15 执行 `ots upgrade` 成功，拿到 4 条比特币区块证明（block 974817 / 974819 / 974839 / 974847），已推送到 GitHub（commit `63816b6`）。
- 关联仓库：`E:\avatar_feel`
- 关联文档：`项目总计划.md`、`著作权登记-材料清单与待办.md`

---

## 1. 已经做了什么

2026-11-06 用 OpenTimestamps 给 `E:\avatar_feel\evidence\manifest.sha256` 盖了时间戳：

```text
被存证文件：evidence/manifest.sha256
内容哈希：  c42e95076da9172cb9018b17726d6a0c2c078e8ba388e1d3adc32bc043f7f826
证明文件：  evidence/manifest.sha256.ots（约 1.1KB）
提交日历：  a/b.pool.opentimestamps.org、finney.eternitywall.com、btc.calendar.catallaxy.com
提交记录：  git commit 543ae28（已推送到 GitHub）
当前状态：  PendingAttestation × 4（等待聚合进比特币区块）
```

`manifest.sha256` 是 25 个文件的哈希清单，所以它实际锁定了 `c6e428a` 那个提交的整份内容。

---

## 2. 升级结果（2026-11-07 已完成）

```text
$ ots upgrade evidence/manifest.sha256.ots
Got 1 attestation(s) from https://finney.calendar.eternitywall.com
Got 1 attestation(s) from https://alice.btc.calendar.opentimestamps.org
Got 1 attestation(s) from https://bob.btc.calendar.opentimestamps.org
Got 1 attestation(s) from https://btc.calendar.catallaxy.com
Success! Timestamp complete
```

写入证明的区块：

```text
BitcoinBlockHeaderAttestation(974817)
BitcoinBlockHeaderAttestation(974819)
BitcoinBlockHeaderAttestation(974839)
BitcoinBlockHeaderAttestation(974847)
```

`.ots` 从 1155 字节（pending）变成 5369 字节（完整）。最早锚定在 block 974817。

之后**每次发版**要再存证时，重复这个流程：重新生成清单 → `ots stamp` → 等几小时 → `ots upgrade`。

---

## 3. 怎么验证（任何时候都能做）

```powershell
cd E:\avatar_feel

# 离线查看证明结构
.venv\Scripts\ots.exe info evidence\manifest.sha256.ots

# 完整验证（升级完成后；需要能访问比特币节点或公开 API，客户端参数见 --help）
.venv\Scripts\ots.exe verify evidence\manifest.sha256.ots
```

第三方验证不需要信任本仓库：`.ots` 文件自带到区块头的 Merkle 路径，任何 OTS 实现都能独立核对。

---

## 4. 注意事项

1. **不要重新生成 `manifest.sha256` 之后继续用这份旧 `.ots` 当证明**。内容变了哈希就对不上；
   想再存证就重新生成清单并 `ots stamp` 一次，旧证明留作历史版本。
2. **每次发版再存证一次**：重新生成清单 → `ots stamp` → 等确认 → `ots upgrade`。多次存证形成时间链。
3. 本机 `.venv` 里的 `python-bitcoinlib` 打过 Windows 补丁（找不到 OpenSSL 时用 `MagicMock` 占位），
   只影响本机工具，不影响证明本身。换机器重装 `opentimestamps-client` 时如果 import 崩溃，照此处理。
4. 时间戳证明的是“内容在某时刻前已存在”，不证明“你是谁”；作者身份靠仓库提交、版权声明和软著登记补。

---

## 5. 之后要做的事（清单）

- [x] 执行 `ots upgrade`（2026-11-07 09:15）
- [x] 确认输出 `Success! Timestamp complete`，区块 974817–974847
- [x] 提交并推送 `.ots`（avatar_feel commit `63816b6`）
- [ ] 下次发版时重新 `ots stamp`，保持时间链
- [ ] 若之后升级长期失败（日历异常），改用国内可信时间戳服务或区块链存证平台重做一次

---

## 6. 成本

- OpenTimestamps：0 元，不暴露源码，只提交哈希
- 全球日历公开可用，不需要注册账号
- 唯一成本是等待时间
