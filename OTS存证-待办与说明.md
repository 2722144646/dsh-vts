# OTS 存证：待办与说明（之后做）

- 记录时间：2026-11-06
- 状态：**已盖时间戳，待完成比特币锚定**（不是没做，是等日历聚合）
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

## 2. 差的那一步

等几小时（通常 2–6 小时，取决于日历的聚合周期），在 `E:\avatar_feel` 下执行：

```powershell
cd E:\avatar_feel
.venv\Scripts\ots.exe upgrade evidence\manifest.sha256.ots
```

**成功的样子**：

```text
Success! Bitcoin block 8xxxxx attests existence as of 2026-11-06 XX:XX:XX UTC
```

**还是 pending 的样子**：

```text
Calendar https://...: Pending confirmation in Bitcoin blockchain
Failed! Timestamp not complete
```

pending 就再等，不用改任何东西。

成功之后：

```powershell
git add evidence/manifest.sha256.ots
git commit -m "Complete OTS Bitcoin attestation"
git push origin main
```

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

1. **升级前不要重新生成 `manifest.sha256`**。一旦清单内容变了，哈希就对不上，旧证明作废。
2. **每次发版再存证一次**：重新生成清单 → `ots stamp` → 等确认。多次存证形成时间链。
3. 本机 `.venv` 里的 `python-bitcoinlib` 打过 Windows 补丁（找不到 OpenSSL 时用 `MagicMock` 占位），
   只影响本机工具，不影响证明本身。换机器重装 `opentimestamps-client` 时如果 import 崩溃，照此处理。
4. 时间戳证明的是“内容在某时刻前已存在”，不证明“你是谁”；作者身份靠仓库提交、版权声明和软著登记补。

---

## 5. 之后要做的事（清单）

- [ ] 等日历确认后执行 `ots upgrade`
- [ ] 确认输出包含 `Success! Bitcoin block ...`
- [ ] 提交并推送 `.ots`
- [ ] 下次发版时重新 `ots stamp`，保持时间链
- [ ] 若升级长期失败（日历异常），改用国内可信时间戳服务或区块链存证平台重做一次

---

## 6. 成本

- OpenTimestamps：0 元，不暴露源码，只提交哈希
- 全球日历公开可用，不需要注册账号
- 唯一成本是等待时间
