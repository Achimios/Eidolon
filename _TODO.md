# 待办

## 下一步
- [ ] 

## 想法池（💡 有灵感再升级）
- [ ] 

## 感官层升级（👤2026-10-19「we shall upgrade the sense system a lot」）
- [ ] **乐句 / 小节结构**：下拍（每小节第一拍）、乐句边界（v2 粗剪里是手推的 6.4 s 格子：用户锚点 13.08 / 19.5 吸附到起音 13.112 / 19.507 推出）→ 自动检测（自相似矩阵 / 新颖度曲线）
- [ ] **每个起音的强度**写进 events（现在 threat 按帧取会落在冲击之前，要 ±3 帧取最大，见 animatic_v2.onset_strength）
- [~] **乐器分层**：✅ Demucs htdemucs 四轨（`tools/separate.py`，.venv 里跑，5070 上 15.5 s，四轨相加 vs 原混音残差 3.2%）→ `Projekt_Visual/MV_Cant_Kill_Us/分轨/`。发现：人声几乎只在 breakdown（−33 dB）、**rebuild（−20 dB，全曲最强）**、drop3（−31 dB）出现，前半首基本没有；前奏只有贝斯。待做：每轨各出一组感官通道写进感官.json（格式先改规格再通知 Visual）。
- [ ] **预期**（something is coming）：张力的斜率 + 乐句位置 → 「要来了」通道（用户 38.5 s 听出来的那种）
- [ ] bpm 150（不是 75）：4 小节 = 6.4 s 和用户的两个切点吻合 → 写进规格
