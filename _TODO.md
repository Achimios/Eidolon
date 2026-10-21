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

## 💡 下一部试验（👤2026-10-19）
- **Dune 预告片的音乐**：完全不同的节奏 —— 稳、克制，情绪蕴含在相对静止的画面里的场景和元素中。人物动画少，很多画面可以用 Seedream / Seedance 出。用来检验「克制」和「Thymos 驱动」，和 Glitch Mob 的高能量互补。
- Billie Eilish 那种需要真人动画 / 实拍的 MV：以后再说，先数字。👤2026-10-20：顺序 = Glitch Mob → Dune 预告片 → **Billie 的 MV 作为最后一部**（「to earn the dating chance 😉」）。
- 实拍感画面怎么来（Billie 用得上）：@anabology 的做法（按原作者说法）= 模仿一套超长 prompt + 给 Midjourney 权限 + 一个 Moodboard，Opus 5.5 跑了 12 小时。我们的路：moodboard 也要（= 图片参考，低熵注入），但 prompt 不靠「超长」，靠 Thymos + 画面语法 + 参考图 + Blender 镜头指导；Seedream/Seedance 补质感。到时两条都试，比分歧点。
- 👤2026-10-21：再加一部 **Ash Thorp 那种纯 2D motion design**（类似之前的 M O D U L A R 参考）。素材储备：**PCB 设计的视觉词汇**（BGA 扇出、差分对、蛇形等长线、飞线束、铺铜；布线可以「生长」——迷宫 / Lee 波前布线算法当生长过程），参考片《29 届 Mentor Graphics PCB 设计大赛》已交子代理拆（`Projekt_Visual/_参考库/分析_PCB设计大赛.md`），**留给以后，别分散 Glitch Mob 主线**。
- 👤2026-10-21：**粒子海**（参考 `Projekt_Visual/_参考库/2026-10-21_用户给的/粒子海_鲸.webp`：网格 / 点云化的鲸跃出点阵的海，配荧光绿 UI 条块）。我们的版本：物体（Tripo 模型）采样成点，海 = 点阵高度场（可以是频谱瀑布图 / 真实波动方程），HUD 写真参数。
