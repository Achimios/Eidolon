# 从哪继续

> ⭐接手第一眼读这个。只写「现在在做什么 / 下一步」，保持一屏；每天收工更新，旧内容移 `_过期的/`。

**最后更新：2026-10-19**

- 2026-10-19 建骨架。负责会话：原 SimuWeb 会话（会话名 ReadScreen，以后也可能处理 2_ReadScreen_HID）。背景交接见 `../__会话交接_从SimuWeb迁来.md`。
- 👤定位：Synapsis = 导演编剧；3_VisualProcess = 现场 live rig（The Glitch Mob 的 The Blade 那种）；Eidolon = 后台。见 `AGENTS.md`。
- 环境实测：Blender 5.2.1 LTS 命令行后台能跑（`E:/2_Wares/Blender Foundation/Blender 5.2/blender.exe -b`）。【没验到】C4D 2025 的 c4dpy（`E:/2_Wares/Cinema 4D 2025/c4dpy.exe`，可能要登录 Maxon 账号）。
- 项目：用 The Glitch Mob《Can't Kill Us》的音乐重做 MV 全部镜头（本地试验）。作品文件夹待建：`Projekt_Visual/MV_Cant_Kill_Us/`。
- ✅ 2026-10-19：概念静帧 10 张（`tools/stills.py` → `Projekt_Visual/MV_Cant_Kill_Us/静帧/`），给用户剪映蒙太奇用。👤概念「数学当主题」已认可（「Yeah, cool」），草案在 `Projekt_Visual/MV_Cant_Kill_Us/概念_草案.md`。Tripo 先不买（粒子不需要网格；人形轮廓先用免费方案）。
- ✅ 原版拆镜头已做：`Projekt_Visual/MV_Cant_Kill_Us/拆镜头/原版拆镜头.md`（音乐段落、两次吸气 0:46 / 2:34）。
- ✅ 感官层：`src/sensorium.py` + `tools/sense_song.py` → `Projekt_Visual/MV_Cant_Kill_Us/感官.json`（自检 5 项；hunger 峰值独立印证了手拆的两次吸气）。接口规格 `__Eidolon_详细规则.md`（stems 部分还是草案）。前奏样片和 SIGNAL BEHAVIOR 海报已发用户，等反馈（样片轨道偏小）。
- ⭐**哲学和手法已成文**：`_设计/哲学.md`、`_设计/方法_剪辑与情绪.md`（2026-10-19，用户要求「别切到别的对话就忘了」）。
- 粗剪：v2 克制版（0-52）、v3 全曲（原片当素材重剪 + 文字卡）在 `Projekt_Visual/MV_Cant_Kill_Us/样片/`。Demucs 分轨已出（人声只在 breakdown 后出现）。
- 👉 下一步（新）：①v2/v3 改成读 `规则/剪辑规则.json`（现在硬编码）；②每轨感官通道 + 乐句自动检测进感官.json；③入侵改有机形；④吸气镜头的点云山重做；⑤等用户：3D 模型、概念图、对 v3 的反馈。
- 👉 下一步：①前奏镜头「延迟嵌入」做成动的（只要音频，零资产）；②定 stems 通道格式写成 `__Eidolon_详细规则.md`，给 VisualProcess 读；③人形轮廓来源（免费人体 → 采点）。
