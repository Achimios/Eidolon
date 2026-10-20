# 从哪继续

> ⭐接手第一眼读这个。只写「现在在做什么 / 下一步」，保持一屏；每天收工更新，旧内容移 `_过期的/`。

**最后更新：2026-10-20**

- ⭐**先读** `_设计/哲学.md`（§1–14，👤原话 + 落地；§13 下面 a–e 和「盲读再贴读」**待用户过**）→ `_设计/方法_剪辑与情绪.md`（手法 + 实现状态）。会话背景：`../__会话交接_从SimuWeb迁来.md`。负责会话名 ReadScreen（之后也可能处理 2_ReadScreen_HID）。
- 👤定位：Synapsis = 导演编剧；3_VisualProcess = 现场 live rig（Visual 会话负责）；Eidolon = 后台（本仓库）。
- 作品：`Projekt_Visual/MV_Cant_Kill_Us/`（The Glitch Mob《Can't Kill Us》只取音乐，重做全部镜头，本地试验）。
  - ⭐`Thymos.md` 已有👤原话：Halo 4 结局 —— 坠入 11 维、吸收前文明的残响而不是更深的失控、顿悟更高目的、**she will be back, to John**（不要 Halo 5）。下面是我的段落映射草案，待用户过。视觉用第二章蓝色全息女性，不用 Cortana 官方形象；John 只用暗示。
  - 剪辑：`剪辑表.txt`（纯文字，用户可改）→ `python tools/edl.py preview` 20 秒出无效果预览；完整版 `样片/05_第二版剪辑_全曲.mp4`。
  - 原片后处理试验：`tools/fx_footage.py dct|droste`。👤：效果好，但**别铺在复杂原片上**（两套结构打架）—— 用在简单画面 / 只在情绪需要处插入。
  - 感官：`感官.json`（5 通道 + 起音 / 拍点）；`分轨/`（Demucs 四轨，人声只在 breakdown 后出现）；`锚点.json`（用户切点 13.08 / 19.5 → 6.4 s 乐句格子贯穿全曲）。
- 生成：`src/ark.py`（火山方舟，key 在 `config/creds.local.json`，不进 git、不打印）。👤分工：**草图 Seedream 5.0 flash，精图 5.0 pro；视频草稿 Seedance 2.0 fast，最终 2.0 或 2.5**。第一张 flash 图 14 s 出（`生成/thymos_01_坠入高维_flash.jpg`），评价：技术 OK 但偏通用「AI 美术」，下一批 prompt 要按我们的语法写（黑底 + 单光形 + 一个红、符号说真话、克制）。Tripo key 等用户给（API v3 已读文档：text/image→model、rig、retarget，GLB，下载链接 5 分钟过期）。
- 环境：`.venv`（torch 2.11 cu128 + Demucs 4.1）；Blender 5.2.1 命令行可用；C4D c4dpy 未验。
- 👉 下一步：①按 Thymos 出一批 flash 草图（每段一张，当画面锚点）→ 用户挑 → pro 精图；②用干净的生成画面做数学效果（不再铺原片）；③v2/v3 改读 `规则/剪辑规则.json`；④感官升级（乐句自动检测 beat_this、每轨通道、起音强度）—— 装 beat_this / SAM2 等待用户批；⑤Tripo 人形 → Blender 镜头指导 → Seedance。
