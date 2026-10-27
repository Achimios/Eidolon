# 从哪继续

> ⭐接手第一眼读这个。只写「现在在做什么 / 下一步」；旧内容移 `_过期的/`。

**最后更新：2026-10-22（素材生成日）**
- 👤第二章的素材 / voice 已搬到 `MV_Cant_Kill_Us/素材`、`voice`（工具路径已改）。Seedream / Seedance flash 随便用；Tripo 👤网页做（抽奖）。

- ⭐**先读** `AGENTS.md` → `_设计/哲学.md`（§1–14）→ `_设计/方法_剪辑与情绪.md`（⭐§一½ 用户耳朵→声音锚点、§四⅞ 粒子、§四⅞½ 猛烈从哪来、§四⅞¾ 镜头运动）→ `_设计/效果库.md`（所有数学效果）→ `_设计/接口_生成.md` / `接口_Tripo.md`。会话名 ReadScreen。
- 作品：`Projekt_Visual/MV_Cant_Kill_Us/`。`Thymos.md`（为什么）→ `物件.md`（用什么，物件每次推进）→ `情绪大图/草案.json`（每段一格 = **大分镜**，之后每格拆多个镜头进 `剪辑表.txt`）。参考库 `Projekt_Visual/_参考库/`。

## ⭐主线（别丢）
感知 → 拆解 → 构建；情绪是共同坐标。Thymos：她被炸碎 → 坠入 11 维 → 狂语（残响想同化她）→ 吸收 → 空 → 顿悟 → 重组 → 回到 John（面罩反光暗示）。
**现在的位置**：👤「现在专注素材生成，之后一起剪」；⚠️「目前粗剪全局节奏不如最早版本」→ 重剪前先拆早期版节奏。粗剪最新 `样片/13g_粗剪v4.6`（剪辑表 `剪辑表_v4_0-97.txt`）。
- 今天的新素材 / 工具（每个都有效果库行 + 人读说明）：开头 `opening.py`（弦→团、高解析瀑布图、macro 式镜头，`15_开头_v3`）；碎裂 v5 `particle_shatter.py`（2D 机理 + 视频源，`19_碎裂_v5b_*`、余波_3 延长）；面部瓦解 `face_glitch.py`（⭐👤偏好：瀑布图行 + 本片配色；门控 = 强度不是透明度；`17_…_v4/v4b_v2`，脸在右、原速 + 结尾定格到 51.52；加全面抹除中）；多层重影 `layer_ghost.py`（替代透镜重影 `lens_ghost.py`，👤：只是带纹理蒙版 + 散射的多层叠加，不要扭曲；做中）；最优传输 `ot_flow.py`（👤：算法好，太黑、运镜诡异；想做两个画面之间的传输 + 多层感，粒子相撞无所谓 → 可回到 sliced OT）；四维插入 `four_d.py --insert`（v5 = 👤 Tripo 的 `3D/蜷缩_无材质_低模.glb`，`--mesh-lateral sym`（这个模型侧向斜在 60°，x / z 都错）；surface 平切面自发光 `--cut-glow`；`14_四维插入_57.94_v5_tripo蜷缩` / `73.90_v5_…_surface`，等 👤 看）；`push_remap.py`（推近/跳切）；`trash.py`（回收站）；`blender_pose.py`（自动绑骨，👤嫌诡异，弃用）。
- 生成：Seedream 简单图 flash、复杂图 pro；**先挑后做**；Seedance 草稿走 2.5 草稿模式；本地视频当参考只能走 TOS（👤要开，等 AK/SK 进 creds）或👤网页端（`生成/给网页端_运动参考_v1/`）。👤网页生成的在 `素材/`（全息女性视频运镜描述、MEMS 运动1/2）。
- ⚠️删除：只能删自己这一轮的中间文件，用户看过的一律不删（AGENTS.md）。感官层已修 42 ms（时间 = 播放时间，锚点 rev3）。

## 👤已定（2026-10-20/21）
- 视频 Seedance 2.5 样片模式（480p → 正式）；原片 = 脚手架最终全换；MV 先看画面再读 Thymos；John = **面罩反光**；HUD 英文；抽搐音 **先机械后有机**（drop1 MEMS → drop2 过渡 → drop3 克拉尼排成她的脸）。
- 镜头运动不从静止开始 / 不停在静止；进缓只改速度不加晃；特别的声音配独立镜头；运动要沿结构传播（不是各点独立抖）；MEMS 要每个局部元素带信息地动、更猛、能配合镜头（真 3D 最好）；四维物体倾斜摆放、切面动得快（无定态感）。
- 模型以后**先灰模**，确认后再上材质（或叫用户网页做）。样片不覆盖，按版本另起名。
- 先行者符文：👤自己找 / 设计。

## 样片现状（`样片/`）
- 碎裂：`08_粒子_碎裂_v4`（学 The Crew，⭐）。MEMS：`09b` 细颤（👤：还不够，整体在抖）→ ✅`09c_MEMS耳蜗_drop1`（64 根悬臂 = 机械频谱仪，真 3D，`tools/mems_cochlea.py`，等👤看）。
- 卡拉比丘：`11c` 局部 v2（四推 + 1:24.91 诡异独立镜头，👤「太酷了，景深太多」）→ 子代理在做 `11c_局部_v3`（只改速度、景深减小）和 `11b_全局_v2_晶格传播`（图拉普拉斯上的波）。
- 四维切片：`10`（纯数学，👤「相当好」）、`12`（她的模型扩 4D）。
- 静帧 11–21 见 `静帧/`（`_新增_总览.jpg`、`19b`、`20`、`21`）。风格渲染 `生成/风格渲染/`。Tripo 模型 `3D/`（她 / 先行者步枪 / 机械臂 / 躯体核心，final 预设；余额 750）。

## 感官层（2026-10-21 升级，Visual 已通知）
`感官.json` 新增 `downbeats` / `phrases`（吸附重拍）/ `motifs`，beats 换 beat_this（150.0）。生成：`sense_song.py` → `merge_events.py`。`real_ratio` 按段在 `草案.json`。

## 等👤
情绪大图逐格感觉（主线）；HUD 每行措辞；风格渲染 D3 的脸能否接受；PCB 11/12/18 留不留。

## TODO（`_TODO.md`）
Dune 预告片 → Billie MV；Ash Thorp 式纯 2D motion design（PCB 素材，`_参考库/PCB_细节图/`）；粒子海。旧粗剪脚本 v2/v3 改读规则文件。Seedance 视频还没真调过（开通要方舟余额 > 200 元【没验到】）。
- 👤选中风格渲染 A1/A2/B/D1/D3，不一定亚克力、可以更暗；短插入镜头（~1 s）切面运动先快后缓（见 `MV/物件.md` 末尾）。

## ⏸ 2026-10-22 晚（压缩前）
- 在跑：① 四维「浮现·从手」真 4D 重做（👤：上一版用错模型、本质是 3D 遮罩；用 Tripo 蜷缩模型 + Golden Age 观感，`样片/四维切片/14_四维浮现_从手_4D_v1`）；② Neural4D 接口调研 + `src/neural4d.py` + `_设计/接口_Neural4D.md`（key 在 creds `neural4d`，花钱前先报价）。
- 已出待👤看：粗剪 `样片/粗剪/13i_粗剪v4.8`（edl 漂移已修）；面部瓦解 v4c（抹除 v2）→ 碎裂 v5c（余波_3 选 A/B）；多层重影 v3 全强度 / A 回忆 / B 清晰分裂；最优传输 3D v2 参考机位 / v3 视频源；四维 Golden Age spin/twist（剪辑表没放，槽位要👤定）；离子晶体关键帧 a–d（回想段，替 Kuramoto 的 Seedance 版）。
- 👤手动：网页端 Seedream / Seedance（走廊 C、S3、离子晶体、骨架重出）。Seedance 批次 1 的 S3 👤不满意 → 重出；骨架视频运动怪 → 先重出静帧。
- 节奏规则：`拆镜头/节奏对比_早期vs_v4.6.md`。复现：`-给人类的解释/换电脑怎么复现.md`（两个仓库都没有远端，等👤定）。⚠️Tripo key 建议换。

## 🌙 2026-10-24 晚（停在这）
- ⭐**剪辑真源 = Synapsis 节点**（👤选 A）：`MV_Cant_Kill_Us/__synapsis/剪辑/`（6 段落 → 35 镜头，段落时间自动算）；预览 `tools/edl.py preview --synapsis 剪辑`。素材库 = 画面类型卡（每卡留「👤要求 / AI 回报」）。截帧 `cli.js frame`。
- 👤已剪：四维生长插在 2:08 第 8 帧（作「成形」镜头，接记忆图谱）。⚠️粒子海原槽 2:08.36 冲突，待👤定（往后挪 / 别处 / 给 Between Two Points）。
- 记忆图谱最新 v9（1:55.51–2:08.31）；结尾四声处理待 John 视频（建议：分辨率逐声翻倍 + 第四声视频开始播）。
- 四维：真4D blender v2（多中心生长 + xw 旋转 + 只留亮边）；Seedance 已用 Blender 参考出了好的 curled girl 视频（`素材/curled girl/`）。
- 感官空间：新子代理重做 `25_感官空间_正面_v1`（照审计图 / 人声条带配色）待👤看。
- 👤明天：多生成视频素材（`生成/_待生成清单.md`）；走廊 C 多出几张 Seedream（图 2 用 A/D 关键帧或走廊背影截帧）；**拍一张自己的背影**当 John 剪影参考（无标志、通用装甲或普通男人二选一）。⚠️**S1 升正式版 10-29 前**。

## ⭐ 2026-10-26：转向「自己合成音乐」（停在这）
- 👤：Piccolo / Dune 已手动丢回收站、重来；Billie 还在看（画风好，太阴郁）。Fable 压缩前写了 `Projekt_Visual/_Fable_教训.md`。⭐Seedream 必须带参考图当锚点，再从一堆里挑（记忆 seedream-needs-reference）。
- ⭐新方向：自己的 **soft composer**（Nirvana 胜过 Mozart，能玩的乐器）+ Suno（👤网页端）。理念、处理器家族、生长算法、FIR/IIR 生长 = 极点即情绪：`_设计/理念_示波器与处理器.md`。
- ✅示波器模拟器 `tools/scope_sim.py`（GPU，比实时快；光束亮度 ∝ 1/速度 + 余辉 + 辉光），用 Triple Hit 的 192 kHz 无损原曲对照作者视频：留出的 4 帧相关系数 0.89–0.98。样片 `Projekt_Visual/样片_示波器/`。差距：特亮点的光晕偏小；满屏噪声偏密。
- 参考：Triple Hit（CC BY-NC-SA，`Assets/BUS ERROR Collective - Triple Hit/`）；osci-render（GPL，可当乐器）；mg-styles-15 clone 在 `Vault_Automata/_外部仓库/`（没跑）。
- ✅合成器词表 v3（ReadScreen 过审，2026-10-27；v3 修了频谱缺泛音、加了左右声道波形；排队等👤看，不催）：`-给人类的解释/合成器词表.html`，素材 `Projekt_Visual/样片_示波器/词表/v3/`（每词：示波器用 wav + 听的 listen.wav + 带声 mp4 + 四格 + 频谱）；积木 `src/synthkit.py`，生成 `tools/synth_vocab.py` / `synth_vocab_page.py`。.venv 新装 pedalboard 0.9.25、pyloudnorm 0.2.0（👤允许下载）。
- 📝作曲器草案 `_设计/设计_作曲器.md`（层 / 场景 / 旋钮 / take / 显影；事件轨 `作曲.json` = 真值并进感官.json；三个选项 A Remix Can't Kill Us / B Suno / C 全自合成），⭐**八个开放问题等 👤**，拍板前不写代码。
- ✅👤选 A → **Remix 示波器 demo v4**（ReadScreen 过审、已 Cortana 告诉 👤）：`Projekt_Visual/MV_Cant_Kill_Us/样片/Remix_示波器/demo_Remix示波器_25.95-57.98_v4.mp4` + A/B `AB_原曲同段_同LUFS_v4.wav`，分镜 / 改动 `分镜_v4.md`。工具 `tools/remix_cku.py audio|board|render --ver`；径向饱和反馈环 `synthkit.feedback_delay_sat_xy`。**等 👤耳朵判好不好听。**
- 输入方式定了（作曲器 §四½：锁调电脑键盘 / 哼唱 / 拍 / XY 推子 / 挑变体，HTML）→ 👤另开会话「HTML composer」在做。
- 待办（不开工、不推给👤）：她的 SDF 切片 → x / y / z 三路音频 → 三维投影，第四路接四维切片（理念 §九）。
- ✅（2026-10-28 已改：notes / hits 读 audible；clips 跨场景仍按 mute + 场景逐段算，旧导出没字段时现算并打日志）给 Fable（2026-10-27，它离线没转到）：HTML composer 的 `作曲.json` 的 notes / hits 新增 `"audible"`（按 mute + 场景算，独奏不参与）→ `tools/composer_render.py` 改成直接读它，删掉自己的 audibleAt 判断。新层会自动加入已有场景。
- ⭐👤2026-10-27 放慢：一次只递给👤一件要看的东西，他回了再递下一件。现在排第一 = Remix demo v4；词表 v3、作曲器草案已审过，排队。

## 🌙 2026-10-24 深夜：Fable 三条 MV（另一个会话「Music video project」做，我交叉审）
- ⭐先看 `Projekt_Visual/_Fable_今晚汇总.md`。Piccolo（`MV_Piccolo/样片/03_…v3b`）、Happier Than Ever（`MV_Happier_Than_Ever/样片/0{1,2}_…v5`）、Dune 原创预告《静海》（`MV_Dune/样片/A_…v3` + `D_…v4`，音床是自合成的 `音床_v2.wav`，没下载任何东西）。
- 感受器给两首歌跑了；新工具 `tools/sense_lyrics.py`（whisperX，独立 `.venv_whisperx`；HTE 用👤给的歌词强制对齐 → `歌词_v4.json`）。ffmpeg 已加进用户 PATH。
- 等👤：音床用耳朵判；Dune 浪是「笔画」还是「水墙」；Dune 真音频来源（`MV_Dune/音频候选来源.md`）；网页端视频 prompt 清单（HTE / Dune 各自 `生成/prompt清单_网页端_v1.md`）。

## ☀ 2026-10-23 早：给👤看的（夜里三个子代理自迭代 + 主会话筛过一轮）
- 远端：GitHub 私有 `Achimios/Eidolon`、`Achimios/Projekt_Visual`（push 走代理 7897）。图片瘦身：参考库截图 / 接触表不进 git，>2 MB 图 pre-commit 拦。
- 剪辑表 v4：57.94 = Golden Age spin（twist 不用）；51.52 碎裂换 `19_碎裂_v5c_连续_B长推`（我选 B）。
- 四维浮现 v2（N4D 蜷缩、侧面近景 + roll）；👤要开头更慢 → 明天做（费 GPU）。
- 粒子海 v4（`样片/粒子海/`，酸绿 / 黑白红，建议 2:08.36–2:14.78；束提前 0.5 s 离水，因果让步待👤判）。
- 几何符号：倒数 v8（`样片/几何符号/`，建议 3:12.32–3:18.76）；筛 v6（`MotionDesign/几何符号/`，独立）。旧版全留，👤定。
- 信息可视化：A 记忆图谱 v3（`样片/信息可视化/`，建议 1:55.51–2:08.31；CONFIDENCE 真值 0.31、第 28/36 个才找到 John，👤定故事）；B 成分树 v3（`MotionDesign/信息可视化/`）。
- Neural4D：👤实测非 PBR 无材质 10 点 / 个；两个新模型 `3D/N4D_*.glb`。
- 👤今天：即梦生成（走廊 C、S3、离子晶体首帧 b/d、骨架），然后我接剪。
