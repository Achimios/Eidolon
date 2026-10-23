# Eidolon 详细规则（对外接口规格）

> ⭐给下游读（3_VisualProcess、Synapsis、以后的 NPC）。**Eidolon 是上游，格式由这边定；下游每次现读、不存副本、读不到就报，不静默回退。**
> 状态：感官层 = **已实现**（2026-10-19）；stems = **草案**（还没开始渲染，字段可能会变）。
> 两份数据都放在作品文件夹（`Projekt_Visual/<作品>/`），不在本仓库。

## 一、感官（`<作品>/感官.json`）—— 已实现

一段音乐「被生物感觉到」的样子。规则读它，不读手写的参数曲线（👤：感知 → 解构 → 构建；锚点 + 规则涌现）。

```json
{
 "source": "原始音频 / 视频文件名",
 "fps": 24,
 "duration": 295.12,
 "bpm": 150.12,
 "channels": {
  "heartbeat": [0~1, 每帧一个],
  "threat":    [...],
  "hunger":    [...],
  "tension":   [...],
  "rhythm":    [...]
 },
 "events": {"onsets": [秒, ...], "beats": [秒, ...], "downbeats": [秒, ...], "phrases": [秒, ...]},
 "beat_source": "beat_this:final0",
 "motifs": {"<名>": {"template": [t0, t1], "track": "other", "hits": [秒, ...]}}
}
```
- ⭐**2026-10-21：时间 = 播放时间**（修正原曲音频起点 0.042 s 的偏移；之前的 感官.json 早 42 ms）。
  `sensorium.load_audio / decode` 探测音频流第一个包的 pts，在前面补零 → 样本 i ↔ 播放时间 i/sr；beat_this（`beats.py`）、
  Demucs 分轨（`separate.py`，旧分轨已在文件头补 1852 个零）、`motif_find.py` 都走它。旧件留 `*.bak_2026-10-21_offset`。
  ⚠️锚点.json 的 `phrase_grid.start = 13.11` 是当初吸附到**旧**（早 42 ms）起音推出来的；不改它，`phrases` 只吸上 13/47 格（改成 13.15 是 44/47）。
- ⭐**2026-10-21 升级**（`tools/merge_events.py`，在 sense_song 之后跑）：
  - `beats` 换成 beat_this 的拍（独立模型，150.00 BPM；用户的乐句格子 43 点里 39 点 ±30 ms 内对上重拍）。`bpm` 同步换成 150.0（倍拍问题解决：是 150 不是 75）。
  - `downbeats` = 每小节第一拍。`phrases` = 锚点.json 的乐句格子（用户两个切点推出的 6.4 s = 4 小节），每格吸附到 ±40 ms 内最近的重拍（《Can't Kill Us》47 格吸上 43 格）。
  - `motifs` = 用户耳朵指出的某个声音在全曲的每次出现（`tools/motif_find.py`，⚠️要按声音所在的分轨找：鼓每个乐句重复同一型，混进来会误报 —— 👤纠正过）。
  - 字段只增不改名：只读 `onsets` / `beats` 的下游不用改。

| 通道 | 意思 | 怎么算的 | 典型接法 |
|---|---|---|---|
| `heartbeat` | 心跳 | < 120 Hz 的能量（贝斯 / 底鼓） | 呼吸、缩放、Kuramoto 的固有频率 |
| `threat` | 威胁 | 短窗频谱通量的冲击，180ms 指数衰减 | 故障插针、沙堆加沙、逃跑 |
| `hunger` | 饥饿 | 慢响度（2 秒窗）的反面，只在有音乐的部分里归一化 | 消散、DCT 压缩、寻找 |
| `tension` | 张力 | 频谱质心 | 逻辑斯蒂的 r、颜色温度 |
| `rhythm` | 节律 | 拍子相位 0→1，每拍一圈 | 同步、闪烁的相位 |

- 每个通道长度 = `floor(duration × fps)`（只含完整落在音频里的帧），第 i 个值对应时刻 `i / fps` 秒；越界的帧夹到最后一个值。（2026-10-19 更正：原写 round，和代码不符，Visual 会话发现）
- `onsets` 精度：合成点击音实测误差 ≤ 6ms（10ms 分析帧量化）。
- `bpm`：sensorium 自己估的可能差两倍；merge_events 之后用 beat_this 的值（《Can't Kill Us》= 150.0，已和用户的乐句格子交叉验证）。
- 生成：`python tools/sense_song.py <音频或视频> --out <作品文件夹>`，会同时出一张 `感官_条带.jpg` 给人看。
- ⚠️完整顺序：`separate.py`（分轨）→ `sense_song.py`（**整个重写** 感官.json，会丢掉下面的 voice）→ `beats.py` → `merge_events.py`（保留 voice / expect）→ `sense_voice.py`（加 voice）→ `sense_expect.py`（加 expect；读 voice，voice 重算后要重跑）。

### 人声 `voice`（2026-10-23 新增，`src/voice.py` + `tools/sense_voice.py`）
从 Demucs 人声分轨（`分轨/vocals.wav`）算的**纯声学特征**（不识别歌词）。只新增顶层 `voice` 节点，其余字段一个字都不动（写入时逐字段比对；《Can't Kill Us》旧件留 `感官.json.bak_2026-10-23_voice`）。
```json
"voice": {
 "source": "分轨/vocals.wav", "method": "…", "loudness_ref_db": -12.3, "floor_db": 40.0,
 "loudness": [0~1, 每帧], "voiced": [0~1], "f0": [Hz 或 null], "vibrato": [0~1], "vibrato_cents": [音分 或 null], "vibrato_hz": [Hz 或 null],
 "phrases": [{"start": 秒, "end": 秒, "f0_med": Hz, "f0_lo": Hz, "f0_hi": Hz, "peak": 0~1, "tail_vibrato": 0~1}, ...],
 "breaths": [[秒, 秒], ...], "sections": [[秒, 秒], ...]
}
```
| 字段 | 意思 | 怎么算的 | 典型接法 |
|---|---|---|---|
| `loudness` | 人声响度 | 40 ms RMS 的 dB、50 ms 平滑；0 = 比人声峰值（99.5 百分位，`loudness_ref_db`）低 `floor_db` dB | 粒子海的浪高 / 亮度跟她的音量 |
| `voiced` | 有声概率 | ⭐默认：torchcrepe full 的 periodicity（3 帧平均）的 logistic（0.45 → 0.5）× 响度门；回退：YIN 的 CMNDF 最低谷 d 的 logistic（d = 0.2 → 0.5） | 「她在唱」的软开关 |
| `f0` | 基频（Hz） | ⭐默认（2026-10-23 v2）：crepe full 定八度（viterbi 路径 + ±4 格加权平均，**不用** torchcrepe 自带的 ±20 音分抖动）→ 在 crepe 周期 ±10% 里找 YIN 的谷细化 → 5 帧中值。回退：YIN（40 ms 窗，65–1100 Hz）+ 八度跳变修正。用的哪种写在 `method`。**只在有声帧给，其余 `null`** | 音高 → 高度 / 色相；下游自己决定 null 时保持还是回落 |
| `vibrato` | 颤音强度 | 0.6 s 窗里 f0（音分）= 二次趋势 + 正弦 的最小二乘，扫 3.5–9 Hz；正弦解释方差 > 0.5、频率在 4–8 Hz、半幅 > 8 音分才算；= 解释比 × min(1, 半幅/40) | 尾音的颤 → 粒子的细颤 |
| `vibrato_cents` / `vibrato_hz` | 颤音半幅 / 频率 | 同上（已补 YIN 窗的盒式平均）；不可信的帧为 `null` | —— |
| `phrases` | 乐句 | 有声段间空隙 < 0.18 s 并成一句，短于 0.2 s 的扔掉；start / end = 首 / 末有声帧中心；`tail_vibrato` = 最后 0.5 s 的平均颤音 | 一句一个「浪」 |
| `breaths` | 换气 | 相邻两句之间 0.18–1.0 s 的空隙；更长的是休止 | 吸气 / 回落 |
| `sections` | 人声段 | 乐句间空隙 < 3 s 并成一段 | 「女声在哪」的大区段 |

- 逐帧字段的长度 = channels 的帧数（同 fps，第 i 个 ↔ `i / fps` 秒，播放时间）。
- 精度：合成锯齿 / 正弦 f0 误差 < 5 音分、无八度错；合成 ±50 音分 5.5 Hz 颤音检出 5.5 Hz / ≈49 音分；乐句边界 ±30 ms（`tests/test_sensorium_voice.py`，YIN 和 crepe 两条路径跑同一组断言）。crepe 单用（`crepe_raw`）合成信号上整体偏 +5 音分、颤音深度少 17% —— 所以要 YIN 细化。
- ⭐**真实人声对照**（`tools/voice_compare.py`，没有真值；YIN 与 librosa pYIN 同源，它们一致不算交叉验证）：

  | | 新歌 BTP | 《Can't Kill Us》 |
  |---|---|---|
  | 门以上有声比例 yin / crepe / pyin | 0.55 / 0.67 / 0.78 | 0.50 / 0.50 / 0.74 |
  | crepe(细化) vs pyin：中位音分差 / 八度 / 有声一致 | 4.3 / 15.6% / 0.77 | 4.8 / 5.7% / 0.69 |
  | yin vs pyin：同上 | 3.3 / 5.6% / 0.75 | 4.8 / 4.7% / 0.73 |
  | 八度分歧谐波仲裁 crepe : yin | 55 : 45（≈ 平） | 73 : 27 |
  | 门以上人声能量落在乐句里 yin → crepe | 74% → 86% | 78% → 85% |

  谐波仲裁 = 候选 f 的整数倍谐波 vs 半整数倍的对数幅度差（与两种算法都无关；合成音上 f / 2f / f/2 / 3f 全选对）。⚠️BTP 约 16% 的有声帧上几种算法差一个八度、仲裁也近乎打平 —— 这首的人声八度本身含糊（可能有叠唱 / 处理），音高 → 画面时别依赖绝对八度。⚠️crepe 对分轨里串进来的有音高的乐器（BTP 前奏的拨弦）也给有声，出了几个 < 0.5 s 的假乐句。
- 诊断图：`<作品>/人声_条带.jpg`（v1 = YIN）、`人声_条带_v2.jpg`（crepe 默认）。新图不覆盖旧图，自动 `_vN`。
- 生成：`python tools/sense_voice.py <作品文件夹> [--method crepe|crepe_raw|yin]`（要先有 `感官.json` 和 `分轨/vocals.wav`）；改完 voice 要重跑 `sense_expect.py`。

### 期待 / 意外 `expect`（2026-10-23 新增，`src/expect.py` + `tools/sense_expect.py`）
> ⚠️**这是积木 / 审计，不是导演。** 👤「不要太数学化，核心还是跟着 Thymos 和 emotion flow。数学是 building block 和审计视野等作用。」
> 它不改剪辑、不改画面、不替谁做决定；规则可以读它当一个参数，审计图把分歧点标出来给人看。意义由故事（Thymos）给，意外度只说「哪里罕见」（哲学 §15）。

每个声音事件在**她此刻所在的情绪段落**里有多罕见（意外度，比特），再乘上**她此刻在等什么 / 怕什么**（显著度）。同样的鼓点，在「憋住」段是意外，在 drop 段是日常 —— 所以上下文是情绪段落，不是全曲一个分布。只新增顶层 `expect` 节点（放在 `voice` 后面），其余字段一个字都不动（写入时逐字段比对；旧件留 `感官.json.bak_2026-10-23_expect`）。
```json
"expect": {
 "version": "2026-10-23", "role": "积木 / 审计，不是导演 …", "method": "…", "context": "情绪大图/草案.json 的段（15 段）" | "滑动窗口 12.8 s（这首没有情绪大图）",
 "expect_source": "情绪大图/期待.json（⚠️草稿，待👤过）" | null, "salience_note": null | "为什么显著度留空", "params": {…},
 "segments": [[start, end, "id", "名"], ...] | null,
 "events": [{"t": 秒, "s": "onset|motif|voice_on|voice_off|absence", "m": 大小, "bits": 比特, "rate": 比特, "mag": 比特, "seg": "段 id" | null, "sal": 显著度 | null}, ...],
 "surprise": [每帧，比特], "absence": [每帧，比特], "salience": [每帧] | null
}
```
| 字段 | 意思 | 怎么算的 |
|---|---|---|
| `events[].s` | 事件流 | `onset` = events.onsets（大小 `m` = 起音那一帧的 threat）；`motif` = motifs 的 hits；`voice_on` / `voice_off` = voice.phrases 的起 / 止（`m` = peak / 乐句长度）；`absence` = 该来的起音没来（`m` = 这段空隙的长度，秒） |
| `bits` | 意外度 = `rate + mag`（上限 12） | 只用**这一刻之前**听到的（期待只能从已经听到的东西里长出来）。记忆：同段全记，之前段落的 3 s 内淡掉（刚换段时还带着上一段的期待，约 2 小节适应过来）；没有情绪大图时用 12.8 s 滑动窗口 |
| `rate` | 「这一拍里来一个」有多罕见 | λ = 记忆里的事件率（带半个事件的全曲先验）；p = min(1, λ·拍长)，−log2 p。`voice_off` 不算（结束由开始决定） |
| `mag` | 「这么大或更大」有多罕见 | 记忆里 ≥ 它的比例（秩，自己算一个，带全曲分布的先验），−log2 p。`motif` 不算（大小恒定） |
| `absence`（每帧） | 缺席：从上一个起音到现在一个都没来，按此刻的期待有多罕见 | λ·gap / ln2；超过 4 比特记一个 `absence` 事件（「等太久了」被感到的那一刻）；第一个起音之前、最后一个之后 = 0 |
| `sal` / `salience` | 显著度 = 乘子 × 门 × 意外度 | 乘子 = `期待.json` 该段 `attend[事件流]`（没列 = 0.3）；门 = logistic，中心按 `tolerance`：低 0.5 / 中 1.5 / 高 3 比特。**没有 期待.json 或没有情绪大图 → null**，`salience_note` 写原因 |
| `surprise`（每帧） | 意外度的包络 | 事件处冲到 bits，180 ms 指数衰减（同 threat）；absence 不进它，单独一条 |

- `情绪大图/期待.json`（作品数据，和草案.json 同 id）：每段 `expecting`（她在等什么）、`fear`、`tolerance`（低 / 中 / 高）给人读、给人改；`attend` 是从前三个推出的机器字段。⚠️现在是 Eidolon 起草的**草稿，待👤逐段过**。
- 审计图 `<作品>/感官_期待审计.jpg`：段落色带（= 她在等什么）+ 意外度 + absence + 显著度；给了 `--edl` 再加剪辑（切点 / 长码 / 闪帧）+ 样片逐帧运动（6×6 块里最大那块的帧差）。
  - ▼红 = 显著度（没有就意外度）在审计区前 8% 且 ≥ 2，但 ±0.2 s 没切点、前 0.6 s 内没开长码镜头、样片也没有运动尖峰 —— **画面没回应**。
  - ▲橙 = 剪辑给了长码（停留 ≥ 邻近中位 1.5 倍且 ≥ 2.4 s / 静帧慢推 / ⭐），但那一刻（切点前 0.1 到后 0.6 s）意外度 < 3 比特且低于本段六成起音 —— **长码给了平常的一刻**（常常是故事给的，合理；只是标出来）。
  - 闪帧后接回同一素材 = 原镜头的续，不算新开的长码。
- ⚠️已知局限：样片运动只看帧差，镜头本来就一直在动时的「速度鼓包」抓不到（所以加了「前 0.6 s 开了长码镜头」那条）；剪辑表只有 0–97 s 是真剪辑。起音强度（threat）在强段常常饱和到 ≈ 1，大小的区分度有限。
- 生成：`python tools/sense_expect.py <作品文件夹> [--edl 剪辑表 --video 按这份剪辑表出的样片 --audit-end 秒] [--dry]`。剪辑表和样片只用于审计图，不写进 `感官.json`。

## 二、stems（`<作品>/stems/<镜头id>/`）—— 草案

每个镜头一个文件夹，同一帧号对齐（从 0 开始，4 位数）。

| 文件 | 格式 | 内容 | 现场拿来干什么 |
|---|---|---|---|
| `beauty/####.png` 或 `beauty.mp4` | sRGB 8bit | 成品画面 | 底图 |
| `depth/####.png` | 16bit 灰度，近白远黑，`meta.json` 记 near/far（米） | 深度 | 2.5D 视差、按深度景深 / 雾 |
| `mask/####.png` | 8bit，每个像素 = 物体编号（0 = 背景），`meta.json` 记 编号 → 名字 | 物体遮罩 | **真抠像**（光环背光不再靠亮度猜） |
| `motion/####.png` | 16bit RG，像素位移 /（2 × max）+ 0.5，`meta.json` 记 max（像素） | 运动矢量（到下一帧） | **真 datamosh**（用真实运动去拖） |
| `meta.json` | —— | fps、分辨率、帧数、上面的换算参数、`ran_with` | —— |

- `ran_with` 指纹：`<场景文件>@<Eidolon git 短哈希>#<镜头id>/<参数 hash8>`（和 Synapsis 约定的格式一致）。
- 渲染器：Blender（Cycles / EEVEE 的渲染层 + 合成器输出）。【没验到】具体输出设置。

## 三、剪辑规则（`规则/剪辑规则.json` + 作品覆盖）—— 格式 v0（2026-10-19，和 Visual 会话商定）

- 两层：通用默认 = 本仓库 `规则/剪辑规则.json`；作品覆盖 = `Projekt_Visual/<作品>/剪辑规则.json`，只写不同的条目，**按 id 合并**（同 id 字段逐个覆盖，新 id 追加）。两边都现读、不存副本。
- 每条：`id`、`priority`（同帧撞上大的赢）、`enabled`、`trigger`、`action`、`in`（在某个 section 里覆盖的参数）。
- `trigger.on` = 感官.json 的事件名（`beats` / `onsets`，以及升级后的 `downbeats` / `phrases`）；`trigger.section` / `in.<名字>` = 锚点.json 的 `kind`（如 `rise`）。
- `action.type` 是**动作名**，不写实现：Eidolon 离线（Python）和 VisualProcess 现场（GLSL）各实现一次。现有动作：`cut` `cutin` `blink` `break` `bump` `strobe` `invade`。新增动作先在这里登记。
- ✅`downbeats` / `phrases` 已进感官.json（2026-10-21，见第一节）。

## 四、锚点（`Projekt_Visual/<作品>/锚点.json`）—— 作品数据，多写入方
- 不属于任何一个软件。写入方：用户在对话里说的（Eidolon 会话代写，`source:"chat"`）+ VisualProcess 现场 rig 里用户手动标的（`source:"vp"`）。
- 写的规矩：**按 id 增 / 改 / 删单条**；写前读，`rev` 对不上就重读再合并；写完 `rev+1`。`by:user` 的条目规则不能改。
- 演奏录下来的逐帧覆盖**不放这里**，放 `演奏/take_*.json`（VisualProcess 定格式）；用户挑中的再「提升」成锚点。

## 五、改了这份要做什么
- 改格式 = 改上游：先在这里改并写日期，再给 Visual（3_VisualProcess）和 Synapsis 项目构建 会话发消息。下游自己适配，不在这边加兼容层。
