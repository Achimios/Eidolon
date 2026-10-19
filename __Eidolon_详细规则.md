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
 "events": {"onsets": [秒, ...], "beats": [秒, ...]}
}
```

| 通道 | 意思 | 怎么算的 | 典型接法 |
|---|---|---|---|
| `heartbeat` | 心跳 | < 120 Hz 的能量（贝斯 / 底鼓） | 呼吸、缩放、Kuramoto 的固有频率 |
| `threat` | 威胁 | 短窗频谱通量的冲击，180ms 指数衰减 | 故障插针、沙堆加沙、逃跑 |
| `hunger` | 饥饿 | 慢响度（2 秒窗）的反面，只在有音乐的部分里归一化 | 消散、DCT 压缩、寻找 |
| `tension` | 张力 | 频谱质心 | 逻辑斯蒂的 r、颜色温度 |
| `rhythm` | 节律 | 拍子相位 0→1，每拍一圈 | 同步、闪烁的相位 |

- 每个通道长度 = `floor(duration × fps)`（只含完整落在音频里的帧），第 i 个值对应时刻 `i / fps` 秒；越界的帧夹到最后一个值。（2026-10-19 更正：原写 round，和代码不符，Visual 会话发现）
- `onsets` 精度：合成点击音实测误差 ≤ 6ms（10ms 分析帧量化）。
- ⚠️`bpm` 可能差两倍（半拍 / 倍拍）：真歌没有标准答案可对。《Can't Kill Us》算出 150.12，【没验到】真实速度是 150 还是 75。
- 生成：`python tools/sense_song.py <音频或视频> --out <作品文件夹>`，会同时出一张 `感官_条带.jpg` 给人看。

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
- ⚠️`downbeats` / `phrases` 还没进感官.json（v2 粗剪里是从用户锚点手推的 6.4 s 格子），感官层升级后补上。

## 四、锚点（`Projekt_Visual/<作品>/锚点.json`）—— 作品数据，多写入方
- 不属于任何一个软件。写入方：用户在对话里说的（Eidolon 会话代写，`source:"chat"`）+ VisualProcess 现场 rig 里用户手动标的（`source:"vp"`）。
- 写的规矩：**按 id 增 / 改 / 删单条**；写前读，`rev` 对不上就重读再合并；写完 `rev+1`。`by:user` 的条目规则不能改。
- 演奏录下来的逐帧覆盖**不放这里**，放 `演奏/take_*.json`（VisualProcess 定格式）；用户挑中的再「提升」成锚点。

## 五、改了这份要做什么
- 改格式 = 改上游：先在这里改并写日期，再给 Visual（3_VisualProcess）和 Synapsis 项目构建 会话发消息。下游自己适配，不在这边加兼容层。
