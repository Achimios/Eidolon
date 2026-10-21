# 接口：Tripo 3D API V3 能做什么（2026-10-20 查官方文档 + 实测）

> 来源：developers.tripo3d.ai/en/docs（V3 开发者文档）、/en/pricing（价目表）、迁移指南 v2→v3。客户端：`src/tripo.py`。
> ⚠️**V2（`api.tripo3d.ai/v2/openapi/task`）2026-11-01 停用** → 只用 V3：`https://openapi.tripo3d.ai/v3`，`Authorization: Bearer <key>`（V2/V3 共用 key）。
> 【没验到】= 文档写了但我们还没实测。**1 credit = 0.01 美元。** 失败 / 取消的任务不扣钱（建任务时冻结，成功才扣）。

## 实测过的（2026-10-20）
| 调用 | 结果 |
|---|---|
| `GET /account/balance` | ✅ 鉴权通过，返回 `{balance, frozen}`。**当时余额 0.00** |
| `GET /account/usage` | ✅ 返回 `[]`（还没用过） |
| `GET /tasks/{不存在的 UUID}` | ✅ HTTP 404，code 2001 |
| `GET /tasks/task_xxx` | ⚠️ HTTP 400 code 1004「expected UUID format」→ **真实 task_id 是 UUID**，文档示例里的 `task_abc123` 只是示意 |
| `POST /generation/text-to-model`（草稿预设） | ❌ HTTP 403 code 2010「credit 不够」→ 没扣钱。**生成 → 轮询 → 下载这条路【没验到】**，充值后跑一次 |

错误格式统一：`{"code", "status":"error", "message", "suggestion", "request_id"}`；成功：`{"code":0, "data":{...}}`。

## 端点
| 能力 | 端点 | 价格（credit） | 备注 |
|---|---|---|---|
| 文生模型 | `POST /generation/text-to-model` | H：白模 **10** / 标准贴图 20；P1：30 / 40；P2：100 / 110 | prompt ≤1024 字符，negative_prompt ≤255 |
| 图生模型 | `POST /generation/image-to-model` | H：20 / 30；P1：40 / 50；P2：100 / 110 | `input` = file_token / URL / 出图 task_id（三选一） |
| 多视图生模型 | `POST /generation/multiview-to-model` | 同图生 | `inputs: [{"front":…},{"left":…}]`，front 必填，≥2 张；也可以传 image-to-multiview 的 task |
| 上传 | `POST /files`（multipart，字段 `file`） | 免费 | → `file_token`。图 ≤20MB（jpg/png），模型 ≤150MB（glb/gltf/fbx/obj/stl）。>60MB 用 `POST /files/presign`（PUT 直传，30 分钟有效）【没验到】 |
| 查任务 | `GET /tasks/{id}`；批量 `POST /tasks/list` | 免费 | 状态 queued → running → success / failed / banned / expired / cancelled |
| 余额 / 流水 | `GET /account/balance`、`GET /account/usage` | 免费 | ✅实测 |
| 绑骨前检查 | `POST /animations/rig-check` | **免费** | → `{riggable, rig_type}` |
| 自动绑骨 | `POST /animations/rig` | 25 | 见下 |
| 套动作 | `POST /animations/retarget` | **10 / 个动作** | 一次 ≤5 个 |
| 重上贴图 | `POST /models/texture` | fast / 标准 10，HD 20，8K 30 | texture_prompt 可给 text / image / 多视图 |
| 转格式 | `POST /models/convert` | 基础 5；带 quad / face_limit / texture_size / pivot / scale 等 = 高级 10 | GLTF / FBX / USDZ / OBJ / STL / 3MF；`fbx_preset: blender / 3dsmax / mixamo / bake_scale` |
| 重拓扑 / 减面 | `POST /mesh/decimate` | v3.0 三角 35 / 四边 40；v2.0 30；v1.0 10 | v3.0 可同时 `smart_uv`；v2/v3 face_limit 三角 500–20k、四边 500–10k |
| 分件 | `POST /mesh/segment` | 40 | v2.0-20260430 Beta 带语义标签 |
| 智能分件 / 补全 / Smart UV | `/mesh/smartsegment`、`/mesh/complete`、Smart UV | 55–85 / 30–50 / 20 | 没细看 |

**轮询**：官方建议 1–2 s 一次；查询接口限额比生成高。客户端默认 2 s。超限 → HTTP 429（1007 频率 / 2000 并发，看 `Retry-After`）。
**并发**（按账号、按类别）：H 系列生成 10，P 系列 5，出图 1，动画 10，贴图/转格式 5，网格操作 10。
⚠️**`model_url` 5 分钟就过期** → 成功后立刻下载（`tripo.py` 就是这么做的）。

## 生成参数（text / image / multiview 通用）
| 参数 | 取值 | 说明 |
|---|---|---|
| `model` | `v3.1-20260211`（默认，H3.1）/ `v3.0-20250812` / `v2.5-20250123` / **`P1-20260311`** / **`P2-20260801`**（Preview） | P 系列必须显式写 model，否则跑 H3.1 |
| `texture` | 默认 true | **白模：`texture:false` 且 `pbr:false`**（pbr=true 会强制开 texture） |
| `pbr` | 默认 **true** | base_color / metallic / roughness / normal。⚠️默认就开，要白模必须显式关 |
| `texture_quality` | `fast`（要 `texture_version: v3.5-20260815`）/ `standard` / `detailed`（HD +10）/ `extreme`（8K +20） | |
| `texture_version` | `v3.5-20260815`（新，支持 fast 和 `delight` 去光照）/ `v3.0-20250812`（默认） | |
| `geometry_quality` | `standard` / `detailed`（Ultra，+20） | 只 ≥v3.0；P1 不支持 |
| `face_limit` | 不填 = 自适应 | 上限：v3.1 三角 150 万（Ultra 200 万），quad 15 万；P1 50–20k；P2 三角 48–50k、quad 48–25k。官方建议游戏 5–10 万，网页/移动 1–5 万 |
| `quad` | 默认 false（+5） | **开了强制输出 FBX**；不填 face_limit 时默认 1 万面。P1 不支持 |
| `smart_low_poly` | 默认 false（+10） | 手工风格干净拓扑，面数固定三角 500–20k / quad 500–10k；复杂物体可能失败 |
| `generate_parts` | 默认 false（+20） | 要求 texture=false、pbr=false，且不传 quad / smart_low_poly |
| `auto_size` | 默认 false | 按真实尺寸，单位米 |
| `export_orientation` | `+x`（默认）/ `-x` / `+y` / `-y` | ⚠️**后面要 rig / retarget / texture / convert 就别设**：会静默转错（任务照样 success）。要转向最后用 convert 做 |
| `orientation`（图生 / 多视图） | `default` / `align_image` | 只在 texture=true 时生效 |
| `texture_alignment` | `original_image` / `geometry` | 图生 / 多视图 |
| `enable_image_autofix` | 默认 false | 图生：先修输入图 |
| `model_seed` / `texture_seed` / `image_seed` | int | 同 seed 同几何；换贴图不换形 = 固定 model_seed 改 texture_seed |
| `export_uv` | 默认 true | false 更快更小，UV 留给贴图阶段 |
| `compress` | `geometry` | meshopt 压缩 |

输出：`output.model_url`（glb；quad 时 fbx）+ `output.rendered_image_url`（预览图）+ `credits_consumed`。

## 动画
**顺序**：生成 → （所有网格编辑：分件 / 减面 / 转格式**都在绑骨前**做完）→ rig-check（免费）→ rig → retarget。
| | 参数 |
|---|---|
| rig `model` | `v1.0-20240301`（**默认**，只 biped，90+ 预设）/ `v2.5-20260210`（非人形）/ **`v3.0-20260909`（人形推荐，必须显式写）** |
| `rig_type` | biped / quadruped / hexapod / octopod / avian / serpentine / aquatic / others（others 要 v3.0） |
| `spec`（只 biped） | `tripo`（默认）/ `mixamo` / `actorcore` / `unity` / `unreal` / `vrm`（后四个 v3.0 才有） |
| `out_format` | glb / fbx |
| retarget 预设（v2.5 / v3.0 骨架） | biped：`preset:idle walk run dive climb jump slash shoot hurt fall turn`；`preset:quadruped:walk`；`preset:hexapod:walk`、`preset:octopod:walk`、`preset:serpentine:march`、`preset:aquatic:march` |
| retarget 预设（v1.0 骨架） | `preset:biped:xxx` 90+ 个：idle、walk、run、dance_01–06、sit、bow、cheer、cry、hug、cast_a_spell、hit_to_*、scared_*、sing_*、swagger… |
| `animation` / `animations` | 单个 / 列表（**≤5 个**，互斥） |
| `animate_in_place` | 默认 false；true = 原地播放（无根位移） |
| `export_with_geometry` | 默认 true |

⚠️取舍：v3.0 骨架质量好但只有 11 个人形动作；v1.0 骨架有 90+ 动作。要跳舞 / 情绪动作只能 v1.0。【没验到】两者实际效果。

## 推荐预设（`tripo.PRESETS`）
| 用途 | 参数 | 单价 | 说明 |
|---|---|---|---|
| **草稿 / 镜头指导**（`draft`） | `v3.1-20260211`，`texture:false`，`pbr:false`，`face_limit:10000` | 文生 **10** / 图生 20 | Blender 里摆机位、给 Seedance 当首帧几何参考。【没验到】出图质量 |
| **动画角色**（`character`） | `v3.1-20260211`，`texture:true`，`pbr:false`，`face_limit:30000`，三角面，不设朝向 | 生成 20（图生 30）+ rig 25 + 动作 10×n | 绑骨用 `rig(model="v3.0-20260909", spec="mixamo" 或 "tripo")`。quad 会变 FBX、rig 吃不吃四边【没验到】→ 先三角 |
| **静态成片资产**（`final`） | `v3.1-20260211`，`pbr:true`，`texture_quality:detailed`，`geometry_quality:detailed` | 文生 20+10+20 = **50** / 图生 60 | 要 8K 贴图改 `extreme`（再 +10） |
| 低面数游戏件（备选） | `P1-20260311`，face_limit 150–20k | 30–50 | P2 干净四边形但 100+ credit，太贵，确定要四边形再用 |

一个完整动画角色（图生 + rig + 3 个动作）≈ 30 + 25 + 30 = **85 credit ≈ 0.85 美元**（按价目表算，【没验到】）。

## ✅ 实测（2026-10-20）
- **API 积分和网页版会员点数是两个账户。** 网页订阅的 3200 点 API 用不了；会员权益里另外领了 1000 API 积分之后，`GET /account/balance` 才变成 1000。
- `text_to_model`，用 draft 预设（H3.1、无贴图、10k 面）：**10 积分，131 秒**。产出 glb 300KB + 预览 webp + sidecar json → `Projekt_Visual/MV_Cant_Kill_Us/3D/测试_人台.glb`。建任务 → 轮询 → 下载这条主路已验证。
- 官方 CLI `tripo-cli` 0.7.1 已全局安装（npm）。key 用环境变量 `TRIPO_API_KEY` 现读 creds 传进去，`TRIPO_REGION=ov`，`TRIPO_ANALYTICS_ENABLED=0`；`tripo doctor` 全过。CLI 适合手动试 / `--dry-run` 看计划；流水线用 `src/tripo.py`。
- CLI 的经验：面数预算 ≤2 万的低模用 `tripo-p1`，其他默认 `tripo-v3.1`；`tripo-p2` 是预览版，四边面要 `-p quad=true`（输出 FBX）。`mesh decimate --model v3.0 --smart-uv` 可以重拓扑 + 展 UV。

## 👤2026-10-21：先灰模，确认再上材质
- 下次生成先出**灰模**（`texture=False, pbr=False`，几何精度照样拉满 `geometry_quality=detailed`），用户确认几何合格后再用 `tripo.texture(task_id, ...)` 补材质（API 有独立的贴图任务）；或者叫用户去网页端手动做。
- 「她」模型没有彩色贴图是因为参考图本身是灰色不透明雕塑（全息体没法三维重建）；全息效果在合成里做。
- 2026-10-21 实测：4 个 final 预设模型各 60 积分、每个 64–69 MB、约 190–200 万面（四维切片前要减面）。
