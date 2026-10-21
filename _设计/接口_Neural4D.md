# 接口：Neural4D API v1 能做什么（2026-10-22 查官方文档 + 免费实测）

> 来源：docs.neural4d.com（`/llms.txt` 索引下的全部英文页：v1 Quickstart、Common headers、Upload a file、Create 3D model generation tasks、Retrieve a task；旧接口的 Quickstart、Query remaining balance、Convert model format、Generate model from text / image、Retrieve model、Query job progress）+ 官网 /pricing、/api。客户端：`src/neural4d.py`，离线自检 `tests/test_neural4d.py`。
> v1 base：`https://api.neural4d.com/openapi/v1`，`Authorization: Bearer <key>`，可选 `X-Request-Id`（1–128 个 `[A-Za-z0-9._-]`，原样回传，查日志用）。⚠️**v1 只对付费订阅用户开放**（403 `subscription_required`）。
> 旧接口 base：`https://alb.neural4d.com:3000/api`（全是 POST，驼峰参数）。v1 里**没有**余额和转格式，只能走旧接口。
> 【没验到】= 文档写了但我们还没实测。

## 实测过的（2026-10-22，全部免费，余额前后都是 4945 → 没扣钱）
| 调用 | 结果 |
|---|---|
| 旧 `POST /queryPointsInfo`（余额） | ✅ **v1 的 key 在旧接口上也能用**。返回 `total_points 4945`、`monthly_bonus_points 4945`、`permanent_points 0`、`year_points 0`、`credit_cycle "MONTH"`、`last_credits_refresh 2026-09-09`。比文档多了 `weekly_bonus_points / credit_cycle / next_refresh_date / last_credits_refresh` 字段 |
| v1 `GET /tasks/task-info?id=<不存在>` | ✅ HTTP 404 `task_not_found` → 鉴权通过 |
| v1 `GET /tasks/task-info?uuid=<不存在>` | ✅ 同上 404 |
| v1 `POST /3dmodels/generations`（故意填错 model） | ✅ HTTP 400 `invalid_request`，没建任务、没扣钱 |
| v1 `POST /files`（8×8 灰色 png） | ✅ 201，返回 file id（UUID）。上传免费 |
| **生成 → 轮询 → 下载** | 【没验到】要花点数，先报价等用户点头（见下「价格」） |

错误格式统一：`{"error": {"code", "message", "details"?}}`，**按 `error.code` 判断**，别看 message。成功不包壳，直接是对象。

## 端点（v1）
| 能力 | 端点 | 价格 | 备注 |
|---|---|---|---|
| 上传 | `POST /files`（multipart，字段 `file`） | 免费 ✅ | → `id`（UUID）。图 ≤20 MiB（jpg/png/gif/webp/bmp/heic/heif），视频 ≤100 MiB，音频 ≤15 MiB；超了 400 `file_too_large` |
| 文生 / 单图 / 批量图 / 多视图生模型 | `POST /3dmodels/generations`（同一个端点，按字段区分模式） | 文档示例：**一个模型 30 credit**，两张图批量 60【没验到选项是否加价】 | 返回 **202** + 批次 `id`（形如 `normal-3d-<uuid>`）+ 每个子模型的 `uuid` + `usage.credits` |
| 查任务 | `GET /tasks/task-info?id=…` 或 `?uuid=…`（二选一） | 免费 ✅ | 状态 `queued → processing → succeeded / failed`；批次和每个子模型各有状态 |
| 出图 / 出视频 | `POST /images/generations`、`/videos/generations` | — | 没细看（转售 gpt-image-2 / nano-banana / seedream-4.5 / flux-2 / seedance-2.0 / veo-3.1 / kling-3.0 等）。我们出图出视频走方舟 |

### 旧接口（`alb.neural4d.com:3000/api`）
| 能力 | 端点 | 价格 | 备注 |
|---|---|---|---|
| 余额 | `POST /queryPointsInfo` | 免费 ✅ | `statusType 0` = 有效 |
| 转格式 | `POST /convertToFormat` `{uuid, modelType, modelSize}` | **10 point** | `modelType`：fbx / glb / obj / stl / blend / usdz；`modelSize` = 目标尺寸**毫米**（≥2）。`statusType 1` = 转换中 → 重发同一请求轮询；fbx 返回 zip。⚠️**v1 生成的 uuid 能不能用【没验到】** |
| 查模型 / 进度 | `POST /retrieveModel`、`/queryJobProgress` | 免费 | 旧版用，v1 有 task-info 就不用了 |
| 旧版生成 | `/generateModelWithText`、`/generateModelWithImage`（multipart，≤50 MB） | — | 旧参数 `modelCount`（默认 **4**！）、`disablePbr`、`onlyGenerateMesh`、`faceNum`。不用 |
| Q 版角色 | `/generateChibiModelWithImage` | 30 point | 要提前配置 style tag，不用 |
| 人像检测 | `/detectPortrait` | — | 不用 |

## 生成参数（v1 `POST /3dmodels/generations`）
| 参数 | 取值 | 说明 |
|---|---|---|
| `model` | **只有** `dreamtech/neural4d-2.5` | 必填 |
| `prompt`（文生） | ≤500 字符 | 有内容审核（`invalid_prompt`） |
| `input_references`（图生） | `[{"type":"image","file_id"或"url"}]`，1 张 = 单图，**2–10 张 = 批量（每张出一个模型）** | jpg/png/webp，≤20 MiB；URL 必须 **HTTPS**、不带账密 |
| `multi_view_images`（多视图） | `[{"view_type","file_id"或"url"}]`，2–6 张，**必须有 front 和 back**，视角 front/back/left/right/top/bottom 各最多一次 | ≤10 MiB/张；**强制 high 三角**，不能传 `mesh_quality / topology / pose`；面数 50–100 万（默认 50 万） |
| `with_texture` | 默认 **false** | ⭐**默认就是灰模**（和 Tripo 相反） |
| `with_pbr` | 默认 false | true 要求 with_texture=true |
| `mesh_quality` | `standard`（默认）/ `high` / `extra_high` | |
| `topology` | `triangle`（默认）/ `quad` | 四边面输出 **OBJ**（+mtl+贴图） |
| `target_polycount` | 三角 standard **10–50 万**（默认 50 万，步长 1000）；三角 high / extra_high **50–100 万**（默认 100 万）；quad **1000–10 万**（默认 5 万，步长 100） | ⚠️**三角面最低 10 万**，出不了低模。不合规 400 `invalid_target_polycount`（客户端先本地校验） |
| `pose` | `auto`（默认）/ `A-pose` / `T-pose` | 文生 + 单图有效；批量 / 多视图忽略 |
| `image_enhancement` | 默认 false | 图生前先增强输入图 |
| `n` | 1–4 | 一次出几个（文生 / 单图 / 多视图）。**批量图生文档里没有 n** |

**没有的东西**（对比 Tripo 要记住）：seed、负向提示词、朝向、真实尺寸、分件、只换贴图、减面 / 重拓扑、绑骨、动作。

### 输出（`GET /tasks/task-info` → `output.models[]`）
| 模型类型 | `model_urls` | `texture_urls` | 文件数 |
|---|---|---|---|
| 三角（灰模 / 贴图 / PBR） | `glb` | 无（材质在 glb 里） | 1 |
| 四边灰模 | `obj` | 无 | 1 |
| 四边 + 普通贴图 | `obj` + `mtl` | `albedo` | 3 |
| 四边 + PBR | `obj` + `mtl` | `albedo / normal / roughness / metallic` | 6 |

另有 `thumbnail_urls.front`（预览图），以及服务器回报的 `mesh_quality / pose / target_polycount / topology / with_texture / with_pbr`。
链接是**临时签名 URL**，过期了**重新查一次任务就能拿新链接**（比 Tripo 的 5 分钟硬过期宽松；具体多久【没验到】）。贴图文件名要和 mtl 里的引用一致 → 客户端按 URL 原文件名存。

### 错误码要点
- 401 `invalid_api_key`；403 `insufficient_credits` / `subscription_required`。
- 429 `generation_limit_exceeded`（额度 / 并发）/ `rate_limit_exceeded` / `reference_import_busy`，看 `Retry-After`。⚠️**文档没给具体限额数字**【没验到】。官网价目表说各档有「并发任务数」，但页面上是动态加载的，没抓到数。
- ⚠️**500 `task_registration_failed`**（或 `error.details.registration_failed=true`）：任务其实已经排上、已经扣钱了，只是登记失败 → **绝对不要重发**，会再扣一次。`error.details.task` 里有 task id 和 uuid。客户端不会自动重试生成请求。
- 408 / 413 / 415 / 422：参考图下载超时 / 太大 / 格式不对 / 下不下来。

## 价格【没验到】
- 文档里**没有价目表**。唯一的数字是示例里 `usage.credits`：文生一个模型 30，两张批量 60 → 看起来**每个模型 30 credit**。贴图 / PBR / high / extra_high / quad 加不加价不知道。
- 余额是**订阅的月度点数**（4945，`credit_cycle MONTH`）。看起来和网页版是同一个池（不像 Tripo 那样 API 和网页分账）——**推断，没对照验证**。
- 官网 /api 页说「高精度生成约 0.15 美元 / 次」；第三方页面（GetApp）列的订阅：Go 19.9 美元 / 1500 点、Plus 39.9 / 5000、Pro 99.9 / 25000。按 Plus 算 1 点 ≈ 0.008 美元 → 30 点 ≈ **0.24 美元**。都是二手数，以真扣的为准。
- 真跑一次生成后，看 202 返回的 `usage.credits` 和余额差，把这一节改成实测。

## 推荐预设（`neural4d.PRESETS`）
| 用途 | 参数 | 说明 |
|---|---|---|
| **灰模草稿**（`draft`） | standard 三角，无贴图，**10 万面**（下限） | 镜头指导 / 形状确认 |
| **灰模高精**（`grey_hi`） | extra_high 三角，无贴图，100 万面 | 👤先灰模确认几何；四维切片前要减面 |
| **四边灰模**（`quad`） | quad，无贴图，5 万面 → OBJ | Tripo 要 P2（100+ credit）才有干净四边；这里是同一个模型的一个开关。拓扑质量【没验到】 |
| **成片**（`final`） | high 三角，贴图 + PBR，100 万面 | 价格【没验到】 |

## 和 Tripo 比（就我们的需求）
| 需求 | Tripo V3 | Neural4D v1 |
|---|---|---|
| **先灰模，确认后再上贴图** | ✅ `/models/texture` 在**同一个几何**上补贴图（10–30） | ❌ **没有单独贴图接口、也没有 seed** → 要贴图只能重新生成，几何会变。办法：①一开始就 `with_texture=true` 多出几个（`n` 最多 4）挑；②灰模定了以后拿 glb 去 Tripo / 别处上贴图（Tripo 吃不吃外部上传的模型做 texture【没验到】） |
| 低面数 | ✅ 1 万面草稿；P1 低模 | ❌ 三角最低 10 万；要低就得 quad（≥1000）或回 Blender 减面 |
| 高精几何 | Ultra ~200 万面（我们实测 final 每个 64–69 MB） | 官网号称 2048³ 体素分辨率、50–100 万面。哪个细【没验到】，要拿同一张图对比 |
| 四边面 | quad 开关（+5，强制 FBX，上限 15 万）/ P2 干净四边（贵） | quad 开关，1000–10 万，出 OBJ |
| 绑骨 / 动作 | ✅ rig + retarget（25 + 10/个） | ❌ 没有。只有 `pose: A-pose / T-pose` 让外部绑骨（Mixamo / Blender Rigify）好做。官网说「rigging-ready」= 只是姿势和 UV 好 |
| 多视图 | front 必填 + ≥1 张 | front + back 必填，最多 6 张（含 top / bottom） |
| 一次多个变体 | 一个任务一个 | `n` 1–4 / 批量图 2–10 张 |
| 格式 | glb / fbx；convert 5–10 | glb / obj；转 fbx / stl / blend / usdz 走旧接口 10 point |
| 链接过期 | 5 分钟硬过期 | 重新查任务拿新链接 |
| 余额 | API 和网页分账，余额要另领 | 看起来和订阅同池（推断） |

**结论（暂定，等真跑一次再定）**：灰模 + 高精几何 + 四边面 → Neural4D 值得拿同一张图和 Tripo 对比一次；要绑骨动画、要「同一几何补贴图」、要低模 → 还是 Tripo。

## 真跑一次的报价（等用户点头）
最小的一次：`text_to_model("a simple humanoid mannequin in A-pose", ..., preset="draft")`，n=1，文档示例价 **≈30 credit**（余额 4945 的 0.6%）。
更有用的一次：拿「她」的灰色参考图跑 `image_to_model(..., preset="grey_hi", pose="A-pose")`，和 Tripo 的 final 结果对比几何，价格同样【没验到】（预计 30 起）。
