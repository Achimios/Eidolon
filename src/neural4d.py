# -*- coding: utf-8 -*-
""">v<☁生成-Neural4D API v1 出 3D 模型（异步任务 + 轮询）。key 从 config/creds.local.json 的 neural4d 读，绝不打印。
只用标准库 urllib。成功后立刻下载（签名 URL 会过期，过期了重新查任务拿新链接），旁边写同名 .json 记请求参数 / task id / 花了多少 credit。
能力表、价格、和 Tripo 的对比：_设计/接口_Neural4D.md。
⚠️v1 只有「生成」「上传」「查任务」三件事：**没有单独上贴图、没有绑骨、没有 seed** → 灰模确认后再上贴图 = 重新生成（几何会变）。
余额 / 转格式只在旧接口（alb.neural4d.com:3000/api）里有，key 通不通用见文档的实测表。
"""
import json
import mimetypes
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid as _uuid
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
CREDS = ROOT / "config" / "creds.local.json"
DEFAULT_BASE = "https://api.neural4d.com/openapi/v1"
LEGACY_BASE = "https://alb.neural4d.com:3000/api"   # 旧接口：余额 / 转格式
MODEL = "dreamtech/neural4d-2.5"                    # v1 目前唯一的 3D 模型

# 推荐预设（见 _设计/接口_Neural4D.md）。v1 默认就是无贴图（with_texture=false），这里显式写出来防默认值变。
PRESETS = {
    # 灰模草稿：标准质量三角面，取最低面数 10 万（下限就是 10 万，比 Tripo 草稿 1 万高一个量级）
    "draft": {"with_texture": False, "with_pbr": False, "mesh_quality": "standard", "topology": "triangle", "target_polycount": 100000},
    # 灰模高精：几何拉满给用户确认形状（100 万面，四维切片前要减面）
    "grey_hi": {"with_texture": False, "with_pbr": False, "mesh_quality": "extra_high", "topology": "triangle", "target_polycount": 1000000},
    # 四边面灰模：出 OBJ，1k–10 万面（Tripo 要 P2 才有干净四边）
    "quad": {"with_texture": False, "with_pbr": False, "topology": "quad", "target_polycount": 50000},
    # 成片：PBR + 高质量几何
    "final": {"with_texture": True, "with_pbr": True, "mesh_quality": "high", "topology": "triangle", "target_polycount": 1000000},
}

TERMINAL_OK = "succeeded"
TERMINAL_BAD = "failed"
VIEWS = ("front", "back", "left", "right", "top", "bottom")


class Neural4DError(RuntimeError):
    def __init__(self, msg, status=None, code=None, retry_after=None, details=None):
        super().__init__(msg)
        self.status, self.code, self.retry_after, self.details = status, code, retry_after, details


def _cfg():
    c = json.loads(CREDS.read_text("utf-8"))["neural4d"]
    return c.get("base_url", DEFAULT_BASE).rstrip("/"), c["api_key"]


def _headers(key, ctype="application/json"):
    h = {"Authorization": "Bearer " + key, "X-Request-Id": "eidolon-" + _uuid.uuid4().hex[:16]}
    if ctype:
        h["Content-Type"] = ctype
    return h


def _open(r, method, path, timeout):
    """→ (HTTP 状态, json)。错误统一成 Neural4DError（带 error.code / Retry-After），消息里绝不带请求头"""
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            raw = resp.read()
            return resp.status, (json.loads(raw) if raw.strip() else {})
    except urllib.error.HTTPError as e:
        raw = e.read()[:2000]
        try:
            j = json.loads(raw)
        except ValueError:
            j = {}
        err = j.get("error") if isinstance(j.get("error"), dict) else {}
        ra = e.headers.get("Retry-After") if e.headers else None
        raise Neural4DError(
            f"Neural4D {method} {path} -> HTTP {e.code} {err.get('code') or ''}: "
            f"{err.get('message') or j.get('message') or raw[:300].decode('utf-8', 'replace')}",
            status=e.code, code=err.get("code"), retry_after=float(ra) if ra and ra.replace('.', '', 1).isdigit() else None,
            details=err.get("details")) from None


def _req(method, path, body=None, query=None, timeout=120, base=None):
    b, key = _cfg()
    url = (base or b) + path + ("?" + urllib.parse.urlencode(query) if query else "")
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    r = urllib.request.Request(url, data=data, method=method, headers=_headers(key, "application/json" if data is not None else None))
    return _open(r, method, path, timeout)[1]


def _multipart(field, path):
    """>v<☁生成-Neural4D-手搓 multipart/form-data（标准库没有）→ (body, content_type)"""
    p = Path(path)
    b = "----eidolon" + _uuid.uuid4().hex
    mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    head = (f"--{b}\r\nContent-Disposition: form-data; name=\"{field}\"; filename=\"{p.name}\"\r\n"
            f"Content-Type: {mime}\r\n\r\n").encode("utf-8")
    return head + p.read_bytes() + f"\r\n--{b}--\r\n".encode(), "multipart/form-data; boundary=" + b


def upload(path):
    """>v<☁生成-Neural4D-上传本地图片 → file id（UUID）。图 ≤20 MiB（多视图用的图 ≤10 MiB，生成时再查一次）"""
    base, key = _cfg()
    body, ctype = _multipart("file", path)
    r = urllib.request.Request(base + "/files", data=body, method="POST", headers=_headers(key, ctype))
    return _open(r, "POST", "/files", 300)[1]["id"]


def _is_uuid(s):
    try:
        return str(_uuid.UUID(s)) == s.lower()
    except (ValueError, AttributeError, TypeError):
        return False


def _ref(src, uploader=None):
    """本地路径 → 上传拿 file_id；https URL → url；UUID → 当作已上传的 file_id。⚠️只收 https（http 会被 400）"""
    s = str(src)
    if s.startswith("https://"):
        return {"url": s}
    if s.startswith("http://"):
        raise ValueError("Neural4D 只接受 HTTPS 参考图 URL")
    if _is_uuid(s):
        return {"file_id": s}
    return {"file_id": (uploader or upload)(s)}


# ---------- 请求体拼装（纯函数，离线可测）----------

def _check_common(p):
    if p.get("with_pbr") and not p.get("with_texture"):
        raise ValueError("with_pbr=true 要求 with_texture=true（白模两个都关）")
    n = p.get("n")
    if n is not None and not 1 <= int(n) <= 4:
        raise ValueError("n 只能 1–4")


def check_polycount(topology, mesh_quality, count, multiview=False):
    """文档范围：三角 standard 10–50 万；high / extra_high 50–100 万（步长 1000）；quad 1000–10 万（步长 100）；
    多视图固定 high 三角 50–100 万。不合规直接抛，免得白跑一趟 400"""
    if count is None:
        return
    count = int(count)
    if multiview:
        lo, hi, step = 500_000, 1_000_000, 1000
    elif topology == "quad":
        lo, hi, step = 1000, 100_000, 100
    elif mesh_quality in ("high", "extra_high"):
        lo, hi, step = 500_000, 1_000_000, 1000
    else:
        lo, hi, step = 100_000, 500_000, 1000
    if not (lo <= count <= hi and count % step == 0):
        raise ValueError(f"target_polycount={count} 不在 {lo}–{hi}（步长 {step}）里")


def _params(preset, extra):
    p = {**(PRESETS[preset] if preset else {}), **extra}
    _check_common(p)
    check_polycount(p.get("topology", "triangle"), p.get("mesh_quality", "standard"), p.get("target_polycount"))
    return p


def text_body(prompt, preset="draft", **extra):
    if not prompt or len(prompt) > 500:
        raise ValueError("prompt 不能空，≤500 字符")
    return {"model": MODEL, "prompt": prompt, **_params(preset, extra)}


def image_body(images, preset="draft", uploader=None, **extra):
    """images = 1 张（单图，可 n=1–4）或 2–10 张（批量，每张出一个；文档没给批量的 n / pose）"""
    imgs = [images] if isinstance(images, (str, Path)) else list(images)
    if not 1 <= len(imgs) <= 10:
        raise ValueError("input_references 1–10 张")
    p = _params(preset, extra)
    if len(imgs) > 1 and ("n" in p or "pose" in p):
        raise ValueError("批量图生（≥2 张）文档里没有 n / pose 参数")
    return {"model": MODEL, "input_references": [{"type": "image", **_ref(s, uploader)} for s in imgs], **p}


def multiview_body(views, preset="draft", uploader=None, **extra):
    """views = {"front": ..., "back": ..., 可选 left/right/top/bottom}。多视图强制 high 三角，
    不能传 mesh_quality / topology / pose（预设里有就剥掉），面数 50–100 万"""
    if "front" not in views or "back" not in views or not 2 <= len(views) <= 6:
        raise ValueError("多视图必须有 front + back，2–6 张")
    bad = set(views) - set(VIEWS)
    if bad:
        raise ValueError(f"未知视角 {bad}，只能 {VIEWS}")
    p = {k: v for k, v in {**(PRESETS[preset] if preset else {}), **extra}.items()
         if k not in ("mesh_quality", "topology", "pose")}
    if (PRESETS.get(preset) or {}).get("target_polycount", 0) < 500_000 and "target_polycount" not in extra:
        p.pop("target_polycount", None)   # 预设的低面数在多视图里不合法 → 用服务器默认 50 万
    _check_common(p)
    check_polycount("triangle", "high", p.get("target_polycount"), multiview=True)
    mv = [{"view_type": k, **_ref(v, uploader)} for k, v in views.items()]
    return {"model": MODEL, "multi_view_images": mv, **p}


# ---------- 查任务 / 轮询 / 下载 ----------

def task(tid=None, uuid=None):
    """>v<☁生成-Neural4D-查任务（免费）。id = 批次 task id（normal-3d-…）；uuid = 子模型 UUID；二选一"""
    if (tid is None) == (uuid is None):
        raise ValueError("task() 要 id 或 uuid 二选一")
    return _req("GET", "/tasks/task-info", query={"id": tid} if tid else {"uuid": uuid})


def wait(tid, poll=5.0, timeout_s=900, quiet=False):
    """>v<☁生成-Neural4D-轮询到终态（文档：每隔几秒查一次）。429 按 Retry-After 退避。成功返回整个 task dict"""
    t0 = time.time()
    last = None
    while True:
        try:
            st = task(tid)
        except Neural4DError as e:
            if e.status == 429:
                time.sleep(e.retry_after or poll * 2)
                continue
            raise
        s = st.get("status")
        subs = [m.get("status") for m in (st.get("output") or {}).get("models") or []]
        if not quiet and (s, subs) != last:
            print(f"[neural4d {tid}] {s} {subs}", flush=True)
            last = (s, subs)
        if s == TERMINAL_OK:
            return st
        if s == TERMINAL_BAD:
            raise Neural4DError(f"Neural4D 任务 {tid} failed: {st.get('error')}", code=(st.get("error") or {}).get("code"))
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"Neural4D 任务 {tid} 超时（{timeout_s}s），状态 {s}")
        time.sleep(poll)


def _download(url, out):
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=600) as resp:
        out.write_bytes(resp.read())
    return out


def _url_suffix(url):
    return Path(urlparse(url).path).suffix.lower()


def out_paths(out, n):
    """n=1 → 原名；n>1 → 名字_1 / 名字_2 …（一个请求出多个模型）"""
    out = Path(out)
    return [out] if n == 1 else [out.with_name(f"{out.stem}_{i + 1}{out.suffix}") for i in range(n)]


def plan_files(m, out):
    """一个子模型 → [(url, 本地路径)]。三角 → glb；四边 → obj (+mtl +贴图，贴图名必须和 mtl 里引用的一致 → 用 URL 里的原文件名)"""
    out = Path(out)
    jobs = []
    urls = m.get("model_urls") or {}
    main = urls.get("glb") or urls.get("obj")
    if not main:
        raise Neural4DError(f"子模型 {m.get('uuid')} 成功但没有 model_urls：{list(urls)}")
    main_path = out.with_suffix(_url_suffix(main) or ("." + (m.get("format") or "glb")))
    jobs.append((main, main_path))
    if urls.get("mtl"):
        jobs.append((urls["mtl"], main_path.with_suffix(".mtl")))
    for kind, u in (m.get("texture_urls") or {}).items():
        name = Path(urlparse(u).path).name or f"{main_path.stem}_{kind}.png"
        jobs.append((u, main_path.parent / name))
    th = (m.get("thumbnail_urls") or {}).get("front")
    if th:
        jobs.append((th, Path(str(main_path) + ".preview" + (_url_suffix(th) or ".png"))))
    return jobs


def sidecar(st, m, api, body, src, files):
    """sidecar 内容：请求参数 + task id + 子 uuid + 花费 + 服务器回报的几何参数。只放 body 本身（没有 key）"""
    keep = ("format", "mesh_quality", "pose", "target_polycount", "topology", "with_texture", "with_pbr",
            "image_enhancement", "mode", "created_at", "updated_at")
    return {"api": api, "task_id": st.get("id"), "uuid": m.get("uuid"), "model": st.get("model"), **body,
            "input_src": src, "usage_credits_task": (st.get("usage") or {}).get("credits"),
            "usage_credits_item": (m.get("usage") or {}).get("credits") if isinstance(m.get("usage"), dict) else None,
            "server": {k: m.get(k) for k in keep if k in m}, "files": [Path(p).name for _, p in files]}


def _finish(tid, api, body, out, src=None):
    """等任务 → 立刻下载每个子模型 + 预览 → 每个模型旁写 sidecar。返回 ([模型路径...], task dict)"""
    st = wait(tid)
    models = (st.get("output") or {}).get("models") or []
    paths = []
    for m, o in zip(models, out_paths(out, len(models))):
        if m.get("status") != TERMINAL_OK:
            print(f"[neural4d] 子模型 {m.get('uuid')} 状态 {m.get('status')}，跳过", flush=True)
            continue
        files = plan_files(m, o)
        for u, p in files:
            _download(u, p)
        main = files[0][1]
        Path(str(main) + ".json").write_text(json.dumps(sidecar(st, m, api, body, src, files), ensure_ascii=False, indent=1), "utf-8")
        paths.append(main)
    return paths, st


def _gen(body, out, src=None):
    j = _req("POST", "/3dmodels/generations", body)
    tid = j["id"]
    print(f"[neural4d] 建任务 {j.get('mode')} -> {tid}，预扣 {(j.get('usage') or {}).get('credits')} credit", flush=True)
    return _finish(tid, "/3dmodels/generations", body, out, src)


def text_to_model(prompt, out, preset="draft", **extra):
    """>v<☁生成-Neural4D-文生模型。preset = draft / grey_hi / quad / final / None；extra 原样进请求（覆盖预设）"""
    return _gen(text_body(prompt, preset, **extra), out)


def image_to_model(images, out, preset="draft", **extra):
    """>v<☁生成-Neural4D-图生模型。images = 本地路径（自动上传）/ https URL / file_id；列表 2–10 张 = 批量（每张一个模型）"""
    body = image_body(images, preset, **extra)
    return _gen(body, out, src=[str(s) for s in images] if isinstance(images, (list, tuple)) else str(images))


def multiview_to_model(views, out, preset="draft", **extra):
    """>v<☁生成-Neural4D-多视图生模型。views = {"front":…, "back":…, …}，强制 high 三角 50–100 万面"""
    body = multiview_body(views, preset, **extra)
    return _gen(body, out, src={k: str(v) for k, v in views.items()})


# ---------- 旧接口（alb.neural4d.com:3000/api）----------

def balance():
    """>v<☁生成-Neural4D-余额（旧接口 POST /queryPointsInfo，免费）→ {total_points, monthly_bonus_points, permanent_points, year_points, expiration_day}"""
    j = _req("POST", "/queryPointsInfo", base=LEGACY_BASE)
    if j.get("statusType") != 0:
        raise Neural4DError(f"Neural4D 余额查询失败：{j.get('message')}")
    return j.get("data")


def convert(model_uuid, out, model_type="fbx", model_size=2, poll=5.0, timeout_s=600):
    """>v<☁生成-Neural4D-转格式（旧接口，**10 point**）。model_type = fbx / glb / obj / stl / blend / usdz；
    model_size = 目标尺寸（毫米，≥2）。statusType 1 = 转换中 → 重发同一请求轮询。fbx 返回的是 zip"""
    body = {"uuid": model_uuid, "modelType": model_type, "modelSize": model_size}
    t0 = time.time()
    while True:
        j = _req("POST", "/convertToFormat", body, base=LEGACY_BASE)
        s = j.get("statusType")
        if s == 0:
            u = j["modelUrl"]
            p = _download(u, Path(out).with_suffix(_url_suffix(u) or Path(out).suffix))
            Path(str(p) + ".json").write_text(json.dumps({"api": "legacy /convertToFormat", **body,
                                                          "scaleNumber": j.get("scaleNumber")}, ensure_ascii=False, indent=1), "utf-8")
            return p
        if s != 1:
            raise Neural4DError(f"Neural4D 转格式失败：{j.get('message')}")
        if time.time() - t0 > timeout_s:
            raise TimeoutError("Neural4D 转格式超时")
        time.sleep(poll)
