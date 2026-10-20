# -*- coding: utf-8 -*-
""">v<☁生成-Tripo V3 出 3D 模型（异步任务 + 轮询）。key 从 config/creds.local.json 的 tripo 读，绝不打印。
只用标准库 urllib。生成物一成功就下载（⚠️model_url 只活 5 分钟），旁边写同名 .json 记请求参数 / task_id / 花了多少 credit。
能力表与推荐预设：_设计/接口_Tripo.md。⚠️V2（api.tripo3d.ai/v2）2026-11-01 停用，这里只走 V3。
"""
import json
import mimetypes
import time
import urllib.request
import uuid
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
CREDS = ROOT / "config" / "creds.local.json"
DEFAULT_BASE = "https://openapi.tripo3d.ai/v3"

# 推荐预设（见 _设计/接口_Tripo.md；credit 是官网价目表 2026-10-20）
PRESETS = {
    # 草稿 / 镜头指导：H3.1 白模，10 credit（文生）/ 20（图生）
    "draft": {"model": "v3.1-20260211", "texture": False, "pbr": False, "face_limit": 10000},
    # 动画角色：贴图但不 PBR（绑骨后再看要不要换贴图），不设朝向，三角面（rig 吃 GLB）
    "character": {"model": "v3.1-20260211", "texture": True, "pbr": False, "face_limit": 30000},
    # 静态成片资产：PBR + HD 贴图 + Ultra 几何
    "final": {"model": "v3.1-20260211", "texture": True, "pbr": True, "texture_quality": "detailed", "geometry_quality": "detailed"},
}


def _cfg():
    c = json.loads(CREDS.read_text("utf-8"))["tripo"]
    return c.get("base_url", DEFAULT_BASE).rstrip("/"), c["api_key"]


def _open(r, method, path, timeout):
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            j = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Tripo {method} {path} -> HTTP {e.code}: {e.read()[:500].decode('utf-8', 'replace')}") from None
    if j.get("code", 0) != 0:
        raise RuntimeError(f"Tripo {method} {path} -> code {j.get('code')}: {j.get('message')} / {j.get('suggestion')}")
    return j.get("data")


def _req(method, path, body=None, timeout=120):
    base, key = _cfg()
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    r = urllib.request.Request(base + path, data=data, method=method,
                               headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    return _open(r, method, path, timeout)


def _multipart(field, path):
    """>v<☁生成-Tripo-手搓 multipart/form-data（标准库没有）→ (body, content_type)"""
    p = Path(path)
    b = "----eidolon" + uuid.uuid4().hex
    mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    head = (f"--{b}\r\nContent-Disposition: form-data; name=\"{field}\"; filename=\"{p.name}\"\r\n"
            f"Content-Type: {mime}\r\n\r\n").encode("utf-8")
    return head + p.read_bytes() + f"\r\n--{b}--\r\n".encode(), "multipart/form-data; boundary=" + b


def upload(path):
    """>v<☁生成-Tripo-上传本地图片 / 模型 → file_token（图 ≤20MB，模型 ≤150MB；>60MB 官方建议 /files/presign，这里没做）"""
    base, key = _cfg()
    body, ctype = _multipart("file", path)
    r = urllib.request.Request(base + "/files", data=body, method="POST",
                               headers={"Content-Type": ctype, "Authorization": "Bearer " + key})
    return _open(r, "POST", "/files", 300)["file_token"]


def _input(src):
    """本地路径 → 先上传拿 file_token；task_ / file_ / http 开头原样用"""
    s = str(src)
    if s.startswith(("http://", "https://", "task_", "file_")) or (len(s) == 36 and s.count("-") == 4):
        return s
    return upload(s)


def balance():
    """>v<☁生成-Tripo-余额 {balance, frozen}（不花钱）"""
    return _req("GET", "/account/balance")


def usage():
    return _req("GET", "/account/usage")


def task(tid):
    return _req("GET", f"/tasks/{tid}")


def wait(tid, poll=2.0, timeout_s=900, quiet=False):
    """>v<☁生成-Tripo-轮询到终态。官方建议 1–2 s 一次（查询接口限额比生成高）。成功返回整个 task dict"""
    t0 = time.time()
    last = None
    while True:
        st = task(tid)
        s = st.get("status")
        if not quiet and (s, st.get("progress")) != last:
            print(f"[tripo {tid}] {s} {st.get('progress')}%", flush=True)
            last = (s, st.get("progress"))
        if s == "success":
            return st
        if s in ("failed", "banned", "expired", "cancelled"):
            raise RuntimeError(f"Tripo 任务 {tid} {s}: {st.get('error_code')} {st.get('error_message')}")
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"Tripo 任务 {tid} 超时（{timeout_s}s），状态 {s}")
        time.sleep(poll)


def _download(url, out):
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=600) as resp:
        out.write_bytes(resp.read())
    return out


def _url_suffix(url):
    return Path(urlparse(url).path).suffix.lower()


def _fix_suffix(out, url):
    """quad=true 会强制出 FBX；文件名后缀跟着服务器实际给的格式走，别把 fbx 存成 .glb"""
    out = Path(out); suf = _url_suffix(url)
    return out.with_suffix(suf) if suf and suf != out.suffix.lower() else out


def _finish(tid, api, body, out, src=None):
    """等任务 → 立刻下载 model_url（5 分钟过期）+ 预览图 → 写 sidecar。返回 (模型路径, task dict)"""
    st = wait(tid)
    o = st.get("output") or {}
    url = o.get("model_url") or o.get("pbr_model") or o.get("model")
    if not url:
        raise RuntimeError(f"Tripo 任务 {tid} 成功但 output 里没有模型链接：{list(o)}")
    p = _download(url, _fix_suffix(out, url))
    side = {"api": api, "task_id": tid, "type": st.get("type"), **body, "input_src": src,
            "credits_consumed": st.get("credits_consumed"), "created_at": st.get("created_at"),
            "completed_at": st.get("completed_at"), "output_keys": sorted(o)}
    if o.get("rendered_image_url"):
        pv = Path(str(p) + ".preview" + (_url_suffix(o["rendered_image_url"]) or ".png"))
        _download(o["rendered_image_url"], pv)
        side["preview"] = pv.name
    Path(str(p) + ".json").write_text(json.dumps(side, ensure_ascii=False, indent=1), "utf-8")
    return p, st


def _gen(path, body, out, src=None):
    tid = _req("POST", path, body)["task_id"]
    print(f"[tripo] 建任务 {path} -> {tid}", flush=True)
    return _finish(tid, path, body, out, src)


def _params(preset, extra):
    return {**(PRESETS[preset] if preset else {}), **extra}


def text_to_model(prompt, out, preset="draft", **extra):
    """>v<☁生成-Tripo-文生模型。preset = draft / character / final / None；extra 原样进请求（覆盖预设）。
    ⚠️后面要 rig / retarget / texture / convert 就别传 export_orientation（会静默转错，最后用 convert 再转）"""
    body = {"prompt": prompt, **_params(preset, extra)}
    return _gen("/generation/text-to-model", body, out)


def image_to_model(image, out, preset="draft", **extra):
    """>v<☁生成-Tripo-图生模型。image = 本地路径（自动上传）/ URL / file_token / 出图 task_id"""
    body = {"input": _input(image), **_params(preset, extra)}
    return _gen("/generation/image-to-model", body, out, src=str(image))


def multiview_to_model(views, out, preset="draft", **extra):
    """>v<☁生成-Tripo-多视图生模型。views = {"front": ..., "left": ..., "back": ..., "right": ...}，front 必填，至少 2 张"""
    if "front" not in views or len(views) < 2:
        raise ValueError("multiview 需要 front + 至少再一张")
    inputs = [{k: _input(v)} for k, v in views.items()]
    body = {"inputs": inputs, **_params(preset, extra)}
    return _gen("/generation/multiview-to-model", body, out, src={k: str(v) for k, v in views.items()})


def rig_check(src):
    """>v<☁生成-Tripo-能不能绑骨（免费）→ {"riggable": bool, "rig_type": "biped"…}"""
    tid = _req("POST", "/animations/rig-check", {"input": _input(src)})["task_id"]
    return wait(tid).get("output")


def rig(src, out, model="v3.0-20260909", rig_type="biped", spec="tripo", out_format="glb", **extra):
    """>v<☁生成-Tripo-自动绑骨（25 credit）。人形用 v3.0-20260909（必须显式指定，默认还是 v1.0）。
    spec 只对 biped 生效：tripo / mixamo / actorcore / unity / unreal / vrm。返回 (路径, task)；task["task_id"] 给 retarget 用"""
    body = {"input": _input(src), "model": model, "rig_type": rig_type, "spec": spec, "out_format": out_format, **extra}
    return _gen("/animations/rig", body, out, src=str(src))


def retarget(rig_task_id, animations, out, out_format="glb", animate_in_place=False, **extra):
    """>v<☁生成-Tripo-套预设动作（10 credit / 个，一次 ≤5 个）。animations = 字符串或列表，
    v2.5/v3.0 骨架用 preset:idle / preset:walk…；v1.0 骨架用 preset:biped:xxx（90+ 个）"""
    anims = [animations] if isinstance(animations, str) else list(animations)
    if len(anims) > 5:
        raise ValueError("retarget 一次最多 5 个动作")
    body = {"input": rig_task_id, "out_format": out_format, "animate_in_place": animate_in_place, **extra}
    body.update({"animation": anims[0]} if len(anims) == 1 else {"animations": anims})
    return _gen("/animations/retarget", body, out)


def convert(src, out, format="FBX", **extra):
    """>v<☁生成-Tripo-转格式（GLTF/FBX/USDZ/OBJ/STL/3MF；5 credit，带 quad/face_limit/texture_size 等算高级 10）"""
    body = {"input": _input(src), "format": format, **extra}
    return _gen("/models/convert", body, out, src=str(src))


def decimate(src, out, model="v3.0", **extra):
    """>v<☁生成-Tripo-智能减面 / 重拓扑（v3.0 三角 35 / 四边 40；v2.0 30；v1.0 10，face_limit 必填）"""
    body = {"input": _input(src), "model": model, **extra}
    return _gen("/mesh/decimate", body, out, src=str(src))


def texture(src, out, **extra):
    """>v<☁生成-Tripo-重新上贴图（10–30 credit）；texture_prompt={"text": …} / {"image": …} / {"images": [...]}"""
    body = {"input": _input(src), **extra}
    return _gen("/models/texture", body, out, src=str(src))
