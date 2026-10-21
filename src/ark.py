# -*- coding: utf-8 -*-
""">v<☁生成-火山方舟-Seedream 出图 / Seedance 出视频（异步任务 + 轮询）。key 从 config/creds.local.json 读，绝不打印。
只用标准库 urllib。生成物下载到作品文件夹，旁边写一个同名 .json 记 prompt / 模型 / 参数（ran_with 的来源）。
"""
import base64
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CREDS = ROOT / "config" / "creds.local.json"


def _cfg():
    c = json.loads(CREDS.read_text("utf-8"))["volc_ark"]
    return c["base_url"], c["api_key"], c.get("models", {})


def _req(method, path, body=None, timeout=120):
    base, key, _ = _cfg()
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    r = urllib.request.Request(base + path, data=data, method=method,
                               headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Ark {method} {path} -> HTTP {e.code}: {e.read()[:500].decode('utf-8', 'replace')}") from None


def model(name):
    return _cfg()[2].get(name, name)


def _download(url, out):
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=300) as resp:
        out.write_bytes(resp.read())
    return out


_MIME = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", ".webp": "webp", ".bmp": "bmp", ".wav": "wav", ".mp3": "mp3"}


def media(src):
    """>v<☁生成-素材-本地文件 → data URI（图片 / 音频可以 base64；⚠️视频不行，只能 URL）；http 开头原样返回"""
    s = str(src)
    if s.startswith(("http://", "https://", "data:", "asset://")):
        return s
    p = Path(s)
    kind = "audio" if p.suffix.lower() in (".wav", ".mp3") else "image"
    return f"data:{kind}/{_MIME[p.suffix.lower()]};base64," + base64.b64encode(p.read_bytes()).decode()


def _short(v):
    """sidecar 里别存整段 base64"""
    if isinstance(v, str) and v.startswith("data:"):
        return v[:40] + f"…({len(v)} chars)"
    if isinstance(v, list):
        return [_short(x) for x in v]
    if isinstance(v, dict):
        return {k: _short(x) for k, x in v.items()}
    return v


def image(prompt, out, mdl="seedream_pro", size="2K", refs=None, sources=None, **extra):
    """>v<☁生成-出图-Seedream；返回本地路径。refs = 参考图（本地路径或 url，pro/flash ≤10 张，prompt 里叫「图1、图2…」）。
    sources = refs 的人类可读来源（写进 sidecar）。extra 原样进请求（如 layer_decomposition / output_format）"""
    body = {"model": model(mdl), "prompt": prompt, "response_format": "url", "size": size, "watermark": False, **extra}
    if refs:
        body["image"] = [media(r) for r in refs] if len(refs) > 1 else media(refs[0])
    r = _req("POST", "/images/generations", body, timeout=300)
    p = _download(r["data"][0]["url"], out)
    side = {"api": "images/generations", **_short(body), "refs_src": [str(x) for x in (sources or refs or [])], "usage": r.get("usage")}
    if len(r["data"]) > 1:  # 图层拆分：其余图层存成 out_layerN.png
        side["layers"] = []
        for d in r["data"][1:]:
            lp = _download(d["url"], Path(out).with_name(f"{Path(out).stem}_layer{d.get('z_index')}.png"))
            side["layers"].append({"file": lp.name, **{k: d.get(k) for k in ("z_index", "name", "description", "bounding_box")}})
    Path(str(p) + ".json").write_text(json.dumps(side, ensure_ascii=False, indent=1), "utf-8")
    return p


def img(src, role="reference_image"):
    """>v<☁生成-素材-Seedance 图片项；role = first_frame / last_frame / reference_image（三种模式互斥）"""
    return {"type": "image_url", "image_url": {"url": media(src)}, "role": role}


def aud(src):
    return {"type": "audio_url", "audio_url": {"url": media(src)}, "role": "reference_audio"}


def vid(url):
    """⚠️视频只能 https URL / asset://。2026-10-21 实测：base64、Files API 的 file-id、file:// 都被 400 拒
    （见 _设计/接口_生成.md「本地视频当参考」）→ 本地路径在这里直接报错，别白建任务"""
    s = str(url)
    if not s.startswith(("http://", "https://", "asset://")):
        raise ValueError(f"Seedance 参考视频只收 http(s) URL 或 asset://，收到 {s[:60]!r}；本地渲染要先传到能 GET 的地址（如 TOS 预签名 URL）")
    return {"type": "video_url", "video_url": {"url": s}, "role": "reference_video"}


def video(content, out, mdl="seedance_25", ratio="16:9", duration=5, poll=5, timeout_s=1800, generate_audio=False, **extra):
    """>v<☁生成-出视频-Seedance 异步任务：建任务 → 轮询 → 下载。content 是 Ark 的 content 列表（text + img()/aud()/vid()）。
    首帧 / 首尾帧模式下 2.5 只接受 ratio="adaptive"。默认不生成声音（音乐用原曲）。
    extra：draft=True（2.5 样片 480p）/ return_last_frame=True（尾帧另存 out.last.jpg）/ resolution / output_format="mov" …"""
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    body = {"model": model(mdl), "content": content, "ratio": ratio, "duration": duration, "watermark": False, "generate_audio": generate_audio, **extra}
    task = _req("POST", "/contents/generations/tasks", body)
    tid = task["id"]
    t0 = time.time()
    while True:
        st = _req("GET", f"/contents/generations/tasks/{tid}")
        s = st.get("status")
        if s == "succeeded":
            url = st["content"]["video_url"]
            p = _download(url, out)
            if st["content"].get("last_frame_url"):
                _download(st["content"]["last_frame_url"], Path(str(p) + ".last.jpg"))
            Path(str(p) + ".json").write_text(json.dumps({"api": "contents/generations/tasks", "task_id": tid, **_short(body), "usage": st.get("usage"),
                                                          "seed": st.get("seed"), "resolution": st.get("resolution"), "draft": st.get("draft")}, ensure_ascii=False, indent=1), "utf-8")
            return p
        if s in ("failed", "cancelled", "expired"):
            raise RuntimeError(f"Seedance 任务 {tid} {s}: {st.get('error')}")
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"Seedance 任务 {tid} 超时（{timeout_s}s），状态 {s}")
        time.sleep(poll)
