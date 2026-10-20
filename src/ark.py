# -*- coding: utf-8 -*-
""">v<☁生成-火山方舟-Seedream 出图 / Seedance 出视频（异步任务 + 轮询）。key 从 config/creds.local.json 读，绝不打印。
只用标准库 urllib。生成物下载到作品文件夹，旁边写一个同名 .json 记 prompt / 模型 / 参数（ran_with 的来源）。
"""
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


def image(prompt, out, mdl="seedream_pro", size="2K", **extra):
    """>v<☁生成-出图-Seedream；返回本地路径。extra 原样进请求（如 image=参考图 url）"""
    body = {"model": model(mdl), "prompt": prompt, "response_format": "url", "size": size, "stream": False, "watermark": False, **extra}
    r = _req("POST", "/images/generations", body)
    url = r["data"][0]["url"]
    p = _download(url, out)
    Path(str(p) + ".json").write_text(json.dumps({"api": "images/generations", **{k: v for k, v in body.items()}, "usage": r.get("usage")}, ensure_ascii=False, indent=1), "utf-8")
    return p


def video(content, out, mdl="seedance_25", ratio="16:9", duration=5, poll=5, timeout_s=900, **extra):
    """>v<☁生成-出视频-Seedance 异步任务：建任务 → 轮询 → 下载。content 是 Ark 的 content 列表（text / image_url / video_url / audio_url + role）"""
    body = {"model": model(mdl), "content": content, "ratio": ratio, "duration": duration, "watermark": False, **extra}
    task = _req("POST", "/contents/generations/tasks", body)
    tid = task["id"]
    t0 = time.time()
    while True:
        st = _req("GET", f"/contents/generations/tasks/{tid}")
        s = st.get("status")
        if s == "succeeded":
            url = st["content"]["video_url"]
            p = _download(url, out)
            Path(str(p) + ".json").write_text(json.dumps({"api": "contents/generations/tasks", "task_id": tid, **body, "usage": st.get("usage")}, ensure_ascii=False, indent=1), "utf-8")
            return p
        if s in ("failed", "cancelled", "expired"):
            raise RuntimeError(f"Seedance 任务 {tid} {s}: {st.get('error')}")
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"Seedance 任务 {tid} 超时（{timeout_s}s），状态 {s}")
        time.sleep(poll)
