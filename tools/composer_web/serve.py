#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🎛作曲器-本地服务-放页面 + 读作品（感官.json 的重拍、分轨切片）+ 存 作曲_vN.json / 会话 / 哼唱 wav。
只监听 127.0.0.1。存档一律按版本号另起文件名，文件已存在就拒绝（409），绝不覆盖。
用法：.venv/Scripts/python.exe tools/composer_web/serve.py [--port 8765] [--root C:/Users/Administrator/Desktop/Projekt_Visual] [--no-browser]
输出：<作品>/作曲/作曲_vN.json（真值，给画面 / 感受器 / Python 渲染读）、作曲_vN_会话.json（能接着玩）、哼唱_vN_第K遍.wav（24 bit）。
"""
import argparse
import base64
import io
import json
import re
import sys
import threading
import webbrowser
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = Path(r"C:/Users/Administrator/Desktop/Projekt_Visual")
OUT_DIR = "作曲"
VER_RE = re.compile(r"^作曲_v(\d+)(?:_会话)?\.json$")
MAX_BODY = 400 * 1024 * 1024


# ---------- 纯函数（测试直接调）----------
def safe_work(root: Path, name: str) -> Path:
    """作品名只能是 root 下面的一级文件夹（不许 .. / 斜杠）。"""
    if not name or "/" in name or "\\" in name or name in (".", "..") or name.startswith("."):
        raise ValueError(f"作品名不合法：{name!r}")
    p = (root / name).resolve()
    if p.parent != root.resolve() or not p.is_dir():
        raise ValueError(f"没有这个作品：{name!r}")
    return p


def safe_name(name: str) -> str:
    """作曲/ 里的文件名：只许一级、不许路径。"""
    if not name or any(c in name for c in "/\\:") or name.startswith(".") or ".." in name:
        raise ValueError(f"文件名不合法：{name!r}")
    return name


def next_version(out: Path) -> int:
    n = 0
    if out.is_dir():
        for f in out.iterdir():
            m = VER_RE.match(f.name)
            if m:
                n = max(n, int(m.group(1)))
            m2 = re.match(r"^哼唱_v(\d+)_", f.name)
            if m2:
                n = max(n, int(m2.group(1)))
    return n + 1


def list_works(root: Path):
    out = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and not d.name.startswith((".", "_")):
            out.append({"name": d.name, "has_sense": (d / "感官.json").is_file(), "has_stems": (d / "分轨").is_dir()})
    return out


def read_sense(work: Path):
    j = json.loads((work / "感官.json").read_text(encoding="utf-8"))
    ev = j.get("events", {})
    return {"bpm": j.get("bpm"), "duration": j.get("duration"), "downbeats": ev.get("downbeats", []), "beat_source": j.get("beat_source")}


def list_stems(work: Path):
    d = work / "分轨"
    if not d.is_dir():
        return []
    return sorted(f.name for f in d.iterdir() if f.suffix.lower() == ".wav" and f.is_file())


def stem_slice(work: Path, name: str, t0: float, t1: float) -> bytes:
    """分轨的 [t0, t1) 秒（歌曲播放时间）→ 16 bit wav 字节。超出文件的部分补静音。"""
    p = work / "分轨" / safe_name(name)
    info = sf.info(str(p))
    sr = info.samplerate
    a, z = int(round(t0 * sr)), int(round(t1 * sr))
    if z <= a or t1 - t0 > 900:
        raise ValueError("时间段不合法")
    pre = max(0, -a)
    x, _ = sf.read(str(p), start=max(0, a), stop=max(0, min(z, info.frames)), dtype="float32", always_2d=True)
    n = z - a
    buf = np.zeros((n, x.shape[1] if x.size else info.channels), dtype=np.float32)
    buf[pre:pre + len(x)] = x
    bio = io.BytesIO()
    sf.write(bio, np.clip(buf, -1, 1), sr, subtype="PCM_16", format="WAV")
    return bio.getvalue()


def save_bundle(work: Path, body: dict):
    """写 作曲_vN.json + 会话 + 录音。任何一个目标文件已存在 → FileExistsError（不写任何东西）。"""
    ver = str(body.get("ver", ""))
    if not re.fullmatch(r"v\d+", ver):
        raise ValueError(f"版本号不合法：{ver!r}")
    out = work / OUT_DIR
    out.mkdir(exist_ok=True)
    truth_p, sess_p = out / f"作曲_{ver}.json", out / f"作曲_{ver}_会话.json"
    audio = body.get("audio") or []
    targets = [truth_p, sess_p] + [out / safe_name(a["name"]) for a in audio]
    for p in targets:
        if p.exists():
            raise FileExistsError(p.name)
    if not isinstance(body.get("truth"), dict) or not isinstance(body.get("session"), dict):
        raise ValueError("缺 truth / session")
    written = []
    for a in audio:
        x = np.frombuffer(base64.b64decode(a["b64"]), dtype="<f4")
        sr = int(a["sr"])
        if not (8000 <= sr <= 192000):
            raise ValueError("采样率不合法")
        p = out / safe_name(a["name"])
        sf.write(str(p), np.clip(x, -1, 1), sr, subtype="PCM_24")
        written.append(p.name)
    truth = dict(body["truth"])
    truth["saved"] = datetime.now().isoformat(timespec="seconds")
    truth_p.write_text(json.dumps(truth, ensure_ascii=False, indent=1), encoding="utf-8")
    sess_p.write_text(json.dumps(body["session"], ensure_ascii=False), encoding="utf-8")
    return {"dir": str(out), "written": [truth_p.name, sess_p.name] + written, "truth": truth_p.name, "session": sess_p.name}


def list_sessions(work: Path):
    out = work / OUT_DIR
    if not out.is_dir():
        return []
    fs = [f for f in out.iterdir() if f.name.endswith("_会话.json")]
    fs.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return [{"name": f.name, "mtime": datetime.fromtimestamp(f.stat().st_mtime).strftime("%m-%d %H:%M")} for f in fs]


# ---------- HTTP ----------
class Handler(SimpleHTTPRequestHandler):
    root = DEFAULT_ROOT

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(HERE), **kw)

    def log_message(self, fmt, *args):
        if "/api/" in (self.path or ""):
            sys.stderr.write("%s\n" % (fmt % args))

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _send(self, code, data, ctype="application/json; charset=utf-8"):
        b = data if isinstance(data, (bytes, bytearray)) else json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _err(self, code, msg):
        self._send(code, msg.encode("utf-8"), "text/plain; charset=utf-8")

    def do_GET(self):
        u = urlparse(self.path)
        if not u.path.startswith("/api/"):
            return super().do_GET()
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/api/works":
                return self._send(200, list_works(self.root))
            work = safe_work(self.root, q.get("work", ""))
            if u.path == "/api/sense":
                if not (work / "感官.json").is_file():
                    return self._err(404, "这个作品没有 感官.json")
                return self._send(200, read_sense(work))
            if u.path == "/api/stems":
                return self._send(200, list_stems(work))
            if u.path == "/api/stem":
                return self._send(200, stem_slice(work, q["name"], float(q["t0"]), float(q["t1"])), "audio/wav")
            if u.path == "/api/next":
                return self._send(200, {"ver": f"v{next_version(work / OUT_DIR)}"})
            if u.path == "/api/sessions":
                return self._send(200, list_sessions(work))
            if u.path == "/api/file":
                p = work / OUT_DIR / safe_name(q.get("name", ""))
                if not p.is_file():
                    return self._err(404, "没有这个文件")
                ctype = "audio/wav" if p.suffix.lower() == ".wav" else "application/json; charset=utf-8"
                return self._send(200, p.read_bytes(), ctype)
            return self._err(404, "没有这个接口")
        except (ValueError, KeyError) as e:
            return self._err(400, str(e))
        except FileNotFoundError as e:
            return self._err(404, str(e))

    def do_POST(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            n = int(self.headers.get("Content-Length", "0"))
            if n <= 0 or n > MAX_BODY:
                return self._err(413, "请求太大 / 是空的")
            body = json.loads(self.rfile.read(n).decode("utf-8"))
            work = safe_work(self.root, q.get("work", ""))
            if u.path == "/api/save":
                return self._send(200, save_bundle(work, body))
            return self._err(404, "没有这个接口")
        except FileExistsError as e:
            return self._err(409, f"已存在：{e}（不覆盖）")
        except (ValueError, KeyError) as e:
            return self._err(400, str(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    Handler.root = Path(a.root)
    Handler.extensions_map.update({".js": "text/javascript", ".mjs": "text/javascript"})
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://127.0.0.1:{a.port}/"
    print(f"作曲器：{url}  （作品根目录 {Handler.root}；Ctrl+C 停）")
    if not a.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
