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
sys.path.insert(0, str(HERE.parent))   # tools/trash.py
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
    """写 作曲_vN.json + 会话 + 录音。整包要么全写成、要么一个都不落：
    先写临时名，全部成功再逐个改成正式名；任何一个正式名已存在 → FileExistsError（409，不覆盖）。"""
    ver = str(body.get("ver", ""))
    if not re.fullmatch(r"v\d+", ver):
        raise ValueError(f"版本号不合法：{ver!r}")
    if not isinstance(body.get("truth"), dict) or not isinstance(body.get("session"), dict):
        raise ValueError("缺 truth / session")
    audio = body.get("audio") or []
    # 录音名只许是这一版自己的名字，且彼此不重（防同批互相盖、防盖掉 作曲_vN.json）
    name_re = re.compile(rf"^(哼唱_{ver}_第\d+遍|素材_{ver}_[A-Za-z0-9]+)\.wav$")
    names = [str(a.get("name", "")) for a in audio]
    for nm in names:
        if not name_re.match(nm):
            raise ValueError(f"录音文件名不合法：{nm!r}")
    if len(set(names)) != len(names):
        raise ValueError("同一批里录音文件名重复")
    pcm = []
    for a in audio:
        sr = int(a["sr"])
        if not (8000 <= sr <= 192000):
            raise ValueError("采样率不合法")
        x = np.frombuffer(base64.b64decode(a["b64"]), dtype="<f4")
        # 拖进来的立体声文件：b64 = 第 1 个声道，extra_channels = 其余声道（同长），叠成多声道 wav
        ex = [np.frombuffer(base64.b64decode(e), dtype="<f4") for e in (a.get("extra_channels") or [])]
        if ex:
            if any(len(e) != len(x) for e in ex):
                raise ValueError("声道长度不一致")
            x = np.stack([x] + ex, 1)
        pcm.append((x, sr))
    out = work / OUT_DIR
    out.mkdir(exist_ok=True)
    truth_p, sess_p = out / f"作曲_{ver}.json", out / f"作曲_{ver}_会话.json"
    finals = [truth_p, sess_p] + [out / nm for nm in names]
    for p in finals:
        if p.exists():
            raise FileExistsError(p.name)
    tag = f".tmp_{datetime.now().strftime('%H%M%S%f')}_"
    tmps = [p.with_name(tag + p.name) for p in finals]
    try:
        truth = dict(body["truth"])
        truth["saved"] = datetime.now().isoformat(timespec="seconds")
        tmps[0].write_text(json.dumps(truth, ensure_ascii=False, indent=1), encoding="utf-8")
        tmps[1].write_text(json.dumps(body["session"], ensure_ascii=False), encoding="utf-8")
        for (x, sr), tp in zip(pcm, tmps[2:]):
            sf.write(str(tp), np.clip(x, -1, 1), sr, subtype="PCM_24", format="WAV")
        for tp, p in zip(tmps, finals):
            tp.rename(p)          # Windows 上目标已存在会报错 → 不会覆盖
    except Exception:
        # 只清这一次自己写出的临时文件（进回收站，不直接删）
        left = [str(tp) for tp in tmps if tp.exists()]
        if left:
            from trash import trash
            trash(*left)
        raise
    return {"dir": str(out), "written": [p.name for p in finals], "truth": truth_p.name, "session": sess_p.name}


FINAL = "当前.txt"          # 定稿指针：一行 作曲_vN.json；下游（sense_song / 剪辑 / 画面）只读它
FINAL_LOG = "定稿记录.txt"   # 每次定稿追加一行，可追溯


def read_final(work: Path):
    p = work / OUT_DIR / FINAL
    return p.read_text(encoding="utf-8").strip() if p.is_file() else None


def set_final(work: Path, name: str):
    """把 作曲_vN.json 定为定稿。指针文件本来就是要改的（不是生成物）；换之前的那一行留在定稿记录里。"""
    if not re.fullmatch(r"作曲_v\d+\.json", name or ""):
        raise ValueError(f"只能定稿 作曲_vN.json：{name!r}")
    out = work / OUT_DIR
    if not (out / name).is_file():
        raise FileNotFoundError(name)
    prev = read_final(work)
    tmp = out / f".tmp_{FINAL}"
    tmp.write_text(name + "\n", encoding="utf-8")
    tmp.replace(out / FINAL)
    with open(out / FINAL_LOG, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')}\t{name}\t（之前：{prev or '无'}）\n")
    return {"final": name, "prev": prev}


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

    def _foreign(self):
        """别的网页 / DNS 重绑定发来的请求：Host 必须是本机这个端口；带 Origin 就必须是自己。"""
        port = self.server.server_address[1]
        ok = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if self.headers.get("Host", "") not in ok:
            return "Host 不对"
        origin = self.headers.get("Origin")
        if origin is not None and origin not in {f"http://{h}" for h in ok}:
            return "不接受别的网页发来的请求"
        return None

    def _err(self, code, msg):
        self._send(code, msg.encode("utf-8"), "text/plain; charset=utf-8")

    def do_GET(self):
        u = urlparse(self.path)
        bad = self._foreign()
        if bad:
            return self._err(403, bad)
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
            if u.path == "/api/final":
                return self._send(200, {"final": read_final(work)})
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
        bad = self._foreign()
        if bad:
            return self._err(403, bad)
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            return self._err(415, "只收 application/json")   # 简单请求（text/plain）不会触发预检，挡掉
        try:
            n = int(self.headers.get("Content-Length", "0"))
            if n <= 0 or n > MAX_BODY:
                return self._err(413, "请求太大 / 是空的")
            body = json.loads(self.rfile.read(n).decode("utf-8"))
            work = safe_work(self.root, q.get("work", ""))
            if u.path == "/api/save":
                return self._send(200, save_bundle(work, body))
            if u.path == "/api/final":
                return self._send(200, set_final(work, str(body.get("name", ""))))
            return self._err(404, "没有这个接口")
        except FileExistsError as e:
            return self._err(409, f"已存在：{e}（不覆盖）")
        except (ValueError, KeyError) as e:
            return self._err(400, str(e))
        except FileNotFoundError as e:
            return self._err(404, f"没有这个文件：{e}")


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
