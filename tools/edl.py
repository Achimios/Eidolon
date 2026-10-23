#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<✂剪辑表-入口-纯文字剪辑表 + 不叠效果的快速预览（👤2026-10-19：「最快的方法其实是纯文字」「focus on 片子的 feelings, no jam」）。
  python tools/edl.py export    # 把第二版剪辑的结构导出成 Projekt_Visual/MV_Cant_Kill_Us/剪辑表.txt（已存在就不覆盖，除非 --force）
  python tools/edl.py preview   # 读剪辑表，拼原片片段 + 原曲，480p，不叠任何效果 → 样片/预览_剪辑表.mp4
剪辑表一行一段（# 开头是注释，空行忽略）：
  开始  素材  [速度]  [# 备注]
  开始 = m:ss.ss（这一段从歌的第几秒开始；结束 = 下一行的开始）
  素材：
    原片@m:ss.ss            原版 MV 从这一秒开始
    第二章:shot5@3.2        第二章素材里文件名含 shot5 的视频，从 3.2 秒开始
    样片:文件名@秒          样片文件夹里的视频
    作品:相对作品文件夹的路径@秒  作品文件夹里任意视频（如 情绪大图/_情绪大图_动态_0-97.mp4）
    图:相对作品文件夹的路径  静帧
    文字:要显示的话          黑底文字卡（需要 3D / 还没有画面的地方）
    黑                      黑场
  速度：可选，默认 1（0.5 = 慢放，2 = 快放）
  preview 选项：--edl 另一份剪辑表  --end 只出到第几秒  --out 输出  --hd（1280×720）
  ⭐--synapsis <页id>：剪辑真源 = Synapsis 时间轴页（👤2026-10-24 定），读 `cli.js timeline <页> --json`；
     节点属性 src（剪辑表语法）> 第一张图（静帧）> 文字卡；只取主轨，空隙补黑
"""
import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stills as S  # noqa: E402

MV = S.MV
EDL = MV / "剪辑表.txt"
PW, PH, FPS = 854, 480, 24
END = 290.0
FCN = "C:/Windows/Fonts/msyh.ttc"


def fmt(t):
    return f"{int(t // 60)}:{t % 60:05.2f}"


def parse_time(s):
    m, sec = s.split(":") if ":" in s else ("0", s)
    return int(m) * 60 + float(sec)


def export(force=False):
    if EDL.exists() and not force:
        sys.exit(f"已存在，不覆盖（你可能改过）：{EDL}；要重新导出加 --force")
    import animatic_v3 as V3
    segs = V3.plan(52.0, END, np.random.default_rng(11))       # 和第二版剪辑同一个 seed = 同一个结构
    L = ["# 剪辑表 —— 《Can't Kill Us》（从第二版剪辑导出，2026-10-19）",
         "# 一行一段：开始  素材  [速度]  [# 备注]。结束 = 下一行的开始。格式说明见 4_Eidolon/tools/edl.py 开头。",
         "# 改完说一声，或自己跑：python 4_Eidolon/tools/edl.py preview  → 样片/预览_剪辑表.mp4（不叠效果，几分钟内出）",
         "",
         f"0:00.00   样片:03_粗剪v2_0-52.mp4@0     1      # 前奏 + Kuramoto（v2 克制版，带效果）"]
    for ts, dur, src, speed, upr, _shot in segs:
        sec = V3.section_at(ts)[0]
        note = sec + ("  uprise" if upr else "")
        L.append(f"{fmt(ts):9s} 原片@{fmt(src):10s} {speed:<5g}  # {note}")
    L.append(f"{fmt(END):9s} 黑")
    EDL.write_text("\n".join(L) + "\n", "utf-8", newline="\n")
    print("->", EDL, len(segs) + 2, "行")


LINE = re.compile(r"^\s*(\d+:\d+(?:\.\d+)?)\s+(\S.*?)(?:\s+([\d.]+))?\s*(?:#.*)?$")


def read_edl(edl=None):
    rows = []
    for raw in Path(edl or EDL).read_text("utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        body = s.split("#", 1)[0].rstrip()
        m = LINE.match(body)
        if not m:
            sys.exit(f"看不懂这一行：{raw}")
        t, src, speed = m.group(1), m.group(2).strip(), m.group(3)
        if src.startswith("文字:") and speed is None:
            pass
        rows.append((parse_time(t), src, float(speed) if speed else 1.0))
    rows.sort(key=lambda r: r[0])
    return rows


SYN_CLI = "C:/Users/Administrator/Desktop/Obsidian_V_Synapsis/core/cli.js"


def rows_from_timeline(tl):
    """Synapsis 时间轴导出（cli.js timeline <页> --json）→ read_edl 同格式的行（👤2026-10-24：剪辑真源 = Synapsis 节点）。
    只取主轨（track 0）；t0 ≥ t1 的条跳过；条之间的空隙补「黑」。
    素材：节点属性 src（剪辑表语法，如 样片:xxx.mp4@1.2）优先；否则第一张图 → 静帧（慢推）；都没有 → 文字卡（标签）。"""
    items = sorted((it for it in tl["items"] if int(it.get("track") or 0) == 0
                    and it.get("t0") is not None and it.get("t1") is not None and it["t0"] < it["t1"]),
                   key=lambda it: it["t0"])
    rows, end = [], 0.0
    for it in items:
        t0, t1 = float(it["t0"]), float(it["t1"])
        if t0 > end + 1e-6:
            rows.append((end, "黑", 1.0))
        if it.get("src"):
            src = it["src"]
        elif it.get("pics"):
            src = "图:" + it["pics"][0]
        else:
            src = "文字:" + (it.get("label") or it["id"])
        rows.append((max(t0, end), src, float(it.get("speed") or 1.0)))
        end = max(end, t1)
    rows.append((end, "黑", 1.0))
    return rows


def _timeline_json(page, ws):
    import json
    r = subprocess.run(["node", SYN_CLI, "timeline", page, "--json", "--ws", ws], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return json.loads(r.stdout) if r.returncode == 0 else None


def flatten_items(tl, fetch, depth=0):
    """段落节点本身也是时间轴页（剪辑 → 段落 → 镜头）：没写 src 的条若能作为时间轴页打开，就展开成它的子条。
    子条的 t0/t1 本来就是成片时间，不用再换算。fetch(页id) → 导出 dict 或 None。"""
    out = []
    for it in tl["items"]:
        sub = None if it.get("src") or depth > 4 else fetch(it["id"])
        if sub and sub.get("items"):
            out += flatten_items(sub, fetch, depth + 1)
        else:
            out.append(it)
    return out


def read_synapsis(page, ws=None):
    """调 Synapsis CLI 拿时间轴页（下游现读上游，不存副本）；嵌套的段落页自动展开"""
    ws = ws or str(MV)
    tl = _timeline_json(page, ws)
    if tl is None:
        sys.exit(f"Synapsis 时间轴读不到：{page}（不是 timeline 页或不存在）")
    items = flatten_items(tl, lambda p: _timeline_json(p, ws))
    return rows_from_timeline({"items": items})


def resolve(src):
    """素材 → (视频路径或 None, 起点秒, 静帧图或 None)"""
    if src == "黑":
        return None, 0, Image.new("RGB", (PW, PH))
    if src.startswith("文字:"):
        im = Image.new("RGB", (PW, PH)); d = ImageDraw.Draw(im); f = ImageFont.truetype(FCN, 26)
        txt = src[3:].strip().strip('"「」')
        tw = d.textlength(txt, font=f)
        d.text(((PW - tw) / 2, PH / 2 - 16), txt, font=f, fill=(225, 225, 225))
        return None, 0, im
    if src.startswith("图:"):
        im = Image.open(MV / src[2:].strip()).convert("RGB")
        w, h = im.size; r = max(PW / w, PH / h)                      # 铺满后居中裁（不拉伸）
        im = im.resize((round(w * r), round(h * r)), Image.LANCZOS); x0, y0 = (im.width - PW) // 2, (im.height - PH) // 2
        return None, 0, im.crop((x0, y0, x0 + PW, y0 + PH))
    path, at = src, "0"
    if "@" in src:
        path, at = src.rsplit("@", 1)
    t = parse_time(at)
    if path == "原片":
        return S.MUSIC, t, None
    if path.startswith("第二章:"):
        key = path.split(":", 1)[1]
        hits = sorted((S.MV / "素材").glob(f"*{key}*.mp4"))
        if not hits:
            sys.exit(f"第二章素材里找不到：{key}")
        return hits[0], t, None
    if path.startswith("样片:"):
        return MV / "样片" / path.split(":", 1)[1], t, None
    if path.startswith("作品:"):
        return MV / path.split(":", 1)[1], t, None
    sys.exit(f"不认识的素材：{src}")


def count_frames(ff, path):
    """数视频帧（流拷贝，不解码）"""
    err = subprocess.run([ff, "-i", str(path), "-map", "0:v", "-c", "copy", "-f", "null", "-"], capture_output=True, text=True,
                         encoding="utf-8", errors="replace").stderr
    m = re.findall(r"frame=\s*(\d+)", err)
    return int(m[-1]) if m else -1


def preview(edl=None, end=END, out=None, hd=False, synapsis=None):
    global PW, PH
    if hd:
        PW, PH = 1280, 720
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    rows = [r for r in (read_synapsis(synapsis) if synapsis else read_edl(edl)) if r[0] < end]
    out = Path(out) if out else MV / "样片" / "预览_剪辑表.mp4"
    END_ = end
    with tempfile.TemporaryDirectory() as td:
        parts = []
        for k, (t, src, speed) in enumerate(rows):
            nxt = min(rows[k + 1][0], END_) if k + 1 < len(rows) else END_
            # 按绝对帧号切：每段帧数 = round(结束×fps) − round(开始×fps)，误差不累积（按秒四舍五入时全曲漂了快 1 秒）
            nf = int(round(nxt * FPS)) - int(round(t * FPS))
            dur = nf / FPS
            if nf <= 0:
                continue
            video, at, still = resolve(src)
            p = Path(td) / f"{k:04d}.mp4"
            vf = f"scale={PW}:{PH}:force_original_aspect_ratio=increase,crop={PW}:{PH},setsar=1,fps={FPS}"
            if still is not None:
                img = Path(td) / f"{k:04d}.png"; still.save(img)
                if src.startswith("图:"):   # 静帧不静止（👤镜头不从静止开始）：匀速慢推，速度列 = 推多少（默认 1 → 推 8%）
                    vf = (f"scale={PW * 2}:{PH * 2},zoompan=z='1+{0.08 * speed:.3f}*on/{nf}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
                          f":d=1:s={PW}x{PH}:fps={FPS},setsar=1")
                cmd = [ff, "-loglevel", "error", "-y", "-loop", "1", "-i", str(img), "-vf", vf, "-frames:v", str(nf)]
            else:
                # 素材不够长时用最后一帧补齐（tpad）：少一帧，后面所有切点都会提前（2026-10-22 实测 13g 漂了 0.71 s）
                cmd = [ff, "-loglevel", "error", "-y", "-ss", f"{at:.3f}", "-t", f"{dur * speed + 0.1:.3f}", "-i", str(video),
                       "-vf", f"setpts=PTS/{speed},{vf},tpad=stop_mode=clone:stop_duration={dur + 1:.3f}", "-frames:v", str(nf)]
            subprocess.run(cmd + ["-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p", str(p)], check=True)
            got = count_frames(ff, p)
            if got != nf:
                sys.exit(f"第 {k} 段（{fmt(t)} {src}）帧数 {got} ≠ {nf}：后面的切点会漂，停下")
            parts.append(p)
        lst = Path(td) / "list.txt"
        lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), "utf-8")
        subprocess.run([ff, "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-t", str(END_), "-i", str(S.MUSIC),
                        "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-t", str(END_), str(out)], check=True)
    total = count_frames(ff, out)
    want = int(round(END_ * FPS))
    print("->", out, len(parts), "段", f"帧数 {total}/{want}" + ("" if total == want else "  ⚠️对不上"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "preview"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--edl", default=None); ap.add_argument("--end", type=float, default=END)
    ap.add_argument("--out", default=None); ap.add_argument("--hd", action="store_true")
    ap.add_argument("--synapsis", default=None, help="Synapsis 时间轴页 id（剪辑真源；给了就不读 --edl）")
    a = ap.parse_args()
    export(a.force) if a.cmd == "export" else preview(a.edl, a.end, a.out, a.hd, a.synapsis)
