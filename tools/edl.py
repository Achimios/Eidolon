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
    图:相对作品文件夹的路径  静帧
    文字:要显示的话          黑底文字卡（需要 3D / 还没有画面的地方）
    黑                      黑场
  速度：可选，默认 1（0.5 = 慢放，2 = 快放）
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


def read_edl():
    rows = []
    for raw in EDL.read_text("utf-8").splitlines():
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
        return None, 0, Image.open(MV / src[2:].strip()).convert("RGB").resize((PW, PH))
    path, at = src, "0"
    if "@" in src:
        path, at = src.rsplit("@", 1)
    t = parse_time(at)
    if path == "原片":
        return S.MUSIC, t, None
    if path.startswith("第二章:"):
        key = path.split(":", 1)[1]
        hits = sorted((S.PROJ / "第二章_墙里的东西" / "素材").glob(f"*{key}*.mp4"))
        if not hits:
            sys.exit(f"第二章素材里找不到：{key}")
        return hits[0], t, None
    if path.startswith("样片:"):
        return MV / "样片" / path.split(":", 1)[1], t, None
    sys.exit(f"不认识的素材：{src}")


def preview():
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    rows = read_edl()
    out = MV / "样片" / "预览_剪辑表.mp4"
    with tempfile.TemporaryDirectory() as td:
        parts = []
        for k, (t, src, speed) in enumerate(rows):
            nxt = rows[k + 1][0] if k + 1 < len(rows) else END
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
                cmd = [ff, "-loglevel", "error", "-y", "-loop", "1", "-i", str(img), "-vf", vf, "-frames:v", str(nf)]
            else:
                cmd = [ff, "-loglevel", "error", "-y", "-ss", f"{at:.3f}", "-t", f"{dur * speed + 0.1:.3f}", "-i", str(video),
                       "-vf", f"setpts=PTS/{speed},{vf}", "-frames:v", str(nf)]
            subprocess.run(cmd + ["-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p", str(p)], check=True)
            parts.append(p)
        lst = Path(td) / "list.txt"
        lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), "utf-8")
        subprocess.run([ff, "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-t", str(END), "-i", str(S.MUSIC),
                        "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", str(out)], check=True)
    print("->", out, len(parts), "段")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "preview"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    export(a.force) if a.cmd == "export" else preview()
