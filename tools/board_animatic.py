#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🎬样片-情绪大图动态版：把 `情绪大图/草案.json` 的每格图按段落时间铺到原曲上，配「她此刻」字幕、时间码、感官条，
再把动机检索（`动机_*.json`）的每次出现标出来 —— 让用户不用自己对音乐（👤：「现在得我自己对音乐吧？」）。
不加任何运动 / 缩放（克制；这是看节奏和情绪的草稿，不是成片）。
用法：python tools/board_animatic.py [--start 0] [--end 97] [--w 1280]
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

MV = Path("C:/Users/Administrator/Desktop/Projekt_Visual/MV_Cant_Kill_Us")
MUSIC = MV.parent / "Assets" / "The Glitch Mob - Can't Kill Us [2020 Edit]_190189282.mp4"
FPS = 24
RED = (232, 44, 52)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=97.0)
    ap.add_argument("--w", type=int, default=1280)
    a = ap.parse_args()
    W = a.w; H = W * 9 // 16; BAR = 54
    spec = json.loads((MV / "情绪大图" / "草案.json").read_text("utf-8"))
    sense = json.loads((MV / "感官.json").read_text("utf-8"))
    motifs = []
    for p in MV.glob("动机_*.json"):
        m = json.loads(p.read_text("utf-8"))
        motifs += [(h["t"], m["name"], k + 1, m["template"][1] - m["template"][0]) for k, h in enumerate(m["hits"])]
    fz = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", max(18, W // 52))
    fs = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", max(14, W // 80))
    fm = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", max(12, W // 90))
    imgs = {}
    for s in spec["段"]:
        p = MV / "情绪大图" / f"{s['id']}.jpg"
        imgs[s["id"]] = Image.open(p).convert("RGB").resize((W, H), Image.LANCZOS) if p.exists() else Image.new("RGB", (W, H))
    thr = np.array(sense["channels"]["threat"]); ten = np.array(sense["channels"]["tension"])
    import imageio_ffmpeg
    out = MV / "情绪大图" / f"_情绪大图_动态_{int(a.start)}-{int(a.end)}.mp4"
    proc = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                             "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H + BAR}", "-r", str(FPS), "-i", "-",
                             "-ss", str(a.start), "-t", str(a.end - a.start), "-i", str(MUSIC),
                             "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                             "-c:a", "aac", "-b:a", "192k", "-shortest", str(out)], stdin=subprocess.PIPE)
    n = int((a.end - a.start) * FPS)
    for i in range(n):
        t = a.start + i / FPS
        seg = next((s for s in spec["段"] if s["t"][0] <= t < s["t"][1]), spec["段"][-1])
        fr = Image.new("RGB", (W, H + BAR), (8, 9, 10))
        fr.paste(imgs[seg["id"]], (0, 0))
        d = ImageDraw.Draw(fr)
        # 字幕：她此刻（段落开头 3 秒亮，之后变暗）
        age = t - seg["t"][0]
        c = 235 if age < 3 else 150
        line = seg["她此刻"]
        tw = d.textlength(line, font=fz)
        d.rectangle([(W - tw) / 2 - 14, H - 64, (W + tw) / 2 + 14, H - 22], fill=(0, 0, 0))
        d.text(((W - tw) / 2, H - 62), line, font=fz, fill=(c, c, c))
        d.text((16, 12), f"{seg['id']}  {seg['名']}", font=fz, fill=(220, 224, 228))
        if seg.get("记忆"):
            d.text((16, 12 + fz.size + 8), "◇ " + seg["记忆"], font=fs, fill=(150, 156, 162))
        # 动机：出现那一刻右上角亮红标签
        for (mt, name, k, ln) in motifs:
            if mt <= t < mt + ln:
                lab = f"{name} #{k}"
                d.rectangle([W - 16 - d.textlength(lab, font=fz) - 16, 10, W - 10, 14 + fz.size + 10], outline=RED, width=2)
                d.text((W - 16 - d.textlength(lab, font=fz) - 8, 14), lab, font=fz, fill=RED)
        # 底栏：时间码 + 当前段内的感官条 + 段落刻度
        fi = min(int(t * FPS), len(thr) - 1)
        d.text((12, H + 8), f"{int(t // 60)}:{t % 60:05.2f}", font=fm, fill=(200, 205, 210))
        x0 = 120; x1 = W - 12; span = a.end - a.start
        for s in spec["段"]:
            if a.start <= s["t"][0] <= a.end:
                xx = x0 + (s["t"][0] - a.start) / span * (x1 - x0)
                d.line([(xx, H + 6), (xx, H + BAR - 6)], fill=(70, 74, 78))
        for (mt, name, k, ln) in motifs:
            if a.start <= mt <= a.end:
                xx = x0 + (mt - a.start) / span * (x1 - x0)
                d.line([(xx, H + BAR - 14), (xx, H + BAR - 4)], fill=RED, width=2)
        px = x0 + (t - a.start) / span * (x1 - x0)
        d.line([(px, H + 4), (px, H + BAR - 4)], fill=(240, 240, 240), width=2)
        d.rectangle([x0, H + 10, x0 + thr[fi] * 120, H + 18], fill=(170, 176, 182))
        d.rectangle([x0, H + 22, x0 + ten[fi] * 120, H + 30], fill=RED)
        proc.stdin.write(fr.tobytes())
    proc.stdin.close(); proc.wait()
    print("->", out)


if __name__ == "__main__":
    main()
