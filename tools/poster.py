#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🖼信息图-海报-仿 MACHINE BEHAVIOR（参考 Projekt_Visual/_参考库/信息可视化_machine_behavior.png）的版式：
上：整首歌的真实频谱 + 响度曲线 + 段落竖线；中：两排概念静帧缩略图；下：每个概念一条轨道，只在它负责的那段画出这段音频的波形。
数据全是真的（这首歌的音频 + 概念草案的段落表），没有装饰性的假数字。
用法：python tools/poster.py [--out 路径]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stills as S  # noqa: E402

PW, PH = 1654, 2339          # A4 竖版 200dpi
DUR = 290.0                  # 音乐到 4:50
# 概念草案的段落（秒）：名字, 开始, 结束
LANES = [("DELAY_EMBEDDING", 0, 10), ("KURAMOTO", 10, 50), ("LOGISTIC", 52, 110), ("REACTION_DIFFUSION", 52, 110),
         ("DLA", 110, 126), ("SANDPILE", 128, 154), ("DCT", 154, 172), ("DROSTE", 172, 204),
         ("OPTIMAL_TRANSPORT", 204, 282), ("ATTRACTOR", 282, 290)]
SECTIONS = [(0, "INTRO"), (10, "BUILD"), (46, "INHALE"), (52, "DROP 1"), (110, "BRIDGE"), (128, "DROP 2"),
            (154, "BREAKDOWN"), (172, "REBUILD"), (204, "DROP 3"), (282, "OUTRO")]
WHITE, DIM, LINE, RED = (230, 233, 236), (120, 126, 132), (70, 74, 80), (255, 42, 52)


def fnt(sz, bold=False):
    return ImageFont.truetype("C:/Windows/Fonts/consolab.ttf" if bold else S.FONT, sz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(S.MV / "静帧" / "海报_signal_behavior.jpg"))
    a = ap.parse_args()
    sr = 11025
    x, _ = S.audio(sr=sr, start=0, dur=DUR)
    im = Image.new("RGB", (PW, PH), (0, 0, 0))
    d = ImageDraw.Draw(im)
    d.text((90, 70), "SIGNAL BEHAVIOR", font=fnt(30), fill=WHITE)
    d.text((PW - 150, 70), "/ 01", font=fnt(30), fill=WHITE)

    # ---------- 上：频谱 + 响度 ----------
    L, R_ = 200, PW - 120
    gw = R_ - L
    top, sh = 230, 360
    n, hop = 1024, 512
    frames = (len(x) - n) // hop
    win = np.hanning(n)
    idx = np.arange(n)[None, :] + hop * np.arange(frames)[:, None]
    spec = np.abs(np.fft.rfft(x[idx] * win, axis=1))           # (frames, 513)
    fq = np.fft.rfftfreq(n, 1 / sr)
    rows = np.geomspace(40, sr / 2 - 1, sh)                    # 对数频率轴
    ri = np.searchsorted(fq, rows).clip(0, len(fq) - 1)
    sp = np.log1p(spec[:, ri] * 20).T[::-1]                    # 高频在上
    sp = sp / sp.max()
    cols = np.linspace(0, frames - 1, gw).astype(int)
    img = (np.clip(sp[:, cols] ** 2.6 * 1.25, 0, 1) * 235).astype(np.uint8)  # 压暗：只让强的频带亮
    im.paste(Image.fromarray(img).convert("RGB"), (L, top))
    d.text((70, top - 4), "SPECTRUM", font=fnt(15), fill=DIM)
    d.text((70, top + 18), "40 Hz - 5.5 kHz", font=fnt(13), fill=LINE)
    d.text((L + 6, top - 26), "LOG FREQUENCY  //  HOP 46 ms", font=fnt(13), fill=DIM)
    # 响度曲线（每 2 秒 RMS，dB）
    ly0 = top + sh + 60; lh = 200
    blk = sr * 2
    rms = np.array([np.sqrt(np.mean(x[i:i + blk] ** 2)) for i in range(0, len(x) - blk, blk)])
    db = 20 * np.log10(rms + 1e-6)
    v = (db - db.min()) / (db.max() - db.min())
    pts = [(L + gw * (i * 2 + 1) / DUR, ly0 + lh - v[i] * lh) for i in range(len(v))]
    d.line(pts, fill=WHITE, width=3)
    d.text((70, ly0 + lh - 10), "LOUDNESS", font=fnt(15), fill=DIM)
    d.text((70, ly0 + lh + 12), f"{db.min():.0f} .. {db.max():.0f} dB", font=fnt(13), fill=LINE)
    # 段落竖线（穿过频谱和响度）+ 两次吸气标红
    for si, (t, name) in enumerate(SECTIONS):
        xx = L + gw * t / DUR
        dy = 34 if si % 2 else 0   # 隔一个往下错一行，挨得近的标签不叠
        red = name in ("INHALE", "BREAKDOWN")
        d.line([(xx, top), (xx, ly0 + lh + 24 + dy)], fill=RED if red else LINE, width=1)
        d.text((xx + 4, ly0 + lh + 26 + dy), f"{name}", font=fnt(12), fill=RED if red else DIM)
        d.text((xx + 4, ly0 + lh + 42 + dy), f"{int(t)//60}:{int(t)%60:02d}", font=fnt(12), fill=LINE)
    # 主角最早成形的一刻（0:46）连一条引线，像原版的 ROBOT VISION 标注
    xx = L + gw * 46 / DUR
    d.line([(xx, top + 40), (xx - 60, top - 60), (L - 120, top - 60)], fill=WHITE, width=1)
    d.text((L - 120, top - 84), "US  >  FIRST SYNC", font=fnt(14), fill=WHITE)

    # ---------- 中：两排缩略图 ----------
    ty = ly0 + lh + 120
    files = sorted((S.MV / "静帧").glob("[0-9]*.jpg"))[:10]
    tw = (gw + 80 - 4 * 16) // 5; th = int(tw * 9 / 16)
    for i, fpath in enumerate(files):
        r, c = divmod(i, 5)
        xx = L - 80 + c * (tw + 16); yy = ty + r * (th + 54)
        im.paste(Image.open(fpath).resize((tw, th)), (int(xx), int(yy)))
        d.rectangle((xx, yy, xx + tw, yy + th), outline=DIM, width=1)
        nm, s0, s1 = LANES[i]
        d.text((xx, yy + th + 6), f"{i + 1:02d} {nm.replace('_', ' ')}", font=fnt(12), fill=DIM)
        d.text((xx + tw - 90, yy + th + 6), f"{s0//60}:{s0%60:02d}-{s1//60}:{s1%60:02d}", font=fnt(12), fill=LINE)

    # ---------- 下：概念轨道 ----------
    by = ty + 2 * (th + 54) + 80
    lane_h = 78
    d.rectangle((L, by - 14, R_, by - 8), fill=(170, 170, 170))
    for i, (nm, s0, s1) in enumerate(LANES):
        y0 = by + i * lane_h
        d.rectangle((L, y0, R_, y0 + lane_h), outline=LINE, width=1)
        d.text((40, y0 + lane_h // 2 - 8), "_" + nm[:19], font=fnt(12), fill=DIM)
        xa, xb = L + gw * s0 / DUR, L + gw * s1 / DUR
        d.rectangle((xa, y0 + 1, xb, y0 + lane_h - 1), fill=(34, 36, 40))
        d.rectangle((xa, y0 + 1, xb, y0 + 6), fill=(150, 154, 158))
        seg = x[int(s0 * sr):int(s1 * sr)]
        cols_n = max(2, int(xb - xa))
        env = np.array([np.sqrt(np.mean(c ** 2)) for c in np.array_split(seg, cols_n)])  # RMS 不用峰值：母带压得很满，峰值画出来是实心块
        env = (env / (env.max() + 1e-9)) ** 3
        mid = y0 + lane_h // 2 + 3
        for j, e in enumerate(env):
            hgt = 1 + e * (lane_h * 0.34)
            d.line([(xa + j, mid - hgt), (xa + j, mid + hgt)], fill=WHITE, width=1)
    for t in range(0, int(DUR) + 1, 30):   # 时间刻度 + 竖网格
        xx = L + gw * t / DUR
        d.line([(xx, by), (xx, by + lane_h * len(LANES))], fill=(40, 43, 47), width=1)
        d.text((xx - 14, by + lane_h * len(LANES) + 10), f"{t//60}:{t%60:02d}", font=fnt(12), fill=DIM)

    # ---------- 页脚 ----------
    d.text((90, PH - 50), "eidolon // visual process // synapsis", font=fnt(15), fill=DIM)
    d.text((PW - 640, PH - 50), "the glitch mob - can't kill us [2020 edit] - local study", font=fnt(15), fill=LINE)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out, quality=92)
    print("->", out)


if __name__ == "__main__":
    main()
