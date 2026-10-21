#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🎬样片-MEMS 细颤（v2，按👤批评重做）：底图 = 静帧 13 的精细工程图；抽搐音那一秒，**每一个细部各自高频颤**，
不是大物体整体摆（👤：「这种颤动明显是偏全局的细节，而非大物体整体动」「频率要更快」「不够狠」）。
做法：工程图的线条按连通块拆成几百个零件（GPU 上标签传播）；零件 i 的位移 d_i = A · env(t) · x(t + φ_i) · ê_i，
x = 「other」分轨（抽搐音所在层）的真实波形，φ_i = 每个零件读波形的时间偏移（0–30 ms）→ 全画面同时、各不相同地颤；
每帧 3 次子曝光叠加（快门内的颤 → 毛刺状拖影）。小零件颤得更狠（质量小）。#2 比 #1 狠。
用法：python tools/mems_shiver.py [--start 61.5] [--end 77.1]
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

MV = Path("C:/Users/Administrator/Desktop/Projekt_Visual/MV_Cant_Kill_Us")
MUSIC = MV.parent / "Assets" / "The Glitch Mob - Can't Kill Us [2020 Edit]_190189282.mp4"
FPS, SUB = 24, 3
dev = "cuda" if torch.cuda.is_available() else "cpu"


def components(mask):
    """8 连通标签：每个像素取自己的序号，反复和邻居取最大，收敛即为连通块"""
    H, W = mask.shape
    m = torch.tensor(mask, device=dev)
    lab = torch.where(m, torch.arange(1, H * W + 1, device=dev, dtype=torch.float32).view(H, W), torch.zeros((), device=dev))
    for _ in range(4000):
        nl = F.max_pool2d(lab[None, None], 3, 1, 1)[0, 0] * m
        if torch.equal(nl, lab):
            break
        lab = nl
    u, inv = torch.unique(lab, return_inverse=True)
    return inv.cpu().numpy(), len(u)              # 0 = 背景


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=float, default=61.5)
    ap.add_argument("--end", type=float, default=77.1)
    ap.add_argument("--amp", type=float, default=9.0, help="px，最大位移（小零件再乘最多 2）")
    a = ap.parse_args()
    base = np.asarray(Image.open(MV / "静帧" / "13_mems.jpg").convert("RGB")).astype(np.float32) / 255
    H, W, _ = base.shape
    lum = base.mean(2)
    keep = np.zeros((H, W), bool); keep[H - 210:, :1100] = True; keep[:90, W - 520:] = True   # HUD / 标题不颤
    mask = (lum > 0.08) & ~keep
    lab, n = components(mask)
    sizes = np.bincount(lab.ravel(), minlength=n).astype(np.float32); sizes[0] = 1
    rng = np.random.default_rng(13)
    phi = rng.uniform(0, 0.030, n)                                  # 每个零件读波形的时间偏移
    ang = rng.normal(0, 0.35, n) + rng.choice([0, np.pi], n)        # 方向：以水平（梳齿方向）为主
    ex, ey = np.cos(ang), np.sin(ang) * 0.6
    gain = np.clip((400 / sizes) ** 0.25, 0.6, 2.0); gain[0] = 0     # 小零件颤得更狠
    ys, xs = np.nonzero(lab > 0); li = lab[ys, xs]
    col = base[ys, xs]                                              # 每个前景像素的颜色（含原图里的红点）
    static = base * (lab == 0)[..., None]                           # 背景 + HUD 不动

    p = next(MV.glob("分轨/**/other.wav"))
    x, sr = sf.read(p); x = x.mean(1) if x.ndim > 1 else x
    k = int(sr * 0.02); env = np.sqrt(np.convolve(x ** 2, np.ones(k) / k, mode="same"))
    motif = json.loads((MV / "动机_抽搐.json").read_text("utf-8"))
    hits = [h["t"] for h in motif["hits"] if a.start <= h["t"] < a.end]
    dur = motif["template"][1] - motif["template"][0]
    emax = {h: env[int(h * sr):int((h + dur) * sr)].max() + 1e-9 for h in hits}
    xmax = {h: np.abs(x[int(h * sr):int((h + dur) * sr)]).max() + 1e-9 for h in hits}

    f = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 18)
    import imageio_ffmpeg
    out = MV / "样片" / "09b_MEMS细颤_drop1.mp4"
    proc = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                             "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                             "-ss", str(a.start), "-t", str(a.end - a.start), "-i", str(MUSIC),
                             "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16",
                             "-c:a", "aac", "-b:a", "192k", "-shortest", str(out)], stdin=subprocess.PIPE)
    black = np.zeros((H, W, 3), np.uint8).tobytes()
    n_fr = int((a.end - a.start) * FPS)
    for i in range(n_fr):
        t = a.start + i / FPS
        kk = next((q for q, h in enumerate(hits) if h <= t < h + dur), None)
        if kk is None:
            proc.stdin.write(black); continue
        h0 = hits[kk]; A = a.amp * (1.0 + 0.7 * kk)
        acc = np.zeros((H, W, 3), np.float32); dmax = 0.0
        for s in range(SUB):                                        # 快门内 3 次子曝光
            ts = t + s / (FPS * SUB)
            e = env[int(ts * sr)] / emax[h0]
            samp = x[np.clip(((ts + phi) * sr).astype(int), 0, len(x) - 1)] / xmax[h0]
            d = A * e * samp * gain                                 # 每个零件的位移幅度（px）
            dx, dy = (d * ex)[li], (d * ey)[li]
            tx = np.clip(np.round(xs + dx).astype(int), 0, W - 1); ty = np.clip(np.round(ys + dy).astype(int), 0, H - 1)
            np.add.at(acc, (ty, tx), col)
            dmax = max(dmax, float(np.abs(d).max()))
        img = static + np.clip(acc / SUB * 1.6, 0, 1)
        fr = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        dr = ImageDraw.Draw(fr)
        dr.rectangle([40, 30, 1020, 92], fill=(6, 7, 8))
        dr.text((48, 36), f"TWITCH {kk + 1}/{len(hits)}  @ {h0:.2f} s    t = {t:6.2f} s    parts = {n - 1}", font=f, fill=(150, 160, 168))
        dr.text((48, 62), f"d_i = {A:.0f} px * env(t) * x_other(t + phi_i) * g_i,  phi_i in [0, 30] ms,  max |d| = {dmax:4.1f} px", font=f, fill=(150, 160, 168))
        proc.stdin.write(fr.tobytes())
    proc.stdin.close(); proc.wait()
    print("->", out, "parts", n - 1)


if __name__ == "__main__":
    main()
