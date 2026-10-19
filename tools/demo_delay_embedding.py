#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🎬样片-延迟嵌入-前奏 10 秒：这首歌自己的低音 (x(t), x(t-τ), x(t-2τ)) 从一条线展开成三维轨道，配原曲。
画面：黑底冷灰，笔头（最近 50ms）红色；嵌入维度 1→3 在前 5 秒展开；镜头慢转；HUD 写真实参数。
用法：python tools/demo_delay_embedding.py [--start 0] [--dur 10] [--out 路径]
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stills as S  # noqa: E402

FPS = 24


def smooth(t, a, b):
    u = np.clip((t - a) / (b - a), 0, 1)
    return u * u * (3 - 2 * u)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--dur", type=float, default=10.0)
    ap.add_argument("--out", default=str(S.MV / "样片" / "01_delay_embedding_10s.mp4"))
    a = ap.parse_args()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)

    sr = 11025
    x, _ = S.audio(sr=sr, start=a.start, dur=a.dur + 1.0)
    k = int(sr / 120)
    xl = np.convolve(np.convolve(x, np.ones(k) / k, mode="same"), np.ones(k) / k, mode="same")
    xl /= (np.abs(xl).max() + 1e-9)          # 全段统一缩放，画面不跟着音量抖
    tau = int(sr * 0.012)
    rms = np.sqrt(np.convolve(x ** 2, np.ones(sr // 20) / (sr // 20), mode="same"))
    rms /= rms.max() + 1e-9

    W, H = S.W, S.H
    acc = np.zeros((H, W))
    f = ImageFont.truetype(S.FONT, 18)
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.Popen([ff, "-y", "-loglevel", "error",
                             "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                             "-ss", str(a.start), "-t", str(a.dur), "-i", str(S.MUSIC),
                             "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16",
                             "-c:a", "aac", "-b:a", "256k", "-shortest", str(out)], stdin=subprocess.PIPE)
    n = int(a.dur * FPS)
    win = int(sr * 0.5)                       # 每帧画最近 0.5 秒的轨道
    scale = H * 0.36
    t0 = time.time()
    for i in range(n):
        t = i / FPS
        e = int(t * sr) + 2 * tau + win
        seg = xl[e - win - 2 * tau:e]
        A, B, C = seg[2 * tau:], seg[tau:-tau], seg[:-2 * tau]
        u = smooth(t, 1.0, 5.0)               # 维度展开：一条线 → 三维
        B, C = B * u, C * smooth(t, 2.5, 6.0)
        th = 0.6 + 0.09 * t; ph = 0.35 + 0.04 * t
        px = A * np.cos(th) - C * np.sin(th)
        py = (A * np.sin(th) + C * np.cos(th)) * np.sin(ph) + B * np.cos(ph)
        tt = np.linspace(0, 1, 6)[:, None]
        lx = (px[:-1] * (1 - tt) + px[1:] * tt).ravel(); ly = (py[:-1] * (1 - tt) + py[1:] * tt).ravel()
        acc = acc * 0.86 + S.splat(lx * scale + W / 2, ly * scale + H / 2)
        v = np.clip(np.log1p(acc) / 4.0, 0, 1) ** 0.6
        head = int(sr * 0.05)                 # 笔头：最近 50ms
        hx = px[-head:]; hy = py[-head:]
        hh = np.clip(S.blur(S.splat(hx * scale + W / 2, hy * scale + H / 2), 2) * 4, 0, 1)
        img = S.colorize(v) * (0.75 + 0.5 * rms[min(int(t * sr), len(rms) - 1)]) + hh[..., None] * S.RED
        img = S.glow(np.clip(img, 0, 1), 4, 0.5)
        im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        d = ImageDraw.Draw(im)
        dim = 1 + u + smooth(t, 2.5, 6.0)
        hud = [f"x(t) = Can't Kill Us  t = {a.start + t:6.2f} s   lowpass 120 Hz",
               f"tau = {tau / sr * 1000:.1f} ms   embed dim = {dim:.2f}",
               f"rms = {rms[min(int(t * sr), len(rms) - 1)]:.3f}"]
        for j, line in enumerate(hud):
            d.text((48, H - 48 - 26 * (len(hud) - j)), line, font=f, fill=(150, 160, 168))
        d.text((W - 48 - 230, 40), "01  DELAY EMBEDDING", font=f, fill=(110, 118, 124))
        proc.stdin.write(im.tobytes())
        if i % 48 == 0:
            print(f"frame {i}/{n}  {time.time() - t0:.0f}s", flush=True)
    proc.stdin.close(); proc.wait()
    print("->", out)


if __name__ == "__main__":
    main()
