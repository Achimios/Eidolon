#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🎬样片-粒子-她坠入高维：生成的「她」（设定图）被采样成 N 个点；每个低音起音从胸口剥下一圈，
剥下的点流进这首歌自己的延迟嵌入轨道 (x(t), x(t-τ), x(t-2τ))。点数守恒：intact + in orbit = N。
生成模型只负责她的样子；粒子的每一步都来自音乐（👤：要有实质信息，不要唯美的假粒子）。
用法：python tools/particle_her.py [--start 0] [--dur 13.11] [--img 设定图] [--out 路径]
"""
import argparse
import json
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


def lowfreq_noise(h, w, cell, rng):
    """有机边缘用的低频噪声（随机格子双线性放大）"""
    g = rng.random((h // cell + 2, w // cell + 2))
    return np.asarray(Image.fromarray((g * 65535).astype(np.uint16)).resize((w, h), Image.BICUBIC)) / 65535


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--dur", type=float, default=13.11)          # 前奏到第一个乐句边界
    ap.add_argument("--img", default=str(S.MV / "生成" / "角色_她_正面_灰_pro.jpg"))
    ap.add_argument("--N", type=int, default=250000)   # 9 万点时剥开处只看到黑洞，点太稀，碎屑撑不起「她」的密度
    ap.add_argument("--keep", type=float, default=0.38, help="片尾仍完整的比例（脸最后走）")
    ap.add_argument("--out", default=str(S.MV / "样片" / "06_粒子_她坠入高维_前奏.mp4"))
    a = ap.parse_args()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    W, H = S.W, S.H
    rng = np.random.default_rng(7)

    # ---- 她：设定图 → 亮度 → N 个点（按亮度加权采样）
    im = np.asarray(Image.open(a.img).convert("L").resize((W, H), Image.LANCZOS)) / 255.0
    lum = np.clip((im - 0.06) / 0.94, 0, 1)
    p = (lum ** 1.4).ravel(); p /= p.sum()
    idx = rng.choice(W * H, size=a.N, p=p)
    sy, sx = np.divmod(idx, W)
    px0 = sx + rng.random(a.N); py0 = sy + rng.random(a.N)
    bright = lum.ravel()[idx]

    # ---- 剥离场：离胸口的距离 + 低频噪声（边缘有机，不是圆）
    ys, xs = np.nonzero(lum > 0.25)
    chest = np.array([xs.mean(), ys.min() + 0.42 * (ys.max() - ys.min())])   # 身体纵向 42% 处 ≈ 胸口
    yy, xx = np.mgrid[0:H, 0:W]
    dist = np.hypot(xx - chest[0], (yy - chest[1]) * 0.8)
    field = dist / dist[lum > 0.25].max() + 0.22 * lowfreq_noise(H, W, 60, rng) + 0.16 * lowfreq_noise(H, W, 9, rng)  # 大块 + 碎边
    head_pen = np.clip((chest[1] - 0.18 * (ys.max() - ys.min()) - yy) / 200.0, 0, 1) * 0.6   # 头部更晚走
    field = field + head_pen
    pf = field[sy, sx]
    R_end = np.quantile(pf, 1 - a.keep)            # 片尾的剥离半径 → 正好剩 keep 比例

    # ---- 音乐：感官层的起音 + 威胁强度 → 剥离半径 R(t) 一步一步往外走
    sense = json.loads((S.MV / "感官.json").read_text("utf-8"))
    thr = np.array(sense["channels"]["threat"])
    ons = [t for t in sense["events"]["onsets"] if a.start <= t < a.start + a.dur]
    st = np.array([thr[max(0, int(t * FPS) - 3):int(t * FPS) + 4].max() for t in ons])
    steps = st / st.sum() * R_end
    n = int(round(a.dur * FPS))

    def R_at(t):                                     # 每个起音 120ms 内把这一步走完（冲击 → 衰减）
        tt = a.start + t
        return sum(s * smooth(tt, o, o + 0.12) for o, s in zip(ons, steps))

    # ---- 轨道：这一段低音的延迟嵌入（和 01 样片同一套参数）
    sr = 11025
    x, _ = S.audio(sr=sr, start=a.start, dur=a.dur + 1.0)
    k = int(sr / 120)
    xl = np.convolve(np.convolve(x, np.ones(k) / k, mode="same"), np.ones(k) / k, mode="same")
    xl /= (np.abs(xl).max() + 1e-9)
    tau = int(sr * 0.012); win = int(sr * 0.5)
    xl = np.concatenate([np.zeros(win + 2 * tau), xl])
    orbit_c = np.array([W * 0.70, H * 0.46]); scale = H * 0.30
    slot = rng.random(a.N)                           # 每个点在轨道窗口里的位置（固定 → 点沿轨道流动）

    det_t = np.full(a.N, np.inf)                     # 每个点被剥下的时刻
    cx, cy = px0.copy(), py0.copy()
    f = ImageFont.truetype(S.FONT, 18)
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.Popen([ff, "-y", "-loglevel", "error",
                             "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                             "-ss", str(a.start), "-t", str(a.dur), "-i", str(S.MUSIC),
                             "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16",
                             "-c:a", "aac", "-b:a", "256k", "-shortest", str(out)], stdin=subprocess.PIPE)
    acc = np.zeros((H, W)); t0 = time.time(); m_ema = 0.3
    for i in range(n):
        t = i / FPS
        R = R_at(t)
        newly = (pf < R) & np.isinf(det_t)
        det_t[newly] = t
        gone = np.isfinite(det_t)

        # 轨道上的目标位置：最近 0.5 s 的 (x(t), x(t-τ), x(t-2τ))，慢转
        e = int(t * sr) + 2 * tau + win
        seg = xl[e - win - 2 * tau:e]
        m_ema = 0.85 * m_ema + 0.15 * max(np.abs(seg).max(), 1e-3)   # 自适应：前奏低音很轻，按最近振幅放大（HUD 写 gain）
        seg = seg / m_ema
        A, B, C = seg[2 * tau:], seg[tau:-tau], seg[:-2 * tau]
        th = 0.6 + 0.09 * t; ph = 0.35 + 0.04 * t
        ox = A * np.cos(th) - C * np.sin(th)
        oy = (A * np.sin(th) + C * np.cos(th)) * np.sin(ph) + B * np.cos(ph)
        j = (slot[gone] * (len(ox) - 1)).astype(int)
        tx = orbit_c[0] + ox[j] * scale; ty = orbit_c[1] + oy[j] * scale
        age = t - det_t[gone]
        rate = 0.02 + 0.55 * smooth(age, 0.25, 1.6)  # 刚剥下时先在原地碎开，越久越被轨道吸住（吸住后贴紧轨道，形状才看得清）
        cx[gone] += (tx - cx[gone]) * rate; cy[gone] += (ty - cy[gone]) * rate

        # 画：还在的部分直接用原图（被剥离的像素挖掉），剥下的点 splat
        hole = smooth(R - field, -0.004, 0.012)      # 剥离边缘 ≈ 几个像素的软边
        body = im * (1 - hole)
        rim = np.clip((1 - np.abs(R - field) / 0.01), 0, 1) * (lum > 0.2) * (R > 0)   # 正在裂开的那条边
        fresh = 1 + 3.0 * (1 - smooth(age, 0.0, 0.6))     # 刚碎下来的点亮，像碎屑；进轨道后变细
        dots = S.splat(cx[gone], cy[gone], weight=(bright[gone] * 0.9 + 0.1) * fresh)
        acc = acc * 0.35 + dots
        v = np.clip(np.log1p(acc) / 3.0, 0, 1) ** 0.85
        img = np.clip(body[..., None] * S.GREY * 1.05 + S.colorize(v), 0, 1)
        # 红：轨道笔头（最近 50ms = 「此刻」的音乐）
        hd = int(sr * 0.05)
        hh = np.clip(S.blur(S.splat(orbit_c[0] + ox[-hd:] * scale, orbit_c[1] + oy[-hd:] * scale), 2) * 4, 0, 1)
        img = img * (1 - hh[..., None]) + hh[..., None] * S.RED
        img = img + rim[..., None] * 0.35 * S.GREY
        img = S.glow(np.clip(img, 0, 1), 3, 0.35)
        fr = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        d = ImageDraw.Draw(fr)
        ng = int(gone.sum()); k_on = sum(1 for o in ons if o <= a.start + t)
        hud = [f"x(t) = Can't Kill Us  t = {a.start + t:6.2f} s   lowpass 120 Hz   tau = {tau / sr * 1000:.1f} ms   dim = 3",
               f"onsets {k_on:2d}/{len(ons)}   front R = {R:.3f}   orbit gain = {1 / m_ema:5.2f}",
               f"N = {a.N:,}   intact {a.N - ng:6,}   in orbit {ng:6,}   mass in = mass out"]
        for jj, line in enumerate(hud):
            d.text((48, H - 48 - 26 * (len(hud) - jj)), line, font=f, fill=(150, 160, 168))
        d.text((W - 48 - 260, 40), "THYMOS 01  FALL INTO 11D", font=f, fill=(110, 118, 124))
        proc.stdin.write(fr.tobytes())
        if i % 48 == 0:
            print(f"frame {i}/{n}  {time.time() - t0:.0f}s  intact {a.N - ng}", flush=True)
    proc.stdin.close(); proc.wait()
    print("->", out)


if __name__ == "__main__":
    main()
