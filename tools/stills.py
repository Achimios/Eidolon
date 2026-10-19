#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🖼概念静帧-入口-《Can't Kill Us》概念草案的静帧（给用户在剪映里蒙太奇用）。
每张都是真算出来的：延迟嵌入（这首歌自己的音频）、Kuramoto、逻辑斯蒂分岔、反应扩散、DLA、沙堆、DCT、Droste、吸引子、最优传输。
风格：黑底、冷灰、只给一个强调色；角落 HUD 小字 = 这张图真实的参数（符号说真话）。
用法：python tools/stills.py [--only name1,name2] [--out 目录]   默认输出到 Projekt_Visual/MV_Cant_Kill_Us/静帧/
"""
import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
PROJ = ROOT.parents[1] / "Projekt_Visual"
MV = PROJ / "MV_Cant_Kill_Us"
MUSIC = PROJ / "Assets" / "The Glitch Mob - Can't Kill Us [2020 Edit]_190189282.mp4"
SRC_IMG = PROJ / "第二章_墙里的东西" / "素材" / "AI girl_corridor_01_Var1.jpg"
W, H = 1920, 1080
GREY = np.array([0.80, 0.85, 0.88])   # 冷灰
RED = np.array([1.0, 0.16, 0.20])     # 唯一强调色
FONT = "C:/Windows/Fonts/consola.ttf"


# ---------- 公共 ----------
def tone(d, gamma=0.45):
    """密度图 → 0~1:log 压缩再 gamma,黑底上细节才出得来"""
    d = np.log1p(d.astype(np.float64))
    m = d.max() or 1.0
    return np.clip(d / m, 0, 1) ** gamma


def colorize(v, accent=None, col=GREY):
    """单色映射;accent 是 0~1 的权重图,把那部分染成强调色"""
    img = v[..., None] * col
    if accent is not None:
        img = img * (1 - accent[..., None]) + v[..., None] * RED * accent[..., None]
    return img


def splat(x, y, w=W, h=H, weight=None):
    """点 → 密度图(x, y 已是像素坐标)"""
    ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    d = np.zeros((h, w))
    np.add.at(d, (y[ok].astype(int), x[ok].astype(int)), 1 if weight is None else weight[ok])
    return d


def fit(px, py, margin=0.12, w=W, h=H, keep_aspect=True):
    """把点云缩放进画幅(留边)"""
    x0, x1, y0, y1 = px.min(), px.max(), py.min(), py.max()
    sx, sy = (w * (1 - 2 * margin)) / (x1 - x0 + 1e-9), (h * (1 - 2 * margin)) / (y1 - y0 + 1e-9)
    if keep_aspect:
        sx = sy = min(sx, sy)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return (px - cx) * sx + w / 2, (py - cy) * sy + h / 2


def blur(img, r=2):
    """便宜的盒式模糊(辉光用),不依赖 scipy"""
    out = img.copy()
    for ax in (0, 1):
        acc = np.cumsum(np.pad(out, [(r + 1, r) if a == ax else (0, 0) for a in range(out.ndim)], mode="edge"), axis=ax)
        hi = np.take(acc, range(2 * r + 1, acc.shape[ax]), axis=ax)
        lo = np.take(acc, range(0, acc.shape[ax] - 2 * r - 1), axis=ax)
        out = (hi - lo) / (2 * r + 1)
    return out


def glow(img, r=6, k=0.6):
    return np.clip(img + k * blur(blur(img, r), r), 0, 1)


def upscale(a, w=W, h=H):
    """小网格 → 画幅(双线性)"""
    return np.asarray(Image.fromarray((np.clip(a, 0, 1) * 65535).astype(np.uint16)).resize((w, h), Image.BILINEAR)) / 65535


def save(img, name, hud, out):
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(im, "RGBA")
    f = ImageFont.truetype(FONT, 18)
    tw = max(d.textlength(line, font=f) for line in hud)
    d.rectangle((36, H - 60 - 26 * len(hud), 60 + tw, H - 36), fill=(0, 0, 0, 170))  # 亮底上 HUD 也读得清
    for i, line in enumerate(hud):
        d.text((48, H - 48 - 26 * (len(hud) - i)), line, font=f, fill=(150, 160, 168))
    d.text((W - 48 - 10 * len(name), 40), name.upper(), font=f, fill=(110, 118, 124))
    p = out / f"{name}.jpg"
    im.save(p, quality=92)
    return p


# ---------- 各概念 ----------
def audio(sr=11025, start=60.0, dur=6.0):
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    raw = subprocess.run([ff, "-loglevel", "error", "-ss", str(start), "-t", str(dur), "-i", str(MUSIC), "-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float64) / 32768, sr


def s_delay_embedding():
    """>v<🖼概念静帧-延迟嵌入-这首歌 1:00 起 6 秒的波形,(x(t), x(t-τ), x(t-2τ)) 转一个角度投影"""
    x, sr = audio(start=60.0, dur=6.0)
    k = int(sr / 120)  # 盒式滤波两次 ≈ 低通 120Hz:只留贝斯 / 底鼓,轨道才是环不是毛球
    x = np.convolve(np.convolve(x, np.ones(k) / k, mode="same"), np.ones(k) / k, mode="same")
    x = x[int(sr * 2.0):int(sr * 3.2)]  # 只画 1.2 秒,线才数得清
    tau = int(sr * 0.012)
    a, b, c = x[2 * tau:], x[tau:-tau], x[:-2 * tau]
    th, ph = 0.7, 0.45
    px = a * np.cos(th) - c * np.sin(th)
    py = (a * np.sin(th) + c * np.cos(th)) * np.sin(ph) + b * np.cos(ph)
    # 线段插值成密集的点,画成连续的轨道
    t = np.linspace(0, 1, 8)[:, None]
    px = (px[:-1] * (1 - t) + px[1:] * t).ravel(); py = (py[:-1] * (1 - t) + py[1:] * t).ravel()
    X, Y = fit(px, py, 0.1)
    v = tone(splat(X, Y), 0.5)
    return glow(colorize(v), 3, 0.5), [f"x(t) = Can't Kill Us @ 1:02, {len(x)/sr:.1f} s, lowpass 120 Hz", f"tau = {tau/sr*1000:.1f} ms   embed dim = 3", "Takens: the song draws its own attractor"]


def s_kuramoto(K=2.2, N=4000, steps=900, dt=0.02, seed=1):
    """>v<🖼概念静帧-Kuramoto-N 个振子,半径 = 固有频率,角度 = 相位;K 过临界后聚成一团"""
    rng = np.random.default_rng(seed)
    w = rng.standard_cauchy(N) * 0.5
    w = np.clip(w, -4, 4)
    th = rng.uniform(0, 2 * np.pi, N)
    trail = np.zeros((H, W))
    for i in range(steps):
        z = np.exp(1j * th).mean()
        th = th + dt * (w + K * np.abs(z) * np.sin(np.angle(z) - th))
        if i > steps - 300:  # 最后 300 步留拖影
            r = 300 + 55 * w
            trail += splat(W / 2 + r * np.cos(th), H / 2 + r * np.sin(th)) * (1 - (steps - i) / 300)
    R = np.abs(np.exp(1j * th).mean())
    v = np.clip(tone(trail, 0.35) * 1.3, 0, 1)
    lock = (np.abs(w) < K * R)  # 被锁住的振子染红
    acc = tone(splat(W / 2 + (300 + 55 * w[lock]) * np.cos(th[lock]), H / 2 + (300 + 55 * w[lock]) * np.sin(th[lock])), 0.5)
    img = colorize(v) * 0.8 + acc[..., None] * RED * 0.9
    return glow(np.clip(img, 0, 1), 4, 0.6), [f"N = {N}   K = {K:.2f}   K_c = {2*0.5:.2f} (Lorentzian, gamma = 0.5)", f"R = |<e^(i theta)>| = {R:.3f}", "red = phase-locked oscillators"]


def s_logistic():
    """>v<🖼概念静帧-逻辑斯蒂-分岔图,r 从 2.8 到 4.0;过 3.5699 之后染红(混沌)"""
    rs = np.linspace(2.8, 4.0, W * 2)
    x = np.full_like(rs, 0.5)
    for _ in range(600):
        x = rs * x * (1 - x)
    d = np.zeros((H, W)); a = np.zeros((H, W))
    for _ in range(500):
        x = rs * x * (1 - x)
        px = ((rs - 2.8) / 1.2 * (W - 1)).astype(int); py = ((1 - x) * (H * 0.8) + H * 0.1).astype(int)
        np.add.at(d, (py, px), 1)
    v = tone(d, 0.55)
    chaos = (np.arange(W) >= (3.5699 - 2.8) / 1.2 * W).astype(float)[None, :] * np.ones((H, 1))
    return glow(colorize(v, chaos * 0.9), 2, 0.4), ["x_(n+1) = r x_n (1 - x_n)", "r : 2.80 -> 4.00", "r_inf = 3.5699...  onset of chaos (red)"]


def s_reaction_diffusion(F=0.037, k=0.06, steps=9000, seed=2):
    """>v<🖼概念静帧-反应扩散-Gray-Scott,270x480 网格,结果放大"""
    h, w = 270, 480
    rng = np.random.default_rng(seed)
    U = np.ones((h, w)); V = np.zeros((h, w))
    for _ in range(14):
        y, x = rng.integers(20, h - 20), rng.integers(20, w - 20)
        V[y - 4:y + 4, x - 4:x + 4] = 0.5; U[y - 4:y + 4, x - 4:x + 4] = 0.25
    Du, Dv = 0.16, 0.08
    for _ in range(steps):
        Lu = np.roll(U, 1, 0) + np.roll(U, -1, 0) + np.roll(U, 1, 1) + np.roll(U, -1, 1) - 4 * U
        Lv = np.roll(V, 1, 0) + np.roll(V, -1, 0) + np.roll(V, 1, 1) + np.roll(V, -1, 1) - 4 * V
        uvv = U * V * V
        U += Du * Lu - uvv + F * (1 - U); V += Dv * Lv + uvv - (F + k) * V
    v = upscale((V - V.min()) / (V.max() - V.min() + 1e-9)) ** 0.8
    edge = np.abs(np.gradient(v)[0]) + np.abs(np.gradient(v)[1])
    rng2 = np.random.default_rng(7); spots = np.clip((upscale(rng2.random((6, 10))) - 0.72) * 6, 0, 1)  # 平滑的几团,不是方块
    acc = np.clip(edge * 25, 0, 1) * spots * 0.8
    return colorize(v * 0.9, acc), [f"Gray-Scott   F = {F}   k = {k}", f"Du = {Du}   Dv = {Dv}   steps = {steps}", "Turing patterns: life out of chemistry"]


def s_dla(n=40000, seed=3):
    """>v<🖼概念静帧-DLA-很多随机游走者同时走,碰到已有晶体就粘上;粘上的先后 = 亮度(早亮晚暗)"""
    rng = np.random.default_rng(seed)
    S = 401; c = S // 2
    grid = np.zeros((S, S), bool); grid[c, c] = True
    order = np.zeros((S, S)); stuck = 1
    rmax = 2
    walkers = None
    while stuck < n:
        if walkers is None or len(walkers) < 600:
            m = 1200; ang = rng.uniform(0, 2 * np.pi, m); R = rmax + 6
            new = np.stack([c + R * np.cos(ang), c + R * np.sin(ang)], 1).astype(int)
            walkers = new if walkers is None else np.concatenate([walkers, new])
        walkers += rng.integers(-1, 2, walkers.shape)
        d = np.hypot(walkers[:, 0] - c, walkers[:, 1] - c)
        walkers = walkers[(d < rmax + 20) & (walkers.min(1) > 1) & (walkers.max(1) < S - 2)]
        y, x = walkers[:, 0], walkers[:, 1]
        nb = grid[y + 1, x] | grid[y - 1, x] | grid[y, x + 1] | grid[y, x - 1]
        if nb.any():
            for yy, xx in walkers[nb]:
                if not grid[yy, xx]:
                    grid[yy, xx] = True; stuck += 1; order[yy, xx] = stuck
                    rmax = max(rmax, int(np.hypot(yy - c, xx - c)) + 1)
            walkers = walkers[~nb]
        if rmax > S // 2 - 25:
            break
    v = np.where(grid, 1 - 0.75 * order / stuck, 0)
    img = np.zeros((H, W)); s = H / S
    big = np.asarray(Image.fromarray((v * 255).astype(np.uint8)).resize((int(S * s), int(S * s)), Image.NEAREST)) / 255
    x0 = (W - big.shape[1]) // 2; img[:, x0:x0 + big.shape[1]] = big[:H]
    tip = np.clip((img > 0) * (1 - img) * 3 - 1.6, 0, 1)  # 最新长出来的尖端染红
    return glow(colorize(img, tip), 3, 0.7), [f"diffusion-limited aggregation   particles = {stuck}", f"radius = {rmax} cells   D_f ~ 1.71 (2D)", "red = newest tips"]


def s_sandpile(grains=2 ** 17):
    """>v<🖼概念静帧-沙堆-阿贝尔沙堆,中心一次放 2^17 粒,反复同时倒塌到稳定;0~3 粒 = 四档灰"""
    S = 241
    h = np.zeros((S, S), np.int64); h[S // 2, S // 2] = grains
    topples = 0
    while True:
        q = h // 4
        if not q.any():
            break
        topples += int(q.sum())
        h -= 4 * q
        h[1:, :] += q[:-1, :]; h[:-1, :] += q[1:, :]; h[:, 1:] += q[:, :-1]; h[:, :-1] += q[:, 1:]
    ys, xs = np.nonzero(h + (np.arange(S)[:, None] * 0))  # 占位,保持形状
    lv = np.array([0.0, 0.35, 0.62, 0.95])[h]
    sz = H - 40
    big = np.asarray(Image.fromarray((lv * 255).astype(np.uint8)).resize((sz, sz), Image.NEAREST)) / 255
    img = np.zeros((H, W)); x0 = (W - sz) // 2; img[20:20 + sz, x0:x0 + sz] = big
    acc = (np.asarray(Image.fromarray(((h == 3) * 255).astype(np.uint8)).resize((sz, sz), Image.NEAREST)) / 255) * 0.0
    crit = np.zeros((H, W)); crit[20:20 + sz, x0:x0 + sz] = 0  # 3 粒的点太多,染红就成一片粉;只灰,红留给现场的雪崩   # 3 粒 = 再加一粒就塌 = 红
    return colorize(img, crit), [f"abelian sandpile   grains = 2^17 = {grains}", f"topplings = {topples:,}", "brightest = 3 grains: one more and it avalanches"]


def _dct_mat(n):
    k = np.arange(n)[:, None]; i = np.arange(n)[None, :]
    M = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.sqrt(2 / n); M[0] /= np.sqrt(2)
    return M


def s_dct(keep=(1.0, 0.02, 0.003, 0.0005)):
    """>v<🖼概念静帧-DCT-第二章的 AI 女孩,整幅 DCT 只留最大的一小撮系数;四联:全 → 2% → 0.3% → 0.05%"""
    g = np.asarray(Image.open(SRC_IMG).convert("L").resize((480, 360)), np.float64) / 255
    A, B = _dct_mat(360), _dct_mat(480)
    C = A @ g @ B.T
    panels = []
    for f in keep:
        n = max(1, int(C.size * f))
        th = np.sort(np.abs(C).ravel())[-n]
        rec = A.T @ np.where(np.abs(C) >= th, C, 0) @ B
        panels.append(np.clip(rec, 0, 1))
    pw, ph = W // 4, int(W / 4 * 360 / 480)
    img = np.zeros((H, W)); y0 = (H - ph) // 2
    for i, p in enumerate(panels):
        big = np.asarray(Image.fromarray((p * 255).astype(np.uint8)).resize((pw - 12, ph), Image.BILINEAR)) / 255
        img[y0:y0 + ph, i * pw + 6:i * pw + 6 + big.shape[1]] = big
    acc = np.zeros((H, W)); acc[y0:y0 + ph, 3 * pw:] = 0.8  # 最后一格(几乎死了)染红
    return colorize(img, acc), [f"2D DCT, 480x360 = {C.size} coefficients", "kept: 100% | 2% | 0.3% | 0.05% (" + str(max(1, int(C.size * keep[-1]))) + " coeffs)", "compress until only light is left"]


def s_droste():
    """>v<🖼概念静帧-Droste-Escher 式对数螺旋自相似:复对数 → 旋转缩放 → 指数,图里套图"""
    src = np.asarray(Image.open(SRC_IMG).convert("L").resize((1024, 768)), np.float64) / 255
    sh, sw = src.shape
    r1, r2 = 0.12, 1.0            # 内外半径比 = 每一圈缩小的倍数
    yy, xx = np.mgrid[0:H, 0:W]
    z = ((xx - W / 2) + 1j * (yy - H / 2)) / (H / 2)
    lz = np.log(z + 1e-9)
    period = np.log(r2 / r1)
    alpha = np.arctan(period / (2 * np.pi))
    f = np.cos(alpha) * np.exp(1j * alpha)
    lz = lz / f
    re = np.mod(lz.real - np.log(r1), period) + np.log(r1)
    w = np.exp(re + 1j * lz.imag)
    u = (w.real * 0.5 + 0.5) * (sw - 1); v = (w.imag * 0.5 * sw / sh + 0.5) * (sh - 1)
    ok = (u >= 0) & (u < sw) & (v >= 0) & (v < sh)
    img = np.where(ok, src[np.clip(v, 0, sh - 1).astype(int), np.clip(u, 0, sw - 1).astype(int)], 0)
    ring = np.clip(1 - np.abs(np.mod(np.log(np.abs(z) + 1e-9) / period, 1) - 0.5) * 40, 0, 1) * 0.6
    return colorize(img ** 1.2, ring), [f"Droste: w = exp( log(z) / f ),  f = cos(a) e^(ia)", f"scale per turn = {r2/r1:.1f}x   a = {np.degrees(alpha):.1f} deg", "falling into yourself, forever"]


def s_attractor(n=6_000_000, a=-1.7, b=1.3, c=-0.1, d=-1.21):
    """>v<🖼概念静帧-吸引子-Clifford 吸引子,6M 个点的密度;扰动后回来的那条轨道染红"""
    x = np.zeros(2000); y = np.zeros(2000)
    rng = np.random.default_rng(4); x += rng.uniform(-0.1, 0.1, 2000); y += rng.uniform(-0.1, 0.1, 2000)
    xs, ys = [], []
    for i in range(n // 2000):
        x, y = np.sin(a * y) + c * np.cos(a * x), np.sin(b * x) + d * np.cos(b * y)
        if i > 20:
            xs.append(x.copy()); ys.append(y.copy())
    X, Y = fit(np.concatenate(xs), np.concatenate(ys), 0.08)
    v = tone(splat(X, Y), 0.42)
    # 一条被踢出去的轨道:从远处出发,几十步内被拉回吸引子
    px, py = [2.5], [2.5]
    for _ in range(60):
        px.append(np.sin(a * py[-1]) + c * np.cos(a * px[-1])); py.append(np.sin(b * px[-2]) + d * np.cos(b * py[-1]))
    allx, ally = np.concatenate(xs), np.concatenate(ys)
    sx = (W * 0.84) / (allx.max() - allx.min()); sy = (H * 0.84) / (ally.max() - ally.min()); s = min(sx, sy)
    cx, cy = (allx.max() + allx.min()) / 2, (ally.max() + ally.min()) / 2
    tx = (np.array(px) - cx) * s + W / 2; ty = (np.array(py) - cy) * s + H / 2
    t = np.linspace(0, 1, 30)[:, None]
    lx = (tx[:-1] * (1 - t) + tx[1:] * t).ravel(); ly = (ty[:-1] * (1 - t) + ty[1:] * t).ravel()
    acc = np.clip(blur(splat(lx, ly), 1) * 3, 0, 1)
    img = colorize(v) + acc[..., None] * RED
    return glow(np.clip(img, 0, 1), 3, 0.5), [f"Clifford attractor  a={a} b={b} c={c} d={d}", f"points = {n:,}", "kicked out (red) -> pulled back. can't kill us"]


def s_optimal_transport(N=60000, iters=60, seed=5):
    """>v<🖼概念静帧-最优传输-切片最优传输:一团均匀圆盘 → 女孩的亮度分布;画中间时刻 t=0.5 + 流线"""
    rng = np.random.default_rng(seed)
    g = np.asarray(Image.open(SRC_IMG).convert("L").resize((320, 240)), np.float64)
    g = np.clip(g / 255 - 0.35, 0, 1) ** 2  # 只要亮的部分(全息人形),不然目标是一整块长方形
    p = (g / g.sum()).ravel()
    idx = rng.choice(p.size, N, p=p)
    tgt = np.stack([idx % 320 + rng.random(N), idx // 320 + rng.random(N)], 1)
    tgt = (tgt - [160, 120]) / 120
    ang = rng.uniform(0, 2 * np.pi, N); rr = np.sqrt(rng.random(N)) * 0.35
    src = np.stack([rr * np.cos(ang) - 1.6, rr * np.sin(ang)], 1)
    tgt[:, 0] += 0.9
    X = src.copy()
    for _ in range(iters):   # 切片 OT:随机方向上一维排序配对,沿该方向挪
        th = rng.uniform(0, np.pi); dvec = np.array([np.cos(th), np.sin(th)])
        a, b = X @ dvec, tgt @ dvec
        ia, ib = np.argsort(a), np.argsort(b)
        delta = np.empty(N); delta[ia] = b[ib] - a[ia]
        X += 0.9 * delta[:, None] * dvec
    cost = np.mean(np.sum((X - src) ** 2, 1))
    d = np.zeros((H, W))
    for t in np.linspace(0, 1, 9):   # 位移插值的几个时刻叠在一起 = 流
        P = src * (1 - t) + X * t
        d += splat(P[:, 0] * 360 + W / 2, P[:, 1] * 360 + H / 2) * (0.35 if 0 < t < 1 else 1.0)
    v = tone(d, 0.5)
    mid = src * 0.5 + X * 0.5
    acc = np.clip(tone(splat(mid[:, 0] * 360 + W / 2, mid[:, 1] * 360 + H / 2), 0.5) * 1.2, 0, 1)
    return glow(colorize(v, acc * 0.8), 3, 0.5), [f"sliced optimal transport   N = {N:,} particles   slices = {iters}", f"W2^2 ~ {cost:.3f}   mass in = mass out", "shattered is not dead. it just moved"]


STILLS = {"01_delay_embedding": s_delay_embedding, "02_kuramoto": s_kuramoto, "03_logistic": s_logistic,
          "04_reaction_diffusion": s_reaction_diffusion, "05_dla": s_dla, "06_sandpile": s_sandpile,
          "07_dct": s_dct, "08_droste": s_droste, "09_optimal_transport": s_optimal_transport, "10_attractor": s_attractor}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=str(MV / "静帧"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for name, fn in STILLS.items():
        if a.only and not any(k in name for k in a.only.split(",")):
            continue
        t0 = time.time()
        img, hud = fn()
        p = save(img, name, hud, out)
        print(f"{name}: {time.time() - t0:.1f}s -> {p.name}", flush=True)


if __name__ == "__main__":
    main()
