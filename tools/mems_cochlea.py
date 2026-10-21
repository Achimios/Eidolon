#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🎬样片-MEMS 耳蜗（v3，按👤「太简单 / 不够狠 / 每个局部自己动且带信息 / 要能连续运镜 → 真 3D」重做）。
一台机械频谱仪：64 根悬臂梁指（长度 L_i、厚度 t_i 渐变）挂在一根梳齿驱动的梭子上（梳齿 / 折叠梁 / 锚点借静帧 13）。
真物理：Euler–Bernoulli 一阶 f_i = (1.875² / 2π) · (t_i / L_i²) · √(E / 12ρ)，f_i 对数铺满 80 Hz – 8 kHz；
每根指 = 阻尼谐振子，被「other」分轨（抽搐音所在层）在它自己那一频带的 STFT 能量（÷ 该带在 drop1 的中位数 = 谱对比度）驱动，
画面上按可见频率 5–14 Hz 慢放（越长越慢，顺序保真）→ 声音的谱怎么变，哪几根指就在那一刻拍得最狠：机械演出的频谱图。
形状 = 悬臂一阶模态 φ(ξ)，按转角积分（不伸长，大挠度不穿帮），峰值 ≈ 0.35–0.45 L；每帧 6 次子曝光（运动模糊叠影）；梭子几乎不动。
红 = 此刻 |w_tip|/L 最大的那根指（= 主导频率）。第二次（75.43 s）更狠、镜头更近，几根指尖开始卷曲（有机的前兆 → drop3 克拉尼）。
渲染：3D 线框（薄板挤出）→ 透视相机（一直在动，抽搐处加速）→ GPU 上沿线密采样双线性泼溅 + 深度衰减。
用法：python tools/mems_cochlea.py [--preview 帧号,帧号]   输出 样片/09c_MEMS耳蜗_drop1.mp4 + 静帧/13b_mems_cochlea.jpg（不覆盖旧文件）
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

MV = Path("C:/Users/Administrator/Desktop/Projekt_Visual/MV_Cant_Kill_Us")
MUSIC = MV.parent / "Assets" / "The Glitch Mob - Can't Kill Us [2020 Edit]_190189282.mp4"
OUT = MV / "样片" / "09c_MEMS耳蜗_drop1.mp4"
STILL = MV / "静帧" / "13b_mems_cochlea.jpg"
FONT = "C:/Windows/Fonts/consola.ttf"
FPS, SUB, W, H, SS = 24, 6, 1920, 1080, 2
dev = "cuda" if torch.cuda.is_available() else "cpu"
GREY = torch.tensor([0.80, 0.85, 0.88], device=dev)
RED = torch.tensor([1.0, 0.16, 0.20], device=dev)

# 两次抽搐（动机_抽搐.json hits[0:2]），每段一个插入镜头，硬切
SHOTS = [dict(t0=62.2, t1=64.0, hit=62.624, peak=0.40, curl=0.0),
         dict(t0=75.0, t1=76.8, hit=75.43, peak=0.46, curl=1.0)]

# ---------- 器件（µm） ----------
E, RHO = 160e9, 2330.0                      # poly-Si
C1 = 1.875 ** 2 / (2 * np.pi) * np.sqrt(E / (12 * RHO))   # f = C1 · t / L²（SI）
N = 64
F_LO, F_HI = 80.0, 8000.0
f_i = np.geomspace(F_LO, F_HI, N)
L_i = np.geomspace(4000.0, 1000.0, N)       # 长 → 短（低频在左）
t_i = f_i * (L_i * 1e-6) ** 2 / C1 * 1e6    # 由 f、L 反解厚度（µm）
FV_i = np.geomspace(5.0, 14.0, N)           # 画面可见频率（Hz），保持顺序
PITCH, WF = 60.0, 40.0
x_i = (np.arange(N) - (N - 1) / 2) * PITCH
T_DRAW = 3.0                                 # 厚度画大 3 倍（真厚 1–6 µm 在这尺度看不见）

# 悬臂一阶模态 φ(ξ)，tip 归一 = 1；我们要的是 φ'(ξ)
_bl, _sg = 1.8751, 0.7341


def _phi(xi):
    b = _bl * xi
    return (np.cosh(b) - np.cos(b) - _sg * (np.sinh(b) - np.sin(b)))


K = 40
XI = np.linspace(0, 1, K)
_d = np.gradient(_phi(XI), XI)
DPHI = _d / _phi(np.array([1.0]))[0]        # ∫0^1 DPHI = 1


# ---------- 音频 → 每根指的驱动 ----------
def band_drive():
    """返回 (times, D[N, T])：每根指所在频带的 STFT 幅度 ÷ 该带在 52–90 s 的中位数，再减 1 截负"""
    x, sr = sf.read(next(MV.glob("分轨/**/other.wav")), dtype="float32")
    x = x.mean(1)
    a, b = int(50.0 * sr), int(90.0 * sr)
    seg = torch.tensor(x[a:b], device=dev)
    nfft, hop = 4096, 220
    S = torch.stft(seg, nfft, hop, window=torch.hann_window(nfft, device=dev), return_complex=True).abs() ** 2
    fr = np.fft.rfftfreq(nfft, 1 / sr)
    edges = np.geomspace(F_LO / 1.075, F_HI * 1.075, N + 1)
    B = torch.zeros(N, S.shape[1], device=dev)
    for i in range(N):
        m = (fr >= edges[i]) & (fr < edges[i + 1])
        if m.sum() == 0:
            m = np.abs(fr - f_i[i]) == np.abs(fr - f_i[i]).min()
        B[i] = S[torch.tensor(m, device=dev)].mean(0)
    amp = B.sqrt()
    tt = 50.0 + np.arange(S.shape[1]) * hop / sr
    ref = amp[:, (tt > 52) & (tt < 90)].median(1).values[:, None] + 1e-6
    lev = torch.clamp(amp / ref - 1.0, min=0)
    sm = F.avg_pool1d(amp[None], 5, 1, 2, count_include_pad=False)[0]          # ~25 ms 平滑
    flux = torch.clamp(sm[:, 4:] - sm[:, :-4], min=0) / ref                     # 起音（颤音的每一下）
    flux = F.pad(flux, (4, 0))
    D = (lev + 2.5 * flux).cpu().numpy()
    return tt, D


def simulate(tt, D, t0, t1, zeta=0.07, pre=1.5, fs=2400):
    """x'' + 2ζω x' + ω² x = ω² · d(t) · cos(ωt + φ)，从 t0−pre 起算（入镜时已经在动）。返回 (ts, X[N, T])"""
    ts = np.arange(t0 - pre, t1 + 0.1, 1 / fs)
    d = np.stack([np.interp(ts, tt, D[i]) for i in range(N)])
    w = 2 * np.pi * FV_i[:, None]
    ph = np.random.default_rng(7).uniform(0, 2 * np.pi, (N, 1))
    force = w ** 2 * d * np.cos(w * ts[None] + ph)
    X = np.zeros_like(d); x = np.zeros(N); v = np.zeros(N); h = 1 / fs
    for k in range(len(ts)):                        # 半隐式欧拉
        v += h * (force[:, k] - 2 * zeta * w[:, 0] * v - w[:, 0] ** 2 * x)
        x += h * v
        X[:, k] = x
    return ts, X


# ---------- 3D 几何 ----------
def box_edges(x0, x1, y0, y1, z0, z1):
    p = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float)
    idx = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    return np.stack([np.stack([p[a], p[b]]) for a, b in idx])


def hatch(x0, x1, y0, y1, z, step=70.0):
    segs = []
    for c in np.arange(x0 - (y1 - y0), x1, step):     # x − y = c 的 45° 线，裁到矩形里
        a = (max(x0, c + y0), max(x0, c + y0) - c)
        b = (min(x1, c + y1), min(x1, c + y1) - c)
        if b[0] > a[0]:
            segs.append([[a[0], a[1], z], [b[0], b[1], z]])
    return np.array(segs, float).reshape(-1, 2, 3)


TH = 20.0   # 框架画的厚度


def frame_geometry(dy):
    """梭子（几乎不动，dy µm）+ 转子梳齿 + 定子 + 折叠梁 + 锚点。返回 (segs[M,2,3], inten[M])"""
    S, I = [], []

    def add(s, w):
        S.append(s); I.append(np.full(len(s), w))
    add(box_edges(-1950, 1950, -220 + dy, 0 + dy, -TH, 0), 0.95)                 # 梭子
    for xr in np.arange(-1800, 1801, 48):                                         # 转子齿
        add(box_edges(xr - 8, xr + 8, -420 + dy, -220 + dy, -TH, 0), 0.55)
    for xs in np.arange(-1776, 1777, 48):                                         # 定子齿
        add(box_edges(xs - 8, xs + 8, -460, -260, -TH, 0), 0.4)
    add(box_edges(-1900, 1900, -640, -460, -TH, 0), 0.6)                          # 定子锚
    add(hatch(-1900, 1900, -640, -460, 0.0), 0.22)
    s = np.linspace(0, 1, 24); sh = 3 * s ** 2 - 2 * s ** 3
    for sd in (1, -1):                                                            # 折叠梁 + 桁架 + 锚
        xe = sd * 1950; xt = sd * 2650
        add(box_edges(min(xt, xt + sd * 40), max(xt, xt + sd * 40), -300 + dy / 2, 80 + dy / 2, -TH, 0), 0.6)
        ax0, ax1 = sorted([sd * 2050, sd * 2200])
        add(box_edges(ax0, ax1, -170, -50, -TH, 0), 0.6); add(hatch(ax0, ax1, -170, -50, 0.0, 40), 0.25)
        for yb in (-260, 40):                                                     # 内梁：梭子 → 桁架
            xs_ = xe + sd * s * 700; ys_ = yb + dy - dy / 2 * sh
            pl = np.stack([xs_, ys_, np.full_like(s, -TH / 2)], 1)
            add(np.stack([pl[:-1], pl[1:]], 1), 0.5)
        for yb in (-150, -70):                                                    # 外梁：锚 → 桁架
            xs_ = sd * 2200 + sd * s * 450; ys_ = yb + dy / 2 * sh
            pl = np.stack([xs_, ys_, np.full_like(s, -TH / 2)], 1)
            add(np.stack([pl[:-1], pl[1:]], 1), 0.5)
    return np.concatenate(S), np.concatenate(I)


def finger_geometry(a, curl, dy):
    """a[N] = 尖端挠度/L（小角意义），curl[N] = 指尖卷曲量（0..1）。
    转角 θ(s) = a·φ'(ξ) + curl·卷曲项，按弧长积分（不伸长）→ 中心线 (y, z)。返回 segs[N, E, 2, 3]"""
    th = a[:, None] * DPHI[None] + curl[:, None] * 1.35 * np.pi * np.clip((XI[None] - 0.68) / 0.32, 0, 1) ** 2
    ds = L_i[:, None] / (K - 1)
    cy = np.concatenate([np.zeros((N, 1)), np.cumsum(np.cos(0.5 * (th[:, 1:] + th[:, :-1])) * ds, 1)], 1)
    cz = np.concatenate([np.zeros((N, 1)), np.cumsum(np.sin(0.5 * (th[:, 1:] + th[:, :-1])) * ds, 1)], 1)
    ny, nz = -np.sin(th), np.cos(th)
    hw = WF / 2; ht = (t_i * T_DRAW)[:, None] / 2 + 3
    edges = []
    for sx in (-1, 1):
        for sz in (1, -1):
            px = np.broadcast_to(x_i[:, None] + sx * hw, (N, K))
            py = cy + sz * ht * ny + dy; pz = cz + sz * ht * nz - ht
            edges.append(np.stack([px, py, pz], -1))                   # [N, K, 3]
    segs = [np.stack([e[:, :-1], e[:, 1:]], 2) for e in edges]         # [N, K-1, 2, 3]
    # 尖端截面 + 根部截面（矩形）
    for j in (K - 1, 0):
        c = [edges[0][:, j], edges[2][:, j], edges[3][:, j], edges[1][:, j]]
        segs.append(np.stack([np.stack([c[q], c[(q + 1) % 4]], 1) for q in range(4)], 1))
    return np.concatenate(segs, 1)


# ---------- 相机与光栅 ----------
def look_at(eye, tgt):
    f = tgt - eye; f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return np.stack([r, u, f])


def camera(shot, t):
    """连续运动；抽搐处时间扭曲 → 速度鼓包（不从静止开始）"""
    k = SHOTS.index(shot)
    tau = (t - shot["t0"]) + 0.35 * (1 + np.tanh((t - shot["hit"] - 0.08) / 0.12))
    if k == 0:   # 从指尖前方(+y)斜上方回望：频率轴 = 画面横轴，挠度 = 竖向
        R, az, el = 7300 - 320 * tau, np.radians(62 + 6.0 * tau), np.radians(24 - 2.0 * tau)
        tgt = np.array([150 - 80 * tau, 1900.0, -60.0])
    else:        # 更近、反向绕，盯住基频 + 谐波那几根
        R, az, el = 3900 - 300 * tau, np.radians(100 - 8.0 * tau), np.radians(34 - 2.0 * tau)
        tgt = np.array([450 + 60 * tau, 1100.0, -150.0])
    eye = tgt + R * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    return eye, look_at(eye, tgt), R


def project(P, eye, Rm, fpx):
    c = (P - eye) @ Rm.T
    z = c[..., 2]
    u = fpx * c[..., 0] / z + W * SS / 2
    v = -fpx * c[..., 1] / z + H * SS / 2
    return u, v, z


def raster(buf, segs, inten, eye, Rm, fpx, zn, zf):
    """segs[M,2,3] (torch) → 沿线密采样，双线性泼溅进 buf[H*SS, W*SS]"""
    u, v, z = project(segs, eye, Rm, fpx)
    ok = (z.min(1).values > 50)
    u, v, z, inten = u[ok], v[ok], z[ok], inten[ok]
    fade = torch.clamp(1 - 0.75 * (z.mean(1) - zn) / (zf - zn), 0.18, 1.0)
    w = inten * fade
    ln = torch.hypot(u[:, 1] - u[:, 0], v[:, 1] - v[:, 0])
    n = torch.clamp((ln / 0.6).ceil().long(), 1, 4000)
    rep = torch.repeat_interleave(torch.arange(len(n), device=dev), n)
    start = torch.cumsum(n, 0) - n
    s = (torch.arange(len(rep), device=dev) - start[rep] + 0.5) / n[rep]
    pu = u[rep, 0] + s * (u[rep, 1] - u[rep, 0]); pv = v[rep, 0] + s * (v[rep, 1] - v[rep, 0])
    pw = w[rep] * (ln[rep] / n[rep])
    x0 = pu.floor(); y0 = pv.floor(); fx = pu - x0; fy = pv - y0
    x0 = x0.long(); y0 = y0.long()
    Hh, Ww = buf.shape
    flat = buf.view(-1)
    for dx, dy_, ww in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xx = x0 + dx; yy = y0 + dy_
        m = (xx >= 0) & (xx < Ww) & (yy >= 0) & (yy < Hh)
        flat.index_add_(0, (yy[m] * Ww + xx[m]), (pw * ww)[m])


def gblur(img, r):
    k = torch.exp(-0.5 * (torch.arange(-3 * r, 3 * r + 1, device=dev, dtype=torch.float32) / r) ** 2); k /= k.sum()
    x = img.permute(2, 0, 1)[:, None]
    x = F.conv2d(F.pad(x, (3 * r, 3 * r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    x = F.conv2d(F.pad(x, (0, 0, 3 * r, 3 * r), mode="replicate"), k.view(1, 1, -1, 1))
    return x[:, 0].permute(1, 2, 0)


# ---------- 主渲染 ----------
def render_frame(shot, t, sim, fnt, fnt_s, red_state):
    ts, X = sim
    k = SHOTS.index(shot)
    g = torch.zeros(H * SS, W * SS, device=dev); r = torch.zeros_like(g)
    fpx = 0.5 * W * SS / np.tan(np.radians(42) / 2)
    render_red_only(shot, t, sim, red_state)               # 红 = |w_tip|/L 包络最大的那根（滞回），先定再画，HUD 和画面一致
    for q in range(SUB):
        tq = t + (q / SUB - 0.5) / FPS                      # 360° 快门内 6 次子曝光
        j = np.searchsorted(ts, tq)
        a = disp(X[:, j], shot)
        # 卷曲：只在第 2 次，挑能量大的几根，抽搐开始后逐渐长出来
        grow = shot["curl"] * np.clip((tq - shot["hit"]) / 0.6, 0, 1) ** 1.5
        curl = grow * shot["curl_w"] * np.sign(a + 1e-9) * np.clip(np.abs(a) / 0.3, 0.3, 1.0)
        dy = 5.0 * np.sin(2 * np.pi * 3.1 * tq) * min(1.0, np.abs(a).mean() * 4)    # 梭子：几乎不动
        eye, Rm, R = camera(shot, tq)
        zn, zf = R - 2600, R + 2600
        fs = finger_geometry(a, curl, dy)                    # [N, E, 2, 3]
        red_i = red_state["i"]
        isred = np.zeros(fs.shape[:2], bool); isred[red_i] = True
        fr_s, fr_i = frame_geometry(dy)
        segs = torch.tensor(np.concatenate([fs.reshape(-1, 2, 3), fr_s]), device=dev, dtype=torch.float32)
        ew = np.ones(fs.shape[:2]); ew[:, K - 1:2 * (K - 1)] = 0.3; ew[:, 3 * (K - 1):4 * (K - 1)] = 0.3   # 下棱淡
        inten = np.concatenate([(np.where(isred, 0.0, 1.0) * ew).reshape(-1) * 0.85, fr_i])
        rint = np.concatenate([np.where(isred, 1.0, 0.0).reshape(-1) * 1.1, np.zeros(len(fr_i))])
        it = torch.tensor(inten, device=dev, dtype=torch.float32)
        rt = torch.tensor(rint, device=dev, dtype=torch.float32)
        Rm_t = torch.tensor(Rm, device=dev, dtype=torch.float32); eye_t = torch.tensor(eye, device=dev, dtype=torch.float32)
        raster(g, segs, it / SUB, eye_t, Rm_t, fpx, zn, zf)
        raster(r, segs, rt / SUB, eye_t, Rm_t, fpx, zn, zf)
    g = F.avg_pool2d(g[None, None], SS)[0, 0]; r = F.avg_pool2d(r[None, None], SS)[0, 0]
    gv = 1 - torch.exp(-1.9 * g); rv = 1 - torch.exp(-2.2 * r)
    img = gv[..., None] * GREY * (1 - rv[..., None]) + rv[..., None] * RED
    img = torch.clamp(img + 0.35 * gblur(img, 3) + 0.25 * gblur(img, 10), 0, 1)
    img = img + torch.tensor([0.020, 0.024, 0.028], device=dev)
    yy = torch.arange(H, device=dev, dtype=torch.float32)[:, None, None]
    img = img * torch.clamp(1 - 0.85 * (yy - (H - 170)) / 60, 0.15, 1.0)        # HUD 底下压暗
    im = Image.fromarray((torch.clamp(img, 0, 1) * 255).byte().cpu().numpy())
    d = ImageDraw.Draw(im)
    # 频率刻度（3D 投影到梭子前缘下方）+ 红指标签
    eye, Rm, R = camera(shot, t)
    fp1 = fpx / SS
    for fq in (100, 300, 1000, 3000, 8000):
        i = int(np.argmin(np.abs(np.log(f_i / fq))))
        P = np.array([[x_i[i], -40.0, 0.0], [x_i[i], -110.0, 0.0]])
        c = (P - eye) @ Rm.T
        uu = fp1 * c[:, 0] / c[:, 2] + W / 2; vv = -fp1 * c[:, 1] / c[:, 2] + H / 2
        d.line([(uu[0], vv[0]), (uu[1], vv[1])], fill=(120, 128, 134), width=1)
        d.text((uu[1] + 4, vv[1] + 2), f"{fq if fq < 1000 else str(fq // 1000) + 'k'} Hz", font=fnt_s, fill=(120, 128, 134))
    ri = red_state["i"]
    j = np.searchsorted(ts, t); a_now = disp(X[ri, j], shot)
    grow = shot["curl"] * np.clip((t - shot["hit"]) / 0.6, 0, 1) ** 1.5
    fs_now = finger_geometry(disp(X[:, j], shot), grow * shot["curl_w"] * np.sign(disp(X[:, j], shot) + 1e-9) * 0.6, 0.0)
    P = fs_now[ri, K - 2, 1][None] + np.array([[40.0, 0, 60.0]])       # 红指上棱的指尖
    c = (P - eye) @ Rm.T
    uu = fp1 * c[0, 0] / c[0, 2] + W / 2; vv = -fp1 * c[0, 1] / c[0, 2] + H / 2
    uu, vv = min(max(uu, 20), W - 330), min(max(vv, 90), H - 215)    # 别压到 HUD
    d.text((uu + 8, vv - 8), f"f1 = {f_i[ri]:,.0f} Hz\nL = {L_i[ri]:,.0f} um  t = {t_i[ri]:.2f} um", font=fnt_s, fill=(235, 60, 66))
    hud = [f"f1 = (1.875^2/2pi) (t/L^2) sqrt(E/12 rho)    poly-Si E = {E/1e9:.0f} GPa  rho = {RHO:.0f}    {N} cantilevers, f1 = 80 Hz .. 8 kHz (log)",
           f"L = 4000 -> 1000 um   t = {t_i[0]:.2f} -> {t_i[-1]:.2f} um (drawn x{T_DRAW:.0f})   drive = |STFT(other)| per band / drop-1 median",
           f"each finger: damped resonator, zeta = 0.07, shown at 5..14 Hz ({f_i[0]/FV_i[0]:.0f}x .. {f_i[-1]/FV_i[-1]:.0f}x slower)   mode 1, display gamma 0.65, 6 exposures/frame",
           f"TWITCH #{k + 1}  @ {shot['hit']:.2f} s    t = {t:6.2f} s    red = argmax |w_tip|/L = {f_i[ri]:,.0f} Hz   w_tip/L = {abs(a_now):.2f}"
           + ("    tips curling" if shot["curl"] and t > shot["hit"] + 0.2 else "")]
    for jj, line in enumerate(hud):
        d.text((48, H - 44 - 26 * (len(hud) - jj)), line, font=fnt, fill=(150, 160, 168))
    d.text((W - 48 - 300, 40), "13b  MEMS  COCHLEA", font=fnt, fill=(110, 118, 124))
    return im, float(np.abs(disp(X[:, j], shot)).sum())


GAMMA = 0.65


def disp(x, shot):
    """画面挠度 = sign · (|x|/max)^0.65 · 峰值（显示伽马：让谐波那几根也拍得开，HUD 写明）"""
    return np.sign(x) * (np.abs(x) / shot["xmax"]) ** GAMMA * shot["peak"]


def prepare():
    tt, D = band_drive()
    sims = []
    for sh in SHOTS:
        ts, X = simulate(tt, D, sh["t0"], sh["t1"])
        m = (ts >= sh["t0"]) & (ts <= sh["t1"])
        sh["xmax"] = np.abs(X[:, m]).max()
        # 卷曲的指：这一段能量最大的 6 根
        e = np.abs(X[:, m]).max(1)
        cw = np.zeros(N); cw[np.argsort(e)[-6:]] = 1.0
        sh["curl_w"] = cw
        sims.append((ts, X))
    return sims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", type=str, default="", help="只渲染这些全局帧号，存到 --pdir")
    ap.add_argument("--pdir", type=str, default=".")
    a = ap.parse_args()
    sims = prepare()
    fnt = ImageFont.truetype(FONT, 17); fnt_s = ImageFont.truetype(FONT, 14)
    frames = [(k, sh["t0"] + i / FPS) for k, sh in enumerate(SHOTS) for i in range(int(round((sh["t1"] - sh["t0"]) * FPS)))]
    if a.preview:
        for fi in map(int, a.preview.split(",")):
            k, t = frames[fi]
            st = {"i": N // 2}
            for tw in np.arange(SHOTS[k]["t0"], t, 1 / FPS)[-12:]:          # 预热红指的滞回状态
                render_red_only(SHOTS[k], tw, sims[k], st)
            im, _ = render_frame(SHOTS[k], t, sims[k], fnt, fnt_s, st)
            im.save(Path(a.pdir) / f"cochlea_{fi:03d}.png"); print("frame", fi, t)
        return
    assert not OUT.exists() and not STILL.exists(), "不覆盖旧文件"
    import imageio_ffmpeg
    (s0, s1) = SHOTS
    af = (f"[1:a]atrim={s0['t0']}:{s0['t1']},asetpts=PTS-STARTPTS,afade=t=out:st={s0['t1'] - s0['t0'] - 0.01}:d=0.01[a1];"
          f"[1:a]atrim={s1['t0']}:{s1['t1']},asetpts=PTS-STARTPTS,afade=t=in:d=0.005[a2];[a1][a2]concat=n=2:v=0:a=1[a]")
    proc = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                             "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                             "-i", str(MUSIC), "-filter_complex", af, "-map", "0:v", "-map", "[a]",
                             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17", "-c:a", "aac", "-b:a", "192k",
                             "-shortest", str(OUT)], stdin=subprocess.PIPE)
    best, best_im = -1, None
    st = None; last_k = -1
    for fi, (k, t) in enumerate(frames):
        if k != last_k:
            st = {"i": N // 2}; last_k = k
        im, score = render_frame(SHOTS[k], t, sims[k], fnt, fnt_s, st)
        if k == 1 and SHOTS[1]["hit"] <= t <= SHOTS[1]["hit"] + 1.04 and score > best:   # 静帧：第 2 次抽搐那 1.04 s 内最狠的一帧
            best, best_im = score, im
        proc.stdin.write(im.tobytes())
        if fi % 12 == 0:
            print("frame", fi, "/", len(frames), flush=True)
    proc.stdin.close(); proc.wait()
    best_im.save(STILL, quality=94)
    print("->", OUT, "\n->", STILL)


def render_red_only(shot, t, sim, st):
    ts, X = sim
    j = np.searchsorted(ts, t)
    env = np.abs(X[:, j - 60:j + 1]).max(1)
    cand = int(np.argmax(env))
    if env[cand] > 1.15 * env[st["i"]]:
        st["i"] = cand


if __name__ == "__main__":
    main()
