#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🎬运动参考-MEMS v2（给 Seedance 网页端）：按👤「v1 太糟（灰 CAD、几乎不动）」重做 —— 不用 Blender，自己的 GPU 光栅器。
构图 / 版式对 A2_暗_v1a：中间梭子（矩形底板 + 菱形长板），上下两排长梳齿，左右扇形折叠梁 + 桁架，外圈夹具框，右上一个红点 + 红线。
运动学 = 09c 耳蜗那一套，搬到 A2 的梳齿上：
  · 60 根梳齿 = 60 个阻尼谐振子，频带 120 Hz – 8 kHz 对数铺开，上下两排交错（上排 = 偶数带，下排 = 奇数带，左低右高）；
    每根被「other」分轨在自己那一频带的 STFT 能量（÷ 该带 50–90 s 中位数 − 1，再减去 0.9 × 自身 2 s 因果慢平均 = 适应，+ 起音通量）
    在共振频率上驱动 → 抽搐（62.624 s，基频 ≈ 1.32 kHz + 2.63 / 3.94 / 5.25 / 6.46 kHz 谐波）那几根拍得最狠，其余安静。
  · 画面频率：真实 kHz → 3.5–9 Hz（低频带慢、高频带快，顺序保真），ζ = 0.12；挠度显示伽马 0.9，抽搐窗内最大那根 = 3.2 个齿距（横向）
    + 一半的离面分量（正交相位，椭圆），软限幅 3.8。
  · 梳齿根部的「脊」是一根 1D 阻尼波动方程（c = 45 齿距/s）：被各齿的反作用力 + 梭子的冲击驱动 → 振动沿梳排传开，脊带着齿根上下起伏；
    脊的加速度反过来作为齿根激励（κ = 0.15：同排邻齿被带到 ~15%，看得见但不抢）。
  · 梭子：mems_motion_ref.drive（命中冲击 + 包络通量 ⊛ e^(−t/τ) sin 2π·2.2 Hz）沿梳齿轴平动，桁架走一半，折叠梁 S 形（3s² − 2s³）。
渲染（torch，GPU）：三角形 → 透视光栅（深度+id 打包 scatter_reduce amin）→ 平面法线 PBR-ish 着色：
  冷白聚光（后上方斜切，阴影图 + PCF）+ 一条头顶长条柔光箱的镜面反射（齿面一倾斜，高光就跑 —— 读运动的主要线索）
  + 拉丝金属噪声纹理；光锥里 ray-march 体积雾（3D fBm 噪声，阴影图 → 光束被齿切开）；4 次子曝光（180° 快门）、景深、泛光、冷调、颗粒。
  ⚠️噪声（雾、拉丝、颗粒）是程序噪声 —— 没有找真实记录当源（低熵注入的回退）。
镜头：push_remap.curve 的「进 / 缓」，第 1 帧已在动，命中加速，之后减速不停；直线推进（相机与注视点各自沿直线），不晃。
画面里**没有任何文字**。第 k 帧 = 歌曲时间 t0 + k/fps（播放时间，和 感官.json 同一时基）。
用法：
  python tools/mems_motion_ref_v2.py --work <临时目录>                     # 出视频 + 接触表 + 逐帧 json（不覆盖）
  python tools/mems_motion_ref_v2.py --work <临时目录> --only 0,24,40      # 只渲这几帧 PNG 调构图
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mems_motion_ref as M1  # noqa: E402  驱动包络 / 冲击 → 梭子位移
import push_remap as P  # noqa: E402

MV = Path("C:/Users/Administrator/Desktop/Projekt_Visual/MV_Cant_Kill_Us")
OUT = MV / "生成" / "给网页端_运动参考_v2" / "MEMS_参考视频_v2.mp4"
A2 = MV / "生成" / "风格渲染" / "A2_暗_v1a.jpg"
dev = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- 器件（单位 = 1 个齿距） ----------
NB = 30                                   # 每排齿数
NF = 2 * NB
F_LO, F_HI = 120.0, 8000.0
f_band = np.geomspace(F_LO, F_HI, NF)     # 带中心（第 k 带）
FV = np.geomspace(3.5, 9.0, NF)           # 画面频率
ZETA = 0.12
FLUX_W = 1.0                              # 起音通量权重（09c 用 2.5；这里压低，让持续的抽搐音主导而不是一下尖峰）
LF, WF, TF = 15.0, 0.42, 0.55             # 齿长 / 宽 / 厚
XF = (np.arange(NB) - (NB - 1) / 2) * 1.0
BODY = (16.0, 7.0)                        # 底板半宽 / 半高（y）
SPINE = 1.0                               # 脊宽（y）
Z0, ZT = 0.0, 1.2                         # 结构层底 / 顶
TRUSS_X = 54.0
PEAK_X, PEAK_Z, GAMMA = 3.2, 1.6, 0.9     # 齿尖峰值横向 / 离面（齿距），显示伽马
SHUTTLE_VMAX = 1.5                        # 梭子峰值（齿距）
SPINE_ZMAX = 0.45                         # 脊起伏峰值


def band_of(k):
    """第 k 带 → (排 0=上 / 1=下, 齿号)"""
    return k % 2, k // 2


def finger_band(bank, i):
    return 2 * i + bank


# ---------- 音频 → 驱动 ----------
def band_drive(x, sr, t_lo=50.0, t_hi=90.0, adapt=2.0, keep=0.9):
    """每带 STFT 幅度 ÷ 该带 52–88 s 中位数 − 1（截负）→ 减去 keep × 自身因果慢平均（适应）+ 2.5 × 起音通量。返回 (tt, D[NF, T])"""
    a, b = int(t_lo * sr), int(t_hi * sr)
    seg = torch.tensor(x[a:b], device=dev, dtype=torch.float32)
    nfft, hop = 4096, 220
    S = torch.stft(seg, nfft, hop, window=torch.hann_window(nfft, device=dev), return_complex=True).abs() ** 2
    fr = np.fft.rfftfreq(nfft, 1 / sr)
    r = np.sqrt(f_band[1] / f_band[0])
    B = torch.zeros(NF, S.shape[1], device=dev)
    for k in range(NF):
        m = (fr >= f_band[k] / r) & (fr < f_band[k] * r)
        if m.sum() == 0:
            m = np.abs(fr - f_band[k]) == np.abs(fr - f_band[k]).min()
        B[k] = S[torch.tensor(m, device=dev)].mean(0)
    amp = B.sqrt()
    tt = t_lo + np.arange(S.shape[1]) * hop / sr
    ref = amp[:, (tt > t_lo + 2) & (tt < t_hi - 2)].median(1).values[:, None] + 1e-6
    lev = torch.clamp(amp / ref - 1.0, min=0)
    sm = F.avg_pool1d(amp[None], 5, 1, 2, count_include_pad=False)[0]
    flux = F.pad(torch.clamp(sm[:, 4:] - sm[:, :-4], min=0) / ref, (4, 0))
    lev = lev.cpu().numpy()
    # 适应（像真耳蜗的外毛细胞 / 神经适应）：减去该带自己的因果慢平均（τ = adapt 秒）→ 持续的低频和弦慢慢「听不见」，新出现的谱线才拍得狠
    al = 1 - np.exp(-(hop / sr) / adapt)
    ema = np.zeros_like(lev); acc = lev[:, 0].copy()
    for n in range(lev.shape[1]):
        acc += al * (lev[:, n] - acc); ema[:, n] = acc
    nov = np.clip(lev - keep * ema, 0, None)
    return tt, nov + FLUX_W * flux.cpu().numpy()


def simulate(tt, D, ts, ysh, kappa=0.15, beta=0.6, sigma=4.0, c=45.0, gam=5.0, seed=7):
    """齿：x'' + 2ζω x' + ω² x = ω² d(t) cos(ωt + φ) − κ ẅ_root；脊（每排 NB 个节点）：w'' = c² ∇²w − γ w' + β ω² x + σ ÿ_shuttle·窗。
    ts 等步长；ysh = 梭子位移（同 ts）。返回 X[NF, T]（横向）, Q[NF, T]（正交 = x'/ω）, Wsp[2, NB, T]"""
    fs = 1 / (ts[1] - ts[0]); h = 1 / fs
    d = np.stack([np.interp(ts, tt, D[k]) for k in range(NF)])
    w = 2 * np.pi * FV
    ph = np.random.default_rng(seed).uniform(0, 2 * np.pi, NF)
    yacc = np.gradient(np.gradient(ysh, h), h)
    prof = np.exp(-0.5 * (XF / 9.0) ** 2)                       # 梭子冲击主要从脊中段进
    x = np.zeros(NF); v = np.zeros(NF)
    ws = np.zeros((2, NB)); wv = np.zeros((2, NB)); wa = np.zeros((2, NB))
    X = np.zeros((NF, len(ts))); Q = np.zeros_like(X); Wsp = np.zeros((2, NB, len(ts)))
    kb = np.array([band_of(k) for k in range(NF)])               # [NF, 2]
    for n in range(len(ts)):
        root_acc = wa[kb[:, 0], kb[:, 1]]
        a = w ** 2 * d[:, n] * np.cos(w * ts[n] + ph) - 2 * ZETA * w * v - w ** 2 * x - kappa * root_acc
        v += h * a; x += h * v
        lap = np.zeros_like(ws)
        lap[:, 1:-1] = ws[:, 2:] - 2 * ws[:, 1:-1] + ws[:, :-2]
        lap[:, 0] = ws[:, 1] - ws[:, 0]; lap[:, -1] = ws[:, -2] - ws[:, -1]
        src = np.zeros((2, NB)); src[kb[:, 0], kb[:, 1]] = beta * w ** 2 * x
        wa = c ** 2 * lap - gam * wv + src + sigma * yacc[n] * prof[None]
        wv += h * wa; ws += h * wv
        X[:, n] = x; Q[:, n] = v / w; Wsp[:, :, n] = ws
    return X, Q, Wsp


def disp(x, xmax, peak, clip=3.8):
    """显示挠度：sign·(|x|/xmax)^GAMMA·peak，再软限幅到 clip 齿距（xmax = 抽搐窗内最大）"""
    d = (np.abs(x) / xmax) ** GAMMA * peak
    return np.sign(x) * clip * np.tanh(d / clip)


# ---------- 几何（三角形） ----------
def box_tris(c0, c1):
    """轴对齐盒 → [12, 3, 3]"""
    x0, y0, z0 = c0; x1, y1, z1 = c1
    p = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float)
    f = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6), (0, 4, 5), (0, 5, 1), (1, 5, 6), (1, 6, 2), (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0)]
    return p[np.array(f)]


def prism_tris(poly, z0, z1):
    """凸多边形（逆时针）挤出 → 三角形"""
    poly = np.asarray(poly, float); n = len(poly)
    out = []
    for i in range(1, n - 1):
        out.append([[*poly[0], z1], [*poly[i], z1], [*poly[i + 1], z1]])
        out.append([[*poly[0], z0], [*poly[i + 1], z0], [*poly[i], z0]])
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        out.append([[*a, z0], [*b, z0], [*b, z1]]); out.append([[*a, z0], [*b, z1], [*a, z1]])
    return np.array(out)


def tube_tris(C, U, hw, ht):
    """矩形截面扫掠：C[N, K, 3] 中心线，U[N, K, 3] 截面「上」方向；hw/ht 半宽 / 半厚（标量或 [N]）。返回 [N*M, 3, 3]"""
    T = np.gradient(C, axis=1); T /= np.linalg.norm(T, axis=-1, keepdims=True) + 1e-12
    Wd = np.cross(T, U); Wd /= np.linalg.norm(Wd, axis=-1, keepdims=True) + 1e-12
    Ud = np.cross(Wd, T)
    hw = np.broadcast_to(np.asarray(hw, float).reshape(-1, 1, 1), C.shape[:1] + (1, 1))
    ht = np.broadcast_to(np.asarray(ht, float).reshape(-1, 1, 1), C.shape[:1] + (1, 1))
    cs = [C - hw * Wd - ht * Ud, C + hw * Wd - ht * Ud, C + hw * Wd + ht * Ud, C - hw * Wd + ht * Ud]   # 4 × [N, K, 3]
    tris = []
    for q in range(4):
        a, b = cs[q], cs[(q + 1) % 4]
        a0, a1, b0, b1 = a[:, :-1], a[:, 1:], b[:, :-1], b[:, 1:]
        tris.append(np.stack([a0, b0, b1], 2)); tris.append(np.stack([a0, b1, a1], 2))
    for j in (0, -1):
        c = [cs[q][:, j] for q in range(4)]
        tris.append(np.stack([c[0], c[1], c[2]], 1)[:, None]); tris.append(np.stack([c[0], c[2], c[3]], 1)[:, None])
    return np.concatenate([t.reshape(C.shape[0], -1, 3, 3) for t in tris], 1).reshape(-1, 3, 3)


_bl, _sg = 1.8751, 0.7341


def phi(xi):
    b = _bl * xi
    return (np.cosh(b) - np.cos(b) - _sg * (np.sinh(b) - np.sin(b))) / 2.7242


def static_scene():
    """不动的部分：地板、夹具框、锚、探针杆。返回 (tris, mat)；mat 0 硅 / 1 框 / 2 地板 / 3 红 / 4 红线"""
    S, Mt = [], []

    def add(t, m):
        S.append(t); Mt.append(np.full(len(t), m))
    for gx in np.arange(-200, 200, 25):                        # 地板（切成小块，相机后方的块被裁掉也不破洞）
        for gy in np.arange(-160, 200, 25):
            add(box_tris((gx, gy, -9.0), (gx + 25, gy + 25, -8.0)), 2)
    xi, yi, wr = 64.0, 44.0, 9.0                               # 内框（凸起导轨）
    for (a, b) in [((-xi - wr, -yi - wr), (xi + wr, -yi)), ((-xi - wr, yi), (xi + wr, yi + wr)),
                   ((-xi - wr, -yi), (-xi, yi)), ((xi, -yi), (xi + wr, yi))]:
        L = b[0] - a[0] if b[0] - a[0] > b[1] - a[1] else b[1] - a[1]
        nseg = int(np.ceil(L / 20))
        for s in range(nseg):                                  # 沿长边切段
            if b[0] - a[0] > b[1] - a[1]:
                x0 = a[0] + (b[0] - a[0]) * s / nseg; x1 = a[0] + (b[0] - a[0]) * (s + 1) / nseg
                add(box_tris((x0, a[1], -8.0), (x1, b[1], 3.0)), 1)
            else:
                y0 = a[1] + (b[1] - a[1]) * s / nseg; y1 = a[1] + (b[1] - a[1]) * (s + 1) / nseg
                add(box_tris((a[0], y0, -8.0), (b[0], y1, 3.0)), 1)
    for sx in (-1, 1):                                         # 导轨上的细杆 + 夹块（A2 左右两侧那些）
        for k, off in enumerate((2.0, 4.5, 7.0)):
            x = sx * (xi + off)
            for s in range(6):
                y0 = -yi - 4 + s * (2 * yi + 8) / 6; y1 = y0 + (2 * yi + 8) / 6
                add(box_tris((x - 0.35, y0, 3.0), (x + 0.35, y1, 3.8 + 0.3 * k)), 1)
        for sy in (-1, 1):
            add(box_tris((sx * (xi + 1) - 5, sy * (yi + 1) - 5, 3.0), (sx * (xi + 1) + 5, sy * (yi + 1) + 5, 6.0)), 1)
        for yy in (-30.0, -10.0, 12.0, 32.0):
            add(box_tris((sx * xi - 2.5 * (sx > 0) - 0.5, yy - 1.5, 3.0), (sx * xi + 2.5 * (sx < 0) + 0.5, yy + 1.5, 4.5)), 1)
    xo, yo = 84.0, 62.0                                        # 外框
    for (a, b) in [((-xo, -yo), (xo, -yo + 7)), ((-xo, yo - 7), (xo, yo)), ((-xo, -yo), (-xo + 7, yo)), ((xo - 7, -yo), (xo, yo))]:
        add(box_tris((a[0], a[1], -8.0), (b[0], b[1], 1.5)), 1)
    for sx in (-1, 1):                                         # 桁架外侧的锚（接到框上）
        add(box_tris((sx * (TRUSS_X + 2.5) - 1.5, -2.5, Z0 - 8.0), (sx * (TRUSS_X + 2.5) + 1.5, 2.5, ZT)), 0)
        add(box_tris((min(sx * (TRUSS_X + 4), sx * xi), -1.0, Z0 - 0.5), (max(sx * (TRUSS_X + 4), sx * xi), 1.0, Z0 + 0.6)), 1)
    # 探针细杆（A2 左侧那根竖直的玻璃杆）
    s = np.linspace(0, 1, 30)
    C = np.stack([np.full_like(s, -37.0), -70 + 125 * s, 7.0 - 3.0 * np.sin(np.pi * s)], 1)[None]
    add(tube_tris(C, np.broadcast_to([0, 0, 1.0], C.shape).copy(), 0.3, 0.3), 1)
    return np.concatenate(S), np.concatenate(Mt)


def dynamic_scene(ax, az, wsp, ysh, zsh):
    """会动的部分。ax/az[NF] 齿尖横向 / 离面（齿距），wsp[2, NB] 脊起伏，ysh/zsh 梭子位移。返回 (tris, mat, red_pos)"""
    S, Mt = [], []

    def add(t, m):
        S.append(t); Mt.append(np.full(len(t), m))
    BX, BY = BODY
    add(box_tris((-BX, -BY + ysh, Z0 + zsh), (BX, BY + ysh, ZT + zsh)), 0)                     # 底板
    dia = [(-29, 0), (-15, -4.2), (15, -4.2), (29, 0), (15, 4.2), (-15, 4.2)]                    # 菱形长板（A2 的梭子）
    add(prism_tris([(u, v + ysh) for u, v in dia], ZT + zsh, ZT + 0.7 + zsh), 0)
    for u in (-6.0, -2.0, 2.0, 6.0):                                                             # 板上的小凸台
        add(box_tris((u - 0.8, -1.4 + ysh, ZT + 0.7 + zsh), (u + 0.8, 1.4 + ysh, ZT + 1.0 + zsh)), 0)
    for sx in (-1, 1):                                                                           # 底板边上的耳片
        for sy in (-1, 1):
            add(box_tris((sx * 10 - 1.2, sy * BY - (sy < 0) * 1.6 + ysh, Z0 + zsh), (sx * 10 + 1.2, sy * BY + (sy > 0) * 1.6 + ysh, ZT + zsh)), 0)
    K = 18
    xi_ = np.linspace(0, 1, K); ph_ = phi(xi_)
    red = None
    for bank in (0, 1):
        sg = 1 if bank == 0 else -1
        # 脊：沿 x 的细长条，z 随 wsp 起伏
        xs = np.linspace(XF[0] - 0.8, XF[-1] + 0.8, 4 * NB)
        zs = np.interp(xs, XF, wsp[bank])
        y_sp = sg * (BY + SPINE / 2) + ysh
        C = np.stack([xs, np.full_like(xs, y_sp), (Z0 + ZT) / 2 + zsh + zs], 1)[None]
        add(tube_tris(C, np.broadcast_to([0, 0, 1.0], C.shape).copy(), SPINE / 2, (ZT - Z0) / 2), 0)
        # 齿
        ks = np.array([finger_band(bank, i) for i in range(NB)])
        y0 = sg * (BY + SPINE) + ysh
        Cx = XF[:, None] + ax[ks][:, None] * ph_[None]
        Cy = y0 + sg * LF * xi_[None] * np.ones((NB, 1))
        Cz = (Z0 + ZT) / 2 + zsh + wsp[bank][:, None] + az[ks][:, None] * ph_[None]
        C = np.stack([Cx, Cy, Cz], -1)
        U = np.broadcast_to([0, 0, 1.0], C.shape).copy()
        add(tube_tris(C, U, WF / 2, TF / 2), 0)
        if bank == 0:
            red = C[-1, -1] + np.array([0.0, 0.5, TF / 2 + 0.25])
    # 红点 + 红线（垂到右侧导轨）
    add(box_tris(red - 0.33, red + 0.33), 3)
    s = np.linspace(0, 1, 40)
    end = np.array([64.0, 18.0, 3.0])
    C = red[None] + (end - red)[None] * s[:, None]
    C[:, 2] += -2.5 * np.sin(np.pi * s) + 0.6 * s * 0
    add(tube_tris(C[None], np.broadcast_to([0, 0, 1.0], (1, 40, 3)).copy(), 0.09, 0.09), 4)
    # 折叠梁：每侧 8 根扇形，梭子端随 ysh，桁架端 ysh/2，3s² − 2s³
    s = np.linspace(0, 1, 24); u = 3 * s ** 2 - 2 * s ** 3
    yt = np.array([-6.2, -4.5, -2.7, -0.9, 0.9, 2.7, 4.5, 6.2])
    for sx in (-1, 1):
        xa = sx * BX if True else 0
        Cs = []
        for j, yy in enumerate(yt):
            yb = 0.62 * yy
            x_ = xa + (sx * TRUSS_X - xa) * s
            y_ = (yb + ysh) + (yy - yb) * s - ysh / 2 * u
            z_ = (Z0 + ZT) / 2 + zsh * (1 - 0.5 * u)
            Cs.append(np.stack([x_, y_, z_], 1))
        C = np.stack(Cs)
        add(tube_tris(C, np.broadcast_to([0, 0, 1.0], C.shape).copy(), 0.16, 0.3), 0)
        tx = sx * TRUSS_X
        add(box_tris((tx - 1.0, -7.5 + ysh / 2, Z0 + zsh / 2), (tx + 1.0, 7.5 + ysh / 2, ZT + zsh / 2)), 0)    # 桁架
        for yy in (-5.0, 5.0):                                                                              # 桁架 → 锚的短外梁
            x_ = tx + sx * 2.5 * s
            C = np.stack([x_, yy * (1 - s) + np.sign(yy) * 2.0 * s + ysh / 2 * (1 - u), np.full_like(s, (Z0 + ZT) / 2)], 1)[None]
            add(tube_tris(C, np.broadcast_to([0, 0, 1.0], C.shape).copy(), 0.16, 0.3), 0)
    return np.concatenate(S), np.concatenate(Mt), red


# ---------- 光栅 ----------
DQ = float(2 ** 38)


def look_at(eye, tgt, up=(0, 0, 1.0)):
    f = np.asarray(tgt, float) - eye; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    return np.stack([r, np.cross(r, f), f])


def project(P_, eye, R, fpx, W, H):
    c = (P_ - eye) @ R.T
    z = c[..., 2]
    return torch.stack([fpx * c[..., 0] / z + W / 2, -fpx * c[..., 1] / z + H / 2, z], -1)


def raster(V, H, W, near=0.5, far=3000.0, chunk=12_000_000):
    """V[T, 3, 3]（屏幕 x, y, 视深 z）→ key[H*W]（int64，高位 = 量化深度，低 24 位 = 三角形号；空 = int64 max）"""
    T = V.shape[0]
    buf = torch.full((H * W,), torch.iinfo(torch.int64).max, device=dev, dtype=torch.int64)
    x, y, z = V[..., 0], V[..., 1], V[..., 2]
    x0 = torch.ceil(x.min(1).values - 0.5).clamp(0, W - 1); x1 = torch.floor(x.max(1).values - 0.5).clamp(-1, W - 1)
    y0 = torch.ceil(y.min(1).values - 0.5).clamp(0, H - 1); y1 = torch.floor(y.max(1).values - 0.5).clamp(-1, H - 1)
    area = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (y[:, 1] - y[:, 0]) * (x[:, 2] - x[:, 0])
    ok = (z.min(1).values > near) & (x1 >= x0) & (y1 >= y0) & (area.abs() > 1e-9) & (z.min(1).values < far)
    ids = ok.nonzero()[:, 0]
    if len(ids) == 0:
        return buf
    bw = (x1 - x0 + 1)[ids].long(); bh = (y1 - y0 + 1)[ids].long(); cnt = bw * bh
    cum = torch.cumsum(cnt, 0)
    bounds = [0]
    tot = int(cum[-1]); start = 0
    while start < tot:                                         # 按候选像素数切块
        j = int(torch.searchsorted(cum, torch.tensor(start + chunk, device=dev), right=True))
        j = max(j, bounds[-1] + 1)
        bounds.append(min(j, len(ids)))
        start = int(cum[bounds[-1] - 1])
    for a, b in zip(bounds[:-1], bounds[1:]):
        tid = ids[a:b]; c = cnt[a:b]
        rep = torch.repeat_interleave(torch.arange(len(tid), device=dev), c)
        st = torch.cumsum(c, 0) - c
        loc = torch.arange(len(rep), device=dev) - st[rep]
        t = tid[rep]
        px = x0[t].long() + loc % bw[a:b][rep]; py = y0[t].long() + loc // bw[a:b][rep]
        cx = px.float() + 0.5; cy = py.float() + 0.5
        X_, Y_, Z_ = x[t], y[t], z[t]
        w0 = (X_[:, 2] - X_[:, 1]) * (cy - Y_[:, 1]) - (Y_[:, 2] - Y_[:, 1]) * (cx - X_[:, 1])
        w1 = (X_[:, 0] - X_[:, 2]) * (cy - Y_[:, 2]) - (Y_[:, 0] - Y_[:, 2]) * (cx - X_[:, 2])
        w2 = (X_[:, 1] - X_[:, 0]) * (cy - Y_[:, 0]) - (Y_[:, 1] - Y_[:, 0]) * (cx - X_[:, 0])
        ar = area[t]
        b0, b1, b2 = w0 / ar, w1 / ar, w2 / ar
        ins = (b0 >= 0) & (b1 >= 0) & (b2 >= 0)
        iz = b0 / Z_[:, 0] + b1 / Z_[:, 1] + b2 / Z_[:, 2]
        dep = 1 / iz
        q = ((dep - near) / (far - near)).clamp(0, 1) * DQ
        key = (q.long() << 24) | t
        buf.scatter_reduce_(0, (py * W + px)[ins], key[ins], "amin")
    return buf


def resolve(key, V, W, near=0.5, far=3000.0):
    """key → (有效 mask, 三角形号, 透视校正重心 [M, 3], 视深)。只对有效像素"""
    valid = key != torch.iinfo(torch.int64).max
    pix = valid.nonzero()[:, 0]
    k = key[pix]
    tid = k & 0xFFFFFF
    dep = (k >> 24).double() / DQ * (far - near) + near
    cx = (pix % W).float() + 0.5; cy = (pix // W).float() + 0.5
    X_, Y_, Z_ = V[tid, :, 0], V[tid, :, 1], V[tid, :, 2]
    ar = (X_[:, 1] - X_[:, 0]) * (Y_[:, 2] - Y_[:, 0]) - (Y_[:, 1] - Y_[:, 0]) * (X_[:, 2] - X_[:, 0])
    w0 = ((X_[:, 2] - X_[:, 1]) * (cy - Y_[:, 1]) - (Y_[:, 2] - Y_[:, 1]) * (cx - X_[:, 1])) / ar
    w1 = ((X_[:, 0] - X_[:, 2]) * (cy - Y_[:, 2]) - (Y_[:, 0] - Y_[:, 2]) * (cx - X_[:, 2])) / ar
    w2 = 1 - w0 - w1
    bp = torch.stack([w0 / Z_[:, 0], w1 / Z_[:, 1], w2 / Z_[:, 2]], 1)
    bp = bp / bp.sum(1, keepdim=True)
    return valid, pix, tid, bp, dep.float()


# ---------- 着色 ----------
class Look:
    """灯光 / 材质 / 雾的常量 + 预计算纹理"""

    def __init__(self, seed=3):
        g = torch.Generator(device=dev).manual_seed(seed)
        self.L_pos = np.array([16.0, 110.0, 90.0]); self.L_tgt = np.array([0.0, -4.0, 0.0])
        self.L_R = look_at(self.L_pos, self.L_tgt, up=(0, 1.0, 0))
        self.L_cos_in, self.L_cos_out = np.cos(np.radians(12)), np.cos(np.radians(34))
        self.L_I = 2.2
        self.SMAP, self.L_f = 2048, 0.5 * 2048 / np.tan(np.radians(36))
        # 头顶长条柔光箱（方向空间里的一条带）：中心仰角 / 方位、半高 / 半宽
        self.box_dir = np.array([0.05, 0.37, 0.93]); self.box_dir /= np.linalg.norm(self.box_dir)
        # 拉丝 + 斑驳纹理（世界 xy，±230 一张）
        n = torch.rand(1, 1, 1024, 1024, generator=g, device=dev)
        k = torch.exp(-0.5 * (torch.arange(-24, 25, device=dev).float() / 9) ** 2); k /= k.sum()
        brushed = F.conv2d(F.pad(n, (24, 24, 0, 0), mode="circular"), k.view(1, 1, 1, -1))
        mott = sum(F.interpolate(torch.rand(1, 1, s, s, generator=g, device=dev), size=1024, mode="bicubic", align_corners=False) * w
                   for s, w in ((24, 0.35), (96, 0.35), (384, 0.3)))
        self.tex = torch.cat([(brushed - brushed.mean()) / brushed.std(), (mott - mott.mean()) / mott.std()], 1)   # [1, 2, 512, 512]
        # 雾：3D fBm（64³）
        self.fog = sum(F.interpolate(torch.rand(1, 1, s, s, s, generator=g, device=dev), size=(64, 64, 64), mode="trilinear", align_corners=False) * w
                       for s, w in ((4, 0.45), (8, 0.3), (16, 0.17), (32, 0.08)))
        self.fog = (self.fog - self.fog.min()) / (self.fog.max() - self.fog.min())
        # 材质：反照率、金属度、粗糙度、镜面、自发光
        self.mat = torch.tensor([[0.46, 0.49, 0.53, 0.85, 0.30, 1.0, 0.0],      # 0 硅（亮金属）
                                 [0.07, 0.075, 0.085, 0.75, 0.32, 0.6, 0.0],     # 1 框
                                 [0.008, 0.009, 0.011, 0.2, 0.7, 0.10, 0.0],     # 2 地板
                                 [1.0, 0.08, 0.06, 0.0, 0.5, 0.0, 30.0],        # 3 红点
                                 [0.7, 0.03, 0.03, 0.0, 0.5, 0.2, 1.6]], device=dev)   # 4 红线

    def spot(self, P_):
        """P[M,3] → (到灯的单位向量 l, 聚光强度)"""
        Lp = torch.tensor(self.L_pos, device=dev, dtype=torch.float32)
        ax = torch.tensor(self.L_R[2], device=dev, dtype=torch.float32)
        d = Lp - P_; dist = d.norm(dim=-1, keepdim=True); l = d / dist
        cosang = (-l * ax).sum(-1)
        cone = ((cosang - self.L_cos_out) / (self.L_cos_in - self.L_cos_out)).clamp(0, 1)
        return l, self.L_I * cone ** 2 * (3 - 2 * cone)

    def shadow_map(self, tris):
        V = project(tris, torch.tensor(self.L_pos, device=dev, dtype=torch.float32), torch.tensor(self.L_R, device=dev, dtype=torch.float32),
                    self.L_f, self.SMAP, self.SMAP)
        key = raster(V, self.SMAP, self.SMAP)
        dep = torch.where(key == torch.iinfo(torch.int64).max, torch.tensor(1e9, device=dev, dtype=torch.float64),
                          (key >> 24).double() / DQ * (3000.0 - 0.5) + 0.5).float()
        return dep.view(1, 1, self.SMAP, self.SMAP)

    def visibility(self, smap, P_, bias=0.25, pcf=True):
        c = (P_ - torch.tensor(self.L_pos, device=dev, dtype=torch.float32)) @ torch.tensor(self.L_R.T, device=dev, dtype=torch.float32)
        u = self.L_f * c[:, 0] / c[:, 2] + self.SMAP / 2; v = -self.L_f * c[:, 1] / c[:, 2] + self.SMAP / 2
        offs = [(-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)] if pcf else [(0, 0)]
        vis = torch.zeros_like(u)
        for du, dv in offs:
            g = torch.stack([(u + 0.7 * du) / self.SMAP * 2 - 1, (v + 0.7 * dv) / self.SMAP * 2 - 1], -1).view(1, 1, -1, 2)
            d = F.grid_sample(smap, g, mode="nearest", padding_mode="border", align_corners=False).view(-1)
            vis += (c[:, 2] - bias <= d).float()
        return vis / len(offs)

    def env(self, r):
        """头顶柔光箱：r 与 box_dir 的夹角（分仰角 / 方位）落在长条里 → 亮"""
        bd = torch.tensor(self.box_dir, device=dev, dtype=torch.float32)
        el = torch.asin(r[:, 2].clamp(-1, 1)); el0 = float(np.arcsin(self.box_dir[2]))
        az = torch.atan2(r[:, 0], r[:, 1]); az0 = float(np.arctan2(self.box_dir[0], self.box_dir[1]))
        a = ((el - el0).abs() - np.radians(2.5)).clamp(min=0) / np.radians(2.0)
        b = ((az - az0).abs() - np.radians(22)).clamp(min=0) / np.radians(12)
        core = torch.exp(-a ** 2 - b ** 2)
        halo = 0.35 * torch.exp(-((el - el0) / np.radians(20)) ** 2) * (r @ bd).clamp(min=0)
        return 2.5 * core + halo

    def shade(self, P_, N_, eye, mid, smap, t):
        """P/N[M,3] 世界坐标 / 法线（已朝相机），mid[M] 材质号 → 线性 RGB[M,3]"""
        m = self.mat[mid]
        alb, metal, rough, ks, emit = m[:, :3], m[:, 3:4], m[:, 4:5], m[:, 5:6], m[:, 6:7]
        uv = torch.stack([P_[:, 0] / 230, P_[:, 1] / 230], -1)          # 世界 ±230 一张，不平铺（无接缝）
        tx = F.grid_sample(self.tex, uv.view(1, 1, -1, 2), mode="bilinear", padding_mode="reflection", align_corners=False)[0, :, 0].T
        alb = alb * (1 + 0.08 * tx[:, 1:2]).clamp(min=0.3) * (1 + 0.10 * tx[:, 0:1])
        rough = (rough * (1 + 0.25 * tx[:, 0:1] + 0.15 * tx[:, 1:2])).clamp(0.08, 0.9)
        V_ = torch.tensor(eye, device=dev, dtype=torch.float32) - P_; V_ = V_ / V_.norm(dim=-1, keepdim=True)
        l, Li = self.spot(P_)
        vis = self.visibility(smap, P_ + 0.05 * N_)
        Hh = l + V_; Hh = Hh / Hh.norm(dim=-1, keepdim=True)
        nl = (N_ * l).sum(-1, keepdim=True).clamp(min=0); nv = (N_ * V_).sum(-1, keepdim=True).clamp(min=1e-3)
        nh = (N_ * Hh).sum(-1, keepdim=True).clamp(min=0)
        a2 = (rough ** 2) ** 2
        Dg = a2 / (np.pi * (nh ** 2 * (a2 - 1) + 1) ** 2)
        f0 = 0.04 * (1 - metal) + alb * metal
        Fr = f0 + (1 - f0) * (1 - (Hh * V_).sum(-1, keepdim=True).clamp(0, 1)) ** 5
        spec = Dg * Fr / (4 * nv) * nl
        diff = alb * (1 - metal) * nl / np.pi
        direct = (diff + ks * spec.clamp(max=60)) * (Li * vis)[:, None]
        r = -V_ + 2 * (N_ * V_).sum(-1, keepdim=True) * N_
        envv = self.env(r)[:, None] * f0 * ks * (1 - 0.6 * rough)
        amb = alb * torch.tensor([0.05, 0.06, 0.08], device=dev) * (0.4 + 0.6 * N_[:, 2:3].clamp(min=0))
        return direct + envv + amb + emit * alb

    def haze(self, eye, R, fpx, W, H, depth, smap, t, hit, nstep=40, seed=0):
        """W×H 的体积雾：沿视线从 near 走到表面（depth = 视深），光锥 × 阴影 × 雾密度 × HG 相函数"""
        g = torch.Generator(device=dev).manual_seed(seed)
        yy, xx = torch.meshgrid(torch.arange(H, device=dev).float() + 0.5, torch.arange(W, device=dev).float() + 0.5, indexing="ij")
        Rt = torch.tensor(R, device=dev, dtype=torch.float32)
        Dr = torch.stack([(xx - W / 2) / fpx, -(yy - H / 2) / fpx, torch.ones_like(xx)], -1).view(-1, 3) @ Rt   # 视深 1 的射线
        ln = Dr.norm(dim=-1)
        dmax = depth.view(-1).clamp(max=900.0)
        jit = torch.rand(Dr.shape[0], generator=g, device=dev)
        E = torch.tensor(eye, device=dev, dtype=torch.float32)
        acc = torch.zeros(Dr.shape[0], device=dev); trans = torch.ones_like(acc)
        dmin = 40.0
        hg_g = 0.55
        for i in range(nstep):
            s = dmin + (dmax - dmin).clamp(min=0) * (i + jit) / nstep
            ds = (dmax - dmin).clamp(min=0) / nstep * ln
            Pw = E + Dr * s[:, None]
            l, Li = self.spot(Pw)
            vis = self.visibility(smap, Pw, bias=0.4, pcf=False)
            # 密度：薄雾 + 结构附近的雾团（随时间漂、命中时被梭子推开一点）
            push = 1.0 + 0.25 * np.exp(-max(t - hit, 0) / 0.5) * (t > hit)
            q = torch.stack([(Pw[:, 0] / (45 * push) + 0.035 * t), (Pw[:, 1] / (32 * push) - 0.02 * t), (Pw[:, 2] - 2) / 14 + 0.015 * t], -1)
            n = F.grid_sample(self.fog, q.view(1, 1, 1, -1, 3), mode="bilinear", padding_mode="reflection", align_corners=False).view(-1)
            near_s = torch.exp(-(Pw[:, 0] / 40) ** 2 - (Pw[:, 1] / 24) ** 2 - ((Pw[:, 2] - 2.0) / 3.5) ** 2)
            dens = 0.00012 + 0.4 * near_s * ((n - 0.45) / 0.3).clamp(0, 1) ** 2 + 0.004 * ((n - 0.45) / 0.3).clamp(0, 1)
            vdir = -Dr / ln[:, None]
            cth = (-l * vdir).sum(-1)                               # 光从灯 → 点 → 相机的散射角
            ph = (1 - hg_g ** 2) / (1 + hg_g ** 2 - 2 * hg_g * cth).clamp(min=1e-4) ** 1.5
            acc += trans * dens * ds * Li * vis * ph * FOG_GAIN
            trans = trans * torch.exp(-dens * ds * 0.6)
        return acc.view(H, W), trans.view(H, W)


def gblur(img, r):
    """img[C, H, W]"""
    if r <= 0.3:
        return img
    rr = int(np.ceil(3 * r))
    k = torch.exp(-0.5 * (torch.arange(-rr, rr + 1, device=dev).float() / r) ** 2); k /= k.sum()
    x = img[:, None]
    x = F.conv2d(F.pad(x, (rr, rr, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    x = F.conv2d(F.pad(x, (0, 0, rr, rr), mode="replicate"), k.view(1, 1, -1, 1))
    return x[:, 0]


# ---------- 镜头 ----------
EYE0 = np.array([0.0, -153.0, 265.0]); TGT0 = np.array([0.0, 3.0, 0.0])
PUSH_TO = np.array([4.0, 6.0, 1.0])      # 推向梳齿区（略偏右上，红点那边）


def camera(n, fps, t0, hits, travel=0.30, aim=0.5):
    """push_remap.curve 的「进/缓」→ 相机沿 EYE0→PUSH_TO 直线推 travel 比例，注视点 TGT0→PUSH_TO 走 aim 比例。返回 (eye[n,3], tgt[n,3], v[n])"""
    _, v, pos = P.curve(n, fps, t0, [(h, 1.0) for h in hits], base=0.05, gain=0.7, ramp_frames=2.5, tau=0.45)
    p = pos / pos[-1]
    eye = EYE0[None] + (PUSH_TO - EYE0)[None] * (travel * p)[:, None]
    tgt = TGT0[None] + (PUSH_TO - TGT0)[None] * (aim * p)[:, None]
    return eye, tgt, v


# ---------- 帧 ----------
W, H, SS, FPS = 1280, 720, 2, 24
HFOV = 36.0
FOG_GAIN = float(__import__('os').environ.get('MEMS_FOG', '0.25'))


def render_frame(look, static, state_fn, t_frame, eye_fn, nsub=4, shutter=0.5, hit=62.624, seed=0):
    """state_fn(t) → (ax, az, wsp, ysh, zsh)；eye_fn(t) → (eye, tgt)。返回 uint8[H, W, 3]"""
    Ws, Hs = W * SS, H * SS
    fpx = 0.5 * Ws / np.tan(np.radians(HFOV) / 2)
    st_tris, st_mat = static
    acc = torch.zeros(3, Hs, Ws, device=dev)
    depth_mid = None; smap_mid = None
    for q in range(nsub):
        tq = t_frame + ((q + 0.5) / nsub - 0.5) * shutter / FPS
        ax, az, wsp, ysh, zsh = state_fn(tq)
        dt_, dm_, _ = dynamic_scene(ax, az, wsp, ysh, zsh)
        tris = torch.tensor(np.concatenate([st_tris, dt_]), device=dev, dtype=torch.float32)
        mat = torch.tensor(np.concatenate([st_mat, dm_]), device=dev, dtype=torch.long)
        smap = look.shadow_map(tris)
        eye, tgt = eye_fn(tq)
        R = look_at(eye, tgt)
        V = project(tris, torch.tensor(eye, device=dev, dtype=torch.float32), torch.tensor(R, device=dev, dtype=torch.float32), fpx, Ws, Hs)
        key = raster(V, Hs, Ws)
        valid, pix, tid, bp, dep = resolve(key, V, Ws)
        Pw = (bp[:, :, None] * tris[tid]).sum(1)
        Nn = torch.cross(tris[tid, 1] - tris[tid, 0], tris[tid, 2] - tris[tid, 0], dim=-1)
        Nn = Nn / (Nn.norm(dim=-1, keepdim=True) + 1e-12)
        toE = torch.tensor(eye, device=dev, dtype=torch.float32) - Pw
        Nn = torch.where(((Nn * toE).sum(-1, keepdim=True) < 0), -Nn, Nn)
        col = look.shade(Pw, Nn, eye, mat[tid], smap, tq)
        img = torch.zeros(3, Hs * Ws, device=dev)
        img[:, pix] = col.T
        acc += img.view(3, Hs, Ws) / nsub
        if q == nsub // 2:
            d = torch.full((Hs * Ws,), 2000.0, device=dev); d[pix] = dep
            depth_mid = d.view(Hs, Ws); smap_mid = smap; eye_mid, R_mid = eye, R
    img = F.avg_pool2d(acc[None], SS)[0]                                       # [3, H, W] 线性
    dep = F.avg_pool2d(depth_mid[None, None], SS)[0, 0]
    # 体积雾（半分辨率）
    hw_, hh_ = W // 2, H // 2
    dh = -F.max_pool2d(-dep[None, None], 2)[0, 0]
    fog, trans = look.haze(eye_mid, R_mid, 0.5 * hw_ / np.tan(np.radians(HFOV) / 2), hw_, hh_, dh, smap_mid, t_frame, hit, seed=seed)
    fog = F.interpolate(fog[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
    trans = F.interpolate(trans[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
    fog = gblur(fog[None], 2.0)[0]
    fogc = torch.tensor([0.62, 0.76, 0.92], device=dev)[:, None, None]
    img = img * trans[None] + fog[None] * fogc
    # 景深：对焦在结构中心
    focus = float(np.linalg.norm(eye_mid - np.array([0.0, 2.0, 1.0])))
    coc = (6.0 * (1 - focus / dep.clamp(min=1)).abs()).clamp(0, 9)
    levels = [0.0, 1.2, 2.6, 4.5, 7.0, 9.5]
    stack = torch.stack([gblur(img, r) for r in levels])
    li = torch.tensor(levels, device=dev)
    idx = torch.searchsorted(li, coc.contiguous()).clamp(1, len(levels) - 1)
    lo, hi = li[idx - 1], li[idx]; fr = ((coc - lo) / (hi - lo)).clamp(0, 1)
    a = stack.gather(0, (idx - 1)[None, None].expand(1, 3, H, W))[0]; b = stack.gather(0, idx[None, None].expand(1, 3, H, W))[0]
    img = a * (1 - fr) + b * fr
    # 泛光
    br = (img - 0.9).clamp(min=0)
    img = img + 0.35 * gblur(br, 4) + 0.25 * gblur(br, 14) + 0.12 * gblur(br, 40)
    # 色调：曝光 → filmic → 冷调 → 暗角 → 颗粒
    img = img * 1.1
    img = img * (2.51 * img + 0.03) / (img * (2.43 * img + 0.59) + 0.14)
    img = img.clamp(0, 1) * torch.tensor([0.93, 0.98, 1.04], device=dev)[:, None, None] + torch.tensor([0.004, 0.006, 0.009], device=dev)[:, None, None]
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H, device=dev), torch.linspace(-1, 1, W, device=dev), indexing="ij")
    img = img * (1 - 0.45 * (xx ** 2 * 0.6 + yy ** 2 * 0.9).clamp(0, 1))[None]
    img = img.clamp(0, 1) ** (1 / 2.2)
    gen = torch.Generator(device=dev).manual_seed(1000 + seed)
    img = img + 0.012 * torch.randn(1, H, W, generator=gen, device=dev)
    return (img.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()


# ---------- 计划 ----------
def prepare(hit=62.624, pre=1.0, dur=4.5, fps=FPS, fs=2400.0):
    import soundfile as sf
    x, sr = sf.read(MV / "分轨" / "other.wav", dtype="float32"); x = x.mean(1) if x.ndim > 1 else x
    motif = json.loads((MV / "动机_抽搐.json").read_text("utf-8"))
    tpl = motif["template"][1] - motif["template"][0]
    hits = [h["t"] for h in motif["hits"]]
    t0 = hit - pre; n = int(round(dur * fps))
    t1 = t0 + (n - 1) / fps
    ts = np.arange(t0 - 1.5, t1 + 0.1, 1 / fs)
    te, e = M1.envelope(x, sr)
    ysh_raw, _ = M1.drive(te, e, hits, ts, f_vis=2.2, tau=0.8, flux_gain=0.10, template=tpl, vmax=1.0)
    ysh_raw = ysh_raw * (ts >= ts[0] + 0.3)                    # 起点前的截断瞬态不要
    tt, D = band_drive(x, sr)
    X, Q, Wsp = simulate(tt, D, ts, ysh_raw)
    m = (ts >= t0) & (ts <= t1)
    mt = (ts >= hit) & (ts <= hit + 1.04)                      # 抽搐模板窗
    xmax = np.abs(X[:, mt]).max(); wmax = np.abs(Wsp[:, :, m]).max(); ymax = np.abs(ysh_raw[m]).max()
    ft = t0 + np.arange(n) / fps
    in_clip = [h for h in hits if ft[0] <= h <= ft[-1]]
    eye, tgt, vel = camera(n, fps, t0, in_clip)
    return dict(ts=ts, X=X, Q=Q, Wsp=Wsp, ysh=ysh_raw, xmax=xmax, wmax=wmax, ymax=ymax, t0=t0, n=n, ft=ft,
                hits=in_clip, eye=eye, tgt=tgt, vel=vel, hit=hit)


def make_fns(pl):
    ts = pl["ts"]

    def state(t):
        j = int(np.clip(np.searchsorted(ts, t), 0, len(ts) - 1))
        ax = disp(pl["X"][:, j], pl["xmax"], PEAK_X)
        az = disp(pl["Q"][:, j], pl["xmax"], PEAK_Z)
        wsp = pl["Wsp"][:, :, j] / pl["wmax"] * SPINE_ZMAX
        ysh = pl["ysh"][j] / pl["ymax"] * SHUTTLE_VMAX
        return ax, az, wsp, ysh, 0.25 * ysh

    kk = np.arange(pl["n"])

    def eye_fn(t):
        k = (t - pl["t0"]) * FPS
        e = np.array([np.interp(k, kk, pl["eye"][:, i]) for i in range(3)])
        g = np.array([np.interp(k, kk, pl["tgt"][:, i]) for i in range(3)])
        return e, g
    return state, eye_fn


def contact(frames, pl, out, cols=4, rows=3):
    from PIL import Image, ImageDraw, ImageFont
    ks = np.linspace(0, pl["n"] - 1, cols * rows - 1).round().astype(int)
    tw, th = 480, 270
    sheet = Image.new("RGB", (cols * tw, rows * (th + 22)), (12, 12, 14))
    d = ImageDraw.Draw(sheet)
    fz = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 15)
    tiles = [("参考 A2_暗_v1a", Image.open(A2))] + [(f"#{k:03d}  t={pl['ft'][k]:.3f}", Image.fromarray(frames[k])) for k in ks]
    for i, (lab, im) in enumerate(tiles):
        x, y = (i % cols) * tw, (i // cols) * (th + 22)
        sheet.paste(im.convert("RGB").resize((tw, th), Image.LANCZOS), (x, y + 22))
        d.text((x + 6, y + 2), lab, font=fz, fill=(200, 200, 205))
    sheet.save(out, quality=92)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hit", type=float, default=62.624)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--work", required=True)
    ap.add_argument("--only", default="", help="只渲这些帧号 PNG 到 --work")
    ap.add_argument("--nsub", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists() and not a.only:
        sys.exit(f"已存在，不覆盖：{out}")
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    pl = prepare(hit=a.hit)
    state, eye_fn = make_fns(pl)
    look = Look()
    static = static_scene()
    # 诊断：每根齿在片段内的峰值（显示单位，齿距）
    m = (pl["ts"] >= pl["t0"]) & (pl["ts"] <= pl["ft"][-1])
    pk = np.abs(disp(pl["X"][:, m], pl["xmax"], PEAK_X)).max(1)
    top = np.argsort(pk)[::-1][:8]
    print("最狠的 8 根：", [(f"{f_band[k]:.0f}Hz", "上" if band_of(k)[0] == 0 else "下", band_of(k)[1], round(float(pk[k]), 2)) for k in top])
    print("中位峰值", round(float(np.median(pk)), 2), " 梭子 ymax", round(float(pl["ymax"]), 3))
    if a.only:
        from PIL import Image
        for k in map(int, a.only.split(",")):
            im = render_frame(look, static, state, pl["ft"][k], eye_fn, nsub=a.nsub, hit=a.hit, seed=k)
            Image.fromarray(im).save(work / f"v2_{k:03d}.png"); print("帧", k, flush=True)
        return
    import imageio_ffmpeg
    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "15",
                             "-preset", "slow", "-movflags", "+faststart", "-an", str(out)], stdin=subprocess.PIPE)
    frames = []
    for k in range(pl["n"]):
        im = render_frame(look, static, state, pl["ft"][k], eye_fn, nsub=a.nsub, hit=a.hit, seed=k)
        proc.stdin.write(im.tobytes()); frames.append(im)
        if k % 12 == 0:
            print("帧", k, "/", pl["n"], flush=True)
    proc.stdin.close(); proc.wait()
    contact(frames, pl, out.with_name(out.stem + "_接触表.jpg"))
    j_of = lambda t: int(np.clip(np.searchsorted(pl["ts"], t), 0, len(pl["ts"]) - 1))  # noqa: E731
    side = {"t0": pl["t0"], "fps": FPS, "n": pl["n"], "hit": a.hit, "hits_in_clip": pl["hits"],
            "bands_Hz": [round(float(f), 1) for f in f_band], "band_layout": "第 k 带 → 排 k%2（0=上, 1=下），齿号 k//2（左→右）",
            "visible_Hz": [round(float(f), 2) for f in FV],
            "说明": f"无 HUD。齿：显示挠度 = sign·(|x|/max)^{GAMMA}·{PEAK_X} 齿距（横向）、{PEAK_Z}（离面）；梭子峰值 {SHUTTLE_VMAX} 齿距；脊起伏峰值 {SPINE_ZMAX}。第 k 帧 = 歌曲时间 t0 + k/fps（播放时间）",
            "frames": []}
    for k in range(pl["n"]):
        j = j_of(pl["ft"][k])
        ax = disp(pl["X"][:, j], pl["xmax"], PEAK_X)
        hot = np.argsort(np.abs(ax))[::-1][:3]
        side["frames"].append({"k": k, "t": round(float(pl["ft"][k]), 4), "shuttle": round(float(pl["ysh"][j] / pl["ymax"] * SHUTTLE_VMAX), 3),
                               "hot": [[round(float(f_band[i])), round(float(ax[i]), 2)] for i in hot],
                               "eye": [round(float(v), 2) for v in pl["eye"][k]], "cam_speed": round(float(pl["vel"][k]), 4)})
    out.with_suffix(".json").write_text(json.dumps(side, ensure_ascii=False, indent=1), "utf-8")
    print("写出", out)


if __name__ == "__main__":
    main()
