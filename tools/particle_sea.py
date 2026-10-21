#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🎬样片-粒子海-残响之海（参考 `_参考库/2026-10-21_用户给的/粒子海_鲸.webp`，拆解见 `_设计/效果库.md`「粒子海」行、
`-给人类的解释/粒子海怎么做的.md`）。纯 numpy，CPU 跑（不吃 GPU）。

参考图拆出来的三件事（量过，数字在效果库）：
  ① 海 = 一张真实起伏的面上的「标记点」，每个点是一小段短划；亮度不靠打光，靠**投影密度**：
     波峰掠射处点被透视挤在一起 = 亮线，波背被波峰挡住 = 黑。→ 我们：点跟着水走（拉格朗日标记），
     面 = Tessendorf FFT 海（JONSWAP 谱 + cos^2s 方向扩展 + 深水色散 ω = √(gk) + choppy 水平位移 = Gerstner 的推广），
     点的世界密度 ∝ 1/距离²（等角采样，像从镜头路径上做的 LiDAR 扫描）→ 平水面在屏幕上接近均匀，只剩坡度和遮挡在说话。
     短划 = 曝光时间里的拖影（相机在动）+ 沿水面切向的一小段（= 水面网格的一段），所以短划方向跟着水面、长度跟着镜头速度。
  ② 形体 = 网格模型的「切片线」：模型表面 ∩ 两族等距平面（y = 常数的环、x' = 常数的竖线）= 一张贴着曲面的线框，
     亮度 = 兰伯特（顶光）× 背面剔除 × 深度缓冲遮挡；水面以下的部分看不见（按点所在处的真实水面高度判）。
     形体来自真实模型（默认 3D/N4D_蜷缩抱膝_无材质_100万面.glb），不是程序噪声（低熵注入）。
  ③ 配色：参考图是灰白 + 荧光绿的 UI 块。绿块（UI）不做 —— 那是装饰，不说真话。我们的**颜色 = 状态**（PALETTES）：
     bwr = 全冷灰白 + 一个红心；acid（👤2026-10-22 允许酸性点缀）= 海暗冷灰，正被吸起来的 / 被起音推起来的水是酸绿，
     到了她身上褪成白，心 = 酸黄。一次模拟可同时出几套配色（--palettes bwr,acid）。

动力学（全部闭式 / 可在任意 t 取样）：
  · 吸收（质量守恒）：她的每个线框点 i 都对应海里的一个点（源 = 她脚下周围半径 R 内的水）。释放前它就是海的一部分；
    释放时刻 t_i 起按**临界阻尼弹簧**飞向目标：e(u) = (e0 + (v0 + ω e0) u) e^{−ωu}，x = T + e（无过冲、闭式）。
    → 海 + 飞行中 + 已到达 = 常数（HUD 写这三个数，自检逐帧验）。被吸走的水域会变稀（看得见的代价）。
  · 释放节奏 = 感官层起音：每个起音放一批，批大小 ∝ 威胁^p（强拍一下吞一大口，弱拍零星几粒），另有一点基础细流；
    顺序 = 离水面的高度（先低后高）+ 抖动 → 她从海里往上长出来。
  · 拍点作用在有形的东西上：每个起音在她脚下打出一圈**Cauchy–Poisson 环形波**（深水点源初值问题的精确解：
    η(r,t) = ∫ k η̂0(k) J0(kr) cos(√(gk) t) dk，η̂0 = 高斯初始隆起的 Hankel 变换；短波跑在前面 = 真色散），
    振幅 ∝ 威胁。不推全局放大、不抖画面。
  · 心（全片唯一的点缀）：「下一个小节首拍还在镜头里」的起音中最强那一拍放出的点里，取落在她胸口附近的一个；
    它的 ω 单独算，让它正好在那个首拍到达（晚一拍）。
  · 镜头：不从静止开始 —— 沿路径的位移 s(t) = v_b t + v0 τ (1 − e^{−t/τ})（先快后缓、不归零），高度也随之缓降，
    看向她（带一点滞后）。

用法：
  python tools/particle_sea.py still --t 132.9 [--w 960 --h 540] --out x.jpg          # 单帧（调参用，960 宽 ≈ 1 s）
  python tools/particle_sea.py clip  --t0 128.36 --t1 134.78 --palettes bwr,acid
         --out 样片/粒子海/20_粒子海_吸收_{pal}_v1.mp4 [--w 1920 --h 1080] [--preset absorb] [--set key=value ...]
  {pal} 换成配色中文名（黑白红 / 酸绿）。每条 clip 同时出 <名>_接触表.jpg 和 <名>.json（全部参数 + 每个起音放了多少点
  + 心的到达时刻 + 逐帧 海 / 飞行 / 到达 计数）。输出不覆盖：重名就停（换 _v2 / _v3）。
  预设（PRESETS）：absorb = v1（07 吸收第一乐句）；absorb2 = v2（力更读得出）；finale = 07 最后一个乐句（横移 + 升起）。
  --set 改任意参数（数字 / 逗号分隔的向量）。1080p 约 3.8 s/帧（CPU，numpy；不占 GPU）。
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
G = 9.81
ARRIVE = 5.84            # (1+x)e^{−x} = 0.02 的根：临界阻尼无初速时「剩 2 % = 到达」

# ---------------------------------------------------------------- 参数
BASE = dict(
    # 海（Tessendorf FFT）
    Hs=10.0, Tp=12.0, gamma=3.3, wind_deg=-20.0, spread_s=6.0, L=512.0, grid=256, chop=1.2, sea_seed=7,
    # 海上的标记点：世界密度 ∝ 1/max(d, d0)²，d = 到镜头路径的距离（含镜头高度）
    N_sea=220000, d0=30.0, sea_xmax=420.0, sea_zmax=520.0, fog=260.0, dash_m=0.7, w_sea=1.1,
    # 她（模型线框）
    body=1, center="0,0,0", mesh=str(S.MV / "3D" / "N4D_蜷缩抱膝_无材质_100万面.glb"), body_h=36.0, body_yaw=-15.0, body_pitch=8.0, body_y=-9.0,
    line_n=70, line_eps=0.13, n_surf=500000, n_depth=160000, w_body=0.6, body_seed=3,
    # 吸收
    R_src=4.0, w_src=0.2, w_flight=0.35, pre_absorbed=0.35, end_frac=0.92, trickle=0.10, thr_pow=3.0, omega=7.0, v_up=26.0, order_noise=0.25,
    # 环形波（Cauchy–Poisson）
    ring_A=3.0, ring_b=6.0, ring_decay=3.0, ring_min_threat=0.08, ring_scatter=0.0, ring_kind="impulse", ring_tau=0.6,
    # 镜头
    cam_from="-80,9,-150", cam_to="-24,2.6,-70", v_frac=0.55, tau=1.2, look="-12,16,0", look_lag=0.35, hfov=52.0,
    roll_deg=-4.0, shutter=0.5, K=6, trail_s=0.25, trail_n=10, trail_gain=0.7,
    # 画
    gain=1.3, bloom=0.18, bg=0.012, hud=1,
)
PRESETS = {
    "absorb": {},      # 07 吸收：低空滑向她，起音一口口把海吸进她身体
    # v2：v1 的力读不出来（飞点只是一层薄雾、环形波几乎看不见）→ 源更远、上抛更高、拖尾更长 = 一股股从海里升起的弧；
    #     起音打在水上的冲量加大（先砸出坑再散开的环更明显）
    "absorb2": dict(R_src=6.0, v_up=38.0, omega=5.0, trail_s=0.45, trail_n=14, trail_gain=1.0, w_flight=0.35,
                    ring_A=4.0, ring_tau=1.2, ring_b=7.0),
    # 07 吸收的最后一个乐句（147.56–153.96）：她几乎是完整的了，最后几下强拍把剩下的海吞完 → 接 08「空」。
    # 镜头换一种运动：贴着海面横移（右→左）同时升起（crane up），她在画面左侧，先快后缓
    "finale": dict(pre_absorbed=0.80, end_frac=1.0, trickle=0.05, cam_from="95,3.0,-92", cam_to="38,14,-98",
                   look="14,13,0", roll_deg=3.0, v_frac=0.75, tau=0.9),
}


# 配色：颜色 = 状态（符号说真话）。sea 海里的点 / ring 被起音的环形波推起来的那部分 / fly 正在被吸过去的 / body 已经是她 / heart 那一个点
_G = S.GREY / S.GREY.max()
PAL_NAME = {"bwr": "黑白红", "acid": "酸绿"}
PALETTES = {
    # ① 黑白红：全部冷灰白，只有一个红点（她的心，晚一拍到）
    "bwr": dict(bg=(0.012, 0.012, 0.012), sea=_G, ring=_G, fly=_G, body=_G, heart=tuple(S.RED * 1.05)),
    # ② 酸：海是很暗的冷灰；残响（被吸起来的、被拍点推起来的水）是酸绿 = 外来的东西；一到她身上就褪成白 = 被吸收了。
    #    心 = 酸黄（同一色族里最亮的那一个）。参考图本身就是灰 + 荧光绿，这里绿只给「正在发生的」。
    "acid": dict(bg=(0.008, 0.010, 0.009), sea=(0.50, 0.56, 0.58), ring=(0.62, 1.0, 0.10), fly=(0.70, 1.0, 0.16),
                 body=(0.96, 0.97, 0.93), heart=(1.0, 0.95, 0.12)),
}


def parse_vec(v):
    return np.array([float(x) for x in str(v).split(",")])


# ---------------------------------------------------------------- 数学零件
def j0(x):
    """贝塞尔 J0（Abramowitz & Stegun 9.4.1 / 9.4.3，误差 < 1e-7）；没有 scipy 也能用"""
    x = np.abs(np.asarray(x, dtype=np.float64))
    out = np.empty_like(x)
    s = x <= 3.0
    y = (x[s] / 3.0) ** 2
    out[s] = (1 + y * (-2.2499997 + y * (1.2656208 + y * (-0.3163866 + y * (0.0444479 + y * (-0.0039444 + y * 0.0002100))))))
    xl = x[~s]; u = 3.0 / xl
    f0 = 0.79788456 + u * (-0.00000077 + u * (-0.00552740 + u * (-0.00009512 + u * (0.00137237 + u * (-0.00072805 + u * 0.00014476)))))
    th = xl - 0.78539816 + u * (-0.04166397 + u * (-0.00003954 + u * (0.00262573 + u * (-0.00054125 + u * (-0.00029333 + u * 0.00013558)))))
    out[~s] = f0 * np.cos(th) / np.sqrt(xl)
    return out


def jonswap(omega, Hs, Tp, gamma=3.3):
    """JONSWAP 频谱 S(ω)（m²·s），按 Hs 归一：∫S dω = Hs²/16"""
    wp = 2 * np.pi / Tp
    w = np.maximum(omega, 1e-6)
    sig = np.where(w <= wp, 0.07, 0.09)
    r = np.exp(-((w - wp) ** 2) / (2 * sig ** 2 * wp ** 2))
    Spm = w ** -5.0 * np.exp(-1.25 * (wp / w) ** 4)
    S_ = Spm * gamma ** r
    ww = np.linspace(0.05 * wp, 8 * wp, 20000)
    sg = np.where(ww <= wp, 0.07, 0.09)
    m0 = np.trapezoid(ww ** -5.0 * np.exp(-1.25 * (wp / ww) ** 4) * gamma ** np.exp(-((ww - wp) ** 2) / (2 * sg ** 2 * wp ** 2)), ww)
    return np.where(omega > 0, S_ * (Hs ** 2 / 16) / m0, 0.0)


class Ocean:
    """Tessendorf FFT 海：周期为 L 的方格，h(x,t)、choppy 位移 D(x,t)、坡度 ∂h/∂x、∂h/∂z。"""

    def __init__(self, Hs=3.0, Tp=9.0, gamma=3.3, wind_deg=-20.0, spread_s=6.0, L=512.0, n=256, chop=0.9, seed=7):
        self.L, self.n, self.chop = L, n, chop
        k1 = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
        KX, KZ = np.meshgrid(k1, k1)            # [iz, ix]
        K = np.hypot(KX, KZ)
        self.KX, self.KZ, self.K = KX, KZ, K
        self.w = np.sqrt(G * K)                 # 深水色散
        dk = 2 * np.pi / L
        # S(k,θ) = S(ω) dω/dk · D(θ) / k ；dω/dk = ½ √(g/k)
        Kp = np.maximum(K, 1e-9)
        Sw = jonswap(self.w, Hs, Tp, gamma)
        th = np.arctan2(KZ, KX) - np.deg2rad(wind_deg + 90.0)   # 风向：0° = +z（朝前），角度向 +x 偏
        th = (th + np.pi) % (2 * np.pi) - np.pi
        Dth = np.maximum(np.cos(th / 2), 0) ** (2 * spread_s)   # cos^2s(θ/2)：主要顺风、少量逆风
        tt = np.linspace(-np.pi, np.pi, 4001)
        Dth /= np.trapezoid(np.cos(tt / 2) ** (2 * spread_s), tt)
        Sk = Sw * 0.5 * np.sqrt(G / Kp) * Dth / Kp
        Sk[K == 0] = 0
        rng = np.random.default_rng(seed)
        amp = np.sqrt(Sk * dk * dk / 2)         # 每个模的 rms 振幅；/2：k 和 −k 两项各分一半（否则 4σ = √2·Hs，自检抓到的）
        self.h0 = amp * (rng.normal(size=K.shape) + 1j * rng.normal(size=K.shape)) / np.sqrt(2)
        mk = (-np.arange(n)) % n
        self.h0m = np.conj(self.h0[np.ix_(mk, mk)])               # conj(h0(−k))
        self.Hs, self.Tp = Hs, Tp

    def spectrum_t(self, t):
        e = np.exp(1j * self.w * t)
        return self.h0 * e + self.h0m * np.conj(e)               # 厄米 ⇒ 实场

    def fields(self, t):
        """→ dict(h, dx, dz, hx, hz)，全是 [iz, ix] 的实场；格点 x = ix·L/n"""
        Hk = self.spectrum_t(t)
        n2 = self.n * self.n
        Kp = np.where(self.K > 0, self.K, 1.0)
        f = lambda a: np.real(np.fft.ifft2(a)) * n2               # noqa: E731（离散逆变换 = 模的和）
        return dict(h=f(Hk), dx=f(-1j * self.KX / Kp * Hk) * self.chop, dz=f(-1j * self.KZ / Kp * Hk) * self.chop,
                    hx=f(1j * self.KX * Hk), hz=f(1j * self.KZ * Hk))


def sample_periodic(F, L, x, z):
    """周期场双线性取样：F[iz, ix]，格距 L/n"""
    n = F.shape[0]
    u = (x / L * n) % n; v = (z / L * n) % n
    i0 = np.floor(u).astype(np.int64); j0_ = np.floor(v).astype(np.int64)
    fu = u - i0; fv = v - j0_
    i1 = (i0 + 1) % n; j1 = (j0_ + 1) % n
    return (F[j0_, i0] * (1 - fu) * (1 - fv) + F[j0_, i1] * fu * (1 - fv) + F[j1, i0] * (1 - fu) * fv + F[j1, i1] * fu * fv)


def cauchy_poisson(r, t, A, b, nk=384, kind="lift", tau_i=0.0):
    """深水点源初值问题（Cauchy–Poisson），线性、无粘：
    kind = "lift"：t=0 时水面隆起 A·exp(−r²/b²)、静止放开 → η(r,t) = ∫ k η̂(k) J0(kr) cos(√(gk) t) dk，
                   η̂(k) = A b²/2 · exp(−k²b²/4)（高斯的 Hankel 变换）。
    kind = "impulse"：t=0 时水面平、被一个短促的压力冲量往下打（尺度 tau_i 秒）→ 解 = tau_i · ∂η_lift/∂t
                   = −tau_i ∫ k η̂(k) ω sin(ωt) J0(kr) dk：先砸出一个坑，再一圈圈散开（鼓点 = 一下打在水上）。
    t < 0 返回 0（因果）。"""
    r = np.asarray(r, dtype=np.float64)
    if t < 0:
        return np.zeros_like(r)
    kmax = 7.0 / b
    k = np.linspace(0, kmax, nk)
    w = np.sqrt(G * k)
    base = A * b * b / 2 * np.exp(-(k * b) ** 2 / 4) * k
    hat = base * np.cos(w * t) if kind == "lift" else -tau_i * base * w * np.sin(w * t)
    J = j0(np.outer(r.ravel(), k))
    return (np.trapezoid(J * hat[None], k, axis=1)).reshape(r.shape)


def damped_spring(e0, v0, omega, u):
    """临界阻尼：e(u) = (e0 + (v0 + ω e0) u) e^{−ωu}（e = 位置 − 目标）；u < 0 时 = e0（还没放）"""
    u = np.asarray(u, dtype=np.float64)
    uu = np.maximum(u, 0)[..., None] if np.ndim(e0) > np.ndim(u) else np.maximum(u, 0)
    return (e0 + (v0 + omega * e0) * uu) * np.exp(-omega * uu)


def smooth01(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


# ---------------------------------------------------------------- 她：模型 → 线框点
def rot_y(deg):
    a = np.deg2rad(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_x(deg):
    a = np.deg2rad(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def body_points(p):
    """模型表面 ∩ 两族等距平面 = 贴着曲面的线框。返回 (lines_xyz, lines_n, depth_xyz, info)；世界坐标（米）"""
    import trimesh
    m = trimesh.load(p["mesh"], force="mesh")
    rng = np.random.default_rng(int(p["body_seed"]))
    P, fi = trimesh.sample.sample_surface(m, int(p["n_surf"]), seed=int(p["body_seed"]))
    Nn = m.face_normals[fi]
    lo, hi = m.bounds
    sc = p["body_h"] / (hi[1] - lo[1])
    c = (lo + hi) / 2
    P = (P - c) * sc
    s = p["body_h"] / p["line_n"]                           # 线距（米）
    eps = p["line_eps"]
    fy = (P[:, 1] / s) % 1.0
    xr = P[:, 0] * 0.8 + P[:, 2] * 0.6                       # 竖线那族平面斜 37°（不和视线对齐）
    fx = (xr / s) % 1.0
    on = (np.minimum(fy, 1 - fy) < eps / 2) | (np.minimum(fx, 1 - fx) < eps / 2)
    R = rot_y(p["body_yaw"]) @ rot_x(p["body_pitch"])
    off = np.array([0.0, p["body_y"] + p["body_h"] / 2, 0.0])
    L_ = P[on] @ R.T + off
    Ln = Nn[on] @ R.T
    di = rng.choice(len(P), size=min(int(p["n_depth"]), len(P)), replace=False)
    D_ = P[di] @ R.T + off
    return L_, Ln, D_, dict(scale_m_per_unit=float(sc), line_spacing_m=float(s), n_lines=int(on.sum()))


# ---------------------------------------------------------------- 镜头
class CamPath:
    """沿 from→to 的直线；路程 s(t) = v_b t + v0 τ (1 − e^{−t/τ})，总长 = |to − from|（t = T 时到 to）。"""

    def __init__(self, p, T):
        self.a, self.b = parse_vec(p["cam_from"]), parse_vec(p["cam_to"])
        self.T, self.tau, self.vf = T, p["tau"], p["v_frac"]
        self.look = parse_vec(p["look"]); self.lag = p["look_lag"]
        self.hfov, self.roll = p["hfov"], p["roll_deg"]
        dist = np.linalg.norm(self.b - self.a)
        e = 1 - np.exp(-T / self.tau)
        # 让 v0 τ e = vf·dist，v_b T = (1 − vf)·dist
        self.v0 = self.vf * dist / (self.tau * e); self.vb = (1 - self.vf) * dist / T
        self.dist = dist

    def s(self, t):
        return self.vb * t + self.v0 * self.tau * (1 - np.exp(-t / self.tau))

    def speed(self, t):
        return self.vb + self.v0 * np.exp(-t / self.tau)

    def pose(self, t):
        pos = self.a + (self.b - self.a) * (self.s(t) / self.dist)
        # 看向点带一点滞后：看的是「它 lag 秒前该看的方向」与目标的混合 → 进来时视线略落后于运动
        tgt = self.look + np.array([0.0, 0.0, 0.0])
        lagpos = self.a + (self.b - self.a) * (self.s(max(t - self.lag, -self.lag)) / self.dist)
        fwd = tgt - pos
        fwd_l = tgt - lagpos
        f = 0.75 * fwd / np.linalg.norm(fwd) + 0.25 * fwd_l / np.linalg.norm(fwd_l)
        f /= np.linalg.norm(f)
        r = np.cross(np.array([0.0, 1.0, 0.0]), f); r /= np.linalg.norm(r)
        u = np.cross(f, r)
        if self.roll:
            a = np.deg2rad(self.roll); r, u = r * np.cos(a) + u * np.sin(a), -r * np.sin(a) + u * np.cos(a)
        return pos, r, u, f


def project(P, pose, W, H, hfov):
    pos, r, u, f = pose
    d = P - pos
    zc = d @ f
    fpx = (W / 2) / np.tan(np.deg2rad(hfov) / 2)
    zz = np.maximum(zc, 1e-3)
    X = W / 2 + fpx * (d @ r) / zz
    Y = H / 2 - fpx * (d @ u) / zz
    return X, Y, zc


def splat_bilinear(x, y, w, W, H):
    ok = (x >= 0) & (x < W - 1) & (y >= 0) & (y < H - 1)
    x, y, w = x[ok], y[ok], w[ok]
    ix = x.astype(np.int64); iy = y.astype(np.int64)
    fx = x - ix; fy = y - iy
    i0 = iy * W + ix
    idx = np.concatenate([i0, i0 + 1, i0 + W, i0 + W + 1])
    ww = np.concatenate([w * (1 - fx) * (1 - fy), w * fx * (1 - fy), w * (1 - fx) * fy, w * fx * fy])
    return np.bincount(idx, ww, minlength=W * H).reshape(H, W), float(w.sum())


def zbuffer(X, Y, Z, W, H, cell):
    """最近深度缓冲（cell 像素一格）：按深度从远到近写，后写的覆盖 = 最近的留下"""
    gw, gh = W // cell + 1, H // cell + 1
    ok = (X >= 0) & (X < W) & (Y >= 0) & (Y < H) & (Z > 0)
    gx = (X[ok] / cell).astype(np.int64); gy = (Y[ok] / cell).astype(np.int64); z = Z[ok]
    o = np.argsort(-z)
    buf = np.full(gh * gw, np.inf)
    buf[(gy * gw + gx)[o]] = z[o]
    return buf.reshape(gh, gw)


def min_filter(a, r=1):
    p = np.pad(a, r, mode="edge")
    out = a.copy()
    for i in range(-r, r + 1):
        for j in range(-r, r + 1):
            out = np.minimum(out, p[r + i:r + i + a.shape[0], r + j:r + j + a.shape[1]])
    return out


def box_blur(img, r):
    if r <= 0:
        return img
    out = img
    for ax in (0, 1):
        c = np.cumsum(np.pad(out, [(r + 1, r) if a == ax else (0, 0) for a in range(out.ndim)], mode="edge"), axis=ax)
        sl_hi = [slice(None)] * out.ndim; sl_lo = [slice(None)] * out.ndim
        sl_hi[ax] = slice(2 * r + 1, None); sl_lo[ax] = slice(0, -(2 * r + 1))
        out = (c[tuple(sl_hi)] - c[tuple(sl_lo)]) / (2 * r + 1)
    return out


# ---------------------------------------------------------------- 场景
class Sea:
    def __init__(self, p, t0, t1, sense, W=1920, H=1080, log=print):
        self.p, self.t0, self.t1, self.W, self.H = p, t0, t1, W, H
        T = t1 - t0
        t_w = time.time()
        self.ocean = Ocean(p["Hs"], p["Tp"], p["gamma"], p["wind_deg"], p["spread_s"], p["L"], int(p["grid"]), p["chop"], int(p["sea_seed"]))
        self.cam = CamPath(p, T)
        rng = np.random.default_rng(11)
        # --- 她
        if p["body"]:
            self.BL, self.BN, self.BD, self.binfo = body_points(p)
        else:                                                    # 只有海（「四周全是别的声音」那种镜头）
            z3 = np.zeros((0, 3))
            self.BL, self.BN, self.BD, self.binfo = z3, z3, z3, dict(scale_m_per_unit=0.0, line_spacing_m=0.0, n_lines=0)
        nb = len(self.BL)
        # --- 海上的标记点：密度 ∝ 1/max(d, d0)²（d = 到镜头路径地面投影的距离），在 [−xmax, xmax] × [zmin, zmax] 里拒绝采样
        a, b = self.cam.a, self.cam.b
        zmin = min(a[2], b[2]) - 30
        n = int(p["N_sea"]); xs = []; zs = []
        while sum(len(v) for v in xs) < n:
            m = n * 3
            x = rng.uniform(-p["sea_xmax"], p["sea_xmax"], m); z = rng.uniform(zmin, p["sea_zmax"], m)
            d = np.hypot(self._dist_to_path(x, z), 0.5 * (a[1] + b[1]))   # 到镜头的大致 3D 距离（含镜头高度）
            acc = rng.random(m) < (p["d0"] / np.maximum(d, p["d0"])) ** 2
            xs.append(x[acc]); zs.append(z[acc])
        self.sx = np.concatenate(xs)[:n]; self.sz = np.concatenate(zs)[:n]
        # --- 被她吸走的那些水：源 = 线框点的地面投影 + 径向外推（她脚下周围 R 内）
        foot = self.BL[:, [0, 2]]
        cen = foot.mean(0) if nb else parse_vec(p["center"])[[0, 2]]
        rad = np.percentile(np.linalg.norm(foot - cen, axis=1), 90) if nb else 10.0
        ang = rng.uniform(0, 2 * np.pi, nb)
        rr = rad * (0.3 + p["R_src"] * np.sqrt(-np.log(rng.uniform(1e-3, 1.0, nb))) / 1.6)   # 高斯径向分布 → 软边（不是一块白盘）
        self.ax = cen[0] + 0.35 * (foot[:, 0] - cen[0]) + rr * np.cos(ang)
        self.az = cen[1] + 0.35 * (foot[:, 1] - cen[1]) + rr * np.sin(ang)
        self.center = np.array([cen[0], 0.0, cen[1]]); self.src_rad = float(rad * p["R_src"])
        # --- 释放日程（感官层起音）
        self.sched = self.schedule(sense, nb, rng)
        self.N_total = n + nb
        self.font = ImageFont.truetype(S.FONT, max(12, int(15 * H / 1080)))
        log(f"  场景：海 {n:,} 点 + 她 {nb:,} 点（线距 {self.binfo['line_spacing_m']:.2f} m）  {time.time() - t_w:.1f}s")

    def _dist_to_path(self, x, z):
        a, b = self.cam.a[[0, 2]], self.cam.b[[0, 2]]
        ab = b - a; L2 = ab @ ab
        q = np.stack([x - a[0], z - a[1]], 1)
        s = np.clip(q @ ab / L2, 0, 1)
        dd = q - s[:, None] * ab
        return np.hypot(dd[:, 0], dd[:, 1])

    def schedule(self, sense, nb, rng):
        """每个线框点的释放时刻（歌的播放时间）；之前已吸收的 pre_absorbed 比例记为 −inf"""
        p = self.p
        ons = sense["events"]["onsets"]; thr = np.asarray(sense["channels"]["threat"])
        evs = []
        for o in ons:
            if self.t0 <= o < self.t1 - 0.5:
                i = int(round(o * FPS))
                s = float(thr[max(0, i - 3):i + 4].max())               # 起音强度取 ±3 帧最大（方法 §一）
                evs.append((o, s))
        # 高度排序（先低后高）+ 抖动
        hgt = self.BL[:, 1]
        key = (hgt - hgt.min(initial=0)) / (np.ptp(hgt) + 1e-9 if nb else 1.0) + p["order_noise"] * rng.standard_normal(nb)
        order = np.argsort(key)
        n_pre = int(p["pre_absorbed"] * nb); n_end = int(p["end_frac"] * nb)
        n_ev = n_end - n_pre
        wts = np.array([s ** p["thr_pow"] for _, s in evs]) if evs else np.zeros(0)
        n_trk = int(p["trickle"] * n_ev) if len(evs) else n_ev
        n_burst = n_ev - n_trk
        counts = np.floor(n_burst * wts / max(wts.sum(), 1e-9)).astype(int) if len(evs) else np.zeros(0, int)
        n_trk += n_burst - counts.sum()
        rt = np.full(nb, np.inf)
        rt[order[:n_pre]] = -np.inf
        # 基础细流（均匀）与各起音的批次交错：按时间把「名额」排成一条队列，再按高度顺序发
        times = list(np.sort(rng.uniform(self.t0, self.t1 - 0.6, n_trk)))
        for (o, s), c in zip(evs, counts):
            times += list(o + np.abs(rng.normal(0, 0.03, c)))          # 一批在 ~30 ms 内放完
        times = np.sort(np.asarray(times))
        rt[order[n_pre:n_pre + len(times)]] = times
        # 红点：最强一拍那批里，取最靠近「胸口」（躯干上 60 %、朝镜头一侧）的一个；到达时刻对齐其后第一个小节首拍
        red = -1; red_hit = None; red_arrive = None
        dbs_all = sense["events"]["downbeats"]

        def next_db(o):
            d = [x for x in dbs_all if x > o + 0.15]
            return d[0] if d else o + 0.4
        # 只在「它的下一个小节首拍还在镜头里」的起音里挑最强的（否则红点到不了）；同强取早的
        cand = [e for e in evs if next_db(e[0]) <= self.t1 - 0.1]
        if cand:
            o_max, s_max = max(cand, key=lambda e: (e[1], -e[0]))
            batch = np.nonzero(np.abs(rt - o_max) < 0.2)[0]
            if len(batch):
                cy = hgt.min() + 0.62 * np.ptp(hgt)
                toward = -(self.cam.b - self.center); toward[1] = 0; toward /= np.linalg.norm(toward) + 1e-9
                score = np.abs(self.BL[batch, 1] - cy) / np.ptp(hgt) - 0.3 * (self.BL[batch] - self.center) @ (-toward) / np.ptp(hgt) \
                    + 0.5 * np.abs((self.BL[batch, 0] - self.center[0])) / np.ptp(hgt)
                red = int(batch[np.argmin(score)])
                red_hit = o_max; red_arrive = next_db(o_max)
                rt[red] = o_max
        self.red, self.red_hit, self.red_arrive = red, red_hit, red_arrive
        self.om = np.full(nb, float(p["omega"]))
        if red >= 0:
            self.om[red] = ARRIVE / max(red_arrive - red_hit, 0.1)   # 红点的 ω：在 red_arrive 时剩余距离 2 %（晚一拍到达）
        self.events = evs; self.counts = counts
        self.rt = rt
        return dict(events=[(round(o, 3), round(s, 3), int(c)) for (o, s), c in zip(evs, counts)],
                    n_pre=n_pre, n_end=n_end, n_trickle=int(n_trk), red_hit=red_hit, red_arrive=red_arrive)

    # ---------- 水面高度 = FFT 海 + 所有已触发的环形波
    def ring_centers(self):
        """每个起音的环形波中心：默认都在她脚下；ring_scatter > 0 时散在半径内（「四周全是别的声音」：每个残响冒出来的地方）"""
        if not hasattr(self, "_rc"):
            r = np.random.default_rng(5)
            sc = self.p["ring_scatter"]
            self._rc = []
            for _ in self.events:
                if sc > 0:
                    a = r.uniform(0, 2 * np.pi); d = sc * np.sqrt(r.uniform(0.05, 1))
                    self._rc.append((self.center[0] + d * np.cos(a), self.center[2] + d * np.sin(a)))
                else:
                    self._rc.append((self.center[0], self.center[2]))
        return self._rc

    def ring_height(self, x, z, t):
        out = np.zeros_like(x)
        p = self.p
        if getattr(self, "_ring_t", None) != t:                 # 同一帧里多次调用 → 剖面只算一次
            self._ring_t, self._ring_prof = t, []
            R = 2.2 * np.hypot(p["sea_xmax"], p["sea_zmax"])
            grid = np.linspace(0, R, 3072)
            for (o, s), c in zip(self.events, self.ring_centers()):
                if s < p["ring_min_threat"] or t < o:
                    continue
                u = t - o
                A = p["ring_A"] * s * np.exp(-u / p["ring_decay"])
                self._ring_prof.append((c, grid, cauchy_poisson(grid, u, A, p["ring_b"], kind=p["ring_kind"], tau_i=p["ring_tau"])))
        for c, grid, prof in self._ring_prof:
            out += np.interp(np.hypot(x - c[0], z - c[1]), grid, prof)
        return out

    def surface(self, x, z, F, t, with_ring=True):
        """标记点（静止位置 x, z）在时刻 t 的世界位置 + 切向（沿 x 的一小段）"""
        L = self.ocean.L
        dx = sample_periodic(F["dx"], L, x, z); dz = sample_periodic(F["dz"], L, x, z)
        h = sample_periodic(F["h"], L, x, z); hx = sample_periodic(F["hx"], L, x, z)
        if with_ring:
            h = h + self.ring_height(x, z, t)
        P = np.stack([x + dx, h, z + dz], 1)
        T = np.stack([np.ones_like(hx), hx, np.zeros_like(hx)], 1)
        T /= np.linalg.norm(T, axis=1, keepdims=True)
        return P, T

    def counts_at(self, t):
        """(海里, 飞行中, 已到达)；到达 = 剩余距离 < 2 % 初距（无初速时闭式 (1+ωu)e^{−ωu} < 0.02 ⇔ ωu > ARRIVE = 5.84）"""
        u = t - self.rt
        sea = int((u < 0).sum())
        arrived = int((u * self.om > ARRIVE).sum())
        return self.N_total - len(self.rt) + sea, len(self.rt) - sea - arrived, arrived

    # ---------- 一帧
    def frame(self, t):
        p, W, H = self.p, self.W, self.H
        tl = t - self.t0
        F = self.ocean.fields(t)
        # 海（未被吸走的那部分「吸收源」也在海里）
        P_sea, T_sea = self.surface(self.sx, self.sz, F, t)
        u_rel = t - self.rt
        in_sea = u_rel < 0
        P_src, T_src = self.surface(self.ax, self.az, F, t)
        # 飞行中 / 到达：闭式临界阻尼；e0 = 释放时刻的水面位置 − 目标（用释放时刻的场取样）
        fly = ~in_sea                                           # 含 pre_absorbed（rt = −inf → 早已到达）
        idx = np.nonzero(fly)[0]
        P_fly = np.zeros((0, 3)); w_fly = np.zeros(0)
        if len(idx):
            # 释放时刻水面位置：按释放时刻分组取样代价大 → 用当前场近似水平位置、高度用释放时刻的 FFT 场值
            # （水平 choppy 位移 < 3 m，相对 30+ m 的飞行可忽略；自检里 e(0) = 源 精确成立）
            e0 = P_src[idx] - self.BL[idx]
            omega = self.om[idx][:, None]
            v0 = np.zeros_like(e0); v0[:, 1] = p["v_up"]
            uu = np.clip(u_rel[idx], 0, 1e3)[:, None]
            e = (e0 + (v0 + omega * e0) * uu) * np.exp(-omega * uu)
            P_fly = self.BL[idx] + e

            def fly_at(dt):                                       # 快门内的位置（飞行中的点自己拖出弧线）
                u2 = np.clip(u_rel[idx] + dt, 0, 1e3)[:, None]
                return self.BL[idx] + (e0 + (v0 + omega * e0) * u2) * np.exp(-omega * u2)
            prog = 1 - np.linalg.norm(e, axis=1) / (np.linalg.norm(e0, axis=1) + 1e-9)
            w_fly = p["w_src"] + (p["w_flight"] - p["w_src"]) * smooth01(prog / 0.3)   # 离开水面后是暗的飞点；到达（下面 shade）才亮 → 身体周围不起雾
        src_vis = in_sea                                        # 还没放的（含本片段里轮不到的 rt = +inf）留在海里
        P_s = np.concatenate([P_sea, P_src[src_vis]]); T_s = np.concatenate([T_sea, T_src[src_vis]])
        m_s = np.concatenate([np.ones(len(P_sea)), np.full(int(src_vis.sum()), p["w_src"])])
        # 她的线框点：已经到达的（u ≥ 0）就是 P_fly 里那些；pre_absorbed 的 rt = −inf → u = inf → e = 0
        # 法向（兰伯特 + 背面剔除）
        nrm = self.BN[idx]
        # ---- 镜头：快门内 K 个子样本（相机在动 → 拖影），每个子样本同时沿水面切向偏一小段（短划）
        K = int(p["K"])
        acc = np.zeros((H, W))
        sh = p["shutter"] / FPS
        pose_c = self.cam.pose(tl)
        # 深度缓冲：她（实心采样，在水面以上的部分）+ 海自己
        Xd, Yd, Zd = project(self.BD, pose_c, W, H, self.cam.hfov)
        hb = self.surface_height_at(self.BD[:, 0], self.BD[:, 2], F, t)
        above = self.BD[:, 1] > hb
        cell = max(2, int(round(4 * W / 1920)))
        zb_body = min_filter(zbuffer(Xd[above], Yd[above], Zd[above], W, H, cell), 1)
        Xs, Ys, Zs = project(P_s, pose_c, W, H, self.cam.hfov)
        cs = max(3, int(round(6 * W / 1920)))
        zb_sea = min_filter(zbuffer(Xs, Ys, Zs, W, H, cs), 1)

        def vis_by(zb, X, Y, Z, c, tol):
            gx = np.clip((X / c).astype(np.int64), 0, zb.shape[1] - 1); gy = np.clip((Y / c).astype(np.int64), 0, zb.shape[0] - 1)
            return Z <= zb[gy, gx] * (1 + tol) + 0.3

        fog = lambda Z: np.exp(-np.maximum(Z, 0) / p["fog"])   # noqa: E731
        # 海点可见：没被她挡、没被前面的浪挡
        vs = vis_by(zb_body, Xs, Ys, Zs, cell, 0.0) & vis_by(zb_sea, Xs, Ys, Zs, cs, 0.035)
        w_s = p["w_sea"] * m_s * fog(Zs) * vs
        # 她：背面剔除 + 被浪挡（水面以下）+ 自遮挡
        pos = pose_c[0]
        if len(idx):
            Xb, Yb, Zb = project(P_fly, pose_c, W, H, self.cam.hfov)
            arrived = (u_rel[idx] * self.om[idx] > ARRIVE) | ~np.isfinite(u_rel[idx])
            view = pos - P_fly; view /= np.linalg.norm(view, axis=1, keepdims=True)
            facing = (nrm * view).sum(1)
            light = np.array([0.25, 0.9, -0.35]); light /= np.linalg.norm(light)
            lam = np.clip(nrm @ light, 0, 1)
            rim = (1 - np.clip(facing, 0, 1)) ** 3
            hw = self.surface_height_at(P_fly[:, 0], P_fly[:, 2], F, t)
            under = P_fly[:, 1] < hw
            vis_b = np.where(arrived, (facing > -0.05) & vis_by(zb_body, Xb, Yb, Zb, cell, 0.02), True) & ~under
            shade = np.where(arrived, 0.18 + 0.72 * lam + 0.5 * rim, 1.0)
            w_b = p["w_body"] * shade * np.where(arrived, 1.0, w_fly) * fog(Zb) * vis_b
        # 分层累积（颜色 = 状态，见 PALETTES）：sea 海 / ring 被环形波推起来的那部分海 / body 已到达 / fly 飞行中 + 拖尾
        L = {k: np.zeros((H, W)) for k in ("sea", "ring", "body", "fly")}
        # 环形波占比：这个海点此刻的高度里有多少来自起音（|η_ring| / 0.3 A），颜色就有多少「声音」
        eta_r = np.abs(self.ring_height(np.concatenate([self.sx, self.ax[src_vis]]), np.concatenate([self.sz, self.az[src_vis]]), t))
        rfrac = np.clip(eta_r / (0.3 * p["ring_A"] + 1e-9), 0, 1)
        if len(idx):
            if self.red >= 0:
                w_b = np.where(idx == self.red, 0, w_b)
            # 她身上的点被前面的浪挡住也不画（浪峰比她脚高的时候）
            w_b = w_b * vis_by(zb_sea, Xb, Yb, Zb, cs, 0.02)
            w_arr = w_b * arrived; w_fl = w_b * ~arrived
        # 切向短划长度（世界 m）；拖影来自 K 个子样本的相机位姿
        for j in range(K):
            sj = (j + 0.5) / K - 0.5
            pose_j = self.cam.pose(tl + sj * sh)
            X, Y, _ = project(P_s + T_s * (sj * p["dash_m"]), pose_j, W, H, self.cam.hfov)
            L["sea"] += splat_bilinear(X, Y, w_s * (1 - rfrac) / K, W, H)[0]
            L["ring"] += splat_bilinear(X, Y, w_s * rfrac / K, W, H)[0]
            if len(idx):
                X, Y, _ = project(fly_at(sj * sh), pose_j, W, H, self.cam.hfov)
                L["body"] += splat_bilinear(X, Y, w_arr / K, W, H)[0]
                L["fly"] += splat_bilinear(X, Y, w_fl / K, W, H)[0]
        # 飞行中的点：拖一条「走过的路」（真实的过去位置，越早越淡）→ 吸引力的方向和弧线读得出来
        if len(idx) and p["trail_s"] > 0:
            sub = np.nonzero(w_fl > 0)[0]
            if len(sub):
                M = int(p["trail_n"])
                for m in range(1, M + 1):
                    dt = -p["trail_s"] * m / M
                    X, Y, _ = project(fly_at(dt)[sub], pose_c, W, H, self.cam.hfov)
                    ok_m = (u_rel[idx][sub] + dt) > 0                     # 释放之前没有路
                    L["fly"] += splat_bilinear(X, Y, w_fl[sub] * ok_m * p["trail_gain"] * (1 - m / (M + 1)) / M, W, H)[0]
        # 红点（全片唯一的点缀）：位置
        red_info = None
        if self.red >= 0 and len(idx):
            k = np.nonzero(idx == self.red)[0]
            if len(k) and not under[k[0]] and (not arrived[k[0]] or vis_b[k[0]]):   # 被挡住 / 在水下就不画
                xr, yr, _ = project(P_fly[k], pose_c, W, H, self.cam.hfov)
                if -10 < xr[0] < W + 10 and -10 < yr[0] < H + 10:
                    red_info = (float(xr[0]), float(yr[0]))
        cnt = self.counts_at(t)
        return L, dict(counts=cnt, cam_speed=float(self.cam.speed(tl)), red=red_info)

    def compose(self, L, info, palette="bwr"):
        """分层 → RGB。亮度 = 1 − exp(−g·Σ层)（胶片式饱和）；颜色 = 各层颜色按该像素里各层的份额混合。"""
        p, W, H = self.p, self.W, self.H
        pal = PALETTES[palette]
        # 亮度按 960 宽校准、与分辨率无关：像素面积 ∝ 1/W² → 密度 ×(W/960)²；1920 时点画成 3×3（参考图的短划约 2–3 px 粗）
        r = int(W >= 1440)
        Lb = {k: box_blur(v, r) * (W / 960.0) ** 2 for k, v in L.items()}
        tot = sum(Lb.values()) + 1e-12
        lum = 1 - np.exp(-p["gain"] * 1.6 * tot)
        col = sum(Lb[k][..., None] * np.asarray(pal[k])[None, None] for k in Lb) / tot[..., None]
        img = lum[..., None] * col
        if p["bloom"] > 0:
            br = max(2, int(6 * W / 1920))
            img = img + p["bloom"] * box_blur(box_blur(img, br), br)
        rgb = np.asarray(pal["bg"])[None, None] + img
        if info["red"] is not None:
            x0, y0 = info["red"]
            rr_ = max(2.0, 3.2 * W / 1920)
            ys = slice(max(0, int(y0 - 12)), min(H, int(y0 + 13))); xs = slice(max(0, int(x0 - 12)), min(W, int(x0 + 13)))
            yy, xx = np.mgrid[ys, xs]
            g = np.exp(-(((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * rr_ ** 2)))
            rgb[ys, xs] = rgb[ys, xs] * (1 - g[..., None]) + g[..., None] * np.asarray(pal["heart"])[None, None]
        return np.clip(rgb, 0, 1)

    def surface_height_at(self, x, z, F, t):
        return sample_periodic(F["h"], self.ocean.L, x, z) + self.ring_height(x, z, t)

    def hud(self, im, t, info):
        if not self.p["hud"]:
            return
        d = ImageDraw.Draw(im)
        sea, fly, body = info["counts"]
        nb = len(self.rt)
        lines = [f"t {t:7.2f}s   JONSWAP Hs {self.ocean.Hs:.1f} m  Tp {self.ocean.Tp:.1f} s   omega = sqrt(g k)   cam {info['cam_speed']:5.1f} m/s",
                 f"SEA {sea:,}   IN FLIGHT {fly:,}   ASSIMILATED {body:,} / {nb:,}   SUM {sea + fly + body:,}"]
        fs = self.font.size
        for j, line in enumerate(lines):
            d.text((int(18 * self.W / 1920), self.H - int(14 * self.H / 1080) - (fs + 3) * (len(lines) - j)), line, font=self.font, fill=(118, 126, 132))


# ---------------------------------------------------------------- 出片
def load_sense():
    return json.loads((S.MV / "感官.json").read_text("utf-8"))


def no_overwrite(*paths):
    hit = [str(p) for p in paths if Path(p).exists()]
    if hit:
        raise SystemExit(f"已存在，不覆盖（换 _v2 / _v3）：{hit}")


def contact_sheet(thumbs, path, cols=4, tw=480):
    th = int(tw * thumbs[0][0].size[1] / thumbs[0][0].size[0])
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * th), (10, 10, 12))
    f = ImageFont.truetype(S.FONT, 13); d = ImageDraw.Draw(sheet)
    for i, (im, lab) in enumerate(thumbs):
        x, y = (i % cols) * tw, (i // cols) * th
        sheet.paste(im.resize((tw, th), Image.LANCZOS), (x, y))
        d.text((x + 6, y + 4), lab, font=f, fill=(220, 220, 220))
    sheet.save(path, quality=90)


def build_params(a):
    p = dict(BASE); p.update(PRESETS.get(a.preset, {}))
    for kv in a.set or []:
        k, v = kv.split("=", 1)
        if k not in p:
            raise SystemExit(f"没有这个参数：{k}")
        p[k] = v if isinstance(p[k], str) else type(p[k])(float(v)) if not isinstance(p[k], int) or "." in v else int(v)
    return p


def to_img(rgb):
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["still", "clip"])
    ap.add_argument("--t", type=float, default=132.9, help="still：歌的播放时间")
    ap.add_argument("--t0", type=float, default=128.36); ap.add_argument("--t1", type=float, default=134.78)
    ap.add_argument("--w", type=int, default=1920); ap.add_argument("--h", type=int, default=1080)
    ap.add_argument("--preset", default="absorb"); ap.add_argument("--set", nargs="*")
    ap.add_argument("--palettes", default="bwr", help="逗号分隔（bwr,acid）：一次模拟出几套配色。--out 里写 {pal} 会换成配色中文名；没写且多于一套时文件名加 _<配色>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--thumbs", type=int, default=12)
    a = ap.parse_args(argv)
    p = build_params(a)
    pals = a.palettes.split(",")
    for q in pals:
        if q not in PALETTES:
            raise SystemExit(f"没有这套配色：{q}（有 {list(PALETTES)}）")
    out0 = Path(a.out); out0.parent.mkdir(parents=True, exist_ok=True)
    if "{pal}" in out0.name:
        outs = {q: out0.with_name(out0.name.replace("{pal}", PAL_NAME[q])) for q in pals}
    else:
        outs = {q: (out0 if len(pals) == 1 else out0.with_name(f"{out0.stem}_{q}{out0.suffix}")) for q in pals}
    sense = load_sense()
    if a.mode == "still":
        no_overwrite(*outs.values())
        sea = Sea(p, a.t0, a.t1, sense, a.w, a.h)
        t_w = time.time()
        L, info = sea.frame(a.t)
        for q, o in outs.items():
            im = to_img(sea.compose(L, info, q)); sea.hud(im, a.t, info); im.save(o, quality=92)
            print(f"-> {o}  {time.time() - t_w:.1f}s  {info}")
        return
    sides = {q: o.with_name(o.stem + ".json") for q, o in outs.items()}
    sheets = {q: o.with_name(o.stem + "_接触表.jpg") for q, o in outs.items()}
    no_overwrite(*outs.values(), *sides.values(), *sheets.values())
    sea = Sea(p, a.t0, a.t1, sense, a.w, a.h)
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    n = int(round((a.t1 - a.t0) * FPS))
    procs = {q: subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{a.w}x{a.h}", "-r", str(FPS), "-i", "-",
                                  "-ss", f"{a.t0:.4f}", "-t", f"{n / FPS:.4f}", "-i", str(S.MUSIC),
                                  "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16",
                                  "-c:a", "aac", "-b:a", "256k", "-t", f"{n / FPS:.4f}", str(o)], stdin=subprocess.PIPE)
             for q, o in outs.items()}
    # ⚠️不用 -shortest（AAC 末帧对齐会切掉最后一帧，见 particle_shatter）
    pick = set(np.linspace(0, n - 1, min(a.thumbs, n)).round().astype(int).tolist())
    thumbs = {q: [] for q in pals}; log = []; t_w = time.time()
    for i in range(n):
        t = a.t0 + i / FPS
        L, info = sea.frame(t)
        s_, f_, b_ = info["counts"]
        for q in pals:
            im = to_img(sea.compose(L, info, q)); sea.hud(im, t, info)
            procs[q].stdin.write(im.tobytes())
            if i in pick:
                thumbs[q].append((im.copy(), f"{t:.2f}s  cam {info['cam_speed']:.0f} m/s  body {b_:,}  fly {f_:,}"))
        log.append(dict(t=round(t, 4), sea=s_, flight=f_, body=b_, cam_speed=round(info["cam_speed"], 3),
                        heart=None if info["red"] is None else [round(v, 1) for v in info["red"]]))
        if i % 24 == 0:
            print(f"  {out0.name} frame {i}/{n}  {time.time() - t_w:.0f}s", flush=True)
    for q in pals:
        procs[q].stdin.close(); procs[q].wait()
        contact_sheet(thumbs[q], sheets[q])
    for side in sides.values():
        side.write_text(json.dumps(dict(tool="tools/particle_sea.py", t0=a.t0, t1=a.t1, size=[a.w, a.h], fps=FPS, preset=a.preset,
                                        palettes={q: {k: [round(float(c), 3) for c in v] for k, v in PALETTES[q].items()} for q in pals},
                                        outputs={q: o.name for q, o in outs.items()}, params=p,
                                        body=sea.binfo, schedule=sea.sched, N_total=sea.N_total, per_frame=log,
                                        render_s=round(time.time() - t_w, 1)), ensure_ascii=False, indent=1), "utf-8")
    print("->", list(outs.values()), f"{time.time() - t_w:.0f}s")


if __name__ == "__main__":
    main()
