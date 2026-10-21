#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🔮透镜重影-拆解-参考图 `_参考库/瓦解4.jpg`（隔着微透镜 / 压花玻璃看的人像：有纹理、离散的重影，没有运动模糊）拆成 5 级，
每级一个函数、参数显式。⭐👤：这个效果比之前的 glitch 更适合片中所有「电磁干扰」时刻 → 做成可复用效果，感官驱动。
图像约定：float32 HxWx3 ∈ [0,1]；坐标 = 像素下标（x 向右、y 向下）。纯 numpy（系统 python 也能跑自检）。

⭐当前 = v3（👤2026-10-21 反馈 v1「现在就是方块」：按列分段常数 → 12–14 px 硬边竖条，抛物面透镜在格边跳变）。
物理模型：每个像素看到的是源图上 p = x + s(λ)·D(x) 处的颜色（采样点偏移 = 折射），D = 微透镜项 + 平滑的竖向重影项；全程 2× 超采样。
  1 cap_field / micro_offset / specular   微透镜阵列（周期 period_x × period_y，可错开 stagger）：每格一个 C¹ 圆顶
        h = cos²(πu/2)·cos²(πv/2)（格边上高度和斜率都 = 0 → 没有接缝），D_micro = micro·(P_x/2·sin(πu)cos²(πv/2), P_y/2·cos²(πu/2)sin(πv))；
        micro < 0 → 每格一块倒立的小像。另加每格一个小高光点 exp(−|(u,v) − (−.35,−.4)|²/size²)·(1 − sparkle·ξ(格))，乘局部亮度。
  2 smooth_field   竖向重影（处处光滑、沿 x 没有接缝）：dy = A·N₁(x,t) − m(x)·(y − y_c(x))，m = stretch·(½+½N₂)，y_c = H/2 + yc_amp·H·N₃；
        第一项 = 平滑剪切，第二项 = 绕 y_c 的竖向放大 1/(1−m)（宽的软光带，眉眼被竖着拉长）。n_ghost = 2 套独立场取平均 = 两个离散的像。
        可选 smooth_columns（scatter > 0）：每列一个随机值再沿 x 高斯 σ = seam 抹平（v1 的按列做法的平滑版）。
  3 disperse_sample  色散：n_lambda 个波长 λ ∈ [0,1]（0 = 红端，1 = 蓝端），s(λ) = 1 + dk·(λ − ½) → k_R < k_G < k_B；
        每个波长按 s(λ)·D 采样一次，再用高斯光谱响应（μ_R,G,B = 0.1/0.5/0.9，σ = 0.17，按通道归一化，白进白出）合成 RGB。
        平坦区域各波长采到同一种颜色 → 不出彩；只有明暗边界上各波长采到不同东西 → 彩虹边只在边界（自检：白图 → 白）。
  4 defocus / soften / bloom   轻微失焦（源图高斯 σ_defocus，在透镜之前）+ 透镜之后一点点柔化（soften）+ 高光辉光。
  5 grade            exposure（进透镜前，暗的源拉高）+ 低调：压黑位、gamma、暗角；配色 PALETTES（cold 冷蓝 = 进透镜前先转冷蓝单色，
        彩色只来自色散，dk ×1.6；ref 接近参考图 = 源色 + dk ×2.2 + 饱和度 ×2）。
  v1 的函数（lens_offset 抛物面、column_displacement 按列硬边、displacements_v1）留着给自检 / 复现 24、24b、24c、样片 18_v1。

参考图实测（measure_period，高通后 FFT 边缘谱）：竖向周期 9.1–9.3 px（脸部 / 全图 / 4 块不同区域一致），横向 ≈ 14.5–16 px
  （只在脸部区域稳定）；斜向峰 (fx, fy) ≈ (0.066, 0.109) = (1/15, 1/9.2) → 和**不错位**的矩形网格一致（错位网格的第一个斜向峰
  应在 fy = 1/18.4）。目视：参考图里的弧一列列竖着对齐。→ 默认 stagger = 0，period 14 × 9（720p 帧上，参考图宽 736，脸大小相近）。
  👤原估「hex-staggered、6–8 px」→ 改为「矩形、14 × 9」；stagger 仍留作参数。
  另一点修正：👤原拆解第 2 条是「每格一个 dy」。按格（含行）独立随机 → 马赛克碎块；v1 改成「按列」→ 硬边竖条（👤：方块）；
  v2 列间抹平但幅度大、corr 120 → 像波浪帘子，脸没了（静帧 24*_v2）；v3 = 平滑剪切 + 竖向放大，corr 200（静帧 24*_v3）。

感官驱动（params_from_sense，HUD / sidecar 写同一张表；长度按画高 H / 720 缩放）：
  threat  θ → 平滑剪切 A = (4 + 20·θ)·H/720 px，竖向放大 m = 0.15 + 0.55·θ
  tension τ → 色散 dk = 0.04 + 0.12·τ（蓝端比红端多折射 4–16 %）
  hunger  η → 失焦 σ_defocus = (0.5 + 2.5·η)·H/720 px
时间演化（evolve_params）：玻璃横向以 drift_x px/s 滑动（微透镜和重影场都跟着玻璃走，不闪）；N 的相位按 ω 慢变；
  强起音（该处威胁 ≥ burst_thr，默认 0.5；v1 的 0.3 在 45–50 s 里 9 个起音全触发，太密）→ 脉冲 e(t) = Σ exp(−(t − o)/burst_tau)：
  A ×(1 + burst_gain·e)、m ×(1 + ½·burst_gain·e)（≤ 0.9）、dk ×(1 + 0.5·e)，且换种子（重影场重新生成）。
用于：片中所有「电磁干扰」时刻（👤：比面部瓦解 glitch 更好）。
人读版：-给人类的解释/透镜重影怎么做的.md；效果库行：_设计/效果库.md「透镜重影」。

用法（仓库 .venv 或系统 python；sheet / clip 有 CUDA 时采样走 torch GPU，模糊仍是 numpy FFT）：
  python tools/lens_ghost.py sheet [--t 4.0] [--ver v3]   # 静帧/24_分解 + 24b 冷蓝 + 24c 接近参考 + 24d 3× 放大对比（参考 / 本版 / v1）
  python tools/lens_ghost.py clip [--src 视频] [--start 45.12] [--dur 0] [--palette cold] [--name 18_透镜重影_全息女性_v2] [--hud]
         [--exposure 1.6 --gamma 1.4]（全息女性源偏暗时）[--burst-thr 0.5]
         # → 样片/<name>.mp4（配乐）+ .json（逐帧参数）+ _接触表.jpg
  python tools/lens_ghost.py image 图 [--palette cold] [--out 路径]
  python tools/lens_ghost.py measure 图          # 打印纹理周期（FFT）
"""
import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
from PIL import Image

FPS = 24
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
REF = Path(__file__).resolve().parents[1].parents[1] / "Projekt_Visual" / "_参考库" / "瓦解4.jpg"


def lum(img):
    return img @ LUMA


# ---------------------------------------------------------------- 工具：双线性采样 / 高斯模糊
USE_TORCH = False     # 命令行（sheet / clip）在有 CUDA 时打开；自检始终走 numpy（逐像素相等的检验不受 GPU 浮点影响）


def enable_torch():
    """有 torch + CUDA 就用 GPU 做采样和模糊（同一套数学，结果差 ~1e-5）"""
    global USE_TORCH, _torch
    try:
        import torch
        USE_TORCH = torch.cuda.is_available()
        _torch = torch
    except Exception:  # noqa: BLE001
        USE_TORCH = False
    return USE_TORCH


def _bilinear_torch(img, px, py):
    t = _torch
    H, W, C = img.shape
    it = t.from_numpy(np.ascontiguousarray(img, np.float32)).cuda().permute(2, 0, 1)[None]
    gx = t.from_numpy(np.ascontiguousarray(np.broadcast_to(px, np.broadcast(px, py).shape), np.float32)).cuda()
    gy = t.from_numpy(np.ascontiguousarray(np.broadcast_to(py, np.broadcast(px, py).shape), np.float32)).cuda()
    grid = t.stack([2 * gx / (W - 1) - 1, 2 * gy / (H - 1) - 1], -1)[None]
    out = t.nn.functional.grid_sample(it, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return out[0].permute(1, 2, 0).cpu().numpy()


def bilinear(img, px, py):
    """img (H,W,C) 在浮点坐标 (px, py)（各 HxW）处双线性采样；出界 = 0（黑底）"""
    if USE_TORCH:
        return _bilinear_torch(img, px, py)
    H, W, C = img.shape
    x0 = np.floor(px).astype(np.int64)
    y0 = np.floor(py).astype(np.int64)
    fx = (px - x0).astype(np.float32)[..., None]
    fy = (py - y0).astype(np.float32)[..., None]
    out = np.zeros(px.shape + (C,), np.float32)
    for dy, wy in ((0, 1 - fy), (1, fy)):
        for dx, wx in ((0, 1 - fx), (1, fx)):
            xx, yy = x0 + dx, y0 + dy
            ok = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
            v = img[np.clip(yy, 0, H - 1), np.clip(xx, 0, W - 1)] * ok[..., None]
            out += v * (wx * wy)
    return out


def gaussian_blur(img, sigma):
    """FFT 高斯模糊（反射填充 3σ）。sigma ≤ 0 → 原样返回"""
    if sigma <= 0:
        return img
    p = int(np.ceil(3 * sigma)) + 1
    squeeze = img.ndim == 2
    x = img[..., None] if squeeze else img
    xp = np.pad(x, ((p, p), (p, p), (0, 0)), mode="reflect")
    H, W = xp.shape[:2]
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.rfftfreq(W)[None, :]
    G = np.exp(-2 * (np.pi * sigma) ** 2 * (fx ** 2 + fy ** 2)).astype(np.float32)
    F = np.fft.rfft2(xp, axes=(0, 1)) * G[..., None]
    out = np.fft.irfft2(F, s=(H, W), axes=(0, 1))[p:-p, p:-p].astype(np.float32)
    return out[..., 0] if squeeze else out


# ---------------------------------------------------------------- 1 微透镜阵列
def lens_cells(H, W, period_x=14.0, period_y=9.0, stagger=0.0, origin=(0.0, 0.0)):
    """网格：第 j 行横向错开 (j mod 2)·stagger·period_x。返回 dict：
    i, j（格下标，HxW int）、cx, cy（格中心）、u, v（格内归一化坐标 ∈ [−1,1)）。origin = 网格原点（玻璃漂移用）"""
    ox, oy = origin
    y = np.arange(H, dtype=np.float64)[:, None] - oy
    x = np.arange(W, dtype=np.float64)[None, :] - ox
    j = np.floor(y / period_y).astype(np.int64) + np.zeros((1, W), np.int64)
    off = np.mod(j, 2) * stagger * period_x
    i = np.floor((x - off) / period_x).astype(np.int64)
    cx = (i + 0.5) * period_x + off + ox
    cy = (j + 0.5) * period_y + oy
    X = np.arange(W)[None, :] + 0.0 * j
    Y = np.arange(H)[:, None] + 0.0 * i
    return {"i": i, "j": j, "cx": cx, "cy": cy, "u": (X - cx) / (period_x / 2), "v": (Y - cy) / (period_y / 2)}


def height_field(cells):
    """可视化用的归一化高度 h = 1 − u² − v²（每格一个圆顶）；数学上用的是 h = −½Σk(x − c)²（只差常数和尺度）"""
    return np.clip(1 - cells["u"] ** 2 - cells["v"] ** 2, 0, 1)


def lens_offset(cells, k_x=0.4, k_y=0.4, bow=0.0, period_y=9.0):
    """透镜局部项 D_lens = −∇h + 弯曲 = (k_x(x − c_x), k_y(y − c_y) + bow·period_y·(u² − 1/3))。返回 (dx, dy)"""
    H, W = cells["i"].shape
    X = np.arange(W, dtype=np.float64)[None, :]
    Y = np.arange(H, dtype=np.float64)[:, None]
    dx = k_x * (X - cells["cx"])
    dy = k_y * (Y - cells["cy"]) + bow * period_y * (cells["u"] ** 2 - 1.0 / 3)
    return dx, dy


# ---------------------------------------------------------------- 2 按列的各向异性大位移
def hash01(i, j, seed):
    """每格一个确定的伪随机数 ∈ [0,1)（整数哈希，不依赖 rng 状态 → 同一格同一种子永远一样）"""
    h = (np.asarray(i).astype(np.uint64) * np.uint64(73856093)) ^ (np.asarray(j).astype(np.uint64) * np.uint64(19349663)) \
        ^ np.uint64((seed * 83492791) & 0xFFFFFFFF)
    h = (h ^ (h >> np.uint64(13))) * np.uint64(0x5bd1e995)
    h = h ^ (h >> np.uint64(15))
    return (h & np.uint64(0xFFFFFF)).astype(np.float64) / float(0x1000000)


def fourier_noise(x, t=0.0, corr=40.0, omega=0.6, n=24, seed=0):
    """一维平滑噪声：n 个随机余弦之和，单位方差。波数 ~ N(0, 1/corr)·2π，每个波的相位以 ω·(0.5–1.5) rad/s 漂移"""
    r = np.random.default_rng(seed)
    kx = r.normal(0, 1 / corr, n) * 2 * np.pi
    ph = r.uniform(0, 2 * np.pi, n)
    w = omega * r.uniform(0.5, 1.5, n)
    out = np.zeros(np.shape(x))
    for a in range(n):
        out += np.cos(kx[a] * x + ph[a] + w[a] * t)
    return out * np.sqrt(2.0 / n)


def column_displacement(cells, A=60.0, scatter=0.5, aniso=0.03, t=0.0, corr=40.0, omega=0.6, seed=0,
                        profile="noise", saw_period=120.0, stagger=0.0):
    """一套位移：dy = A·[(1−scatter)·N(c_x, t) + scatter·ξ(列)]，dx = aniso·A·[另一套]。
    「列」= 格的列号 i（stagger ≠ 0 时奇偶行各算一列，键 = (i, j mod 2)）。在格中心求值 → 格内常数、整列相同。
    profile="saw"：N 换成周期柱面 saw = 2·frac(c_x / saw_period) − 1（参考图那条斜的橙带就像这种）"""
    cx, i = cells["cx"], cells["i"]
    j = np.mod(cells["j"], 2) if stagger else np.zeros_like(i)
    if profile == "saw":
        base = 2 * np.mod(cx / saw_period + 0.37 * seed, 1.0) - 1
        base_x = 2 * np.mod(cx / saw_period + 0.61 * seed + 0.5, 1.0) - 1
    else:
        base = fourier_noise(cx, t, corr, omega, seed=seed)
        base_x = fourier_noise(cx, t, corr, omega, seed=seed + 101)
    dy = A * ((1 - scatter) * base + scatter * (2 * hash01(i, j, seed) - 1))
    dx = aniso * A * ((1 - scatter) * base_x + scatter * (2 * hash01(i, j, seed + 7919) - 1))
    return dx, dy


# ---------------------------------------------------------------- 3 色散
MU_RGB = (0.1, 0.5, 0.9)


def spectral_weights(n_lambda=7, sigma=0.17):
    """(n_lambda, 3)：第 i 个波长对 R/G/B 的权重，每个通道按列归一化（白进白出）"""
    lam = np.linspace(0, 1, n_lambda) if n_lambda > 1 else np.array([0.5])
    w = np.exp(-(lam[:, None] - np.array(MU_RGB)[None, :]) ** 2 / (2 * sigma ** 2))
    return lam, (w / w.sum(0, keepdims=True)).astype(np.float32)


def disperse_sample(img, dx, dy, dk=0.05, n_lambda=7, sigma=0.17):
    """out_c(x) = Σ_i w_c(λ_i)·img_c(x + s(λ_i)·D)，s(λ) = 1 + dk·(λ − ½)。dk = 0 → 普通重采样"""
    H, W, _ = img.shape
    X = np.arange(W, dtype=np.float64)[None, :]
    Y = np.arange(H, dtype=np.float64)[:, None]
    if dk == 0:
        return bilinear(img, X + dx, Y + dy)
    lam, w = spectral_weights(n_lambda, sigma)
    out = np.zeros_like(img)
    for li, wl in zip(lam, w):
        s = 1 + dk * (li - 0.5)
        out += bilinear(img, X + s * dx, Y + s * dy) * wl[None, None, :]
    return out


def channel_scale(dk, n_lambda=7, sigma=0.17):
    """每个通道的等效折射倍数 k_c = Σ_i w_c(λ_i)·s(λ_i)（顺序检验用：k_R < k_G < k_B）"""
    lam, w = spectral_weights(n_lambda, sigma)
    s = 1 + dk * (lam - 0.5)
    return (w * s[:, None]).sum(0)


# ---------------------------------------------------------------- 4 失焦 + 辉光
def bloom(img, thr=0.6, sigma=12.0, gain=0.6):
    if gain <= 0:
        return img
    L = lum(img)
    hi = img * (np.clip((L - thr) / max(1e-6, 1 - thr), 0, 1))[..., None]
    return img + gain * gaussian_blur(hi, sigma)


# ---------------------------------------------------------------- 5 配色 / 低调
def pre_cold(img):
    """进透镜前：转成冷蓝单色（彩色只来自色散）"""
    L = lum(img)[..., None]
    return np.clip(L * np.array([0.70, 0.86, 1.10], np.float32), 0, 1)


def post_cold(x):
    return x + np.array([0.0, 0.004, 0.018], np.float32) * (1 - np.clip(lum(x), 0, 1))[..., None]


def saturate(x, s):
    L = lum(x)[..., None]
    return L + (x - L) * s


def pre_ref(img):
    return img


def post_ref(x):
    """接近参考图：饱和度拉高 + 高光偏蓝白"""
    x = saturate(x, 2.0)
    L = np.clip(lum(x), 0, 1)[..., None]
    return np.clip(x * (1 + np.array([-0.08, 0.0, 0.12], np.float32) * L), 0, None)


PALETTES = {"cold": ("冷蓝", pre_cold, post_cold, 1.6), "ref": ("接近参考", pre_ref, post_ref, 2.2)}  # 最后一个 = dk 倍数


def grade(x, black=0.08, gamma=1.7, vignette=0.35):
    H, W, _ = x.shape
    x = np.clip((x - black) / (1 - black), 0, None) ** gamma
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot((xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2))
    return np.clip(x * (1 - vignette * np.clip(r - 0.4, 0, 1) ** 2)[..., None], 0, 1)


# ---------------------------------------------------------------- v2（👤2026-10-21 反馈「现在就是方块」）
# v1 的问题：竖向大位移按列分段常数 → 12–14 px 宽的硬边竖条；微透镜是抛物面 k(x−c)，在格边上跳变 → 方块。
# v2：① 微透镜 = 每格一个 C¹ 圆顶（cos² × cos²，格边上高度和斜率都 = 0 → 没有接缝），折射 + 每格一个小高光点；
#     ② 大位移 = 每列一个值，再沿 x 用高斯（seam px）抹平 → 一阶导连续，没有硬边，仍保留「两个离散的像」；
#     ③ 2× 超采样后 2×2 平均下采样（抗锯齿）。
def cap_field(H, W, period_x=14.0, period_y=9.0, stagger=0.0, origin=(0.0, 0.0)):
    """C¹ 圆顶高度场 h = cos²(πu/2)·cos²(πv/2)（每格一个，峰值 1，格边 h = ∂h = 0）。
    返回 cells, (gx, gy) = (∂h/∂u, ∂h/∂v)·(−1) 的「向外」分量：gx = sin(πu)·cos²(πv/2)，gy = cos²(πu/2)·sin(πv)，各 ∈ [−1,1]"""
    c = lens_cells(H, W, period_x, period_y, stagger, origin)
    u, v = c["u"], c["v"]
    cu2, cv2 = np.cos(np.pi * u / 2) ** 2, np.cos(np.pi * v / 2) ** 2
    c["h"] = cu2 * cv2
    return c, (np.sin(np.pi * u) * cv2, cu2 * np.sin(np.pi * v))


def micro_offset(cells, g, micro=-0.6, period_x=14.0, period_y=9.0):
    """微透镜折射偏移 D = micro·(period_x/2·gx, period_y/2·gy)（= −k∇h，比例并进 micro）。
    micro < 0 → 每格看到一块倒立的小像（球面透镜的实像）；micro > 0 → 正立缩小；0 → 恒等。格边上 D = 0（C¹）"""
    return micro * period_x / 2 * g[0], micro * period_y / 2 * g[1]


def specular(cells, pos=(-0.35, -0.4), size=0.22, sparkle=0.7, seed=0):
    """每格一个小高光点：exp(−((u−u0)² + (v−v0)²)/size²)（光从左上来），亮度按格乘 1 − sparkle·ξ(格)
    （玻璃每个小凸面不完全一样 → 闪烁不规则）。乘局部亮度后加到画面上"""
    g = 1 - sparkle * hash01(cells["i"], cells["j"], seed + 31)
    return (g * np.exp(-((cells["u"] - pos[0]) ** 2 + (cells["v"] - pos[1]) ** 2) / size ** 2)).astype(np.float32)


def blur1d(a, sigma):
    """沿最后一维的高斯模糊（反射边界）"""
    if sigma <= 0:
        return a
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    ap = np.pad(a, [(0, 0)] * (a.ndim - 1) + [(r, r)], mode="reflect")
    return np.apply_along_axis(lambda row: np.convolve(row, k, mode="valid"), -1, ap)


def smooth_columns(W, A=70.0, col_w=14.0, seam=4.0, scatter=0.3, aniso=0.03, t=0.0, corr=35.0, omega=0.6, seed=0,
                   origin_x=0.0):
    """一套大位移（只随 x 变）：每列（宽 col_w，跟着玻璃 origin_x 走）一个值 A[(1−s)N(c_x,t) + s·ξ(列)]，
    再沿 x 高斯 σ = seam 抹平 → 连续、一阶导连续。返回 (dx, dy)，形状 (1, W)，可广播"""
    x = np.arange(W, dtype=np.float64) - origin_x
    i = np.floor(x / col_w).astype(np.int64)
    cx = (i + 0.5) * col_w + origin_x
    z = np.zeros_like(i)
    dy = A * ((1 - scatter) * fourier_noise(cx, t, corr, omega, seed=seed) + scatter * (2 * hash01(i, z, seed) - 1))
    dx = aniso * A * ((1 - scatter) * fourier_noise(cx, t, corr, omega, seed=seed + 101)
                      + scatter * (2 * hash01(i, z, seed + 7919) - 1))
    return blur1d(dx[None, :], seam), blur1d(dy[None, :], seam)


def smooth_field(H, W, A=20.0, stretch=0.4, corr=120.0, t=0.0, omega=0.6, seed=0, origin_x=0.0, yc_amp=0.25):
    """一套竖向重影的平滑位移（v2 主体，处处光滑）：
    dy(x, y) = A·N₁(x, t) − m(x)·(y − y_c(x))，m(x) = stretch·(½ + ½N₂(x))，y_c(x) = H/2 + yc_amp·H·N₃(x)。
    第一项 = 整列平滑上下错（剪切），第二项 = 绕 y_c 的竖向放大 1/(1−m)（宽的、软的竖向光带 —— 参考图里眉眼被竖着拉长）。
    N 都是 x 上相关长 corr 的平滑噪声（随玻璃 origin_x 走）→ 沿 x 没有接缝。返回 (dx, dy)"""
    x = np.arange(W, dtype=np.float64) - origin_x
    y = np.arange(H, dtype=np.float64)[:, None]
    sh = A * fourier_noise(x, t, corr, omega, seed=seed)[None, :]
    m = stretch * (0.5 + 0.5 * fourier_noise(x, t, corr, omega, seed=seed + 5))[None, :]
    yc = H / 2 + yc_amp * H * fourier_noise(x, t, 2 * corr, omega, seed=seed + 9)[None, :]
    return np.zeros((1, W)), sh - m * (y - yc)


@dataclass
class Params:
    # 微透镜（细纹理）
    period_x: float = 14.0      # 720p 画面上（参考图实测 15×9.2 @ 736 宽、脸约 500 px；我们脸约 560 px → 按脸缩放 ≈ 14–17 × 9–10）
    period_y: float = 9.0
    stagger: float = 0.0        # 参考图 FFT：不错位的矩形网格
    micro: float = -0.6         # 折射强度（格半宽的倍数）；负 = 倒立小像
    spec: float = 0.5           # 高光点强度（× 局部亮度）
    spec_size: float = 0.18
    # 大位移（竖向重影）：平滑场 smooth_field + 可选的按列随机项 smooth_columns（scatter > 0 才加）
    A: float = 16.0             # 平滑剪切幅度 px
    stretch: float = 0.45       # 竖向放大强度 m（0 = 不拉伸）
    yc_amp: float = 0.12        # 放大中心 y_c 上下游走的幅度（× 画高）
    corr: float = 200.0         # 横向相关长 px（宽的软光带；120 时像波浪帘子，脸没了 → 24*_v2）
    col_w: float = 14.0         # 列宽 px（只在 scatter > 0 时用）
    seam: float = 4.0           # 列与列之间抹平的 σ px（v1 = 0 → 硬边）
    scatter: float = 0.0
    aniso: float = 0.03
    omega: float = 0.6
    n_ghost: int = 2
    # 色散 / 光学 / 调色
    dk: float = 0.1
    n_lambda: int = 7
    defocus: float = 1.5
    soften: float = 0.4
    bloom_thr: float = 0.5
    bloom_sigma: float = 14.0
    bloom_gain: float = 0.65
    exposure: float = 1.0       # 进透镜前乘的曝光（暗的源可以拉高）
    black: float = 0.08
    gamma: float = 1.7
    vignette: float = 0.35
    strength: float = 1.0
    ss: int = 2                 # 超采样倍数
    seed: int = 3
    t: float = 0.0
    origin_x: float = 0.0
    sense: dict = field(default_factory=dict)

    def hud(self, dk_mul=1.0):
        return [f"micro-lens C1 caps {self.period_x:.0f}x{self.period_y:.0f}px  refraction {self.micro:+.2f}  spec {self.spec:.2f}  ss {self.ss}x",
                f"vertical ghosts {self.n_ghost}  shear A={self.A:.0f}px  stretch m={self.stretch:.2f}  corr {self.corr:.0f}px  column scatter {self.scatter:.2f}",
                f"dispersion dk={self.dk * dk_mul:.3f}{f' (={self.dk:.3f}x{dk_mul})' if dk_mul != 1 else ''} n_lambda={self.n_lambda}  "
                f"defocus {self.defocus:.1f}px  bloom {self.bloom_gain:.2f}  exp {self.exposure:.2f} gamma {self.gamma:.2f}"]


LENGTHS = ("period_x", "period_y", "A", "col_w", "seam", "corr", "defocus", "soften", "bloom_sigma", "origin_x")


def scaled(p: Params, s):
    """超采样：所有长度 × s"""
    q = Params(**asdict(p))
    for k in LENGTHS:
        setattr(q, k, getattr(p, k) * s)
    return q


def params_from_sense(threat, tension, hunger, H=720, base=None, **kw):
    """感官 → 参数（映射表见模块 docstring）"""
    p = Params(**{**(asdict(base) if base else {}), **kw})
    s = H / 720
    p.A = (4 + 20 * threat) * s
    p.stretch = 0.15 + 0.55 * threat
    p.dk = 0.04 + 0.12 * tension
    p.defocus = (0.5 + 2.5 * hunger) * s
    p.sense = {"threat": round(float(threat), 3), "tension": round(float(tension), 3), "hunger": round(float(hunger), 3)}
    return p


MAPPING = {"threat": "shear A = (4 + 20*threat)*H/720 px; vertical stretch m = 0.15 + 0.55*threat (smooth across x)",
           "tension": "dk = 0.04 + 0.12*tension (blue end refracts dk more than red end)",
           "hunger": "defocus sigma = (0.5 + 2.5*hunger)*H/720 px (before the lens)",
           "onsets": "burst e(t) = sum exp(-(t-o)/tau) over onsets with threat >= thr; A *= 1+gain*e, dk *= 1+0.5*e, reseed xi"}


def displacements(H, W, p: Params):
    """→ cells（含 h）, (mx, my) 微透镜偏移, [(dx_g, dy_g)] 每套大位移（(1,W)）"""
    cells, g = cap_field(H, W, p.period_x, p.period_y, p.stagger, (p.origin_x, 0.0))
    micro = micro_offset(cells, g, p.micro, p.period_x, p.period_y)
    cols = []
    for gi in range(max(1, p.n_ghost)):
        dx, dy = smooth_field(H, W, p.A, p.stretch, p.corr, p.t, p.omega, p.seed + 17 * gi, p.origin_x, p.yc_amp)
        if p.scatter > 0:
            cx, cy = smooth_columns(W, p.A, p.col_w, p.seam, 1.0, p.aniso, p.t, p.corr, p.omega, p.seed + 17 * gi, p.origin_x)
            dx, dy = dx + p.scatter * cx, dy + p.scatter * cy
        cols.append((dx, dy))
    return cells, micro, cols


def lens_stage(src, lens, cols, p: Params, dk):
    """微透镜 + 大位移 + 色散：n_ghost 套位移各采一次，取平均"""
    lx, ly = lens
    acc = 0
    for cx, cy in cols:
        acc = acc + disperse_sample(src, p.strength * (lx + cx), p.strength * (ly + cy), dk, p.n_lambda)
    return acc / len(cols)


def upsample(img, s):
    if s == 1:
        return img
    H, W, _ = img.shape
    X = (np.arange(W * s) + 0.5) / s - 0.5
    Y = (np.arange(H * s) + 0.5) / s - 0.5
    return bilinear(img, np.clip(X[None, :], 0, W - 1) + 0 * Y[:, None], np.clip(Y[:, None], 0, H - 1) + 0 * X[None, :])


def downsample(img, s):
    if s == 1:
        return img
    H, W = img.shape[0] // s, img.shape[1] // s
    return img[:H * s, :W * s].reshape(H, s, W, s, -1).mean((1, 3))


def run(img, p: Params, palette="cold", stages=False):
    """整条管线（v2）。stages=True → 额外返回 [(名字, 图)]（累积，每级都过同一个 post + grade）"""
    _, pre, post, dk_mul = PALETTES[palette]
    H0, W0, _ = img.shape
    s = max(1, int(p.ss))
    q = scaled(p, s)
    src = upsample(np.clip(pre(img) * p.exposure, 0, 1), s)
    H, W, _ = src.shape
    cells, micro, cols = displacements(H, W, q)
    sp = specular(cells, size=q.spec_size, seed=q.seed)
    dk = q.dk * dk_mul
    st = []
    fin = lambda x: grade(post(downsample(x, s)), p.black, p.gamma, p.vignette)  # noqa: E731
    zero = [(np.zeros((1, W)), np.zeros((1, W)))]

    def add_spec(x):
        return x + q.spec * sp[..., None] * gaussian_blur(x, q.period_x / 2)

    if stages:
        st.append(("0  source (pre-graded)", fin(src)))
        st.append((f"1  micro-lens C1 caps {p.period_x:.0f}x{p.period_y:.0f}px refraction {p.micro:+.2f} + specular {p.spec}",
                   fin(add_spec(lens_stage(src, micro, zero, q, 0)))))
        st.append((f"2a + vertical ghost 1  shear A={p.A:.0f}px  stretch m={p.stretch:.2f}  corr {p.corr:.0f}px (smooth)",
                   fin(add_spec(lens_stage(src, micro, cols[:1], q, 0)))))
        st.append((f"2b + {len(cols)} ghosts averaged", fin(add_spec(lens_stage(src, micro, cols, q, 0)))))
        st.append((f"3  + dispersion dk={dk:.3f} n_lambda={p.n_lambda} (k_R<k_G<k_B)", fin(add_spec(lens_stage(src, micro, cols, q, dk)))))
    x = add_spec(lens_stage(gaussian_blur(src, q.defocus), micro, cols, q, dk))
    x = gaussian_blur(x, q.soften)
    if stages:
        st.append((f"4a + defocus {p.defocus:.1f}px (before) + soften {p.soften}px (after), {s}x supersampled", fin(x)))
    x = bloom(x, p.bloom_thr, q.bloom_sigma, p.bloom_gain)
    out = fin(x)
    if stages:
        st.append((f"4b + bloom gain={p.bloom_gain} -> 5 grade black={p.black} gamma={p.gamma}", out))
        return out, st, cells, cols
    return out


# ---------------------------------------------------------------- v1（保留以便复现 24/24b/24c 与样片 18_v1）
def displacements_v1(H, W, p, k=0.7, bow=0.3):
    """v1：抛物面微透镜（格边跳变）+ 按列分段常数的竖向位移（硬边）。只给自检 / 复现用"""
    cells = lens_cells(H, W, p.period_x, p.period_y, p.stagger, (p.origin_x, 0.0))
    lens = lens_offset(cells, k, k, bow, p.period_y)
    cols = [column_displacement(cells, p.A, p.scatter, p.aniso, p.t, p.corr, p.omega, p.seed + 17 * g,
                                "noise", 120.0, p.stagger) for g in range(max(1, p.n_ghost))]
    return cells, lens, cols


# ---------------------------------------------------------------- 时间演化
def bursts(t, onsets, threat, fps=FPS, thr=0.3, tau=0.25):
    """强起音脉冲：e(t) = Σ exp(−(t − o)/tau)（t ≥ o），只算该处威胁峰 ≥ thr 的起音。返回 (e, 触发的起音列表)"""
    t = np.asarray(t, np.float64)
    th = np.asarray(threat)
    e = np.zeros_like(t)
    fired = []
    for o in onsets:
        c = int(round(o * fps))
        if c < 0 or c >= len(th) or th[max(0, c - 1):c + 2].max() < thr:
            continue
        m = t >= o - 0.5 / fps
        e[m] += np.exp(-np.clip(t[m] - o, 0, None) / tau)
        fired.append(float(o))
    return e, fired


def evolve_params(p: Params, t, e, n_fired, drift_x=4.0, burst_gain=0.8):
    """t 时刻的参数：玻璃横滑、噪声时间、脉冲放大、每次脉冲换种子"""
    q = Params(**asdict(p))
    q.t = t
    q.origin_x = drift_x * t
    q.A = p.A * (1 + burst_gain * e)
    q.stretch = min(0.9, p.stretch * (1 + 0.5 * burst_gain * e))
    q.dk = p.dk * (1 + 0.5 * e)
    q.seed = p.seed + 1000 * n_fired
    return q


# ---------------------------------------------------------------- 测量：纹理周期
def measure_period(img, hp_sigma=6.0, min_period=3.0, max_period=40.0):
    """高通（减去高斯 hp_sigma）后对亮度做 2D FFT（汉宁窗），在 [min,max] 周期范围里找最强峰。
    返回 {"x": 横向周期（|F| 对 fy 求和后的边缘谱里、|fx| ∈ 带内的最强峰）, "y": 竖向周期（同理，对 fx 求和）,
          "peak": (fx, fy) 2D 全局最强}。用边缘谱而不是只看坐标轴：纹理能量会散在整条 fy = 1/period_y 线上"""
    g = lum(img) if img.ndim == 3 else img
    g = g - gaussian_blur(g, hp_sigma)
    H, W = g.shape
    g = g * np.outer(np.hanning(H), np.hanning(W))
    F = np.abs(np.fft.fft2(g))
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.fftfreq(W)[None, :]
    r = np.hypot(fx, fy)
    band = (r >= 1 / max_period) & (r <= 1 / min_period)
    ax, ay = np.abs(fx[0]), np.abs(fy[:, 0])
    Fx = np.where((ax >= 1 / max_period) & (ax <= 1 / min_period), F.sum(0), 0)
    Fy = np.where((ay >= 1 / max_period) & (ay <= 1 / min_period), F.sum(1), 0)
    Fb = np.where(band, F, 0)
    a, b = np.unravel_index(np.argmax(Fb), F.shape)
    return {"x": float(1 / ax[np.argmax(Fx)]), "y": float(1 / ay[np.argmax(Fy)]),
            "peak": (float(fx[0, b]), float(fy[a, 0]))}


# ---------------------------------------------------------------- IO（复用 face_glitch 的读帧 / 标注 / 不覆盖）
def _F():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import face_glitch as F
    return F


def field_panel(cells, cols):
    """可视化：圆顶高度场（灰）+ 第一套竖向大位移（蓝 = 往下取，橙 = 往上取；沿 x 连续）"""
    h = cells["h"]
    dy = np.broadcast_to(cols[0][1], h.shape) / max(1e-6, np.abs(cols[0][1]).max())
    rgb = np.stack([np.clip(-dy, 0, 1) * 0.8 + 0.3 * h, np.clip(-dy, 0, 1) * 0.4 + 0.3 * h,
                    np.clip(dy, 0, 1) * 0.85 + 0.3 * h], -1)
    return rgb.astype(np.float32)


def load(path):
    return np.asarray(Image.open(path).convert("RGB"), np.float32) / 255


def zoom_sheet(ref, ours, v1, path, F, z=3, box=170, ver="new"):
    """3× 放大对比：同样像素大小的方块（参考图 / v2 / v1），最近邻放大，看纹理颗粒"""
    def crops(img, centers):
        out = []
        for cx, cy in centers:
            c = img[cy - box // 2:cy + box // 2, cx - box // 2:cx + box // 2]
            out.append(F.to_im(c).resize((box * z, box * z), Image.NEAREST))
        return out
    rows = [("REF 瓦解4.jpg (736 wide)", crops(ref, [(560, 300), (180, 380), (470, 860)])),
            (f"OURS {ver} 24b (1280x720)", crops(ours, [(560, 250), (470, 360), (760, 470)]))]
    if v1 is not None:
        rows.append(("OURS v1 24b (old, blocks)", crops(v1, [(560, 250), (470, 360), (760, 470)])))
    S = box * z
    sheet = Image.new("RGB", (3 * S, len(rows) * (S + 30) + 40), (8, 8, 10))
    for r, (name, ims) in enumerate(rows):
        for c, im in enumerate(ims):
            sheet.paste(im, (c * S, 40 + r * (S + 30) + 30))
        from PIL import ImageDraw, ImageFont
        ImageDraw.Draw(sheet).text((10, 40 + r * (S + 30) + 6), f"{name}   {box}x{box}px crops, {z}x nearest",
                                   font=ImageFont.truetype(F.FONT, 18), fill=(220, 224, 228))
    F.label(sheet, "", s=1.1, color=(160, 166, 172), pos="tl",
            lines=["24d  LENS GHOST zoom comparison — same pixel size crops, 3x. tools/lens_ghost.py sheet"])
    sheet.save(path, quality=93)


def cmd_sheet(a):
    F = _F()
    enable_torch()
    img = F.video_frames(F.SRC_VIDEO, a.t, 1)[0]
    H, W, _ = img.shape
    ref = load(REF)
    mp = measure_period(ref[550:1000, 380:700])
    p = params_from_sense(a.threat, a.tension, a.hunger, H, Params())
    out, st, cells, cols = run(img, p, "cold", stages=True)
    # 必须相等：同一套微透镜项作用在斜坡图上（和 tests 同一个量），FFT 周期 == 设定周期。真实画面上 FFT 会被内容（等高线）带偏，不拿来比
    Hr, Wr = int(p.period_y * 24), int(p.period_x * 20)
    _, (lx, ly), _ = displacements(Hr, Wr, Params(**{**asdict(p), "A": 0}))
    ramp = (0.3 * (np.arange(Wr)[None, :] + lx) / Wr + 0.7 * (np.arange(Hr)[:, None] + ly) / Hr)[..., None].repeat(3, -1)
    ours = measure_period(ramp.astype(np.float32), min_period=4)
    s = max(1, p.ss)
    panels = [(f"ref crop  FFT: y {mp['y']:.1f}px  x {mp['x']:.1f}px  peak (fx,fy)=({abs(mp['peak'][0]):.3f},{abs(mp['peak'][1]):.3f})",
               F.to_im(ref[250:250 + 414, 0:736]).resize((W, H), Image.LANCZOS)),
              ("C1 dome height h (grey) + ghost-1 dy (blue down / orange up), smooth across x",
               F.to_im(downsample(field_panel(cells, cols), s)))]
    for i, (name, x) in enumerate(st):
        if i == 1:
            name += f" | ramp FFT {ours['x']:.1f}x{ours['y']:.1f}"
        panels.append((name, F.to_im(x)))
    pw_, ph_ = 640, 360
    cols_n = 3
    rows = (len(panels) + cols_n - 1) // cols_n
    sheet = Image.new("RGB", (cols_n * pw_, rows * ph_ + 40), (8, 8, 10))
    for i, (name, im) in enumerate(panels):
        th = im.resize((pw_, ph_), Image.LANCZOS)
        F.label(th, name[:100], s=0.8, color=(215, 218, 222))
        sheet.paste(th, ((i % cols_n) * pw_, 40 + (i // cols_n) * ph_))
    F.label(sheet, "", s=1.1, color=(160, 166, 172), pos="tl",
            lines=[f"24 {a.ver}  LENS GHOST — C1 micro-lens caps + smooth vertical ghosts + dispersion, {s}x supersampled. tools/lens_ghost.py  "
                   f"t={a.t}s 03_headlift_v3  threat {a.threat} tension {a.tension} hunger {a.hunger}"])
    dst = F.S.MV / "静帧"
    o1 = F.outname(dst / f"24_透镜重影_分解_{a.ver}.jpg"); sheet.save(o1, quality=92)
    o2 = F.outname(dst / f"24b_透镜重影_冷蓝_{a.ver}.jpg")
    F.label(F.to_im(out), "", s=1.0, lines=p.hud(PALETTES["cold"][3]) + ["palette cold: mono cold-blue before lens"]).save(o2, quality=94)
    out_ref = run(img, p, "ref")
    o3 = F.outname(dst / f"24c_透镜重影_接近参考配色_{a.ver}.jpg")
    F.label(F.to_im(out_ref), "", s=1.0, color=(200, 200, 200),
            lines=p.hud(PALETTES["ref"][3]) + ["palette ref: source colour, saturation x2"]).save(o3, quality=94)
    v1p = dst / "24b_透镜重影_冷蓝.jpg"
    v1 = load(v1p) if v1p.exists() and v1p != o2 else None
    o4 = F.outname(dst / f"24d_透镜重影_放大对比_{a.ver}.jpg")
    zoom_sheet(ref, out, v1, o4, F, ver=a.ver)
    for o in (o1, o2, o3, o4):
        print("->", o)
    print("ref period", mp, " ours micro (set %.0fx%.0f)" % (p.period_x, p.period_y), ours, "torch", USE_TORCH)


def cmd_clip(a):
    import imageio_ffmpeg
    F = _F()
    enable_torch()
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    src = Path(a.src)
    sn = json.loads((F.S.MV / "感官.json").read_text("utf-8"))
    ch = sn["channels"]
    frames = F.video_frames(src, a.src_t0, int(round(a.dur * FPS)) if a.dur else 10 ** 6)
    n = len(frames)
    dur = n / FPS
    H, W = frames[0].shape[:2]
    tmp = Path(a.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    vtmp = tmp / f"lens_ghost_{src.stem}_{a.palette}_noaudio.mp4"
    pipe = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                             "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", str(vtmp)],
                            stdin=subprocess.PIPE)
    t_play = a.start + np.arange(n) / FPS
    th = np.array(ch["threat"])
    on = np.array(sn["events"]["onsets"])
    e, fired = bursts(t_play, on[(on >= t_play[0] - 0.5 / FPS) & (on <= t_play[-1])], th, thr=a.burst_thr, tau=a.burst_tau)
    side = {"source": src.name, "src_t0": a.src_t0, "play_start": a.start, "dur": dur, "palette": a.palette,
            "time": "感官.json is in playback time (no offset)", "mapping": MAPPING,
            "evolution": {"drift_x_px_s": a.drift, "burst_gain": a.burst_gain, "burst_tau_s": a.burst_tau,
                          "burst_thr": a.burst_thr, "fired_onsets": fired, "omega_rad_s": Params().omega},
            "exposure": a.exposure, "gamma": a.gamma, "model": "v2 (C1 micro-lens caps, smooth column ghosts, 2x supersampled)",
            "frames": []}
    thumbs, every = [], max(1, n // 16)
    t0 = time.time()
    for i, f in enumerate(frames):
        t = float(t_play[i])
        kf = min(int(round(t * FPS)), len(th) - 1)
        base = params_from_sense(ch["threat"][kf], ch["tension"][kf], ch["hunger"][kf], H, Params(seed=a.seed, exposure=a.exposure, gamma=a.gamma))
        nf = sum(1 for o in fired if o <= t + 0.5 / FPS)
        p = evolve_params(base, t - a.start, float(e[i]), nf, a.drift, a.burst_gain)
        im = F.to_im(run(f, p, a.palette))
        if a.hud:
            F.label(im, "", s=0.8, lines=[f"t = {t:6.3f} s (play)  threat {p.sense['threat']:.2f}  tension "
                                           f"{p.sense['tension']:.2f}  hunger {p.sense['hunger']:.2f}  burst {e[i]:.2f}"] + p.hud(PALETTES[a.palette][3]))
        pipe.stdin.write(np.asarray(im).tobytes())
        if i % every == 0:
            thumbs.append((f"{t:.2f}s thr {p.sense['threat']:.2f} ten {p.sense['tension']:.2f} burst {e[i]:.2f}", im))
        side["frames"].append({"i": i, "t_play": round(t, 3), **p.sense, "burst": round(float(e[i]), 3),
                               "A": round(p.A, 1), "dk": round(p.dk, 4), "dk_effective": round(p.dk * PALETTES[a.palette][3], 4),
                               "defocus": round(p.defocus, 2),
                               "seed": p.seed, "origin_x": round(p.origin_x, 2)})
    pipe.stdin.close()
    pipe.wait()
    print(f"{n} 帧  {time.time() - t0:.1f}s  bursts {fired}")
    out = F.outname(F.S.MV / "样片" / f"{a.name}.mp4")
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(vtmp), "-ss", f"{a.start:.3f}", "-t", f"{dur:.3f}",
                    "-i", str(F.S.MUSIC), "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
                    "-t", f"{dur:.3f}", str(out)], check=True)
    Path(str(out) + ".json").write_text(json.dumps(side, ensure_ascii=False, indent=1), "utf-8")
    F.contact(thumbs, out.with_name(out.stem + "_接触表.jpg"))
    print("->", out)


def cmd_image(a):
    F = _F()
    img = load(a.src)
    o = F.outname(a.out or Path(a.src).with_name(Path(a.src).stem + "_lens_ghost.jpg"))
    F.to_im(run(img, params_from_sense(0.6, 0.6, 0.35, img.shape[0]), a.palette)).save(o, quality=94)
    print("->", o)


def cmd_measure(a):
    print(measure_period(load(a.src)))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s1 = sp.add_parser("sheet")
    s1.add_argument("--t", type=float, default=4.0)
    s1.add_argument("--threat", type=float, default=0.6)
    s1.add_argument("--tension", type=float, default=0.6)
    s1.add_argument("--hunger", type=float, default=0.35)
    s1.add_argument("--ver", default="v3", help="静帧文件名版本后缀（不覆盖：已存在再自动加 _v2 …）")
    s2 = sp.add_parser("clip")
    s2.add_argument("--src", default=None, help="源视频；默认 素材/全息女性视频运镜描述.mp4")
    s2.add_argument("--src-t0", type=float, default=0.0)
    s2.add_argument("--start", type=float, default=45.12, help="歌里的播放时间 = 第 0 帧（感官.json 已是播放时间）")
    s2.add_argument("--dur", type=float, default=0.0, help="秒；0 = 整段")
    s2.add_argument("--palette", default="cold", choices=list(PALETTES))
    s2.add_argument("--name", default="18_透镜重影_全息女性_v2", help="不覆盖：已存在会自动加 _v2、_v3 …")
    s2.add_argument("--seed", type=int, default=3)
    s2.add_argument("--drift", type=float, default=4.0, help="玻璃横向滑动 px/s")
    s2.add_argument("--burst-gain", type=float, default=0.8)
    s2.add_argument("--burst-tau", type=float, default=0.25)
    s2.add_argument("--burst-thr", type=float, default=0.5, help="v1 = 0.3（9 个起音全触发，太密）")
    s2.add_argument("--exposure", type=float, default=1.0, help="进透镜前的曝光（暗的源拉高）")
    s2.add_argument("--gamma", type=float, default=1.7)
    s2.add_argument("--hud", action="store_true", help="画面上写参数")
    s2.add_argument("--tmp", default=str(Path(__file__).resolve().parents[1] / "cache" / "lens_ghost"))
    s3 = sp.add_parser("image"); s3.add_argument("src"); s3.add_argument("--out")
    s3.add_argument("--palette", default="cold", choices=list(PALETTES))
    s4 = sp.add_parser("measure"); s4.add_argument("src")
    a = ap.parse_args()
    if a.cmd == "clip" and a.src is None:
        a.src = str(_F().S.MV / "素材" / "全息女性视频运镜描述.mp4")
    {"sheet": cmd_sheet, "clip": cmd_clip, "image": cmd_image, "measure": cmd_measure}[a.cmd](a)


if __name__ == "__main__":
    main()
