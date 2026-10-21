#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🖼面部瓦解-拆解-参考图 `_参考库/面部瓦解.jpg`（侧脸 glitch art）拆成可组合的几级，每级一个函数、参数显式。
图像约定：float32 HxWx3 ∈ [0,1]；mask = HxW float ∈ [0,1]（她的轮廓，BiRefNet 或亮度阈值）。

管线（编号 = 参考图里看到的东西）：
  1 band_displace    水平切片（高 ~ 几何分布）整条横移，位移重尾（Pareto），偏向左；mask 跟着移
  2 row_smear        每行在轮廓左边界 x_r(y) 取「边界内侧一点」的颜色向左拉长：
                     强度 = exp(-max(0, x_b - x)/L)，x_b = 该切片内最靠左的边界（分段常数 → 台阶状起点），
                     粗粒度随机断点（dropout）+ 每行随机增益 → 长横向条纹
  2' waterfall_rows  可选（替代 2，👤 的「瀑布图」读法）：第 y 行左侧不再是同一颜色，而是
                     该行边界列在过去帧的历史 —— 离边界 d 像素 = 第 d // px_per_frame 帧之前（时间 → x）
  3 pixel_sort_rows  行内阈值排序：亮度 ∈ [lo, hi] 的连续段按亮度排序（亮的在右，向左变暗），只在拖影区
  4 channel_split    RGB 分离：global = 各通道整数平移（精确）；normal = 沿轮廓法向各自位移 d_R/d_G/d_B
                     （参考图：红 / 黄在外缘，绿 / 蓝在内侧弧上）
  5 column_jitter    脸内每一列竖向错位（整数噪声）+ 列增益条纹 → 竖向扫描感
  6 macroblocks      8/16 px 网格上的 B·G·R 三联竖条 / 横短划 + 量化块（块内取均值 = 只剩 DC）
  7 finish           若干切片通道置换 / 反相（只在 mask 内）、色阶量化、底部过曝、灰尘 + 划痕、左缘红竖条
参考图里额外看到的（已并入）：左缘一条纯红竖条（7）；三联条有竖有横（6）；颈 / 肩过曝成白（7）；
  脸内饱和拉高、色相偏（palette_orig 的 saturate）。

配色（PALETTES 表，视频 / 静帧共用）：ours 本片（冷灰白 + 只在轮廓外侧留一个红）/ orig 原配色（饱和 RGB）/
  micro_fringe·micro_streaks·micro_redcyan 微彩（颜色只在轮廓圈 / 少数拖影行里）/ heavy_false·heavy_bands 重彩（接近参考图）。
时间：感官.json 2026-10-21 起已是播放时间 → --sense-offset 默认 0（旧 感官.json 才要 0.042）。
人读版：-给人类的解释/面部瓦解怎么做的.md

感官驱动（params_from_sense，HUD / sidecar 里写同一张表）：
  threat  → 切片数 n_bands = 4 + 22·threat，最大横移 max_dx = 16 + 300·threat，横移概率 0.25 + 0.5·threat
  tension → RGB 分离 d = 2 + 16·tension（R = +d，G = −d/2，B = −d；法向模式）
  hunger  → 拖影长度 L = 40 + 520·hunger
（其余参数固定；种子每 --hold 帧（默认 4）换一次 + 强起音处换。v1 是每 2 帧 → 闪得太碎。）

用法（在仓库 .venv 里）：
  python tools/face_glitch.py sheet      # 静帧/23_面部瓦解_分解 + 23b 本片 + 23c 原配色 + 23d 瀑布图行
  python tools/face_glitch.py palettes   # 静帧/23e–23i 微彩 / 重彩 + 23j 配色对比
  python tools/face_glitch.py clip [--src 视频] [--src-t0 0] [--dur 秒, 0 = 整段] [--start 歌里播放秒数]
         [--palettes ours,micro_fringe,orig] [--name 输出名] [--waterfall] [--hold 4]
         # → 样片/<name>_<配色>.mp4（配乐）+ 同名 .json（逐帧参数）+ _接触表.jpg；遮罩缓存在 cache/face_glitch/
  python tools/face_glitch.py image 图 [--mask 遮罩] [--out 路径]           # 任意图 + 遮罩（无遮罩 → 亮度阈值）
"""
import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stills as S  # noqa: E402

FONT = S.FONT
FPS = 24
SENSE_OFFSET = 0.0            # 感官.json 时间 + 这个 = 播放时间。2026-10-21 起感官层在源头补了 0.042 s，已是播放时间 → 0；
                              # 只有 2026-10-21 之前的旧 感官.json 才要 --sense-offset 0.042（见 _LEARNINGS）
SRC_VIDEO = S.MV / "生成" / "视频" / "03_抬头_v3_仅首帧.mp4"
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def lum(img):
    return img @ LUMA


def leftmost(mask, thr=0.5):
    """每行最靠左的 mask 像素 x；没有 → -1"""
    m = mask > thr
    any_ = m.any(1)
    return np.where(any_, m.argmax(1), -1)


def make_bands(H, mean_h, rng):
    """切片边界：高度 ~ 几何分布（均值 mean_h），铺满整个画高 → [(y0, y1), ...]"""
    out, y = [], 0
    while y < H:
        h = int(rng.geometric(1.0 / max(mean_h, 1.0)))
        out.append((y, min(H, y + h)))
        y += h
    return out


# ---------------------------------------------------------------- 1 切片横移
def band_displace(img, mask, rng, n_bands=12, max_dx=120, p_shift=0.5, alpha=1.3, left_bias=0.7):
    """返回 (img, mask, bands)；bands = [(y0, y1, dx)]。每条切片 np.roll 整体横移（行内容只是循环平移，不丢像素）。
    位移 = 符号 · min(max_dx, s·Pareto(alpha))，s = max_dx/6 → 大部分小、偶尔很大（重尾）"""
    H = img.shape[0]
    bands = []
    out, om = img.copy(), mask.copy()
    for y0, y1 in make_bands(H, H / max(n_bands, 1), rng):
        dx = 0
        if rng.random() < p_shift:
            mag = min(max_dx, (max_dx / 6) * (1 + rng.pareto(alpha)))
            dx = int(round(mag)) * (-1 if rng.random() < left_bias else 1)
        if dx:
            out[y0:y1] = np.roll(img[y0:y1], dx, axis=1)
            om[y0:y1] = np.roll(mask[y0:y1], dx, axis=1)
        bands.append((y0, y1, dx))
    return out, om, bands


def band_boundary(mask, bands, thr=0.5):
    """x_b(y)：每条切片内最靠左的轮廓（分段常数）；x_r(y)：每行自己的最左轮廓"""
    xr = leftmost(mask, thr)
    xb = np.full_like(xr, -1)
    for y0, y1, *_ in bands:
        seg = xr[y0:y1]
        ok = seg >= 0
        if ok.any():
            xb[y0:y1] = np.where(ok, seg[ok].min(), -1)
    return xb, xr


# ---------------------------------------------------------------- 2 向左拖影
def row_smear(img, xr, xb, rng, length=200.0, inset=6, dropout=0.25, seg=64, row_gain=(0.2, 1.0), win=24, group=3.0):
    """x < x_r(y) 的像素 = max(原图, c(y)·env·keep·g)；c(y) = img[y, x_r+inset : x_r+inset+win] 里最亮的那个像素
    （win = 1 → 正好是边界内侧 inset 处那一个像素）。
    env = exp(-max(0, x_b - x)/L)：x_b 到 x_r 之间满强度（台阶），x_b 左边指数衰减。
    行分组（组高 ~ 几何分布，均值 group 行）：组内共用 L = length·lognormal(0, 0.5)、增益 g ~ U(row_gain)、
    断点 keep（seg 像素一格，概率 dropout 断）→ 有厚度的长条纹。length = inf、dropout = 0、row_gain = (1,1) → 恒等于边界色。"""
    H, W, _ = img.shape
    out = img.copy()
    xs = np.arange(W)
    nseg = W // seg + 1
    L_img = lum(img)
    y = 0
    while y < H:
        gh = int(rng.geometric(1.0 / max(group, 1.0)))
        L = length * np.exp(rng.normal(0, 0.5)) if np.isfinite(length) else np.inf
        g = rng.uniform(*row_gain) if row_gain[0] < row_gain[1] else row_gain[0]
        keep_seg = rng.random(nseg) >= dropout if dropout > 0 else np.ones(nseg, bool)
        for yy in range(y, min(H, y + gh)):
            x_r = xr[yy]
            if x_r <= 0:
                continue
            a0 = min(W - 1, x_r + inset)
            a1 = min(W, a0 + max(1, win))
            c = img[yy, a0 + int(np.argmax(L_img[yy, a0:a1]))]
            d = np.maximum(0, xb[yy] - xs[:x_r]) if xb[yy] >= 0 else np.zeros(x_r)
            env = np.exp(-d / L) if np.isfinite(L) else np.ones(x_r)
            keep = keep_seg[xs[:x_r] // seg]
            out[yy, :x_r] = np.maximum(out[yy, :x_r], c[None, :] * (env * keep * g)[:, None])
        y += gh
    return out


class BoundaryHistory:
    """瀑布图读法的历史缓存：每帧推入「每行边界内侧那个像素」(H, 3)，最新在最后"""

    def __init__(self, maxlen=200):
        self.buf, self.maxlen = [], maxlen

    def push(self, img, xr, inset=6):
        H, W, _ = img.shape
        col = np.zeros((H, 3), np.float32)
        ok = xr >= 0
        col[ok] = img[np.arange(H)[ok], np.minimum(W - 1, xr[ok] + inset)]
        self.buf.append(col)
        if len(self.buf) > self.maxlen:
            self.buf.pop(0)

    def array(self):
        return np.stack(self.buf[::-1])     # [0] = 当前帧, [k] = k 帧之前


def waterfall_rows(img, xr, hist, px_per_frame=6, decay=0.0):
    """2'：x < x_r(y) 处 = max(原图, hist[k, y]·exp(-decay·k))，k = (x_r - 1 - x) // px_per_frame；k 超出历史 → 不画。
    hist = BoundaryHistory.array()，形状 (T, H, 3)，[0] 是当前帧"""
    H, W, _ = img.shape
    T = hist.shape[0]
    out = img.copy()
    xs = np.arange(W)
    for y in range(H):
        x_r = xr[y]
        if x_r <= 0:
            continue
        k = (x_r - 1 - xs[:x_r]) // px_per_frame
        ok = k < T
        v = np.zeros((x_r, 3), np.float32)
        v[ok] = hist[k[ok], y] * np.exp(-decay * k[ok])[:, None]
        out[y, :x_r] = np.maximum(out[y, :x_r], v)
    return out


# ---------------------------------------------------------------- 3 行内阈值排序
def pixel_sort_rows(img, region, lo=0.08, hi=0.75, leftward=True):
    """region (H, W) bool 内、亮度 ∈ [lo, hi] 的每个连续段按亮度排序：leftward → 亮的在右、往左变暗。
    每行像素的多重集合不变（只是重排）"""
    out = img.copy()
    L = lum(img)
    sel = region & (L >= lo) & (L <= hi)
    H, W = sel.shape
    for y in range(H):
        r = sel[y]
        if not r.any():
            continue
        d = np.diff(np.concatenate([[0], r.astype(np.int8), [0]]))
        for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            if b - a < 2:
                continue
            o = np.argsort(L[y, a:b], kind="stable")
            out[y, a:b] = img[y, a:b][o if leftward else o[::-1]]
    return out


# ---------------------------------------------------------------- 4 RGB 分离
def shift2d(ch, dx, dy, fill=0.0):
    """整数平移，空出来的填 fill（不循环）"""
    H, W = ch.shape
    out = np.full_like(ch, fill)
    xs0, xs1 = max(0, dx), min(W, W + dx)
    ys0, ys1 = max(0, dy), min(H, H + dy)
    out[ys0:ys1, xs0:xs1] = ch[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    return out


def mask_normal(mask, blur=9):
    """轮廓外法向（模糊后 mask 的负梯度，单位化）"""
    from PIL import ImageFilter
    m = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(blur)), np.float32) / 255
    gy, gx = np.gradient(m)
    n = np.hypot(gx, gy) + 1e-6
    return -gx / n, -gy / n, np.clip(np.hypot(gx, gy) * blur * 4, 0, 1)


def channel_split(img, mask=None, d=(8, -4, -8), mode="normal", dy=(0, 0, 0)):
    """global：通道 c 整体平移 (d[c], dy[c]) 像素（精确，空处填 0）。
    normal：通道 c 在 p 处取 img_c(p - d[c]·n(p))，n = 轮廓外法向，只在轮廓附近生效（远处权重 → 0）。
    d > 0 = 往外推（参考图：红在最外，绿 / 蓝在内）"""
    out = np.empty_like(img)
    if mode == "global" or mask is None:
        for c in range(3):
            out[..., c] = shift2d(img[..., c], int(d[c]), int(dy[c]))
        return out
    H, W, _ = img.shape
    nx, ny, w = mask_normal(mask)
    yy, xx = np.mgrid[0:H, 0:W]
    for c in range(3):
        sx = np.clip(np.rint(xx - d[c] * nx * w), 0, W - 1).astype(int)
        sy = np.clip(np.rint(yy - d[c] * ny * w), 0, H - 1).astype(int)
        out[..., c] = img[sy, sx, c]
    return out


# ---------------------------------------------------------------- 5 竖向列错位
def column_jitter(img, mask, rng, amp=6, gain=0.35, corr=3, per_channel=False):
    """mask 内：每列竖向循环平移 dy(x)（整数，|dy| ≤ amp，相邻 corr 列共用一个值 → 有宽度的竖纹），
    再乘列增益 1 - gain·u(x)，u ~ U(0,1)（竖向扫描条纹）。per_channel → R/G/B 各自一套 dy、增益（灰图也会出彩色竖纹）。
    mask = 全 1、gain = 0 → 每列正好是循环平移"""
    H, W, _ = img.shape
    n = W // corr + 1
    out = np.empty_like(img)
    cols = np.arange(W)
    for c in range(3):
        if c == 0 or per_channel:
            dy = rng.integers(-amp, amp + 1, n)[cols // corr]
            g = (1 - gain * rng.random(W)).astype(np.float32)
            idx = (np.arange(H)[:, None] - dy[None, :]) % H
        out[..., c] = img[idx, cols[None, :], c] * g[None, :]
    m = mask[..., None]
    return out * m + img * (1 - m)


# ---------------------------------------------------------------- 6 宏块
TRIPLET = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], np.float32)   # B G R（参考图从左到右）


def macroblocks(img, region, rng, grid=16, n_triplets=10, n_dashes=12, n_blocks=40, block_region=None):
    """· 三联竖条：网格对齐，3 条宽 grid/2、高 2–4 格、间隔 grid 的 B/G/R 纯色条（位置偏向 region）
       · 三联横短划：同理，高 4 px、长 grid
       · 量化块：block_region 内随机 n_blocks 个 grid×grid 格 → 块内均值（只剩 DC）
    返回 (img, blocks)，blocks = [(y, x)] 量化块左上角（给自检）"""
    H, W, _ = img.shape
    out = img.copy()
    ys, xs = np.nonzero(region[::grid, ::grid])
    if len(ys) == 0:
        ys, xs = np.nonzero(np.ones((H // grid, W // grid), bool))

    def pick():
        i = rng.integers(len(ys))
        return ys[i] * grid, xs[i] * grid

    bw = grid // 2
    for _ in range(n_triplets):
        y, x = pick()
        h = grid * rng.integers(2, 5)
        for c in range(3):
            x0 = x + c * grid
            out[y:y + h, x0:x0 + bw] = TRIPLET[c]
    for _ in range(n_dashes):
        y, x = pick()
        for c in range(3):
            x0 = x + c * int(grid * 1.5)
            out[y:y + 4, x0:x0 + grid] = TRIPLET[c]
    blocks = []
    br = block_region if block_region is not None else region
    by, bx = np.nonzero(br[grid // 2::grid, grid // 2::grid])
    if len(by):
        for i in rng.choice(len(by), min(n_blocks, len(by)), replace=False):
            y, x = by[i] * grid, bx[i] * grid
            out[y:y + grid, x:x + grid] = out[y:y + grid, x:x + grid].reshape(-1, 3).mean(0)
            blocks.append((y, x))
    return out, blocks


# ---------------------------------------------------------------- 7 收尾
PERMS = [(1, 2, 0), (2, 0, 1), (2, 1, 0), (0, 2, 1)]


def channel_ops(img, mask, bands, rng, p_swap=0.15, p_invert=0.06):
    """切片内（只在 mask 里）：概率 p_swap 置换通道、p_invert 反相。返回 (img, ops)"""
    out = img.copy()
    ops = []
    for y0, y1, *_ in bands:
        r = rng.random()
        m = mask[y0:y1, :, None]
        if r < p_swap:
            p = PERMS[rng.integers(len(PERMS))]
            out[y0:y1] = img[y0:y1][..., list(p)] * m + img[y0:y1] * (1 - m)
            ops.append((y0, y1, "swap" + "".join("RGB"[i] for i in p)))
        elif r < p_swap + p_invert:
            out[y0:y1] = (1 - img[y0:y1]) * m + img[y0:y1] * (1 - m)
            ops.append((y0, y1, "invert"))
    return out, ops


def posterize(img, levels=6):
    """量化到 levels 级（0, 1/(L-1), …, 1）"""
    return np.round(np.clip(img, 0, 1) * (levels - 1)) / (levels - 1)


def exposure_clip(img, mask, y0=0.78, gain=3.0):
    """y > y0·H 的 mask 内亮度乘 1 + gain·s(y)（s 从 0 平滑到 1），再截到 1 → 颈肩过曝成白"""
    H = img.shape[0]
    s = np.clip((np.arange(H) / H - y0) / max(1e-6, 1 - y0), 0, 1)
    s = s * s * (3 - 2 * s)
    k = 1 + gain * s[:, None, None] * mask[..., None]
    return np.clip(img * k + 0.25 * s[:, None, None] * mask[..., None], 0, 1)


def dust(img, mask, rng, n=900, n_scratch=6, strength=0.55):
    """背景里的灰尘（1–2 px 亮点，越远离她越少）+ 几道细划痕（短的斜线）"""
    H, W, _ = img.shape
    out = img.copy()
    ys, xs = rng.integers(0, H, n), rng.integers(0, W, n)
    keep = mask[ys, xs] < 0.5
    v = strength * rng.random(n) ** 2
    for y, x, a in zip(ys[keep], xs[keep], v[keep]):
        r = 1 if rng.random() < 0.8 else 2
        out[y:y + r, x:x + r] = np.maximum(out[y:y + r, x:x + r], a)
    for _ in range(n_scratch):
        x, y = rng.integers(0, W), rng.integers(0, H)
        ln = rng.integers(20, 90)
        t = np.arange(ln)
        yy = np.clip(y + t, 0, H - 1)
        xx = np.clip(x + (t * rng.normal(0, 0.3)).astype(int), 0, W - 1)
        out[yy, xx] = np.maximum(out[yy, xx], 0.35 * strength)
    return out


def edge_bar(img, width=10, color=(1.0, 0.0, 0.0)):
    out = img.copy()
    out[:, :width] = color
    return out


# ---------------------------------------------------------------- 配色
def saturate(img, k=1.6):
    g = lum(img)[..., None]
    return np.clip(g + (img - g) * k, 0, 1)


def palette_orig(img, sat=1.7, gamma=0.9):
    return np.clip(saturate(img, sat), 0, 1) ** gamma


def palette_ours(img, mask, ring=26, red_gain=3.0, grey=S.GREY, red=S.RED, side=0.25):
    """冷灰白 + 一个红：红只在轮廓外缘一圈（mask 外 ring 像素宽）、外法向 n_x > side（背对拖影的那一侧，
    参考图的红边也只在外侧）、且该处 R 明显高于 G/B（= RGB 分离的红边）"""
    g = np.clip(lum(img), 0, 1) ** 0.95
    base = g[..., None] * np.asarray(grey, np.float32)[None, None] / max(grey)
    from PIL import ImageFilter
    m8 = Image.fromarray((mask > 0.5).astype(np.uint8) * 255)
    outer = np.asarray(m8.filter(ImageFilter.MaxFilter(2 * ring + 1)), np.float32) / 255
    nx, _, _ = mask_normal(mask, blur=15)
    band = outer * (1 - (mask > 0.5)) * (nx > side)
    a = np.clip((img[..., 0] - np.maximum(img[..., 1], img[..., 2])) * red_gain, 0, 1) * band
    return base * (1 - a[..., None]) + np.asarray(red, np.float32)[None, None] * a[..., None]


def _ring(mask, width):
    """轮廓两侧 width 像素的一圈（膨胀 − 腐蚀）"""
    from PIL import ImageFilter
    m8 = Image.fromarray((mask > 0.5).astype(np.uint8) * 255)
    k = 2 * width + 1
    return (np.asarray(m8.filter(ImageFilter.MaxFilter(k)), np.float32)
            - np.asarray(m8.filter(ImageFilter.MinFilter(k)), np.float32)) / 255


def row_groups_gate(H, rng, frac=0.06, group=4.0):
    """把行按几何分布（均值 group 行）分组，每组以概率 frac 选中 → (H,) bool；外加每组一个随机色相 (H,) ∈ [0,1)"""
    sel = np.zeros(H, bool)
    hue = np.zeros(H, np.float32)
    y = 0
    while y < H:
        h = int(rng.geometric(1.0 / max(group, 1.0)))
        sel[y:y + h] = rng.random() < frac
        hue[y:y + h] = rng.random()
        y += h
    return sel, hue


def hue_rgb(h):
    """色相 → 纯色（HSV 的 S = V = 1），h 可以是数组"""
    h = np.asarray(h, np.float32)[..., None]
    return np.clip(np.abs(((h * 6 + np.array([0, 4, 2], np.float32)) % 6) - 3) - 1, 0, 1)


def _grey(img, grey=S.GREY):
    g = np.clip(lum(img), 0, 1) ** 0.95
    return g, g[..., None] * np.asarray(grey, np.float32)[None, None] / max(grey)


def palette_micro(img, mask, mode="fringe", seed=0, k=1.5, ring=10, frac=0.10):
    """微彩：底子是冷灰白，颜色只在局部出现、而且有来由：
      fringe  ：只在轮廓两侧 ring 像素里保留 RGB 分离产生的色差（× k）—— 颜色 = 真实的通道错位
      streaks ：fringe 的一半强度 + 拖影区里 frac 的行组染一个色相（颜色 × 该处亮度）—— 少数几条彩色扫描线
      redcyan ：色差投影到 红–青 一根轴上（a = R − (G+B)/2：正 → 红，负 → 青），只在轮廓圈里 —— 两色，最克制"""
    rng = np.random.default_rng(seed)
    g, base = _grey(img)
    c = img - lum(img)[..., None]
    rg = _ring(mask, ring)[..., None]
    if mode == "fringe":
        return np.clip(base + c * k * rg, 0, 1)
    if mode == "streaks":
        H, W, _ = img.shape
        out = base + c * (0.5 * k) * rg
        sel, hue = row_groups_gate(H, rng, frac)
        xr = leftmost(mask)
        left = (np.arange(W)[None, :] < xr[:, None]) & sel[:, None]
        tint = hue_rgb(hue)[:, None, :] * np.clip(g * 1.6, 0, 1)[..., None]
        a = left[..., None] * 1.0
        return np.clip(out * (1 - a) + tint * a, 0, 1)
    if mode == "redcyan":
        av = (img[..., 0] - 0.5 * (img[..., 1] + img[..., 2])) * (2 * k) * _ring(mask, 2 * ring)
        red, cyan = np.asarray(S.RED, np.float32), np.array([0.35, 0.85, 0.95], np.float32)
        ap, an = np.clip(av, 0, 1)[..., None], np.clip(-av, 0, 1)[..., None]
        return np.clip(base * (1 - ap - an) + red * ap + cyan * an * g[..., None] ** 0.5, 0, 1)
    raise ValueError(mode)


FALSE_STOPS = [(0.0, (0, 0, 0)), (0.18, (0.05, 0.05, 0.6)), (0.38, (0.55, 0.05, 0.75)), (0.58, (1.0, 0.1, 0.25)),
               (0.78, (1.0, 0.75, 0.1)), (1.0, (1, 1, 1))]


def false_colour(v, stops=FALSE_STOPS):
    xs = np.array([s_[0] for s_ in stops])
    cs = np.array([s_[1] for s_ in stops], np.float32)
    return np.stack([np.interp(v, xs, cs[:, c]) for c in range(3)], -1).astype(np.float32)


def palette_heavy(img, mask, mode="false", seed=0, sat=2.2, frac=0.35, mix=0.6):
    """重彩（接近参考图）：
      false ：亮度 → 假色梯度（黑→蓝→品红→红→黄→白），和饱和 ×sat 的原色按 mix 混 —— 参考图脸上那种蓝 / 品红 / 红
      bands ：frac 的行组（全宽，含拖影）乘一个随机纯色，再整体饱和 ×sat —— 彩色横条纹"""
    rng = np.random.default_rng(seed)
    if mode == "false":
        fc = false_colour(np.clip(lum(img), 0, 1) ** 0.8)
        return np.clip(fc * mix + saturate(img, sat) * (1 - mix), 0, 1)
    if mode == "bands":
        sel, hue = row_groups_gate(img.shape[0], rng, frac)
        tint = 0.35 + 0.65 * hue_rgb(hue)                      # (H,3)，不压到全黑
        t = np.where(sel[:, None], tint, 1.0)[:, None, :]
        return np.clip(saturate(img * t, sat) * 1.1, 0, 1)
    raise ValueError(mode)


# 配色表：名字 → (中文名, 函数(img, mask, seed))；视频 / 静帧都从这里取
PALETTES = {
    "ours": ("本片", lambda x, m, sd: palette_ours(x, m)),
    "orig": ("原配色", lambda x, m, sd: palette_orig(x)),
    "micro_fringe": ("微彩A_边缘", lambda x, m, sd: palette_micro(x, m, "fringe", sd)),
    "micro_streaks": ("微彩B_条纹", lambda x, m, sd: palette_micro(x, m, "streaks", sd)),
    "micro_redcyan": ("微彩C_红青", lambda x, m, sd: palette_micro(x, m, "redcyan", sd)),
    "heavy_false": ("重彩A_假色", lambda x, m, sd: palette_heavy(x, m, "false", sd)),
    "heavy_bands": ("重彩B_条带染色", lambda x, m, sd: palette_heavy(x, m, "bands", sd)),
}


# ---------------------------------------------------------------- 参数 + 管线
@dataclass
class Params:
    seed: int = 7
    place_dx: int = 180              # 把她往右挪（参考图：脸在右，左边全是拖影）
    n_bands: int = 14
    max_dx: int = 140
    p_shift: float = 0.5
    smear_len: float = 220.0
    smear_inset: int = 6
    smear_dropout: float = 0.25
    sort_lo: float = 0.06
    sort_hi: float = 0.7
    split_d: tuple = (9, -4, -9)
    jitter_amp: int = 5
    jitter_gain: float = 0.4
    jitter_rgb: bool = True
    grid: int = 16
    n_triplets: int = 9
    n_dashes: int = 10
    n_blocks: int = 45
    p_swap: float = 0.15
    p_invert: float = 0.05
    levels: int = 7
    clip_y0: float = 0.80
    clip_gain: float = 3.0
    n_dust: int = 900
    bar_w: int = 10
    waterfall: bool = False          # True → 2' 代替 2
    wf_px: int = 6
    wf_decay: float = 0.02
    sense: dict = field(default_factory=dict)

    def hud(self):
        return [f"bands {self.n_bands}  max_dx {self.max_dx}px  p {self.p_shift:.2f}   "
                f"smear L {self.smear_len:.0f}px  drop {self.smear_dropout:.2f}   "
                f"split R{self.split_d[0]:+d} G{self.split_d[1]:+d} B{self.split_d[2]:+d}px (normal)",
                f"sort lum [{self.sort_lo:.2f},{self.sort_hi:.2f}]  jitter ±{self.jitter_amp}px  "
                f"grid {self.grid}  posterize {self.levels}  seed {self.seed}"
                + ("  ROWS = WATERFALL" if self.waterfall else "")]


def params_from_sense(threat, tension, hunger, base=None, **kw):
    """感官 → 参数（映射表见模块 docstring；HUD / sidecar 写同一张表）"""
    p = Params(**{**(asdict(base) if base else {}), **kw})
    p.n_bands = int(round(4 + 22 * threat))
    p.max_dx = int(round(16 + 300 * threat))
    p.p_shift = 0.25 + 0.5 * threat
    d = 2 + 16 * tension
    p.split_d = (int(round(d)), -int(round(d / 2)), -int(round(d)))
    p.smear_len = 40 + 520 * hunger
    p.sense = {"threat": round(float(threat), 3), "tension": round(float(tension), 3), "hunger": round(float(hunger), 3)}
    return p


def place(img, mask, dx):
    if dx == 0:
        return img, mask
    out = np.zeros_like(img)
    om = np.zeros_like(mask)
    if dx > 0:
        out[:, dx:], om[:, dx:] = img[:, :-dx], mask[:, :-dx]
    else:
        out[:, :dx], om[:, :dx] = img[:, -dx:], mask[:, -dx:]
    return out, om


def run(img, mask, p: Params, hist: BoundaryHistory = None, stages=False):
    """整条管线。stages=True → 额外返回 [(名字, 图)]（累积结果）"""
    rng = np.random.default_rng(p.seed)
    st = []
    img, mask = place(img, mask, p.place_dx)
    st.append(("0  source + mask x_r(y)", img))
    x, m, bands = band_displace(img, mask, rng, p.n_bands, p.max_dx, p.p_shift)
    st.append((f"1  band displace  n={p.n_bands} max_dx={p.max_dx} p={p.p_shift:.2f} Pareto a=1.3", x))
    xb, xr = band_boundary(m, bands)
    if hist is not None:
        hist.push(x, xr, p.smear_inset)
    if p.waterfall and hist is not None:
        x = waterfall_rows(x, xr, hist.array(), p.wf_px, p.wf_decay)
        st.append((f"2' waterfall rows  {p.wf_px}px/frame  T={len(hist.buf)}  decay {p.wf_decay}", x))
    else:
        x = row_smear(x, xr, xb, rng, p.smear_len, p.smear_inset, p.smear_dropout)
        st.append((f"2  row smear  L={p.smear_len:.0f} lognorm(0,.5)  dropout {p.smear_dropout}  seg 24", x))
    H, W, _ = x.shape
    left = np.arange(W)[None, :] < (xr[:, None] + 12)
    left &= (xr[:, None] > 0)
    x = pixel_sort_rows(x, left, p.sort_lo, p.sort_hi)
    st.append((f"3  pixel sort rows  lum in [{p.sort_lo},{p.sort_hi}]  leftward, smear region", x))
    x = channel_split(x, m, p.split_d, "normal")
    st.append((f"4  RGB split along normal  dR={p.split_d[0]} dG={p.split_d[1]} dB={p.split_d[2]} px", x))
    x = column_jitter(x, m, rng, p.jitter_amp, p.jitter_gain, per_channel=p.jitter_rgb)
    st.append((f"5  column jitter  |dy|<={p.jitter_amp}px  gain 1-{p.jitter_gain}U  {'per RGB' if p.jitter_rgb else 'mono'}", x))
    bg = ~left & (m < 0.5)
    trip_region = left | (bg & (rng.random(bg.shape) < 0.3))
    x, _ = macroblocks(x, trip_region, rng, p.grid, p.n_triplets, p.n_dashes, p.n_blocks,
                       block_region=(left | (m > 0.5)))
    st.append((f"6  macroblocks  grid {p.grid}  BGR bars {p.n_triplets}+{p.n_dashes}  DC blocks {p.n_blocks}", x))
    x, ops = channel_ops(x, m, bands, rng, p.p_swap, p.p_invert)
    x = posterize(x, p.levels)
    x = exposure_clip(x, m, p.clip_y0, p.clip_gain)
    x = dust(x, m, rng, p.n_dust)
    x = edge_bar(x, p.bar_w)
    st.append((f"7  swap/invert {len(ops)} bands  posterize {p.levels}  clip y>{p.clip_y0}  dust {p.n_dust}  red bar", x))
    return (x, m, st) if stages else (x, m)


# ---------------------------------------------------------------- IO
def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), np.float32) / 255


def lum_mask(img, thr=0.08, blur=6):
    from PIL import ImageFilter
    m = (lum(img) > thr).astype(np.uint8) * 255
    m = Image.fromarray(m).filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(blur))
    return (np.asarray(m, np.float32) / 255 > 0.5).astype(np.float32)


def to_im(x):
    return Image.fromarray((np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8))


def label(im, text, s=1.0, color=(150, 160, 168), pos="bl", lines=None):
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, int(14 * s))
    lines = lines or [text]
    w, h = im.size
    for j, line in enumerate(lines):
        y = h - int(10 * s) - int(18 * s) * (len(lines) - j) if pos == "bl" else int(8 * s) + int(18 * s) * j
        d.text((int(10 * s), y), line, font=f, fill=color)
    return im


def video_frames(path, t0, n):
    """从 t0 起连续 n 帧（RGB float）"""
    import imageio_ffmpeg
    rd = imageio_ffmpeg.read_frames(str(path), input_params=["-ss", f"{t0:.3f}"])
    meta = next(rd)
    W, H = meta["size"]
    out = []
    for buf in rd:
        out.append(np.frombuffer(buf, np.uint8).reshape(H, W, 3).astype(np.float32) / 255)
        if len(out) >= n:
            break
    rd.close()
    return out


def get_matter():
    try:
        import matte
        return matte.Matter()
    except Exception as e:  # noqa: BLE001
        print("BiRefNet 不可用，回退亮度阈值：", e)
        return None


def get_mask(matter, img):
    if matter is None:
        return lum_mask(img), "luminance>0.08"
    a = np.asarray(matter(to_im(img)), np.float32) / 255
    return a, "BiRefNet"


def outname(path):
    """不覆盖：已存在 → _v2, _v3 …"""
    path = Path(path)
    if not path.exists():
        return path
    i = 2
    while (q := path.with_name(f"{path.stem}_v{i}{path.suffix}")).exists():
        i += 1
    return q


# ---------------------------------------------------------------- 命令
def cmd_sheet(a):
    t = a.t
    n_hist = 48
    frames = video_frames(SRC_VIDEO, max(0, t - (n_hist - 1) / FPS), n_hist)
    img = frames[-1]
    matter = get_matter()
    mask, msrc = get_mask(matter, img)
    p = Params()
    out, m, st = run(img, mask, p, stages=True)
    # 2' 瀑布图版：历史 = 前 47 帧（用同一张遮罩 —— 2 s 内头动得少，近似）
    hist = BoundaryHistory()
    pw = Params(waterfall=True)
    for f in frames:
        fi, fm = place(f, mask, pw.place_dx)
        hist.push(fi, leftmost(fm), pw.smear_inset)
    hist.buf.pop()       # run() 里会再推当前帧
    out_w, m_w, st_w = run(img, mask, pw, hist=hist, stages=True)
    ours = palette_ours(out, m)
    orig = palette_orig(out)

    pw_, ph_ = 640, 360
    panels = []
    src = to_im(st[0][1]).convert("RGB")
    d = ImageDraw.Draw(src)
    xr = leftmost(place(img, mask, p.place_dx)[1])
    for y in range(0, len(xr), 2):
        if xr[y] > 0:
            d.point((int(xr[y]), y), fill=(255, 40, 50))
    panels.append((f"0  source t={t:.2f}s (03_headlift_v3) + mask [{msrc}] red = x_r(y)", src))
    for name, x in st[1:]:
        panels.append((name, to_im(x)))
    panels.append((st_w[2][0], to_im(st_w[2][1])))
    panels.append(("final  palette_ours: cold grey + red only on outer ring", to_im(ours)))
    panels.append(("final  palette_orig: saturate x1.7, gamma 0.9", to_im(orig)))
    cols, rows = 3, (len(panels) + 2) // 3
    sheet = Image.new("RGB", (cols * pw_, rows * ph_ + 40), (8, 8, 10))
    for i, (name, im) in enumerate(panels):
        th = im.resize((pw_, ph_), Image.LANCZOS)
        label(th, name, s=0.9, color=(210, 214, 218))
        sheet.paste(th, ((i % cols) * pw_, 40 + (i // cols) * ph_))
    label(sheet, "", s=1.1, color=(160, 166, 172), pos="tl",
          lines=["23  FACE DISINTEGRATION — stage by stage (cumulative). tools/face_glitch.py  seed 7"])
    dst = S.MV / "静帧"
    o1 = outname(dst / "23_面部瓦解_分解.jpg"); sheet.save(o1, quality=92)
    o2 = outname(dst / "23b_面部瓦解_本片配色.jpg")
    label(to_im(ours), "", s=1.0, lines=p.hud()).save(o2, quality=94)
    o3 = outname(dst / "23c_面部瓦解_原配色.jpg")
    label(to_im(orig), "", s=1.0, color=(200, 200, 200), lines=p.hud()).save(o3, quality=94)
    # 瀑布图版也出一张（本片配色），给 👤 对比
    o4 = outname(dst / "23d_面部瓦解_瀑布图行_本片配色.jpg")
    label(to_im(palette_ours(out_w, m_w)), "", s=1.0, lines=pw.hud()).save(o4, quality=94)
    for o in (o1, o2, o3, o4):
        print("->", o)


def cmd_palettes(a):
    """同一张 t=4.0 的瓦解结果（seed 7，和 23b / 23c 同一份）→ 各种配色静帧 + 一张对比表"""
    img = video_frames(SRC_VIDEO, a.t, 1)[0]
    mask, _ = get_mask(get_matter(), img)
    p = Params()
    out, m = run(img, mask, p)
    dst = S.MV / "静帧"
    letters = dict(micro_fringe="23e", micro_streaks="23f", micro_redcyan="23g", heavy_false="23h", heavy_bands="23i")
    tiles = []
    for key in ["ours", "micro_fringe", "micro_streaks", "micro_redcyan", "orig", "heavy_false", "heavy_bands"]:
        zh, fn = PALETTES[key]
        im = to_im(fn(out, m, p.seed))
        tiles.append((key, im))
        if key in letters:
            o = outname(dst / f"{letters[key]}_面部瓦解_{zh}.jpg")
            label(im.copy(), "", lines=p.hud() + [f"palette {key}"]).save(o, quality=94)
            print("->", o)
    tw, th_ = 640, 360
    sheet = Image.new("RGB", (2 * tw, 4 * th_ + 40), (8, 8, 10))
    for i, (name, im) in enumerate(tiles):
        t = im.resize((tw, th_), Image.LANCZOS)
        label(t, name, s=0.9, color=(220, 224, 228))
        sheet.paste(t, ((i % 2) * tw, 40 + (i // 2) * th_))
    label(sheet, "", s=1.1, color=(160, 166, 172), pos="tl",
          lines=["23j  palettes on the same glitch (seed 7, t=4.0 s). tools/face_glitch.py palettes"])
    o = outname(dst / "23j_面部瓦解_配色对比.jpg")
    sheet.save(o, quality=92)
    print("->", o)


def contact(frames, path, cols=4):
    tw, th_ = 480, 270
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * th_), (8, 8, 10))
    for i, (lab, im) in enumerate(frames):
        t = im.resize((tw, th_), Image.LANCZOS)
        label(t, lab, s=0.8, color=(230, 230, 230), pos="tl")
        sheet.paste(t, ((i % cols) * tw, (i // cols) * th_))
    sheet.save(path, quality=88)


def cmd_clip(a):
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    src = Path(a.src)
    sn = json.loads((S.MV / "感官.json").read_text("utf-8"))
    ch = sn["channels"]
    n = int(round(a.dur * FPS)) if a.dur else 10 ** 6
    frames = video_frames(src, a.src_t0, n)
    n = len(frames)
    dur = n / FPS
    tmp = Path(a.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    # 遮罩缓存（同一段源帧只推理一次；缓存在 --tmp 里，gitignore）
    mcache = tmp / f"masks_{src.stem}_{a.src_t0:.3f}_{n}.npy"
    cached = np.load(mcache) if mcache.exists() else None
    matter = None if cached is not None else get_matter()
    masks = []
    on = np.array(sn["events"]["onsets"])
    th = np.array(ch["threat"])
    H, W = frames[0].shape[:2]
    pals = a.palettes.split(",")
    name = a.name or f"17_面部瓦解_试_{a.ver}"
    pipes, vtmp = {}, {}
    for key in pals:
        vtmp[key] = tmp / f"{src.stem}_{a.ver}_{key}{'_wf' if a.waterfall else ''}_noaudio.mp4"
        pipes[key] = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                       "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                       "-crf", "16", str(vtmp[key])], stdin=subprocess.PIPE)
    hist = BoundaryHistory()
    off = a.sense_offset
    side = {"source": src.name, "src_t0": a.src_t0, "play_start": a.start, "dur": dur,
            "sense_offset": off, "palettes": pals, "waterfall": a.waterfall, "place_dx": a.place_dx,
            "mapping": {"threat": "n_bands = 4 + 22*threat; max_dx = 16 + 300*threat; p_shift = 0.25 + 0.5*threat",
                        "tension": "split d = 2 + 16*tension (R=+d, G=-d/2, B=-d, along normal)",
                        "hunger": "smear L = 40 + 520*hunger px",
                        "seed": f"new seed every {a.hold} frames and at onsets with threat >= p88 of the clip"},
            "frames": []}
    t_sense = [a.start + i / FPS - off for i in range(n)]
    i0, i1 = int(round(t_sense[0] * FPS)), int(round(t_sense[-1] * FPS)) + 1
    loc = on[(on >= t_sense[0]) & (on < t_sense[-1] + 1 / FPS)]
    strong = [o for o in loc if th[int(round(o * FPS))] >= np.percentile(th[i0:i1], 88)]
    seed, t0 = 1000, time.time()
    thumbs = {k: [] for k in pals}
    every = max(1, n // 16)
    for i, f in enumerate(frames):
        ts = t_sense[i]
        k = min(int(round(ts * FPS)), len(th) - 1)   # round：浮点 84 - 0.0 → 2015.9999 不能落到前一帧
        if i % a.hold == 0 or any(abs(o - ts) < 0.5 / FPS for o in strong):
            seed += 1
        p = params_from_sense(ch["threat"][k], ch["tension"][k], ch["hunger"][k],
                              Params(seed=seed, waterfall=a.waterfall, place_dx=a.place_dx))
        if cached is not None:
            mask, msrc = cached[i].astype(np.float32) / 255, "BiRefNet (cached)"
        else:
            mask, msrc = get_mask(matter, f)
            masks.append(mask)
        out, m = run(f, mask, p, hist=hist)
        for key in pals:
            im = to_im(PALETTES[key][1](out, m, seed))
            label(im, "", s=0.9, lines=[f"t = {ts + off:6.3f} s (play)   threat {p.sense['threat']:.2f}  "
                                        f"tension {p.sense['tension']:.2f}  hunger {p.sense['hunger']:.2f}   "
                                        f"mask {msrc}   palette {key}"] + p.hud())
            pipes[key].stdin.write(np.asarray(im).tobytes())
            if i % every == 0:
                thumbs[key].append((f"{ts + off:.2f}s  threat {p.sense['threat']:.2f}", im))
        side["frames"].append({"i": i, "t_play": round(ts + off, 3), **p.sense, "seed": seed,
                               "n_bands": p.n_bands, "max_dx": p.max_dx, "split_d": list(p.split_d),
                               "smear_len": round(p.smear_len, 1)})
    for key in pals:
        pipes[key].stdin.close()
        pipes[key].wait()
    if cached is None and matter is not None:
        np.save(mcache, (np.stack(masks) * 255 + 0.5).astype(np.uint8))
    print(f"{n} 帧 × {len(pals)} 配色  {time.time() - t0:.1f}s")
    for key in pals:
        suffix = f"_{PALETTES[key][0]}" if (len(pals) > 1 or a.name) else ""
        out = outname(S.MV / "样片" / f"{name}{suffix}.mp4")
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(vtmp[key]), "-ss", f"{a.start:.3f}",
                        "-t", f"{dur:.3f}", "-i", str(S.MUSIC), "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
                        "-b:a", "256k", "-t", f"{dur:.3f}", str(out)], check=True)
        Path(str(out) + ".json").write_text(json.dumps({**side, "palette": key}, ensure_ascii=False, indent=1), "utf-8")
        contact(thumbs[key], out.with_name(out.stem + "_接触表.jpg"))
        print("->", out)


def cmd_image(a):
    img = load_rgb(a.src)
    mask = (np.asarray(Image.open(a.mask).convert("L").resize(img.shape[1::-1]), np.float32) / 255
            if a.mask else lum_mask(img))
    p = Params(place_dx=a.place_dx, waterfall=False)
    out, m = run(img, mask, p)
    im = PALETTES[a.palette][1](out, m, p.seed)
    o = outname(a.out or Path(a.src).with_name(Path(a.src).stem + "_face_glitch.jpg"))
    to_im(im).save(o, quality=94)
    print("->", o)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s1 = sp.add_parser("sheet"); s1.add_argument("--t", type=float, default=4.0)
    s1b = sp.add_parser("palettes")
    s1b.add_argument("--t", type=float, default=4.0)
    s2 = sp.add_parser("clip")
    s2.add_argument("--src", default=str(SRC_VIDEO), help="源视频（她）")
    s2.add_argument("--start", type=float, default=84.0, help="歌里的播放时间（容器时间）= 第 0 帧")
    s2.add_argument("--dur", type=float, default=2.0, help="秒；0 = 整段源视频")
    s2.add_argument("--src-t0", type=float, default=2.5, help="源视频里从哪一秒取帧")
    s2.add_argument("--place-dx", type=int, default=180, help="把她往右挪多少像素（留出左边给拖影）")
    s2.add_argument("--palettes", default="ours", help="逗号分隔：" + ",".join(PALETTES))
    s2.add_argument("--name", default=None, help="输出名（不含配色后缀）；默认 17_面部瓦解_试_<ver>")
    s2.add_argument("--ver", default="v1")
    s2.add_argument("--hold", type=int, default=4, help="种子保持几帧（强起音处总会换）")
    s2.add_argument("--sense-offset", type=float, default=SENSE_OFFSET, help="感官.json 时间 + 这个 = 播放时间（当前 感官.json 已是播放时间 → 0）")
    s2.add_argument("--waterfall", action="store_true")
    s2.add_argument("--tmp", default=str(Path(__file__).resolve().parents[1] / "cache" / "face_glitch"))
    s3 = sp.add_parser("image"); s3.add_argument("src"); s3.add_argument("--mask"); s3.add_argument("--out")
    s3.add_argument("--place-dx", type=int, default=0); s3.add_argument("--palette", default="orig", choices=list(PALETTES))
    a = ap.parse_args()
    {"sheet": cmd_sheet, "palettes": cmd_palettes, "clip": cmd_clip, "image": cmd_image}[a.cmd](a)


if __name__ == "__main__":
    main()
