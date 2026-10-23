#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🖼现实瓦解-入口-实拍 / 生成视频 / 静帧里的「天、海、墙」慢慢化成像素流苏往下滴，人完好地待在里面（荒诞的平常）。
参考：用户给的草坡看海图 + `Projekt_Visual/_参考库/瓦解_真实场景1/2.jpg`、瓦解1–4。两种手法叠加：
  ① 列拉伸（stretch）：每列从断点 b(x) 起，把断点处往下 L(x)/S 那一小段像素拉成长 L(x) 的条
     （采样 s(y) = b + (y − b)/S，线性插值）；各列 b、L、S 不同 → 锯齿流苏。尖端 1 px 抗锯齿。
  ② 区间排序（sort_runs）：可碎区里亮度 ∈ [lo, hi] 的连续段按亮度排序（亮的在上 = 往下流），
     太暗 / 太亮不动；段外像素逐像素不变，段内像素多重集不变。
图像约定：float32 HxWx3 ∈ [0,1]；遮罩 HxW ∈ [0,1]。

哪里碎（遮罩）：可碎区 zone = 亮度带 [zone_lo, zone_hi]（软边）×（可选）色相带 ×（可选）手绘遮罩
  × (1 − 保护)，保护 = BiRefNet 抠出的主体（外扩 protect_grow px）。合成时主体按 alpha 盖在最上面
  —— 流苏从她身后流过，她不碎。⚠️不用深度模型（没装）。
方向：angle 度，流动方向 = (sin a, cos a)（x 右、y 下）：0 = 向下滴（默认）、90 = 向右、180 = 向上、270 = 向左；
  任意角：先把画面转到「流向 = 画布向下」的画布（双线性），处理完转回来。
⭐时间相干（逐帧乱跳 = 廉价）：
  · 断点 b(x,t)：可碎区在每列的累积质量的 q 分位（q = 低频场 + 每列固定的细锯齿）→ 断点长在可碎区里、跟着内容走；
    可碎区（zone_ema）、断点与可碎段下缘（b_ema）、排序层（sort_ema）都做时间 EMA；拉伸只采样断点所在可碎段内的像素。
  · 长度 L(x,t) = H·length·melt(t)·drive(t)·(0.15 + 0.85·F(x,t)^1.5)·col(x) + 鼓点下坠；
    F = 平滑一维值噪声场（x 方向 Catmull-Rom，时间方向在两张格点之间 smoothstep，整体慢漂 drift px/s）；
    col(x) = 每列固定的随机长度（成组：组宽几何分布 → 几根一起的流苏）。
  · 条内纹理 = 静态竖向噪声以 flow px/s 往下流（液体感），幅度 tex_amp。
  · 排序闪烁：排序输出是输入的顺序统计量 —— 对输入是 1-Lipschitz 的（输入变 ε，输出最多变 ε），
    闪烁只来自段边界跳动 → 段边界用 EMA 过的亮度 + 低频场决定，不用当前帧原始亮度。
驱动：
  · sense：感官.json（时间 = 播放时间）。张力 EMA（τ 1 s）→ drive = 0.55 + 0.45·tension；
    起音里威胁峰 ≥ burst_thr 且离上一个 ≥ burst_gap s 的 → 一次「局部下坠」：在有可碎区的某列附近 burst_w px 里，
    一部分列（不是整块）长度 +burst_amp·H·威胁，形状 (1 − e^(−τ/0.1 s))·e^(−τ/burst_tau)；不全局。
  · time：没有音乐（实拍）→ drive = 0.55 + 0.45·慢场；下坠按泊松过程（burst_rate 次/秒，固定种子）。
  · melt(t)：开场已经碎了 melt0，整段慢慢涨到 1（「一直在发生」，不是爆炸）。
配色：cold（实拍偏冷蓝）/ rwb（红白黑：灰阶高反差；红 = 原图里本来就红的 + 鼓点刚坠下那一截，1.2 s 褪掉）/ none。
静帧：先给慢推镜头（push 1.0 → 1.10，对数匀速 + 轻微横漂），不做锁死的静止。

用法（4_Eidolon 根目录，.venv）：
  python tools/reality_melt.py clip --src 视频或图片 [--t0 0] [--dur 6] [--width 0]
         [--drive sense --start 歌里播放秒数 | --drive time] [--palettes cold,rwb] [--angle 0]
         [--zone-lo 0.2 --zone-hi 1.0] [--hue 0.55,0.2]（色相中心,半宽，0–1）[--hand 手绘遮罩.png]
         [--protect birefnet|none] [--push 1.0,1.10] [--label 素材名] [--set length=0.4,stretch=20 ...] [--hud]
     → 作品/样片/现实瓦解/23_现实瓦解_<素材名>_<配色>_vN.mp4 + _接触表.jpg + .json（参数 + 逐帧量 + 闪烁比）
       不覆盖：自动取下一个 vN。sense 驱动时混入原曲音频。BiRefNet 遮罩缓存在 作品/cache/reality_melt/。
  python tools/reality_melt.py image 图 [--t 2.0] [--palettes cold,rwb]   # 单帧试参数（输出到 scratch 或 --out）
自检：tests/test_reality_melt.py。人读版：-给人类的解释/现实瓦解怎么做的.md（含「想要 X 就调 Y」）。
"""
import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, fields
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stills as S  # noqa: E402

FPS = 24
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
OUT_DIR = S.MV / "样片" / "现实瓦解"
CACHE = S.MV / "cache" / "reality_melt"     # 遮罩缓存 + 无声中间视频；--cache 可改


def lum(img):
    return img @ LUMA


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


# ---------------------------------------------------------------- 平滑随机场
class Field1D:
    """>v<🖼现实瓦解-场-一维平滑随机场 f(x, t) ∈ [0,1]。
    x：格距 scale px 的值噪声，Catmull-Rom 插值（C¹）；t：每 period 秒一张新格点，相邻两张 smoothstep 插值；
    整体以 drift px/s 横漂。多倍频：第二层 scale/3、period/2，权重 0.35。同 (seed, x, t) 结果确定。"""

    def __init__(self, W, scale=160.0, period=6.0, drift=0.0, seed=0, octaves=2):
        self.W, self.layers = W, []
        for o in range(octaves):
            sc, pe = scale / (3 ** o), period / (2 ** o)
            n = int(np.ceil(W / sc)) + 64
            self.layers.append((sc, pe, drift, seed * 1009 + o * 7919, n, 0.65 if o == 0 else 0.35 / (octaves - 1)))
        self._cache = {}

    def _slab(self, sd, k, n):
        key = (sd, k)
        if key not in self._cache:
            self._cache[key] = np.random.default_rng([sd, k & 0x7FFFFFFF]).random(n).astype(np.float64)
            if len(self._cache) > 64:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[key]

    def __call__(self, t, x=None):
        x = np.arange(self.W, dtype=np.float64) if x is None else np.asarray(x, np.float64)
        out = np.zeros_like(x)
        wsum = 0.0
        for sc, pe, dr, sd, n, w in self.layers:
            u = t / pe
            k = int(np.floor(u))
            a = smoothstep(u - k)
            V = (1 - a) * self._slab(sd, k, n) + a * self._slab(sd, k + 1, n)
            xx = (x + dr * t) / sc
            j = np.floor(xx).astype(np.int64)
            f = xx - j
            p0, p1, p2, p3 = (V[(j + d) % n] for d in (-1, 0, 1, 2))
            v = p1 + 0.5 * f * (p2 - p0 + f * (2 * p0 - 5 * p1 + 4 * p2 - p3 + f * (3 * (p1 - p2) + p3 - p0)))
            out += w * v
            wsum += w
        return np.clip(out / wsum, 0.0, 1.0)


def column_groups(W, seed, mean_w=3.0, tail=0.12):
    """每列固定的随机值（成组：组宽 ~ 几何分布，均值 mean_w px）→ 几根并排的流苏一起长短。
    返回 (值 ∈ [0,1], 组号)；tail 的组拿到重尾的长值（偶尔一根特别长）。"""
    rng = np.random.default_rng(seed)
    vals, gid, x, g = np.zeros(W), np.zeros(W, np.int64), 0, 0
    while x < W:
        w = int(rng.geometric(1.0 / mean_w))
        v = rng.random() ** 0.7
        if rng.random() < tail:
            v = 1.0
        vals[x:x + w], gid[x:x + w] = v, g
        x, g = x + w, g + 1
    return vals, gid


# ---------------------------------------------------------------- 两种手法
def sort_runs(img, sortable, descending=True, key=None):
    """>v<🖼现实瓦解-排序-列内区间排序：sortable（HxW bool）为 True 的竖向连续段内按亮度排序。
    descending=True：段内亮的在上（往下流 = 越往下越暗）。sortable 为 False 的像素原样不动（逐像素相等），
    每段像素的多重集不变。key：排序用的亮度（默认 = 当前帧亮度）。全矢量化：每个像素一个段号，
    排序键 = 段号 + 0.999·(1 − 亮度)，按列 argsort。"""
    H, W = sortable.shape
    k = lum(img) if key is None else key
    k = np.clip(k, 0, 1).astype(np.float64)
    s = sortable.astype(bool)
    start = np.ones((H, W), bool)
    start[1:] = ~(s[1:] & s[:-1])          # 连续可排序像素属于同一段；不可排序像素各自一段
    rid = np.cumsum(start, axis=0).astype(np.float64)
    kk = (1 - k) if descending else k
    order = np.argsort(rid + 0.999 * kk, axis=0, kind="stable")
    return np.take_along_axis(img, order[..., None], axis=0)


def breakpoints(zone, q):
    """>v<🖼现实瓦解-断点-每列断点 = 可碎区（软）在这一列的累积质量的 q 分位（q ∈ [0,1]，W 长）。
    返回 (b 浮点行号, mass = 这一列可碎区的总量 px)。断点因此长在可碎区里、随内容走。"""
    H, W = zone.shape
    c = np.cumsum(zone, axis=0)
    mass = c[-1]
    tgt = np.maximum(q * mass, 1e-6)          # 行 y 占 [y, y+1)；b = 累积质量达到 tgt 的连续位置
    idx = (c < tgt[None, :]).sum(0)        # 第一行 c ≥ tgt
    idx = np.clip(idx, 0, H - 1)
    prev = np.where(idx > 0, c[np.maximum(idx - 1, 0), np.arange(W)], 0.0)
    cur = c[idx, np.arange(W)]
    frac = np.where(cur > prev, (tgt - prev) / np.maximum(cur - prev, 1e-9), 0.0)
    b = idx + np.clip(frac, 0, 1)
    return np.clip(b, 0, H - 1), mass


def stretch(img, b, L, S_, tex=None, tex_amp=0.0, s_max=None):
    """>v<🖼现实瓦解-拉伸-列拉伸：y ∈ [b, b+L) 处采样 s(y) = b + (y − b)/S（线性插值），尖端 1 px 抗锯齿。
    b、L、S：W 长（S ≥ 1）。tex：HxW 静态纹理 ∈ [0,1]（已按时间流动过），条内亮度 × (1 + tex_amp·(2·tex − 1))。
    s_max：W 长，采样不越过这一行（= 断点所在可碎段的下缘 → 条的颜色只来自可碎区，不把主体边缘拉出来）。
    返回 (输出, alpha HxW：条覆盖程度)。L = 0 ⇒ 输出 == 输入。"""
    H, W = img.shape[:2]
    y = np.arange(H, dtype=np.float64)[:, None]
    d = y - b[None, :]
    alpha = np.clip(d + 0.5, 0, 1) * np.clip(L[None, :] - d, 0, 1)
    alpha = np.where(L[None, :] > 0, alpha, 0.0)
    s = b[None, :] + np.maximum(d, 0) / np.maximum(S_[None, :], 1.0)
    if s_max is not None:
        s = np.minimum(s, np.maximum(s_max[None, :], b[None, :]))
    s = np.clip(s, 0, H - 1)
    i0 = np.floor(s).astype(np.int64)
    i1 = np.minimum(i0 + 1, H - 1)
    f = (s - i0)[..., None]
    cols = np.arange(W)[None, :]
    streak = img[i0, cols] * (1 - f) + img[i1, cols] * f
    if tex is not None and tex_amp > 0:
        streak = streak * (1 + tex_amp * (2 * tex[..., None] - 1))
    a = alpha[..., None]
    out = img * (1 - a) + np.clip(streak, 0, 1) * a
    return out.astype(np.float32), alpha.astype(np.float32)


# ---------------------------------------------------------------- 任意方向：画布
class Canvas:
    """流向 d = (sin a, cos a) 的画布：画布向下 = 流向。屏幕点 p = cs + (u − ccu)·e + (v − ccv)·d，e = (cos a, −sin a)。
    a 是 90 的倍数时映射落在整数像素上（逐像素精确）。a = 0 直接旁路。"""

    def __init__(self, H, W, angle):
        self.H, self.W, self.a = H, W, float(angle) % 360.0
        r = np.deg2rad(self.a)
        self.d = np.array([np.sin(r), np.cos(r)])      # (x, y)
        self.e = np.array([np.cos(r), -np.sin(r)])
        self.d[np.abs(self.d) < 1e-12] = 0
        self.e[np.abs(self.e) < 1e-12] = 0
        self.Wc = int(np.ceil(abs(W * self.e[0]) + abs(H * self.e[1]) - 1e-9))
        self.Hc = int(np.ceil(abs(W * self.d[0]) + abs(H * self.d[1]) - 1e-9))
        self.cs = np.array([(W - 1) / 2, (H - 1) / 2])
        self.cc = np.array([(self.Wc - 1) / 2, (self.Hc - 1) / 2])
        self.bypass = self.a == 0.0

    def _sample(self, arr, X, Y, order=1):
        from scipy.ndimage import map_coordinates
        if arr.ndim == 2:
            return map_coordinates(arr, [Y, X], order=order, mode="constant", cval=0.0).astype(np.float32)
        return np.stack([map_coordinates(arr[..., c], [Y, X], order=order, mode="constant", cval=0.0)
                         for c in range(arr.shape[2])], -1).astype(np.float32)

    def fwd(self, arr):
        """屏幕 → 画布"""
        if self.bypass:
            return arr
        v, u = np.mgrid[0:self.Hc, 0:self.Wc].astype(np.float64)
        X = self.cs[0] + (u - self.cc[0]) * self.e[0] + (v - self.cc[1]) * self.d[0]
        Y = self.cs[1] + (u - self.cc[0]) * self.e[1] + (v - self.cc[1]) * self.d[1]
        return self._sample(arr, np.round(X, 9), np.round(Y, 9))

    def inv(self, arr):
        """画布 → 屏幕"""
        if self.bypass:
            return arr
        y, x = np.mgrid[0:self.H, 0:self.W].astype(np.float64)
        px, py = x - self.cs[0], y - self.cs[1]
        U = self.cc[0] + px * self.e[0] + py * self.e[1]
        V = self.cc[1] + px * self.d[0] + py * self.d[1]
        return self._sample(arr, np.round(U, 9), np.round(V, 9))


# ---------------------------------------------------------------- 参数
@dataclass
class Params:
    angle: float = 0.0          # 流向（度）：0 下 / 90 右 / 180 上 / 270 左
    zone_lo: float = 0.18       # 可碎区亮度带（软边 zone_soft）
    zone_hi: float = 1.01
    zone_soft: float = 0.06
    hue: tuple = None           # (色相中心, 半宽) ∈ [0,1]；None = 不限色相
    sat_min: float = 0.0        # 色相限定时的最低饱和度
    protect_grow: int = 6       # 保护遮罩外扩 px
    zone_ema: float = 0.15      # 可碎区 / 排序键时间 EMA 的新帧权重（1 = 不平滑）
    b_ema: float = 0.2          # 断点 / 可碎段下缘的时间 EMA 新帧权重（源视频噪声 → 断点不抖；镜头快时调大）
    length: float = 0.50        # 流苏长度尺度（× 画高）
    melt0: float = 0.55         # 开场的碎裂程度（相对整段末尾 = 1）
    q_lo: float = 0.04          # 断点分位的范围：越小断点越靠可碎区上缘
    q_hi: float = 0.55
    q_fine: float = 0.16        # 每列固定的分位锯齿（细流苏）
    stretch: float = 14.0       # 拉伸倍数中值 S（条里看得到的原内容 = L / S）
    stretch_spread: float = 0.9 # S 的每列对数散布
    field_scale: float = 150.0  # 低频场格距（px，按 720 画高）
    field_period: float = 7.0   # 低频场换一张格点的时间（s）—— 越大越慢
    drift: float = 6.0          # 场横漂（px/s）
    flow: float = 22.0          # 条内纹理下流速度（px/s）
    tex_amp: float = 0.10
    sort_amt: float = 0.5       # 排序区占可碎区的比例（低频场阈值）
    sort_lo: float = 0.22       # 排序亮度区间
    sort_hi: float = 0.92
    sort_seg: float = 56.0      # 排序段最长（px，按 720 画高）：每列错开的固定缝把长段切开 → 段边界跳一次只乱一小段
    sort_ema: float = 0.5       # 排序层的时间 EMA 新帧权重（只作用在排序区；1 = 不平滑）
    burst_thr: float = 0.6      # 起音威胁峰阈值（sense）
    burst_gap: float = 1.5      # 两次下坠最短间隔（s）
    burst_rate: float = 0.22    # time 驱动：泊松下坠率（次/s）
    burst_amp: float = 0.22     # 一次下坠的长度（× 画高 × 威胁）
    burst_w: float = 46.0       # 下坠影响的横向范围（px，按 720 画高）
    burst_frac: float = 0.45    # 范围内真的坠下去的列比例
    burst_tau: float = 4.0      # 坠下去之后慢慢缩回的时间常数（s）
    seed: int = 7
    naive: bool = False         # A/B 用：每帧换场 + 不做 EMA（= 逐帧乱跳的对照）


def parse_set(s):
    """「k=v,k=v」→ 覆盖 Params 字段（按默认值的类型转换；hue 用 h:w）"""
    out = {}
    if not s:
        return out
    dflt = Params()
    for kv in s.split(","):
        k, v = (x.strip() for x in kv.split("="))
        if not hasattr(dflt, k):
            raise SystemExit(f"未知参数 {k}")
        d = getattr(dflt, k)
        if k == "hue":
            out[k] = tuple(float(x) for x in v.split(":"))
        elif isinstance(d, bool):
            out[k] = v.lower() in ("1", "true", "yes")
        else:
            out[k] = type(d)(float(v)) if isinstance(d, (int, float)) else v
    return out


def rgb_to_hs(img):
    mx, mn = img.max(-1), img.min(-1)
    c = mx - mn
    sat = np.where(mx > 1e-6, c / np.maximum(mx, 1e-6), 0)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    h = np.zeros_like(mx)
    cc = np.maximum(c, 1e-6)
    h = np.where(mx == r, ((g - b) / cc) % 6, h)
    h = np.where(mx == g, (b - r) / cc + 2, h)
    h = np.where(mx == b, (r - g) / cc + 4, h)
    return np.where(c > 1e-6, h / 6, 0), sat


def zone_raw(img, protect, p: Params, hand=None):
    """可碎区（未平滑）：亮度带 × 色相带 × 手绘 × (1 − 保护)"""
    lm = lum(img)
    z = smoothstep((lm - p.zone_lo) / p.zone_soft + 0.5) * smoothstep((p.zone_hi - lm) / p.zone_soft + 0.5)
    if p.hue is not None:
        h, s = rgb_to_hs(img)
        dh = np.abs(((h - p.hue[0]) + 0.5) % 1.0 - 0.5)
        z = z * smoothstep((p.hue[1] - dh) / 0.04 + 0.5) * smoothstep((s - p.sat_min) / 0.05 + 0.5)
    if hand is not None:
        z = z * hand
    if protect is not None:
        z = z * (1 - protect)
    return z.astype(np.float32)


def grow(mask, r):
    if r <= 0:
        return mask
    from scipy.ndimage import maximum_filter, gaussian_filter
    m = maximum_filter(mask, size=2 * r + 1)
    return np.clip(gaussian_filter(m, r / 2), 0, 1).astype(np.float32)


# ---------------------------------------------------------------- 状态机：逐帧
class Melter:
    """>v<🖼现实瓦解-逐帧-持有时间状态（EMA、场、事件）。step(img, protect, t, drive, bursts) → 输出 + 信息。"""

    def __init__(self, H, W, p: Params, dur):
        self.p, self.H, self.W, self.dur = p, H, W, max(dur, 1e-6)
        self.cv = Canvas(H, W, p.angle)
        Hc, Wc = self.cv.Hc, self.cv.Wc
        self.Hc, self.Wc = Hc, Wc
        u = H / 720.0                              # 像素单位按 720 画高
        self.u = u
        sd = p.seed
        self.F_len = Field1D(Wc, p.field_scale * u, p.field_period, p.drift * u, sd + 1)
        self.F_q = Field1D(Wc, p.field_scale * 1.3 * u, p.field_period * 1.3, -p.drift * 0.7 * u, sd + 2)
        self.F_sort = Field1D(Wc, p.field_scale * 0.8 * u, p.field_period * 0.9, p.drift * 0.5 * u, sd + 3)
        self.col_len, self.gid = column_groups(Wc, sd + 4, mean_w=2.2 + 1.5 * u)
        self.col_q = column_groups(Wc, sd + 5, mean_w=1.8 + u)[0]
        rng = np.random.default_rng(sd + 6)
        ng = int(self.gid.max()) + 1
        self.col_S = (p.stretch * np.exp(p.stretch_spread * rng.standard_normal(ng)))[self.gid].clip(2, 200)
        self.col_burst = rng.random(ng)[self.gid]                      # 下坠时哪几组真的坠
        self.seg_off = (rng.random(ng) * 1000)[self.gid]                # 排序缝每列（组）错开
        trng = np.random.default_rng(sd + 7)                            # 条内纹理：每列独立、竖向相关 ~ 30 px
        from scipy.ndimage import gaussian_filter1d
        P = int(512 * u)
        t0 = gaussian_filter1d(trng.random((P, Wc)).astype(np.float32), 9 * u, axis=0, mode="wrap")
        t0 = (t0 - t0.min()) / (t0.max() - t0.min() + 1e-9)
        self.tex0, self.P = t0, P
        self.zone_s = None
        self.key_s = None
        self.b_s = self.end_s = None
        self.bursts = []                     # (t0, x0, amp)
        self.prev_out = self.prev_src = None

    def add_burst(self, t0, strength, mass):
        p = self.p
        rng = np.random.default_rng([p.seed, int(t0 * 1000)])
        ok = np.flatnonzero(mass > 0.25 * mass.max()) if mass.max() > 0 else np.arange(self.Wc)
        x0 = int(rng.choice(ok)) if len(ok) else self.Wc // 2
        self.bursts.append((float(t0), x0, float(strength)))
        return x0

    def burst_len(self, t):
        """各列因鼓点下坠多出来的长度（px）和「刚坠下」的新鲜度"""
        p = self.p
        x = np.arange(self.Wc)
        add = np.zeros(self.Wc)
        fresh = np.zeros(self.Wc)
        w = p.burst_w * self.u
        for t0, x0, s in self.bursts:
            tau = t - t0
            if tau < 0:
                continue
            shape = np.exp(-0.5 * ((x - x0) / (w / 2.0)) ** 2)
            pick = (self.col_burst < p.burst_frac).astype(np.float64)
            fall = (1 - np.exp(-tau / 0.10)) * np.exp(-tau / p.burst_tau)
            a = p.burst_amp * self.H * s * shape * pick * fall
            add += a
            fresh = np.maximum(fresh, (a > 1) * np.exp(-tau / 1.2))
        return add, fresh

    def step(self, img, protect, t, drive=1.0, gate=1.0, hand=None):
        """img 屏幕帧；protect 主体 alpha（None = 不保护）；t 秒（本段内时间）；drive ∈ [0,1] 总体拉伸；gate 效果强度（0 ⇒ 恒等）"""
        p, cv = self.p, self.cv
        tt = t + (1000.0 * (hash((t, 1)) % 97) if p.naive else 0.0)
        pr = grow(protect, int(round(p.protect_grow * self.u))) if protect is not None else None
        zr = zone_raw(img, pr, p, hand)
        lm = lum(img)
        a = 1.0 if p.naive else p.zone_ema
        self.zone_s = zr if self.zone_s is None else (1 - a) * self.zone_s + a * zr
        self.key_s = lm if self.key_s is None else (1 - a) * self.key_s + a * lm
        Ic = cv.fwd(img)
        Zc = np.clip(cv.fwd(self.zone_s), 0, 1)
        Kc = cv.fwd(self.key_s)
        Hc, Wc = self.Hc, self.Wc
        melt = p.melt0 + (1 - p.melt0) * smoothstep(t / self.dur)
        # 排序：可碎区 × 亮度区间（EMA 键）× 低频场选区
        fs = self.F_sort(tt)
        sel = (fs[None, :] < p.sort_amt * gate) & (Zc > 0.5) & (Kc >= p.sort_lo) & (Kc <= p.sort_hi)
        seg = max(4.0, p.sort_seg * self.u)
        seam = ((np.arange(Hc)[:, None] + self.seg_off[None, :]) % seg) < 1.0
        sel = sel & ~seam
        if gate > 0 and sel.any():
            sc = sort_runs(Ic, sel)
            a2 = 1.0 if p.naive else p.sort_ema
            if a2 < 1.0 and getattr(self, "sort_s", None) is not None:
                sc = np.where(sel[..., None], (1 - a2) * self.sort_s + a2 * sc, sc)
            self.sort_s = sc
            sorted_c = sc
        else:
            sorted_c = Ic
        # 断点 / 长度
        q = np.clip(p.q_lo + (p.q_hi - p.q_lo) * self.F_q(tt) + p.q_fine * (self.col_q - 0.5), 0.0, 1.0)
        b, mass = breakpoints(Zc, q)
        ab = 1.0 if p.naive else p.b_ema
        self.b_s = b if self.b_s is None else (1 - ab) * self.b_s + ab * b
        b = self.b_s
        mw = smoothstep(mass / (0.06 * Hc))
        Lb = self.H * p.length * melt * drive * (0.15 + 0.85 * self.F_len(tt) ** 1.5) * self.col_len * mw
        add, fresh = self.burst_len(t)
        L = (Lb + add * mw) * gate
        sh = p.flow * self.u * t                      # 亚像素下流（整数步进会一帧动一帧不动 = 抖）
        k0, fr_ = int(np.floor(sh)), sh - np.floor(sh)
        tex = (1 - fr_) * self.tex0[(np.arange(Hc) - k0) % self.P] + fr_ * self.tex0[(np.arange(Hc) - k0 - 1) % self.P]
        yy = np.arange(Hc)[:, None]
        below = (yy >= np.ceil(b)[None, :]) & (Zc < 0.5)
        run_end = np.where(below.any(0), below.argmax(0), Hc) - 1.0          # 断点所在可碎段的最后一行
        self.end_s = run_end if self.end_s is None else (1 - ab) * self.end_s + ab * run_end
        run_end = self.end_s
        out_c, alpha = stretch(sorted_c, b, L, self.col_S, tex, p.tex_amp, s_max=run_end)
        # 新鲜下坠段（rwb 的红）：每列 y ∈ [b + Lb, b + L)
        y = np.arange(Hc)[:, None]
        fresh_c = ((y >= (b + Lb * gate)[None, :]) & (y < (b + L)[None, :])) * fresh[None, :] * alpha
        out = cv.inv(out_c) if not cv.bypass else out_c
        alpha_s = cv.inv(alpha) if not cv.bypass else alpha
        fresh_s = cv.inv(fresh_c.astype(np.float32)) if not cv.bypass else fresh_c.astype(np.float32)
        sortm = cv.inv(sel.astype(np.float32)) if not cv.bypass else sel.astype(np.float32)
        if not cv.bypass:      # 画布外 / 旋转边缘：用原图
            cover = cv.inv(np.ones((Hc, Wc), np.float32))
            out = out * cover[..., None] + img * (1 - cover[..., None])
        if protect is not None:
            out = img * protect[..., None] + out * (1 - protect[..., None])
            fresh_s = fresh_s * (1 - protect)
        info = {"melt": float(melt), "drive": float(drive), "L_mean_px": float(L[mw > 0.5].mean()) if (mw > 0.5).any() else 0.0,
                "L_max_px": float(L.max()), "drip_cover": float(alpha_s.mean()), "sort_cover": float(sortm.mean()),
                "b_mean": float(b[mw > 0.5].mean()) if (mw > 0.5).any() else 0.0}
        return np.clip(out, 0, 1).astype(np.float32), {"alpha": alpha_s, "fresh": fresh_s, "sort": sortm, "mass": mass,
                                                       "b": b, "L": L, "info": info}


# ---------------------------------------------------------------- 配色
def palette(img, name, fresh=None, protect=None):
    """cold：实拍偏冷蓝（亮度保留、留 25 % 原饱和）；rwb：灰阶高反差，红只来自原图本来的红 + 刚坠下的那截。"""
    if name == "none":
        return img
    lm = lum(img)[..., None]
    if name == "cold":
        # 渐变映射（v3 起；v2 是线性染蓝 → 发闷）：同 rwb 的 S 曲线，暗部深海军蓝、亮部冰白，再留 15 % 原色度
        tone = smoothstep((np.clip(lm, 0, 1) - 0.02) / 0.9) ** 0.92
        navy, ice = np.array([0.008, 0.022, 0.055], np.float32), np.array([0.80, 0.91, 1.0], np.float32)
        base = navy * (1 - tone) + ice * tone
        out = base + 0.15 * (img - lm)
        if fresh is not None:
            out = out + fresh[..., None] * 0.10 * np.array([0.7, 0.9, 1.0], np.float32)
        return np.clip(out, 0, 1)
    if name == "rwb":
        x = np.clip(lm, 0, 1)
        tone = smoothstep((x - 0.03) / 0.9) ** 0.9                          # 黑位压一点、S 曲线
        out = np.repeat(tone, 3, -1) * np.array([0.97, 0.97, 0.96], np.float32)
        h, s = rgb_to_hs(img)
        redness = smoothstep((0.06 - np.abs(((h - 0.0) + 0.5) % 1.0 - 0.5)) / 0.04 + 0.5) * smoothstep((s - 0.5) / 0.15)
        red = np.asarray(S.RED, np.float32)
        r = redness[..., None]
        out = out * (1 - r) + red * np.clip(tone * 1.6, 0, 1) * r
        if fresh is not None:
            f = np.clip(fresh, 0, 1)[..., None] * 0.85
            out = out * (1 - f) + red * np.clip(0.35 + tone, 0, 1) * f
        return np.clip(out, 0, 1)
    raise ValueError(name)


# ---------------------------------------------------------------- 源：视频 / 静帧慢推
def to_im(x):
    return Image.fromarray((np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8))


def is_image(path):
    return Path(path).suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def push_frame(im: Image.Image, u, out_wh, z=(1.0, 1.10), drift=(0.015, -0.008)):
    """静帧慢推：u ∈ [0,1]，缩放对数匀速 z0 → z1，中心轻微横漂（画宽比例）。先按输出比例裁到最大。"""
    W0, H0 = im.size
    ow, oh = out_wh
    ar = ow / oh
    cw, ch = (W0, W0 / ar) if W0 / H0 < ar else (H0 * ar, H0)
    zz = z[0] * (z[1] / z[0]) ** u
    w, h = cw / zz, ch / zz
    cx = W0 / 2 + drift[0] * W0 * (u - 0.5)
    cy = H0 / 2 + drift[1] * H0 * (u - 0.5)
    box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    return im.transform((ow, oh), Image.EXTENT, box, Image.BICUBIC)


def iter_source(src, t0, n, width, push):
    """逐帧 (RGB float, PIL)；视频按 width 缩放（0 = 原尺寸），图片做慢推。"""
    if is_image(src):
        im = Image.open(src).convert("RGB")
        ow = width or 1280
        oh = int(round(ow * 9 / 16 / 2)) * 2
        for i in range(n):
            fr = push_frame(im, i / max(n - 1, 1), (ow, oh), push)
            yield np.asarray(fr, np.float32) / 255
        return
    import imageio_ffmpeg
    rd = imageio_ffmpeg.read_frames(str(src), input_params=["-ss", f"{t0:.3f}"])
    meta = next(rd)
    W, H = meta["size"]
    k = 0
    for buf in rd:
        fr = Image.frombytes("RGB", (W, H), buf)
        if width and width != W:
            fr = fr.resize((width, int(round(H * width / W / 2)) * 2), Image.LANCZOS)
        yield np.asarray(fr, np.float32) / 255
        k += 1
        if k >= n:
            break
    rd.close()


def src_len(src):
    if is_image(src):
        return None
    import imageio_ffmpeg
    n, _ = imageio_ffmpeg.count_frames_and_secs(str(src))
    return n


def protect_masks(src, frames_fn, n, key, kind):
    """BiRefNet 逐帧（静帧：只算一次原图再跟着慢推走）；缓存 uint8；时间上 EMA 0.5 压一点闪。"""
    if kind == "none":
        return None
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{key}.npy"
    if f.exists():
        m = np.load(f)
        if len(m) >= n:
            return m[:n]
    import face_glitch as Fg
    mt = Fg.get_matter()
    out = []
    for fr in frames_fn():
        a = np.asarray(mt(to_im(fr)), np.uint8) if mt is not None else (Fg.lum_mask(fr) * 255).astype(np.uint8)
        out.append(a)
    m = np.stack(out)
    np.save(f, m)
    return m


# ---------------------------------------------------------------- 驱动
def load_sense():
    return json.loads((S.MV / "感官.json").read_text("utf-8"))


def sense_drive(sn, start, n, p: Params):
    """张力 EMA（τ 1 s）→ drive；起音里威胁峰 ≥ thr、间隔 ≥ gap 的 → 下坠事件（本段内时间, 强度）"""
    ch = sn["channels"]
    te, th = np.asarray(ch["tension"]), np.asarray(ch["threat"])
    on = np.asarray(sn["events"]["onsets"])
    tp = start + np.arange(n) / FPS
    k = np.minimum(np.round(tp * FPS).astype(int), len(te) - 1)
    e = float(te[max(k[0] - 24, 0):k[0] + 1].mean())
    drive = []
    for kk in k:
        e += (te[kk] - e) / FPS
        drive.append(0.55 + 0.45 * e)
    ev, last = [], -1e9
    for o in on[(on >= start - 0.01) & (on < start + n / FPS)]:
        ko = min(int(round(o * FPS)), len(th) - 1)
        pk = float(th[ko:ko + 3].max())
        if pk >= p.burst_thr and o - last >= p.burst_gap:
            ev.append((round(float(o - start), 3), round(pk, 3)))
            last = o
    return np.asarray(drive), ev


def time_drive(n, p: Params):
    F = Field1D(1, 1.0, 5.0, 0.0, p.seed + 11, octaves=1)
    t = np.arange(n) / FPS
    drive = np.array([0.55 + 0.45 * F(tt, np.zeros(1))[0] for tt in t])
    rng = np.random.default_rng(p.seed + 12)
    ev, tt = [], rng.exponential(1 / p.burst_rate) * 0.6
    while tt < n / FPS:
        ev.append((round(float(tt), 3), round(float(0.6 + 0.4 * rng.random()), 3)))
        tt += max(p.burst_gap, rng.exponential(1 / p.burst_rate))
    return drive, ev


# ---------------------------------------------------------------- 输出
def label(im, lines, s=1.0, color=(220, 220, 220)):
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(S.FONT, max(10, int(14 * s)))
    for j, line in enumerate(lines):
        d.text((int(8 * s), int(6 * s) + int(17 * s) * j), line, font=f, fill=color)
    return im


def contact(thumbs, plot, path, cols=4):
    tw = 400
    th_ = int(round(tw * thumbs[0][1].height / thumbs[0][1].width))
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * th_ + plot.height), (8, 8, 10))
    for i, (lab, im) in enumerate(thumbs):
        t = im.resize((tw, th_), Image.LANCZOS)
        label(t, [lab], s=0.75)
        sheet.paste(t, ((i % cols) * tw, (i // cols) * th_))
    sheet.paste(plot.resize((cols * tw, plot.height)), (0, rows * th_))
    sheet.save(path, quality=88)


def series_plot(rows, events, dur, w=1600, h=170):
    """底部条带：L 均值（白）、闪烁比（红，1 = 和源一样）、下坠事件（竖线）"""
    im = Image.new("RGB", (w, h), (14, 14, 16))
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(S.FONT, 12)
    t = np.array([r["t"] for r in rows])
    L = np.array([r["L_mean_px"] for r in rows])
    fl = np.array([r.get("flicker_ratio", np.nan) for r in rows])
    X = lambda tt: int(40 + (w - 60) * tt / max(dur, 1e-6))  # noqa: E731
    for te, s in events:
        d.line([(X(te), 10), (X(te), h - 20)], fill=(120, 40, 40), width=1)
    def poly(v, vmax, col):
        pts = [(X(tt), int(h - 20 - (h - 40) * min(vv / vmax, 1))) for tt, vv in zip(t, v) if np.isfinite(vv)]
        if len(pts) > 1:
            d.line(pts, fill=col, width=2)
    poly(L, max(L.max(), 1), (220, 220, 220))
    poly(fl, 3.0, (230, 70, 70))
    d.text((44, 4), f"white = mean drip length (max {L.max():.0f} px)   red = flicker ratio (2nd time diff) out/src in melt zone (top = 3.0)   "
                    f"dark red lines = bursts", font=f, fill=(170, 170, 170))
    d.text((44, h - 16), "0 s", font=f, fill=(120, 120, 120))
    d.text((w - 70, h - 16), f"{dur:.2f} s", font=f, fill=(120, 120, 120))
    return im


def next_version(stem):
    v = 1
    while (OUT_DIR / f"{stem}_v{v}.mp4").exists() or (OUT_DIR / f"{stem}_v{v}.json").exists():
        v += 1
    return v


def cmd_clip(a):
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    src = Path(a.src)
    p = Params(**{**({"hue": tuple(float(v) for v in a.hue.split(","))} if a.hue else {}),
                  "angle": a.angle, "zone_lo": a.zone_lo, "zone_hi": a.zone_hi, "seed": a.seed, **parse_set(a.set)})
    total = src_len(src)
    n = int(round(a.dur * FPS)) if a.dur else (total - int(round(a.t0 * FPS)))
    if total is not None:
        n = min(n, total - int(round(a.t0 * FPS)))
    push = tuple(float(v) for v in a.push.split(","))
    frames_fn = lambda: iter_source(src, a.t0, n, a.width, push)  # noqa: E731
    first = next(iter_source(src, a.t0, 1, a.width, push))
    H, W = first.shape[:2]
    key = f"{src.stem}_{a.t0:.2f}_{n}_{W}x{H}" + (f"_push{push[0]}-{push[1]}" if is_image(src) else "")
    masks = protect_masks(src, frames_fn, n, key, a.protect)
    hand = None
    if a.hand:
        hand = np.asarray(Image.open(a.hand).convert("L").resize((W, H), Image.BILINEAR), np.float32) / 255
    if a.drive == "sense":
        sn = load_sense()
        drive, events = sense_drive(sn, a.start, n, p)
    else:
        drive, events = time_drive(n, p)
    dur = n / FPS
    m = Melter(H, W, p, dur)
    pals = [s for s in a.palettes.split(",") if s]
    stems = {pl: f"23_现实瓦解_{a.label or src.stem}_{pl}" for pl in pals}
    vers = {pl: next_version(stems[pl]) for pl in pals}
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vtmp = {pl: CACHE / f"{stems[pl]}_v{vers[pl]}_noaudio.mp4" for pl in pals}
    pipes = {pl: subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                                   "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "15",
                                   str(vtmp[pl])], stdin=subprocess.PIPE) for pl in pals}
    rows, thumbs = [], {pl: [] for pl in pals}
    every = max(1, n // 12)
    ev_i, fired = 0, []
    hist_o, hist_s = [], []
    jerk_sum = [0.0, 0.0]
    t_start = time.time()
    for i, fr in enumerate(frames_fn()):
        t = i / FPS
        while ev_i < len(events) and events[ev_i][0] <= t + 1e-6:
            pm = protect_mask_at(masks, i)
            zr = zone_raw(fr, pm, p, hand)
            x0 = m.add_burst(events[ev_i][0], events[ev_i][1], m.cv.fwd(zr).sum(0) if not m.cv.bypass else zr.sum(0))
            fired.append([events[ev_i][0], events[ev_i][1], x0])
            ev_i += 1
        pm = protect_mask_at(masks, i)
        out, inf = m.step(fr, pm, t, float(drive[i]), 1.0, hand)
        region = (inf["alpha"] > 0.5) | (inf["sort"] > 0.5)
        row = {"i": i, "t": round(t, 3), **{k: round(v, 3) for k, v in inf["info"].items()}}
        if pm is not None:
            region &= pm < 0.5
        # 闪烁 = 时间二阶差分 |o_t − 2o_{t−1} + o_{t−2}|（匀速流动 / 平移 ≈ 0，逐帧乱跳 → 大），在效果区里，和源比
        if len(hist_o) == 2 and region.any():
            jo = np.abs(out - 2 * hist_o[1] + hist_o[0]).mean(-1)[region].mean()
            js = np.abs(fr - 2 * hist_s[1] + hist_s[0]).mean(-1)[region].mean()
            jerk_sum[0] += float(jo)
            jerk_sum[1] += float(js)
            row["jerk_out"] = round(float(jo), 5)
            row["jerk_src"] = round(float(js), 5)
            row["flicker_ratio"] = round(float(jo / max(js, 1e-4)), 3)
        hist_o, hist_s = (hist_o + [out])[-2:], (hist_s + [fr])[-2:]
        rows.append(row)
        for pl in pals:
            o = palette(out, pl, inf["fresh"], pm)
            im = to_im(o)
            if a.hud:
                label(im, [f"t {t:5.2f}s  drive {drive[i]:.2f}  melt {inf['info']['melt']:.2f}  L {inf['info']['L_mean_px']:.0f}px"], s=0.8,
                      color=(140, 140, 140))
            pipes[pl].stdin.write(np.asarray(im).tobytes())
            near_ev = any(0 <= t - e[0] < 0.25 and i % 3 == 0 for e in events)
            if i % every == 0 or i == n - 1 or (near_ev and len(thumbs[pl]) < 15):
                thumbs[pl].append((i, f"{t:.2f}s  L {inf['info']['L_mean_px']:.0f}px  drive {drive[i]:.2f}", im))
        if i % 24 == 0:
            print(f"  {i}/{n}  {time.time() - t_start:.0f}s", flush=True)
    for pl in pals:
        pipes[pl].stdin.close()
        pipes[pl].wait()
    fr_ratio = [r["flicker_ratio"] for r in rows if "flicker_ratio" in r]
    side = {"tool": "tools/reality_melt.py clip", "source": str(src), "t0": a.t0, "frames_n": n, "dur": round(dur, 3),
            "size": [W, H], "drive": a.drive, "play_start": a.start if a.drive == "sense" else None,
            "protect": a.protect, "hand": a.hand, "push": push if is_image(src) else None, "params": asdict(p),
            "bursts_fired": fired,
            "flicker_ratio_median": round(float(np.median(fr_ratio)), 3) if fr_ratio else None,
            "flicker_ratio_total": round(jerk_sum[0] / max(jerk_sum[1], 1e-9), 3),
            "flicker_def": "time 2nd difference |o_t - 2o_(t-1) + o_(t-2)| averaged in melt region (not protected); ratio out/src; 1 = as steady as the source",
            "flicker_ratio_p90": round(float(np.percentile(fr_ratio, 90)), 3) if fr_ratio else None,
            "law": {"stretch": "y in [b, b+L): sample s = b + (y-b)/S (linear), tip 1px AA",
                    "breakpoint": "b = q-quantile of cumulative zone mass per column; q = q_lo + (q_hi-q_lo)*F_q(x,t) + q_fine*(col-0.5)",
                    "length": "L = H*length*melt(t)*drive*(0.15+0.85*F_len^1.5)*col_len*mass_w + bursts",
                    "drive": "sense: 0.55+0.45*EMA(tension, 1 s); time: 0.55+0.45*slow field",
                    "burst": "(1-exp(-tau/0.1s))*exp(-tau/burst_tau) * burst_amp*H*strength on burst_frac of the groups within burst_w",
                    "sort": "column runs with EMA-lum in [sort_lo, sort_hi] inside zone where F_sort < sort_amt; bright on top"},
            "frames": rows}
    print(f"{n} 帧 {time.time() - t_start:.0f}s  下坠 {fired}  闪烁比（二阶差分，总）{side['flicker_ratio_total']}  中位 {side['flicker_ratio_median']}")
    plot = series_plot(rows, events, dur)
    for pl in pals:
        out = OUT_DIR / f"{stems[pl]}_v{vers[pl]}.mp4"
        if a.drive == "sense":
            subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(vtmp[pl]), "-ss", f"{a.start:.3f}", "-t", f"{dur:.3f}",
                            "-i", str(S.MUSIC), "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
                            "-t", f"{dur:.3f}", str(out)], check=True)
        else:
            subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(vtmp[pl]), "-c:v", "copy", str(out)], check=True)
        Path(out.with_suffix(".json")).write_text(json.dumps({**side, "palette": pl}, ensure_ascii=False, indent=1), "utf-8")
        th = sorted(thumbs[pl], key=lambda x: x[0])
        if len(th) > 16:
            th = [th[int(round(k))] for k in np.linspace(0, len(th) - 1, 16)]
        contact([(lab, im) for _, lab, im in th], plot, out.with_name(out.stem + "_接触表.jpg"))
        print("->", out)


def protect_mask_at(masks, i):
    if masks is None:
        return None
    return masks[min(i, len(masks) - 1)].astype(np.float32) / 255


def cmd_image(a):
    img = np.asarray(Image.open(a.src).convert("RGB"), np.float32) / 255
    if a.width:
        im = Image.open(a.src).convert("RGB")
        im = im.resize((a.width, int(round(im.height * a.width / im.width))), Image.LANCZOS)
        img = np.asarray(im, np.float32) / 255
    p = Params(**{"angle": a.angle, **parse_set(a.set)})
    H, W = img.shape[:2]
    pm = None
    if a.protect == "birefnet":
        import face_glitch as Fg
        mt = Fg.get_matter()
        pm = np.asarray(mt(to_im(img)), np.float32) / 255 if mt is not None else None
    m = Melter(H, W, p, a.t + 1)
    out, inf = m.step(img, pm, a.t, 1.0)
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    for pl in a.palettes.split(","):
        to_im(palette(out, pl, inf["fresh"], pm)).save(od / f"{Path(a.src).stem}_melt_{pl}.jpg", quality=92)
    print("->", od)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("clip")
    c.add_argument("--src", required=True)
    c.add_argument("--t0", type=float, default=0.0)
    c.add_argument("--dur", type=float, default=0.0)
    c.add_argument("--width", type=int, default=0)
    c.add_argument("--drive", choices=["sense", "time"], default="time")
    c.add_argument("--start", type=float, default=0.0, help="sense 驱动：歌里的播放秒数")
    c.add_argument("--palettes", default="cold,rwb")
    c.add_argument("--angle", type=float, default=0.0)
    c.add_argument("--zone-lo", type=float, default=0.18)
    c.add_argument("--zone-hi", type=float, default=1.01)
    c.add_argument("--hue", default=None)
    c.add_argument("--hand", default=None)
    c.add_argument("--protect", choices=["birefnet", "none"], default="birefnet")
    c.add_argument("--push", default="1.0,1.10")
    c.add_argument("--label", default=None)
    c.add_argument("--seed", type=int, default=7)
    c.add_argument("--set", default=None)
    c.add_argument("--hud", action="store_true")
    c.add_argument("--cache", default=None, help="遮罩缓存 / 无声中间视频放哪（默认 作品/cache/reality_melt）")
    i = sp.add_parser("image")
    i.add_argument("src")
    i.add_argument("--t", type=float, default=2.0)
    i.add_argument("--width", type=int, default=0)
    i.add_argument("--angle", type=float, default=0.0)
    i.add_argument("--palettes", default="cold,rwb")
    i.add_argument("--protect", choices=["birefnet", "none"], default="birefnet")
    i.add_argument("--set", default=None)
    i.add_argument("--out", default=str(Path.cwd()))
    a = ap.parse_args()
    if getattr(a, "cache", None):
        global CACHE
        CACHE = Path(a.cache)
    {"clip": cmd_clip, "image": cmd_image}[a.cmd](a)


if __name__ == "__main__":
    main()
