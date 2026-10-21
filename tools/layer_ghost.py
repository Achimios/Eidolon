#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🖼多层重影-拆解-参考图 `_参考库/瓦解4.jpg`。👤 的读法（权威）：就是「带纹理的蒙版和散射的多层叠加」。
⚠️没有折射 / 扭曲 / 位移场 / 镜头畸变 —— 取代 tools/lens_ghost.py（👤：「太离谱了…甚至给我用上扭曲了」）。

模型（每一项都是线性的、可逆查的）：
  0 palette_pre   源图配色预处理：cold = 亮度 × 冷灰（只留一点色）；ref = 源色饱和 ×1.6（她本来偏蓝）；none = 不动
  1 layers        N 层 = 源图的 N 份拷贝，第 k 层整体平移 dy_k = round(u·spacing·k^growth)（向下；dx_k = round(slant·dy_k)）。
                  纯整数平移 → 离散的重影，没有运动模糊。画边越界默认复制边缘行（零填充会在画顶给每层留一条亮边线）。
                  每层权重 w_k ∝ falloff^k · vis_k（vis = 可见层数的软截断），Σw = 1 → 亮度守恒。
  2 dispersion    每层每个通道再多移 round(u·chan_sep·k·(−1, 0, +1)[c])（R 上 / G 不动 / B 下）+ 每层自己的色调 tint_k；
                  色调按通道归一：Σ_k w_k·tint_k,c = 1 → 层全重合的地方是白的，只有错开的边缘带颜色（像色散）。
  3 texture       每层乘一张真实纹理蒙版 T_k（低熵注入：从参考图自己的亮度高通里抠出来 → 局部方差归一 → 选有纹理的块
                  → 随机块 + 随机翻转 + Hann 窗重叠拼接 → 按 u 缩放）。T = relu(1 + grain·z)^gamma / mean → 均值 = 1（亮度守恒）。
                  tex_mode = carry（默认）：一张纹理贴在源上，跟着每层一起平移 —— 层叠后纹粒在竖向按层距重复；
                  screen：每层自己一张（随机块 / 偏移），固定在画面上。RGB 再各错 tex_rgb 像素 → 闪粒带一点色。
                  ⭐量过参考图：纹粒高通的竖向自相关在 9 px 处 0.77、18 px 0.69、27 px 0.52、36 px 0.39，横向 2 px 外 ≈ 0，
                  R / B 纹粒零位移相关 0.93 → 参考图 = 同一张带纹理的图竖向错 ~9.2 px 叠了 6–8 层（纹理跟着层走）。
                  抠出来的纹理块本身也带着这个竖向 9 px 准周期（它是参考图的真东西，不是规则晶格：横向无周期）。
  4 scatter       每层 = (1−s)·清晰层 + s·模糊层（各向异性高斯：σy = u·scatter_r，σx = aniso·σy）。
                  散射只搬光不造光（能量守恒）；模糊和平移可交换 → 源图只模糊一次再按层平移。纹理也乘在散射部分上。
  5 bloom + tone  高光溢出：max(亮度 − thr, 0) 的大半径模糊加回去（bloom 量）；
                  所有层在黑底上相加；黑位 black；色调曲线 y = (1 − e^(−k·x)) / (1 − e^(−k))（k = 0 → 恒等）。
  u = 像素单位（输出像素 / 参考像素）。默认 1.0 = 按「头宽」对齐：裁肩 v2 里她的头宽 ≈ 500 px，参考图 ≈ 470 px。
      任务单原写「按画宽缩放」（u = 1280/736 = 1.74）—— 放大对比里试过，纹粒和散射相对头都粗了 1.7 倍，不像参考 → 默认改成 1；
      --unit 0 = 按画宽，--unit 1.74 等同。25d 放大对比里两种都摆着。

感官驱动（params_from_sense；HUD / sidecar 写同一张表）。g = 时间门控强度：
  threat  → 层距 spacing = (4 + 16·threat)·(0.3 + 0.7·g)（参考像素）；可见层数 n_vis = 1 + (N − 1)·(0.35 + 0.65·threat)·g
  tension → 通道分离 chan_sep = (0.3 + 2.5·tension)·g（参考像素 / 层）
  hunger  → 散射半径 scatter_r = 10 + 50·hunger（参考像素，σy）；散射量 s = 0.25 + 0.35·g
  grain = 0.6 + 0.4·g；bloom = 0.25 + 0.35·g
时间门控（pulses）：g = max(base, 脉冲)。脉冲只在强起音：起音处威胁峰 ≥ 0.6，且离上一个触发的脉冲 ≥ min_gap（0.45 s）
  —— 150 BPM 下连续 0.4 s 的起音就变成「隔一个」触发。脉冲 = 满幅 2 帧 + exp 衰减（τ = 3 帧）。base 默认 0.25。
时间：感官.json 已是播放时间（offset 0）。
人读版：-给人类的解释/多层重影怎么做的.md

用法（仓库 .venv）：
  python tools/layer_ghost.py sheet [--t 4.0]      # 静帧/25 分解表 + 25b 冷 + 25c 参考色 + 25d 放大对比
  python tools/layer_ghost.py clip [--palette cold] # 样片/21_多层重影_全息女性_v1.mp4（整段裁肩 v2 → 45.12 s 起，配乐）+ .json + 接触表
  python tools/layer_ghost.py sweep                  # 静帧/25e 2×3 扫参（每格 1:1 裁切 + 参数）+ 25b_v3（BEST 组合）
  python tools/layer_ghost.py clip --best --name 21_多层重影_全息女性_v2   # 👤「更多色散、更细腻」的组合
  任意 Params 字段：clip --set "sep_mul=2,chan_g=0.5,tex_scale=0.5,hl_smooth=0.6,spacing_mul=0.8,n_layers=7"
  倍率旋钮（感官映射之后再乘）：spacing_mul、sep_mul、chan_g（G 错位 = chan_g × R/B 错位）、tex_scale（纹粒尺寸）、grain_mul、hl_smooth（白处纹粒减弱）
"""
import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stills as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REF = S.PROJ / "_参考库" / "瓦解4.jpg"
REF_W = 736
SRC_VIDEO = S.MV / "素材" / "全息女性视频运镜描述_裁肩_v2.mp4"
CACHE = ROOT / "cache" / "layer_ghost"
FPS = 24
FONT = S.FONT
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
COLD = np.array([0.80, 0.85, 0.88], np.float32) / 0.88      # 冷灰（S.GREY 归一到白 = 1）


def lum(x):
    return x @ LUMA


# ---------------------------------------------------------------- 基本操作
def shift(ch, dy, dx, edge="zero"):
    """整数平移：out[y, x] = ch[y − dy, x − dx]。精确，不插值。
    越界：edge = "zero" → 0；"replicate" → 取最近的边缘行 / 列（画边不会出现每层一条亮边线）。"""
    H, W = ch.shape[:2]
    if edge == "replicate":
        ys = np.clip(np.arange(H) - dy, 0, H - 1)
        xs = np.clip(np.arange(W) - dx, 0, W - 1)
        return ch[ys][:, xs]
    out = np.zeros_like(ch)
    if abs(dy) >= H or abs(dx) >= W:
        return out
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    out[yd, xd] = ch[ys, xs]
    return out


_DEV = None


def _dev():
    global _DEV
    if _DEV is None:
        import torch
        _DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _DEV


def gblur(x, sy, sx=None):
    """可分离高斯（torch；GPU 有就用）。x: HxW 或 HxWxC float32。σ < 0.3 → 不动。边界复制（不在画边压出暗带）。"""
    import torch
    import torch.nn.functional as Fn
    sx = sy if sx is None else sx
    if sy < 0.3 and sx < 0.3:
        return x.copy()
    a = x if x.ndim == 3 else x[..., None]
    t = torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))[:, None].to(_dev())
    for s, horiz in ((sx, True), (sy, False)):
        if s < 0.3:
            continue
        r = int(np.ceil(3 * s))
        k = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=torch.float32, device=_dev()) / s) ** 2)
        k = k / k.sum()
        if horiz:
            t = Fn.conv2d(Fn.pad(t, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
        else:
            t = Fn.conv2d(Fn.pad(t, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))
    out = t[:, 0].cpu().numpy().transpose(1, 2, 0)
    return out if x.ndim == 3 else out[..., 0]


def tone_curve(x, k):
    """y = (1 − e^(−k·x)) / (1 − e^(−k))；k → 0 是恒等。1 → 1，暗部斜率 k/(1−e^−k) > 1（提一点暗部）、高光压肩。"""
    if k < 1e-6:
        return x
    return (1 - np.exp(-k * x)) / (1 - np.exp(-k))


# ---------------------------------------------------------------- 纹理（低熵注入：从参考图抠）
class TextureBank:
    """参考图亮度 L → 高通 hp = L − G_σ2(L) → z = hp / (sqrt(G_σ6(hp²)) + 0.02)（局部方差归一）
    → 只留「有纹理」的 P×P 块（局部标准差和亮度都够的像素占 ≥ 97 %）→ 每块再标准化（均值 0、方差 1）。"""

    def __init__(self, ref=REF, P=96, stride=16, cache=CACHE):
        cache = Path(cache)
        f = cache / f"texbank_{Path(ref).stem}_{P}_{stride}.npy"
        if f.exists():
            self.patches = np.load(f)
        else:
            im = np.asarray(Image.open(ref).convert("RGB"), np.float32) / 255
            L = lum(im)
            hp = L - gblur(L, 2.0)
            sd = np.sqrt(gblur(hp * hp, 6.0))
            z = hp / (sd + 0.02)
            ok = (sd > np.percentile(sd, 40)) & (gblur(L, 6.0) > 0.10)
            H, W = L.shape
            ps = []
            for y in range(0, H - P + 1, stride):
                for x in range(0, W - P + 1, stride):
                    if ok[y:y + P, x:x + P].mean() >= 0.97:
                        q = z[y:y + P, x:x + P]
                        ps.append((q - q.mean()) / (q.std() + 1e-6))
            self.patches = np.stack(ps).astype(np.float32)
            cache.mkdir(parents=True, exist_ok=True)
            np.save(f, self.patches)
        self.P = self.patches.shape[1]

    def synth(self, H, W, rng):
        """H×W 的 z 场（参考图像素尺度）：随机块 + 随机翻转，半块步长 Hann 重叠相加，除以 sqrt(Σw²) 保方差，再整体标准化。"""
        P, hop = self.P, self.P // 2
        win = np.outer(np.hanning(P), np.hanning(P)).astype(np.float32) + 1e-4
        ny, nx = H // hop + 2, W // hop + 2
        acc = np.zeros(((ny + 1) * hop, (nx + 1) * hop), np.float32)
        wsq = np.zeros_like(acc)
        for i in range(ny):
            for j in range(nx):
                q = self.patches[rng.integers(len(self.patches))]
                if rng.random() < 0.5:
                    q = q[:, ::-1]
                if rng.random() < 0.5:
                    q = q[::-1]
                acc[i * hop:i * hop + P, j * hop:j * hop + P] += q * win
                wsq[i * hop:i * hop + P, j * hop:j * hop + P] += win * win
        oy, ox = int(rng.integers(hop)), int(rng.integers(hop))
        z = (acc / np.sqrt(wsq + 1e-8))[hop + oy:hop + oy + H, hop + ox:hop + ox + W]
        return (z - z.mean()) / (z.std() + 1e-6)


def texture_mask(z, grain, gamma):
    """T = relu(1 + grain·z)^gamma，再除以均值 → mean(T) = 1（亮度守恒）。grain = 0 → 全 1。"""
    if grain <= 0:
        return np.ones_like(z)
    t = np.maximum(0.0, 1.0 + grain * z) ** gamma
    return (t / t.mean()).astype(np.float32)


def frame_texture(bank, H, W, unit, rng):
    """参考尺度合成 → 按 unit 放大到 H×W（双线性）"""
    import torch
    import torch.nn.functional as Fn
    h, w = int(np.ceil(H / unit)) + 2, int(np.ceil(W / unit)) + 2
    z = bank.synth(h, w, rng)
    if abs(unit - 1) < 1e-6:
        z = z[:H, :W]
    else:
        t = torch.from_numpy(z)[None, None]
        t = Fn.interpolate(t, scale_factor=unit, mode="bilinear", align_corners=False, antialias=unit < 1)
        z = t[0, 0, :H, :W].numpy()
    return (z - z.mean()) / (z.std() + 1e-6)


# ---------------------------------------------------------------- 参数
def hue_rgb(h):
    h = np.asarray(h, np.float32)
    return np.clip(np.abs(((h * 6 + np.array([0, 4, 2])) % 6) - 3) - 1, 0, 1)


# 参考图里看到的层色：白（主体）、蓝、橙、青、品红、黄
REF_HUES = [None, 0.60, 0.07, 0.48, 0.86, 0.14, 0.55, 0.02]
COLD_HUES = [None, 0.55, 0.60, 0.52, 0.58, 0.50, 0.62, 0.56]


@dataclass
class Params:
    n_layers: int = 5
    n_vis: float = 5.0              # 可见层数（软截断：第 k 层权重 × clip(n_vis − k, 0, 1)）
    spacing: float = 9.0            # 参考像素（参考图量出来的竖向重复 ≈ 9.2 px）
    growth: float = 1.0             # dy_k = spacing·k^growth
    slant: float = 0.0              # dx_k = slant·dy_k
    falloff: float = 0.65           # w_k ∝ falloff^k
    chan_sep: float = 1.5           # 参考像素 / 层：R −, G 0, B +
    palette: str = "cold"           # cold | ref | none
    tint_sat: float = -1.0          # 层色饱和度；< 0 → 按配色（cold 0.12 / ref 0.85）
    grain: float = 0.45
    gamma: float = 1.3
    tex_rgb: int = 1                # 纹理在 R / B 上各错几个（输出）像素 → 闪粒带色
    tex_mode: str = "carry"         # carry = 一张纹理贴在源上、跟层一起平移（叠出参考图那种竖向 ~9 px 的重复）| screen = 每层自己一张
    scatter: float = 0.5            # s：散射占比
    scatter_r: float = 30.0         # σy（参考像素）
    aniso: float = 0.12             # σx = aniso·σy（参考图的散射几乎全是竖向的）
    bloom: float = 0.4
    bloom_thr: float = 0.55
    bloom_r: float = 40.0           # σy（参考像素）；σx = 0.5·σy
    tone: float = 1.2
    black: float = 0.03             # 色调曲线前的黑位：x → max(x − black, 0)/(1 − black)
    unit: float = 1.0               # 像素单位 u（输出像素 / 参考像素）；0 → 画宽 / 736。默认 1：裁肩 v2 里头宽 ≈ 500 px ≈ 参考图头宽 470 px
    edge: str = "replicate"         # 层平移越界怎么补：replicate（默认）| zero
    # ---- 倍率旋钮（感官映射之后再乘，CLI --set 可改；👤 2026-10-22「更多色散、更细腻」的扫参就是调这些）
    spacing_mul: float = 1.0        # 层距 ×
    sep_mul: float = 1.0            # 通道分离 ×
    chan_g: float = 0.0             # G 通道错位 = chan_g × R/B 的错位量（≠ 0 → 三色彩边：R −s，G +chan_g·s，B +s）
    tex_scale: float = 1.0          # 纹粒尺寸 ×（< 1 更细）
    grain_mul: float = 1.0          # 纹粒强度 ×
    hl_smooth: float = 0.0          # 高光处纹粒减弱：T → 1 + (T − 1)(1 − hl_smooth·L²)，L = 该层源亮度（白处更平，像参考图）
    seed: int = 7
    sense: dict = field(default_factory=dict)

    def hud(self, W=None):
        u = self.unit or ((W or REF_W) / REF_W)
        return [f"layers {self.n_layers} (vis {self.n_vis:.2f})  spacing {self.spacing:.1f}·u  growth {self.growth}  "
                f"falloff {self.falloff}  chan_sep {self.chan_sep:.2f}·u/layer  palette {self.palette}  u={u:.2f}px",
                f"texture: ref high-pass  grain {self.grain:.2f}  gamma {self.gamma}  rgb±{self.tex_rgb}px   "
                f"scatter s {self.scatter:.2f}  σy {self.scatter_r:.0f}·u  σx/σy {self.aniso}   "
                f"bloom {self.bloom:.2f} thr {self.bloom_thr}   tone k {self.tone}   seed {self.seed}",
                f"knobs: spacing x{self.spacing_mul}  sep x{self.sep_mul}  chan_g {self.chan_g}  tex_scale {self.tex_scale}  "
                f"grain x{self.grain_mul}  hl_smooth {self.hl_smooth}"]


# 静帧预设：冷（本片，👤 最喜欢的 17 瀑布图行·本片那种冷灰白、少色）/ 参考色（层距大、层色饱和 → 错开处出彩）
PRESETS = {
    "cold": dict(palette="cold"),
    "ref": dict(palette="ref", n_layers=6, n_vis=6.0, spacing=16.0, falloff=0.9, chan_sep=3.5, tint_sat=1.0),
}


def params_from_sense(threat, tension, hunger, g=1.0, base=None, **kw):
    """感官 + 门控 g → 参数（映射表见模块 docstring）"""
    p = Params(**{**(asdict(base) if base else {}), **kw})
    p.spacing = (4 + 16 * threat) * (0.3 + 0.7 * g)
    p.n_vis = 1 + (p.n_layers - 1) * (0.35 + 0.65 * threat) * g
    p.chan_sep = (0.3 + 2.5 * tension) * g
    p.scatter_r = 10 + 50 * hunger
    p.scatter = 0.25 + 0.35 * g
    p.grain = 0.6 + 0.4 * g
    p.bloom = 0.25 + 0.35 * g
    p.sense = {"threat": round(float(threat), 3), "tension": round(float(tension), 3),
               "hunger": round(float(hunger), 3), "gate": round(float(g), 3)}
    return p


def layer_offsets(p: Params, unit):
    """[(dy_k, dx_k, (sep_R, sep_G, sep_B))]：整数像素"""
    out = []
    for k in range(p.n_layers):
        dy = int(round(unit * p.spacing * p.spacing_mul * (k ** p.growth))) if k else 0
        dx = int(round(p.slant * dy))
        s = unit * p.chan_sep * p.sep_mul * k
        out.append((dy, dx, (int(round(-s)), int(round(p.chan_g * s)), int(round(s)))))
    return out


def layer_weights(p: Params):
    k = np.arange(p.n_layers, dtype=np.float64)
    w = p.falloff ** k * np.clip(p.n_vis - k, 0, 1)
    w[0] = max(w[0], 1e-9)
    return w / w.sum()


def layer_tints(p: Params, w):
    """tint_k (N×3)，按通道归一使 Σ_k w_k·tint_k,c = 1"""
    hues = REF_HUES if p.palette == "ref" else COLD_HUES
    sat = p.tint_sat if p.tint_sat >= 0 else (0.85 if p.palette == "ref" else 0.12)
    t = np.ones((p.n_layers, 3))
    if p.palette != "none":
        for k in range(1, p.n_layers):
            h = hues[k % len(hues)] or 0.0
            t[k] = (1 - sat) + sat * hue_rgb(h) * 1.5
    t = t / (w[:, None] * t).sum(0, keepdims=True)
    return t.astype(np.float32)


def palette_pre(img, palette):
    if palette == "cold":
        return (lum(img)[..., None] * COLD).astype(np.float32)
    if palette == "ref":       # 参考图的大块颜色来自它源图自己的光（蓝 / 暖）→ 我们保留源的蓝并加饱和，彩边交给层色调 + 通道错位
        L = lum(img)[..., None]
        return np.clip(L + 1.6 * (img - L), 0, None).astype(np.float32)
    return img


# ---------------------------------------------------------------- 管线
def run(img, p: Params, bank=None, stages=False):
    """img: HxWx3 float32 [0,1]。stages=True → 额外返回 [(名字, 图)]（每一级单独加上去的结果）。"""
    H, W, _ = img.shape
    u = p.unit or W / REF_W
    rng = np.random.default_rng(p.seed)
    st = []
    src = palette_pre(img, p.palette)
    st.append((f"0  source  palette_pre {p.palette}", src))
    offs = layer_offsets(p, u)
    w = layer_weights(p)
    tint = layer_tints(p, w)
    use_tex = p.grain * p.grain_mul > 0
    gr = p.grain * p.grain_mul
    lsrc = np.clip(lum(src), 0, 1) if p.hl_smooth > 0 else None
    if use_tex and bank is None:
        bank = TextureBank()
    blurred = gblur(src, u * p.scatter_r, u * p.scatter_r * p.aniso) if p.scatter > 0 else None
    sharp_plain = np.zeros_like(src)     # 只有层（无色散 / 纹理 / 散射）—— 给分解表
    disp = np.zeros_like(src)
    tex_only = np.zeros_like(src)
    out = np.zeros_like(src)
    T_show = None
    tr = int(p.tex_rgb)                              # 输出像素
    for k, (dy, dx, sep) in enumerate(offs):
        if w[k] < 1e-6:
            continue
        if not use_tex:
            T = None
        elif p.tex_mode == "carry":    # 一张纹理贴在源上，跟着每一层一起平移
            if T_show is None:
                T0 = texture_mask(frame_texture(bank, H, W, u * p.tex_scale, rng), gr, p.gamma)
            T = np.roll(T0, (dy, dx), axis=(0, 1))     # 循环平移：纹理统计不变，画边不会复制出竖条
        else:                          # screen：每层自己一张（随机块 / 偏移），固定在画面上
            T = texture_mask(frame_texture(bank, H, W, u * p.tex_scale, rng), gr, p.gamma)
        if T_show is None and T is not None:
            T_show = T
        if T is not None and lsrc is not None:     # 高光处纹粒减弱（(T − 1) 乘一个只随亮度变的因子 → 平图上均值仍 = 1）
            Lk = shift(lsrc, dy, dx, p.edge)
            T = 1 + (T - 1) * (1 - p.hl_smooth * Lk * Lk)
        for c in range(3):
            a = w[k] * tint[k, c]
            lay = a * shift(src[..., c], dy + sep[c], dx, p.edge)
            if stages:
                sharp_plain[..., c] += w[k] * shift(src[..., c], dy, dx, p.edge)
                disp[..., c] += lay
            Tc = None
            if T is not None:      # R / B 的纹理循环错开 tr 像素（roll → 均值严格不变）
                Tc = T if (tr == 0 or c == 1) else np.roll(T, tr if c == 0 else -tr, axis=1)
            if stages:
                tex_only[..., c] += lay * (Tc if Tc is not None else 1.0)
            if blurred is not None:
                lay = (1 - p.scatter) * lay + p.scatter * a * shift(blurred[..., c], dy + sep[c], dx, p.edge)
            out[..., c] += lay * (Tc if Tc is not None else 1.0)
    if stages:
        st.append((f"1  {p.n_layers} layers, vis {p.n_vis:.1f}, dy_k = round(u·{p.spacing:.1f}·k^{p.growth})  "
                   f"({', '.join(str(o[0]) for o in offs)} px)  falloff {p.falloff}", sharp_plain))
        st.append((f"2  + per-layer tint & channel shift  R−/B+ {p.chan_sep:.2f}·u·k px", disp))
        if T_show is not None:
            st.append((f"3a texture mask T (layer 0): ref high-pass, grain {p.grain:.2f}, gamma {p.gamma}, mean {T_show.mean():.3f}",
                       np.repeat(np.clip(T_show * 0.35, 0, 1)[..., None], 3, 2)))
        st.append((f"3  + texture mask ({p.tex_mode}: {'one texture, moves with each layer' if p.tex_mode == 'carry' else 'own texture per layer'})", tex_only))
        st.append((f"4  + scatter s={p.scatter:.2f}  σy={u * p.scatter_r:.0f}px σx={u * p.scatter_r * p.aniso:.0f}px", out.copy()))
    if p.bloom > 0:
        hl = np.maximum(lum(out) - p.bloom_thr, 0)
        b = gblur(hl, u * p.bloom_r, u * p.bloom_r * 0.5)
        bc = COLD if p.palette == "cold" else np.ones(3, np.float32)
        out = out + p.bloom * b[..., None] * bc
    if p.black > 0:
        out = np.maximum(out - p.black, 0) / (1 - p.black)
    out = tone_curve(out, p.tone)
    out = np.clip(out, 0, 1).astype(np.float32)
    if stages:
        st.append((f"5  + bloom {p.bloom:.2f} (thr {p.bloom_thr}, σy {u * p.bloom_r:.0f}px)  + tone k={p.tone}  = final", out))
        return out, st
    return out


# ---------------------------------------------------------------- 时间门控
def pulses(t, onsets, threat, fps=FPS, thr=0.6, min_gap=0.45, hold=2, tau=3.0, base=0.25):
    """g(t) = max(base, 脉冲)。起音 o 处的威胁峰 s = max threat[o·fps−1 … o·fps+1] ≥ thr，且离上一个触发 ≥ min_gap 才触发。
    脉冲：满幅 2 帧，之后 exp(−(k−hold+1)/tau)。返回 (g, 触发列表 [(o, s)])"""
    t = np.asarray(t, np.float64)
    th = np.asarray(threat)
    g = np.full(len(t), base, np.float64)
    fired, last = [], -1e9
    for o in onsets:
        if not (t[0] - 0.5 / fps <= o <= t[-1] + 0.5 / fps):
            continue
        c = int(round(o * fps))
        s = float(th[max(0, c - 1):c + 2].max())
        if s < thr or o - last < min_gap:
            continue
        last = o
        fired.append((round(float(o), 3), round(s, 3)))
        j0 = int(np.searchsorted(t, o - 0.5 / fps))
        k = np.arange(len(t)) - j0
        b = np.where((k >= 0) & (k < hold), 1.0, 0.0)
        tail = k >= hold
        b[tail] = np.exp(-(k[tail] - hold + 1) / tau)
        g = np.maximum(g, b)
    return g, fired


# ---------------------------------------------------------------- IO
def video_frames(path, t0, n):
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


def to_im(x):
    return Image.fromarray((np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8))


def label(im, lines, s=1.0, color=(200, 206, 212), pos="bl"):
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, int(14 * s))
    w, h = im.size
    for j, line in enumerate(lines):
        y = h - int(10 * s) - int(18 * s) * (len(lines) - j) if pos == "bl" else int(8 * s) + int(18 * s) * j
        d.text((int(10 * s), y), line, font=f, fill=color)
    return im


def outname(path):
    """不覆盖：已存在 → 同名 _v2, _v3 …（名字里已有 _vN 就把 N 加一）"""
    path = Path(path)
    if not path.exists():
        return path
    stem = path.stem
    import re
    m = re.search(r"_v(\d+)$", stem)
    base, i = (stem[:m.start()], int(m.group(1)) + 1) if m else (stem, 2)
    while (q := path.with_name(f"{base}_v{i}{path.suffix}")).exists():
        i += 1
    return q


# ---------------------------------------------------------------- 命令
def zoom(im, box, k=3):
    return im.crop(box).resize(((box[2] - box[0]) * k, (box[3] - box[1]) * k), Image.NEAREST)


def parse_set(text):
    """--set "sep_mul=1.5,chan_g=0.5,n_layers=7" → dict（按 Params 字段的类型转换）"""
    if not text:
        return {}
    types = {k: type(v) for k, v in asdict(Params()).items()}
    out = {}
    for kv in text.split(","):
        k, v = kv.split("=")
        k = k.strip()
        if k not in types:
            raise SystemExit(f"--set：没有参数 {k}；可用：{', '.join(types)}")
        out[k] = types[k](v) if types[k] is not bool else v.lower() in ("1", "true", "yes")
    return out


# 👤 2026-10-22「更多色散、更细腻」的 2×3 扫参（cold 预设之上的改动）
SWEEP = [
    ("A  baseline (25b_v2)", {}),
    ("B  sep x1.5 + G 0.5 (3-colour fringe)", dict(sep_mul=1.5, chan_g=0.5)),
    ("C  sep x2 + G 0.5", dict(sep_mul=2.0, chan_g=0.5)),
    ("D  grain x0.7 size, hl_smooth 0.6, grain x0.85", dict(tex_scale=0.7, hl_smooth=0.6, grain_mul=0.85)),
    ("E  grain x0.5 size, hl_smooth 0.6, grain x0.85", dict(tex_scale=0.5, hl_smooth=0.6, grain_mul=0.85)),
    ("F  spacing 7 px, 7 layers, falloff 0.75", dict(spacing=7.0, n_layers=7, n_vis=7.0, falloff=0.75)),
]
# 扫参里挑出来的组合（→ 25b_v3 + 样片 21_v2）
BEST = dict(sep_mul=2.0, chan_g=0.5, tex_scale=0.7, hl_smooth=0.6, grain_mul=0.85,
            spacing=7.0, n_layers=7, n_vis=7.0, falloff=0.75)


def cmd_sweep(a):
    """2×3 扫参：每格 = 同一帧的 1:1 原像素裁切（640×720，看得见纹粒），格内写参数；再出最佳组合的整帧"""
    img = video_frames(Path(a.src), a.t, 1)[0]
    bank = TextureBank()
    x0 = a.crop_x
    cw, ch_ = 640, 720
    sheet = Image.new("RGB", (3 * cw, 2 * ch_ + 70), (8, 8, 10))
    for i, (name, kw) in enumerate(SWEEP):
        p = Params(**{**PRESETS["cold"], **kw}, seed=a.seed, unit=a.unit)
        im = to_im(run(img, p, bank)).crop((x0, 0, x0 + cw, ch_))
        ImageDraw.Draw(im).rectangle((0, 0, cw, 30), fill=(0, 0, 0))
        label(im, [name], s=1.0, color=(235, 235, 235), pos="tl")
        label(im, [", ".join(f"{k}={v}" for k, v in kw.items()) or "cold preset"], s=0.8, color=(170, 176, 182))
        sheet.paste(im, ((i % 3) * cw, 70 + (i // 3) * ch_))
    label(sheet, [f"25e  layer ghost parameter sweep, cold preset + changes, frame t={a.t}s, 1:1 crop x={x0}..{x0 + cw}",
                  "knobs: sep_mul (R up / B down x), chan_g (G offset, fraction of R/B), tex_scale (grain size x), "
                  "grain_mul, hl_smooth (less grain in whites), spacing / n_layers / falloff"], s=1.0, color=(170, 176, 182), pos="tl")
    dst = S.MV / "静帧"
    o1 = outname(dst / "25e_多层重影_参数扫_v1.jpg"); sheet.save(o1, quality=93)
    pb = Params(**{**PRESETS["cold"], **BEST}, seed=a.seed, unit=a.unit)
    o2 = outname(dst / "25b_多层重影_冷_v3.jpg")
    label(to_im(run(img, pb, bank)), pb.hud(img.shape[1]), s=0.9, color=(150, 156, 162)).save(o2, quality=94)
    for o in (o1, o2):
        print("->", o)


def cmd_sheet(a):
    img = video_frames(Path(a.src), a.t, 1)[0]
    bank = TextureBank()
    pc = Params(**PRESETS["cold"], seed=a.seed, unit=a.unit)
    pr = Params(**PRESETS["ref"], seed=a.seed, unit=a.unit)
    out_c, st = run(img, pc, bank, stages=True)
    out_r = run(img, pr, bank)
    pw_ = Params(**PRESETS["ref"], seed=a.seed, unit=0.0)          # 任务单原写的「按画宽缩放」，给放大对比
    out_w = run(img, pw_, bank)
    dst = S.MV / "静帧"
    H, W = img.shape[:2]
    # 分解表
    pw, ph = 640, 360
    panels = st + [("final  palette ref (same layers, reference-like hues)", out_r)]
    cols, rows = 3, (len(panels) + 2) // 3
    sheet = Image.new("RGB", (cols * pw, rows * ph + 82), (8, 8, 10))
    for i, (name, x) in enumerate(panels):
        th = to_im(x).resize((pw, ph), Image.LANCZOS)
        label(th, [name], s=0.85)
        sheet.paste(th, ((i % cols) * pw, 82 + (i // cols) * ph))
    label(sheet, ["25  LAYER GHOST = textured mask + scattering, N layers stacked (no warp / displacement). tools/layer_ghost.py"]
          + pc.hud(W), s=1.0, color=(170, 176, 182), pos="tl")
    o1 = outname(dst / "25_多层重影_分解_v1.jpg"); sheet.save(o1, quality=92)
    o2 = outname(dst / "25b_多层重影_冷_v1.jpg"); label(to_im(out_c), pc.hud(W), s=0.9, color=(150, 156, 162)).save(o2, quality=94)
    o3 = outname(dst / "25c_多层重影_参考色_v1.jpg"); label(to_im(out_r), pr.hud(W), s=0.9, color=(170, 170, 170)).save(o3, quality=94)
    # 放大对比：参考 3× vs 我们 3×（同样 200×200 源像素 → 600×600）
    ref = Image.open(REF).convert("RGB")
    rboxes = [(480, 300, 680, 500), (80, 250, 280, 450), (300, 850, 500, 1050)]
    oboxes = a.boxes or [(900, 120, 1100, 320), (560, 200, 760, 400), (700, 480, 900, 680)]
    tiles = []
    for rb, ob in zip(rboxes, oboxes):
        tiles.append([(f"ref 瓦解4  {rb}  x3", zoom(ref, rb)), (f"ours cold u=1  {ob}  x3", zoom(to_im(out_c), ob)),
                      (f"ours ref-colour u=1  {ob}  x3", zoom(to_im(out_r), ob)),
                      (f"ref-colour u=W/736={W / REF_W:.2f} (width-scaled)", zoom(to_im(out_w), ob))])
    z = 600
    cmp_ = Image.new("RGB", (4 * z, 3 * z + 40), (8, 8, 10))
    for r, row in enumerate(tiles):
        for c, (name, im) in enumerate(row):
            im = im.copy()
            ImageDraw.Draw(im).rectangle((0, 0, z, 24), fill=(0, 0, 0))
            ImageDraw.Draw(im).text((8, 4), name, font=ImageFont.truetype(FONT, 14), fill=(230, 230, 230))
            cmp_.paste(im, (c * z, 40 + r * z))
    label(cmp_, [f"25d  3x nearest zoom, 200x200 source px each. ours: frame t={a.t}s of the cropped clip (v2). "
                 f"u = output px per ref px: 1.0 = head-width matched (default), {W / REF_W:.2f} = width-scaled (col 4)"],
          s=1.0, color=(170, 176, 182), pos="tl")
    o4 = outname(dst / "25d_多层重影_放大对比_v1.jpg"); cmp_.save(o4, quality=92)
    for o in (o1, o2, o3, o4):
        print("->", o)


def contact(frames, path, cols=4):
    tw, th_ = 480, 270
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * th_), (8, 8, 10))
    for i, (lab, im) in enumerate(frames):
        t = im.resize((tw, th_), Image.LANCZOS)
        label(t, [lab], s=0.8, color=(230, 230, 230), pos="tl")
        sheet.paste(t, ((i % cols) * tw, (i // cols) * th_))
    sheet.save(path, quality=88)


def cmd_clip(a):
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    src = Path(a.src)
    sn = json.loads((S.MV / "感官.json").read_text("utf-8"))
    ch = sn["channels"]
    frames = video_frames(src, 0, 10 ** 6)
    n = len(frames)
    dur = n / FPS
    H, W = frames[0].shape[:2]
    t_play = a.start + np.arange(n) / FPS
    th = np.asarray(ch["threat"])
    on = np.asarray(sn["events"]["onsets"])
    g, fired = pulses(t_play, on, th, thr=a.thr, min_gap=a.min_gap, base=a.base)
    bank = TextureBank()
    extra = {**(BEST if a.best else {}), **parse_set(a.set)}
    CACHE.mkdir(parents=True, exist_ok=True)
    vtmp = CACHE / f"{a.name}_noaudio.mp4"
    pipe = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                             "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", str(vtmp)],
                            stdin=subprocess.PIPE)
    side = {"tool": "tools/layer_ghost.py clip", "source": src.name, "play_start": a.start, "dur": dur, "frames_n": n,
            "palette": a.palette, "sense_offset": 0.0,
            "model": "N integer-shifted copies x real texture mask (ref high-pass) + anisotropic scatter + bloom, additive on black, tone curve. NO displacement.",
            "mapping": {"threat": "spacing = (4+16*threat)*(0.3+0.7g) ref-px; n_vis = 1+(N-1)*(0.35+0.65*threat)*g",
                        "tension": "chan_sep = (0.3+2.5*tension)*g ref-px per layer (R-, B+)",
                        "hunger": "scatter sigma_y = 10+50*hunger ref-px; scatter s = 0.25+0.35g",
                        "gate": f"g = max({a.base}, pulse); pulse on onsets with threat peak >= {a.thr} and >= {a.min_gap}s after the last fired; "
                                f"full 2 f then exp tau 3 f",
                        "seed": f"new texture/seed every {a.hold} frames and at fired pulses"},
            "knobs": extra, "fired": fired, "frames": []}
    seed, t0 = 100, time.time()
    fired_t = [o for o, _ in fired]
    thumbs, every = [], max(1, n // 16)
    for i, f in enumerate(frames):
        tp = t_play[i]
        k = min(int(round(tp * FPS)), len(th) - 1)
        if i % a.hold == 0 or any(abs(o - tp) < 0.5 / FPS for o in fired_t):
            seed += 1
        p = params_from_sense(ch["threat"][k], ch["tension"][k], ch["hunger"][k], float(g[i]),
                              Params(**{"palette": a.palette, "unit": a.unit, **extra, "seed": seed}))
        out = run(f, p, bank)
        im = to_im(out)
        if a.hud:
            label(im, [f"t = {tp:6.3f} s (play)   gate {g[i]:.2f}   threat {p.sense['threat']:.2f}  tension {p.sense['tension']:.2f}  "
                       f"hunger {p.sense['hunger']:.2f}"] + p.hud(W), s=0.8, color=(130, 136, 142))
        pipe.stdin.write(np.asarray(im).tobytes())
        if i % every == 0:
            thumbs.append((f"{tp:.2f}s  gate {g[i]:.2f}  threat {p.sense['threat']:.2f}", im))
        side["frames"].append({"i": i, "t_play": round(float(tp), 3), **p.sense, "seed": seed,
                               "spacing": round(p.spacing, 2), "n_vis": round(p.n_vis, 2), "chan_sep": round(p.chan_sep, 2),
                               "scatter_r": round(p.scatter_r, 1), "scatter": round(p.scatter, 2)})
    pipe.stdin.close()
    pipe.wait()
    print(f"{n} 帧  {time.time() - t0:.1f}s  脉冲 {fired}")
    out = outname(S.MV / "样片" / f"{a.name}.mp4")
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(vtmp), "-ss", f"{a.start:.3f}", "-t", f"{dur:.3f}",
                    "-i", str(S.MUSIC), "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
                    "-t", f"{dur:.3f}", str(out)], check=True)
    Path(str(out) + ".json").write_text(json.dumps(side, ensure_ascii=False, indent=1), "utf-8")
    contact(thumbs, outname(out.with_name(out.stem + "_接触表.jpg")))
    print("->", out)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s1 = sp.add_parser("sheet")
    s1.add_argument("--src", default=str(SRC_VIDEO))
    s1.add_argument("--t", type=float, default=4.0)
    s1.add_argument("--seed", type=int, default=7)
    s1.add_argument("--unit", type=float, default=1.0, help="像素单位（输出 px / 参考 px）；0 = 画宽 / 736")
    s1.add_argument("--boxes", type=json.loads, default=None, help='放大框 JSON：[[x0,y0,x1,y1],...]（200×200）')
    s3 = sp.add_parser("sweep")
    s3.add_argument("--src", default=str(SRC_VIDEO))
    s3.add_argument("--t", type=float, default=4.0)
    s3.add_argument("--seed", type=int, default=7)
    s3.add_argument("--unit", type=float, default=1.0)
    s3.add_argument("--crop-x", type=int, default=520, help="每格 1:1 裁切的左边 x")
    s2 = sp.add_parser("clip")
    s2.add_argument("--src", default=str(SRC_VIDEO))
    s2.add_argument("--start", type=float, default=45.12, help="第 0 帧 = 歌里的播放秒数")
    s2.add_argument("--palette", default="cold", choices=["cold", "ref", "none"])
    s2.add_argument("--name", default="21_多层重影_全息女性_v1")
    s2.add_argument("--unit", type=float, default=1.0, help="像素单位；0 = 画宽 / 736")
    s2.add_argument("--thr", type=float, default=0.6, help="脉冲：起音处威胁峰 ≥ 这个")
    s2.add_argument("--min-gap", type=float, default=0.45, help="脉冲：离上一个触发至少几秒（150 BPM 连续起音 → 隔一个）")
    s2.add_argument("--base", type=float, default=0.25, help="脉冲之间的底强度")
    s2.add_argument("--hold", type=int, default=4, help="纹理 / 种子保持几帧")
    s2.add_argument("--hud", action=argparse.BooleanOptionalAction, default=True)
    s2.add_argument("--best", action="store_true", help="用扫参挑出的组合 BEST（25b_v3 那组）")
    s2.add_argument("--set", default="", help='覆盖任意 Params 字段，如 "sep_mul=2,chan_g=0.5,tex_scale=0.5,n_layers=7"（在 --best 之上）')
    a = ap.parse_args()
    {"sheet": cmd_sheet, "clip": cmd_clip, "sweep": cmd_sweep}[a.cmd](a)


if __name__ == "__main__":
    main()
