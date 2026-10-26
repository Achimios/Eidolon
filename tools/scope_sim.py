#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<📺示波器-XY-荧光模拟：立体声音频 → 示波器 XY 画面（左声道 = x，右声道 = y），按 `_设计/理念_示波器与处理器.md` §四 的物理。

物理（每一条都是显式参数，见 Params）：
  1 光束：相邻采样点之间画线段（先 resample_poly 做 upsample 倍 sinc 上采样 = DAC 之后的带限重建）。
    每段持续 dt = 1/(sr·upsample)，沉积能量 ∝ dt（单位时间能量恒定），沿线段均匀摊开 → 线密度 ∝ 1/速度：
    慢的地方亮，快跳的地方只剩一条几乎看不见的暗线（不是断点）。能量单位：光束静止一整帧 = 1。
    线段按 step px 细分成子点，双线性溅射，再乘光斑高斯（beam σ）。
  2 荧光余辉：buf ← buf·exp(-1/(fps·persist)) + 本帧沉积；帧内也按 exp(-(t_end - t)/persist) 衰减（先画的已经暗了一点）。
  3 辉光 + 色调：D = 核心 + glow·Σ wₖ·Gauss(核心, rₖ)；E = gain·K₀·D；线性光 lin_c = 1 - exp(-(c_c·E + whiten·E²))，再 sRGB 编码。
    c 的绿通道最大 → 暗处只有荧光色；E² 项让亮核三个通道一起饱和 → 发白（c、whiten 从原视频像素拟合）。
    余辉可以有两个成分：主成分 τ = persist，加一条慢尾（相对幅度 tail、τ = tail_tau，默认关）。
  4 可选 CRT 细节（默认关）：颗粒 grain、刻度网格 graticule（参考视频里有一层暗网格）。
帧的定义：第 i 帧（pts = t0 + i/fps）显示 [pts + lead - 1/fps, pts + lead] 这段光束 + 之前衰减下来的余辉。
  ⭐每帧 = 纯函数(音频, 帧号, 上一帧的衰减缓冲)。上采样按全局下标取（两侧各垫 PAD 个输入采样），所以分块方式不影响结果；
  起点前先预热 warmup 秒（默认 6·persist），余辉 ≥ 99.7% 收敛 → 从任意时刻开始渲（--frame / --t0）结果一致（可续渲）。
GPU：torch（有 CUDA 就用），显存上限 set_per_process_memory_fraction(--vram)；--duty 0.9 = 每帧算完歇 10% 的时间（不碰驱动 / 系统设置）。

用法：
  python tools/scope_sim.py <立体声音频> --out <mp4> [--t0 95 --t1 105] [--fps 60] [--size 1440x1080] [--color green|white|red]
                            [--persist 0.03] [--glow 1] [--gain 1] [--graticule]
  python tools/scope_sim.py <音频> --frame 100 --out x.png        # 单帧 PNG
输出 mp4 会把这段音频（原采样率 → AAC 48k）混回去。不覆盖已有文件（要另起版本名）。
验证：`_验证/`（和 Triple Hit 原视频逐帧对照）的结论写在 -给人类的解释/示波器模拟怎么做的.md。
"""
import argparse
import math
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly

COLORS = {  # 每通道的线性「灵敏度」c：暗处呈现 c 的色相；亮到饱和（+ whiten·E² 项）→ 核心发白
    "green": (0.05, 1.0, 0.17),     # 从 Triple Hit 原视频的像素拟合（见 -给人类的解释/示波器模拟怎么做的.md）
    "white": (0.80, 0.90, 1.0),
    "red": (1.0, 0.10, 0.06),
    "amber": (1.0, 0.45, 0.06),
}
PAD = 64          # 上采样时两侧垫的输入采样数（resample_poly 半长 = 10 个输入采样，够用）
K0 = 2.2e4        # 能量 → 曝光的基准倍数（gain = 1 时；对原视频分箱拟合）


@dataclass
class Params:
    fps: float = 60.0
    W: int = 1440
    H: int = 1080
    color: str = "green"
    persist: float = 0.016      # 余辉主成分时间常数 τ（秒，1/e）；原视频 NNLS 拟合 ≈ 0.016
    tail: float = 0.0           # 慢余辉成分的相对幅度（0 = 关）
    tail_tau: float = 0.15      # 慢余辉时间常数（秒）
    glow: float = 1.0           # 辉光总强度
    glow_radii: tuple = (3.0, 10.0, 32.0, 110.0)     # 1080p 下的 px（按 H/1080 缩放）
    glow_weights: tuple = (0.25, 0.12, 0.10, 0.10)
    gain: float = 1.0
    upsample: int = 4
    scale: float = 1.76         # 幅度 1.0 → scale·H/2 px（原视频拟合 1.76：幅度 ±1 会超出画面，原作就是这样裁的）
    beam: float = 1.0           # 光斑 σ（1080p 下 px）
    lead: float = 1 / 60        # 帧窗口终点相对 pts 的偏移（秒）；原视频 = +1 帧（pts t 的帧画到 t + 1/60）
    step: float = 0.5           # 线段细分间距（px）
    grain: float = 0.0
    graticule: float = 0.0      # 网格线处亮度乘 (1 - graticule)
    grid_px: float = 0.0        # 网格间距 px（0 = 95·H/1080，原视频实测 95 px）
    black: float = 0.0          # 底色（荧光屏本身的暗绿，加在曝光前）
    whiten: float = 0.07        # 核心发白：线性曝光多一项 whiten·E²（所有通道一样）
    response: float = 1.25      # 曝光 = (gain·K0·能量)^γ。⚠️经验值：原视频的亮度对能量是超线性的（分箱拟合 γ ≈ 1.2–1.3），机理没查
    warmup: float = -1.0        # 预热秒数；<0 → 6·persist

    @property
    def px(self):
        return self.H / 1080.0


# ---------------------------------------------------------------- 音频（按全局下标、确定性）
class Audio:
    """按全局上采样下标取样本：j ↔ 时间 j / (sr·up)。分块缓存，结果和分块方式无关（每块两侧各垫 PAD 个输入采样）。"""

    def __init__(self, path, up=4, chunk_sec=2.0):
        self.f = sf.SoundFile(str(path))
        if self.f.channels < 2:
            raise ValueError("要立体声（左 = x，右 = y）")
        self.sr, self.n, self.up = self.f.samplerate, self.f.frames, up
        self.R = self.sr * up
        self.chunk = int(chunk_sec * self.sr)
        self.cache = {}

    def _read(self, a, b):
        """输入采样 [a, b)，越界补 0"""
        out = np.zeros((b - a, 2), np.float32)
        lo, hi = max(a, 0), min(b, self.n)
        if hi > lo:
            self.f.seek(lo)
            out[lo - a:hi - a] = self.f.read(hi - lo, dtype="float32", always_2d=True)[:, :2]
        return out

    def _block(self, k):
        if k not in self.cache:
            if len(self.cache) > 4:
                self.cache.pop(next(iter(self.cache)))
            a = k * self.chunk
            x = self._read(a - PAD, a + self.chunk + PAD)
            y = resample_poly(x, self.up, 1, axis=0).astype(np.float32) if self.up > 1 else x
            self.cache[k] = y[PAD * self.up:(PAD + self.chunk) * self.up]
        return self.cache[k]

    def get(self, ja, jb):
        """上采样后的样本 [ja, jb]（含两端），shape (jb-ja+1, 2)"""
        cu = self.chunk * self.up
        parts, j = [], ja
        while j <= jb:
            k = j // cu
            blk = self._block(k)
            s = j - k * cu
            e = min(cu, jb - k * cu + 1)
            parts.append(blk[s:e])
            j = k * cu + e
        return np.concatenate(parts, 0)


# ---------------------------------------------------------------- 光栅化
def to_px(xy, p):
    """幅度 → 像素（x 向右，y 向上翻成像素向下）"""
    s = p.scale * p.H / 2
    return torch.stack([p.W / 2 + xy[:, 0] * s, p.H / 2 - xy[:, 1] * s], 1)


def splat_path(pts, wseg, H, W, step=0.5, kmax=8192, max_points=12_000_000):
    """折线 pts (N+1, 2) px，每段能量 wseg (N,) 或 (N, C)，沿段均匀摊开 → 双线性溅射，返回 (H, W) 或 (C, H, W) 能量图。"""
    dev = pts.device
    flat = wseg.dim() == 1
    if flat:
        wseg = wseg[:, None]
    img = torch.zeros(H * W, wseg.shape[1], device=dev, dtype=torch.float32)
    p0, d = pts[:-1], pts[1:] - pts[:-1]
    L = torch.linalg.norm(d, dim=1)
    K = torch.clamp(torch.ceil(L / step), 1, kmax).long()
    # 分批（点太多时）
    cs = torch.cumsum(K, 0)
    total = int(cs[-1])
    nb = max(1, math.ceil(total / max_points))
    bounds = [0] + [int(torch.searchsorted(cs, torch.tensor(total * b // nb, device=dev))) + 1 for b in range(1, nb)] + [len(K)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b <= a:
            continue
        Kb = K[a:b]
        idx = torch.repeat_interleave(torch.arange(a, b, device=dev), Kb)
        start = torch.cumsum(Kb, 0) - Kb
        off = torch.arange(idx.numel(), device=dev) - torch.repeat_interleave(start, Kb)
        u = (off.float() + 0.5) / K[idx].float()
        q = p0[idx] + u[:, None] * d[idx]
        w = (wseg / K.float()[:, None])[idx]
        x, y = q[:, 0] - 0.5, q[:, 1] - 0.5          # 像素中心在 (i + 0.5)
        x0, y0 = torch.floor(x), torch.floor(y)
        fx, fy = x - x0, y - y0
        x0, y0 = x0.long(), y0.long()
        for dx, dy, ww in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            xi, yi = x0 + dx, y0 + dy
            ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            img.index_add_(0, (yi[ok] * W + xi[ok]), (w * ww[:, None])[ok])
    return img[:, 0].view(H, W) if flat else img.T.reshape(-1, H, W)


def _gk(sigma, dev):
    r = max(1, int(math.ceil(3 * sigma)))
    t = torch.arange(-r, r + 1, device=dev, dtype=torch.float32)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    return k / k.sum()


def gblur(img, sigma):
    """(H, W) 高斯模糊；σ 大时先 avg-pool 降采样（金字塔近似）"""
    if sigma <= 0.05:
        return img
    H, W = img.shape
    f = max(1, int(sigma // 4))
    x = img[None, None]
    if f > 1:
        x = F.avg_pool2d(F.pad(x, (0, (-W) % f, 0, (-H) % f), mode="replicate"), f)
        s = math.sqrt(max(sigma ** 2 - (f ** 2 - 1) / 12.0, 0.25)) / f   # 扣掉 box 本身的方差
    else:
        s = sigma
    k = _gk(s, img.device)
    r = (k.numel() - 1) // 2
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))
    if f > 1:
        x = F.interpolate(x, scale_factor=f, mode="bilinear", align_corners=False)[..., :H, :W]
    return x[0, 0]


class Scope:
    """逐帧渲染器。state = 衰减缓冲（未加光斑 / 辉光的原始能量图）。"""

    def __init__(self, audio, p: Params, device=None):
        self.a, self.p = audio, p
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.col = torch.tensor(COLORS[p.color], device=self.dev).view(3, 1, 1)
        taus = [p.persist] + ([p.tail_tau] if p.tail > 0 else [])
        self.taus = torch.tensor(taus, device=self.dev)
        self.amps = torch.tensor([1.0] + ([p.tail] if p.tail > 0 else []), device=self.dev)
        self.decay = torch.exp(-1.0 / (p.fps * self.taus)).view(-1, 1, 1)
        self._grid = None

    # 第 i 帧的样本区间（全局上采样下标），相邻帧首尾共享一个点 → 每段只属于一帧
    def window(self, t_end):
        R = self.a.R
        jb = int(round(t_end * R))
        ja = int(round((t_end - 1.0 / self.p.fps) * R))
        return ja, jb

    def deposit(self, t_end):
        """本帧窗口内的新沉积（含帧内衰减）→ (C, H, W)，C = 余辉成分数（主 + 可选慢尾）"""
        p = self.p
        ja, jb = self.window(t_end)
        xy = torch.from_numpy(self.a.get(ja, jb)).to(self.dev)
        pts = to_px(xy, p)
        n = pts.shape[0] - 1
        dt = 1.0 / self.a.R
        age = (jb - (ja + torch.arange(n, device=self.dev, dtype=torch.float32) + 0.5)) * dt
        wseg = dt * p.fps * self.amps[None] * torch.exp(-age[:, None] / self.taus[None])
        return splat_path(pts, wseg, p.H, p.W, p.step)

    def step(self, buf, t_end):
        """纯函数：(上一帧缓冲, 本帧终点时间) → 新缓冲"""
        new = self.deposit(t_end)
        return new if buf is None else buf * self.decay + new

    def warm(self, t_end_first):
        """预热：从 t_end_first - warmup 起逐帧推，返回进入第一帧之前的缓冲"""
        p = self.p
        wu = p.warmup if p.warmup >= 0 else 6 * float(self.taus.max())
        n = int(math.ceil(wu * p.fps))
        buf = None
        for k in range(n, 0, -1):
            buf = self.step(buf, t_end_first - k / p.fps)
        return buf

    def grid(self):
        if self._grid is None:
            p = self.p
            g = p.grid_px or 95 * p.px
            yy, xx = torch.meshgrid(torch.arange(p.H, device=self.dev), torch.arange(p.W, device=self.dev), indexing="ij")
            dx = torch.remainder(xx - p.W / 2 + 0.5, g)
            dy = torch.remainder(yy - p.H / 2 + 0.5, g)
            line = torch.maximum(torch.exp(-0.5 * (torch.minimum(dx, g - dx) / 0.7) ** 2),
                                 torch.exp(-0.5 * (torch.minimum(dy, g - dy) / 0.7) ** 2))
            self._grid = 1 - p.graticule * line
        return self._grid

    def display(self, buf, frame_idx=0):
        """缓冲 → RGB uint8 (H, W, 3)"""
        p = self.p
        core = (p.gain * K0 * gblur(buf.sum(0), p.beam * p.px)).clamp_min(0).pow(p.response)   # 曝光（响应指数 γ）
        D = core.clone()
        for r, w in zip(p.glow_radii, p.glow_weights):
            D += p.glow * w * gblur(core, r * p.px)
        E = (D + p.black)[None]
        lin = 1 - torch.exp(-(E * self.col + p.whiten * E * E))      # 线性光；whiten 让亮核三通道一起饱和 → 发白
        if p.graticule > 0:
            lin = lin * self.grid()
        if p.grain > 0:
            g = torch.Generator(device=self.dev).manual_seed(int(frame_idx) * 7919 + 17)
            lin = lin * (1 + p.grain * torch.randn(1, p.H, p.W, generator=g, device=self.dev))
        lin = lin.clamp(0, 1)
        img = torch.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin.pow(1 / 2.4) - 0.055)   # sRGB 编码
        return (img.clamp(0, 1) * 255 + 0.5).byte().permute(1, 2, 0).contiguous().cpu().numpy()


# ---------------------------------------------------------------- 渲染入口
def render_frame(audio_path, t, p: Params, device=None):
    """单帧：pts = t（窗口终点 t + lead）"""
    a = Audio(audio_path, p.upsample)
    sc = Scope(a, p, device)
    te = t + p.lead
    buf = sc.step(sc.warm(te), te)
    return sc.display(buf, int(round(t * p.fps)))


def render_clip(audio_path, out, t0, t1, p: Params, device=None, duty=1.0, log=print):
    import imageio_ffmpeg
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} 已存在 —— 样片不覆盖，换版本名")
    out.parent.mkdir(parents=True, exist_ok=True)
    a = Audio(audio_path, p.upsample)
    sc = Scope(a, p, device)
    n = int(round((t1 - t0) * p.fps))
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ff, "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{p.W}x{p.H}", "-r", str(p.fps), "-i", "-",
           "-ss", f"{t0:.6f}", "-t", f"{n / p.fps:.6f}", "-i", str(audio_path),
           "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "320k", "-ar", "48000", "-shortest", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    buf = sc.warm(t0 + p.lead)
    t_start = time.time()
    for i in range(n):
        tf = time.time()
        te = t0 + i / p.fps + p.lead
        buf = sc.step(buf, te)
        proc.stdin.write(sc.display(buf, int(round((t0 + i / p.fps) * p.fps))).tobytes())
        if duty < 1.0:
            time.sleep((time.time() - tf) * (1 / duty - 1))
        if (i + 1) % (int(p.fps) * 2) == 0 or i + 1 == n:
            el = time.time() - t_start
            log(f"  {i + 1}/{n} 帧  {el:.0f}s  ({(i + 1) / el:.1f} fps)")
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg 失败")
    return out


def main():
    ap = argparse.ArgumentParser(description="示波器 XY 荧光模拟")
    ap.add_argument("audio")
    ap.add_argument("--out", required=True)
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--frame", type=float, default=None, help="只渲这一时刻的单帧 PNG")
    ap.add_argument("--fps", type=float, default=60)
    ap.add_argument("--size", default="1440x1080")
    ap.add_argument("--color", default="green", choices=sorted(COLORS))
    ap.add_argument("--persist", type=float, default=Params.persist)
    ap.add_argument("--glow", type=float, default=Params.glow)
    ap.add_argument("--gain", type=float, default=Params.gain)
    ap.add_argument("--beam", type=float, default=Params.beam)
    ap.add_argument("--scale", type=float, default=Params.scale)
    ap.add_argument("--upsample", type=int, default=Params.upsample)
    ap.add_argument("--lead", type=float, default=Params.lead)
    ap.add_argument("--grain", type=float, default=0.0)
    ap.add_argument("--graticule", type=float, nargs="?", const=0.35, default=0.0, help="网格（默认关；给了不带值 = 0.35）")
    ap.add_argument("--vram", type=float, default=0.5, help="本进程显存上限（占总显存比例）")
    ap.add_argument("--duty", type=float, default=0.9, help="GPU 占空比：每帧算完歇 (1/duty - 1) 倍时间")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    W, H = (int(v) for v in a.size.lower().split("x"))
    p = Params(fps=a.fps, W=W, H=H, color=a.color, persist=a.persist, glow=a.glow, gain=a.gain, beam=a.beam,
               scale=a.scale, upsample=a.upsample, lead=a.lead, grain=a.grain, graticule=a.graticule)
    dev = "cpu" if a.cpu or not torch.cuda.is_available() else "cuda"
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(a.vram)
    if a.frame is not None:
        from PIL import Image
        out = Path(a.out)
        if out.exists():
            raise FileExistsError(f"{out} 已存在 —— 换版本名")
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(render_frame(a.audio, a.frame, p, dev)).save(out)
        print(out)
        return
    t1 = a.t1 if a.t1 is not None else sf.info(a.audio).duration
    print(f"渲 {a.t0:.2f}–{t1:.2f} s，{W}x{H}@{a.fps}，{dev}，参数 {asdict(p)}")
    print(render_clip(a.audio, a.out, a.t0, t1, p, dev, a.duty))


if __name__ == "__main__":
    main()
