#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🎬样片-开头 0–38.70 s v1（👤2026-10-21 看粗剪 v4.3 的反馈）。
  00 信号 0–13.11：一根弦被这首歌的低音从 0.46 L 处驱动（d'Alembert 外行波，两端吸收）。
     弦 = 延迟线：节点离驱动点 d → 显示的是 x(t − d/c)。把弦「折」起来：
     节点坐标从 (位置, x(s)) 渐变到 (x(s−τ), x(s), x(s−2τ)) —— u = 1 时正好是 Takens 延迟嵌入。
     画面占比按情绪大图 00：弦横跨 80 % 画宽，团占 ~3/4 画高。
  01 回想 13.11–38.71：同一个 Kuramoto（N = 3000，K 每拍 +0.035），四个视角按乐句切：
     wf_song（歌的重分配 STFT 瀑布图，替换旧 raster @≈17 s）/ scatter（叶序圆盘，每点按自己的相位闪）/
     ring（相位环）/ wf_kura（3000 个振子的瞬时频率瀑布图：锁相的塌成一根载波线，替换旧 raster @≈37 s）。
  记忆 blink / cut-in 不在这里（👤剪辑时插）。镜头从不静止：基础速度 + 强起音时只加「速度」（2–3 帧进，指数缓，不归零）；
  没有拍点驱动的全局缩放 / 抖动。红 = 驱动区（弦）/ 锁相边界上的振子（scatter、ring）/ 跟踪的载波（瀑布图）。
用法：
  python tools/opening.py                       # 主片 + 四个视角的素材 + 接触表
  python tools/opening.py --still               # 只出瀑布图静帧（灰 + 色 两张）
  python tools/opening.py --end 3 --no-material # 快速试渲前 3 秒
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stills as S  # noqa: E402

W, H, FPS = 1280, 720, 24
T_END = 38.70
PH0, PHRASE = 13.11, 6.4
SR = 44100
ROWS_PER_S = 96                         # 瀑布图每行 10.4 ms，每帧 4 行
FONT = S.FONT
GOLD = np.pi * (3 - np.sqrt(5))


# ---------------------------------------------------------------- 公共
def smooth(t, a, b):
    u = np.clip((t - a) / (b - a), 0, 1)
    return u * u * (3 - 2 * u)


def splat_aa(x, y, w, h, wt=None):
    """亚像素（双线性）点溅射：线才不锯齿"""
    x0 = np.floor(x); y0 = np.floor(y); fx = x - x0; fy = y - y0
    out = np.zeros(h * w)
    one = np.ones_like(x) if wt is None else wt
    for dx, dy, ww in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi = (x0 + dx).astype(np.int64); yi = (y0 + dy).astype(np.int64)
        ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        out += np.bincount(yi[ok] * w + xi[ok], (ww * one)[ok], minlength=h * w)
    return out.reshape(h, w)


def load_audio(dur=40.0, sr=SR):
    """样本 i ↔ 播放时间 i/sr（走 sensorium.decode：原曲音频晚 0.042 s 开始，从 0 解码要补零）"""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import sensorium
    return sensorium.load_audio(S.MUSIC, 0.0, dur, sr)                 # float64，满幅 = 1（decode 是 int16）


def sense():
    import json
    return json.loads((S.MV / "感官.json").read_text("utf-8"))


def channel(sn, name, t):
    v = sn["channels"][name]
    return v[min(int(t * sn["fps"]), len(v) - 1)]


def strong_onsets(sn, t0, t1, pct=88):
    """强起音（威胁通道 ±3 帧最大值的前 12 %）= 镜头「进」的锚点"""
    on = np.array(sn["events"]["onsets"]); th = np.array(sn["channels"]["threat"]); fps = sn["fps"]
    s = np.array([th[max(0, int(t * fps) - 1):int(t * fps) + 4].max() for t in on])
    m = (on >= t0) & (on < t1)
    return on[m & (s >= np.percentile(s[m], pct))]


class Cam:
    """镜头速度 = 基础速度 + 锚点处的「进」（2.5 帧爬升）接「缓」（指数衰减）；位置 = 速度的积分，永不停"""

    def __init__(self, base, anchors, amp, decay=0.5):
        self.base, self.anchors, self.amp, self.decay = base, np.asarray(anchors), amp, decay
        self.pos = 0.0

    def speed(self, t):
        d = t - self.anchors; d = d[d >= 0]
        kick = (np.minimum(1, d / (2.5 / FPS)) * np.exp(-d / self.decay)).sum() if len(d) else 0.0
        return self.base(t) + self.amp * kick

    def step(self, t):
        self.pos += self.speed(t) / FPS
        return self.pos


def project(X, Y, Z, yaw, pitch, dist, f):
    """绕竖轴 yaw、绕横轴 pitch，透视投影；返回屏幕坐标和深度亮度"""
    x1 = X * np.cos(yaw) + Z * np.sin(yaw); z1 = -X * np.sin(yaw) + Z * np.cos(yaw)
    y2 = Y * np.cos(pitch) - z1 * np.sin(pitch); z2 = Y * np.sin(pitch) + z1 * np.cos(pitch)
    k = f / (z2 + dist)
    return W / 2 + x1 * k, H / 2 + y2 * k, np.clip(k, 0.3, 2.0)


def to_u8(img):
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def hud(im, lines, title, s=1.0):
    d = ImageDraw.Draw(im); f = ImageFont.truetype(FONT, int(15 * s))
    w, h = im.size
    for j, line in enumerate(lines):
        d.text((int(32 * s), h - int(32 * s) - int(21 * s) * (len(lines) - j)), line, font=f, fill=(150, 160, 168))
    d.text((w - int(32 * s) - int(9 * s) * len(title), int(22 * s)), title, font=f, fill=(110, 118, 124))
    return im


# ---------------------------------------------------------------- 00 弦 → 延迟嵌入
class String:
    def __init__(self, sn):
        sr = 11025
        x = load_audio(15.0, sr)
        k = int(sr / 120)
        xl = np.convolve(np.convolve(x, np.ones(k) / k, "same"), np.ones(k) / k, "same")    # ≈ 低通 120 Hz
        n = int(2.0 * sr)
        env = np.sqrt(np.convolve(xl ** 2, np.ones(n) / n, "same")) + 1e-4                     # 自动增益 2 s
        self.x = xl / (2.8 * env); self.sr = sr
        ac = np.correlate(xl[sr:3 * sr], xl[sr:3 * sr], "full")[2 * sr - 1:]
        self.tau = int(np.argmax(ac < 0)) / sr                                                  # τ = 自相关第一个零点
        self.p0 = -0.08                                                                         # 驱动点 0.46 L
        self.N = 2400
        self.pos = np.linspace(-1, 1, self.N)
        self.taper = smooth(1 - np.abs(self.pos), 0, 0.07)                                      # 两端固定、吸收
        self.red = np.abs(self.pos - self.p0) < 0.03
        self.acc = np.zeros((H, W)); self.accr = np.zeros((H, W))
        self.g = 1.0                                                     # 折叠后的显示增益（弱信号时最多 ×2.5，HUD 写出来）
        an = strong_onsets(sn, 0, PH0)
        self.yaw = Cam(lambda t: 0.015 + 0.11 * smooth(t, 5, 11), an, 0.10, 0.6)
        self.Lh, self.G = 0.40 * W, 0.42 * H

    def xs(self, s):
        i = s * self.sr
        i0 = np.clip(np.floor(i).astype(np.int64), 0, len(self.x) - 2); fr = np.clip(i - i0, 0, 1)
        v = self.x[i0] * (1 - fr) + self.x[i0 + 1] * fr
        return np.where(s < 0, 0.0, v)

    def params(self, t):
        hold = 0.006 + 0.40 * smooth(t, 1.2, 8.0) ** 1.4      # 弦上装多少秒（外侧半边）
        u = smooth(t, 6.0, 12.0)                              # 折叠：1 维线 → 3 维嵌入
        amp = (0.05 + 0.10 * smooth(t, 0.5, 6.0)) * H
        lam = 0.08 + 1.4 * smooth(t, 1.5, 9.0) ** 1.5            # 损耗长度（L）：抖动从驱动点往两端传开
        return hold, u, amp, lam

    def frame(self, t):
        yaw0 = self.yaw.step(t)
        hold, u, amp, lam = self.params(t)
        c = (1 - self.p0) / hold
        frame_pts, red_pts = [], []
        sub, shutter = 4, 1 / 120                                    # 短快门：一帧里看清一根波形，不糊成管子
        for j in range(sub):
            ts = t + j * shutter / sub
            s = ts - np.abs(self.pos - self.p0) / c                         # 推迟时间：弦 = 延迟线
            xs0, xs1, xs2 = self.xs(s), self.xs(s - self.tau), self.xs(s - 2 * self.tau)
            if j == 0:                                                   # 吸气前低音变弱 → 团缩成点；显示增益托一下，不让它消失
                r = np.sqrt((xs0 ** 2).mean()) + 1e-6
                self.g += (np.clip(0.45 / r, 1.0, 2.5) - self.g) * 0.25
            Gf = self.G * (1 + (self.g - 1) * u)
            loss = np.exp(-np.abs(self.pos - self.p0) / lam)               # 有损弦：离驱动点越远越小
            tap = (1 - u) * self.taper * loss + u
            X = (1 - u) * self.pos * self.Lh + u * Gf * xs1
            Y = -((1 - u) * amp * xs0 * tap + u * Gf * xs0)
            Z = u * Gf * xs2
            yaw = yaw0 + j * self.yaw.speed(t) * shutter / sub
            pitch = 0.16 + 0.14 * np.sin(0.23 * ts + 0.4)
            px, py, k = project(X, Y, Z, yaw, pitch, 2200 - 22 * ts, 2200)
            tt = np.linspace(0, 1, 3, endpoint=False)[:, None]
            lx = (px[:-1] * (1 - tt) + px[1:] * tt).ravel(); ly = (py[:-1] * (1 - tt) + py[1:] * tt).ravel()
            lw = np.tile((k[:-1] ** 2) * self.taper[:-1] ** 0.5 * (1 - u + u * loss[:-1] ** 0.35), 3)
            rr = np.tile(self.red[:-1], 3)
            frame_pts.append((lx, ly, lw)); red_pts.append((lx[rr], ly[rr], lw[rr]))
        d = sum(splat_aa(a, b, W, H, c_) for a, b, c_ in frame_pts) / sub
        dr = sum(splat_aa(a, b, W, H, c_) for a, b, c_ in red_pts) / sub
        self.acc = self.acc * 0.5 + d; self.accr = self.accr * 0.5 + dr
        v = 1 - np.exp(-self.acc / 2.2); vr = 1 - np.exp(-self.accr / 1.6)
        img = v[..., None] * S.GREY * (1 - vr[..., None]) + vr[..., None] * S.RED
        img = S.glow(np.clip(img, 0, 1), 3, 0.55)
        im = Image.fromarray(to_u8(img))
        return hud(im, [f"x(t) = Can't Kill Us   t = {t:6.2f} s   bass < 120 Hz, AGC 2 s",
                        f"string: driven at 0.46 L   c = {c:6.2f} L/s   holds {hold * 1000:5.0f} ms   loss length {lam:4.2f} L   ends absorb",
                        f"fold u = {u:.2f}   (x(s), x(s-tau), x(s-2tau))   tau = {self.tau * 1000:.1f} ms (1st ACF zero)   dim = {1 + 2 * u:.2f}   display gain x{1 + (self.g - 1) * u:.2f}"],
                   "00  SIGNAL")


# ---------------------------------------------------------------- 重分配 STFT
def reassigned_power(x, sr, centers, n=4096, df=0.5, fmax=3000.0, chunk=96):
    """频率重分配 STFT：每个 bin 的能量挪到它的瞬时频率 f̂ = f_k − Im(X_dh·X_h*)/|X_h|²·sr/2π 上。
    返回 (行, 细 bin) 的功率，单位 = 正弦振幅²（满幅正弦 = 0 dBFS）：整个主瓣加起来 = A²"""
    k = np.arange(n)
    h = 0.5 - 0.5 * np.cos(2 * np.pi * k / n)
    dh = (np.pi / n) * np.sin(2 * np.pi * k / n)
    norm = 4.0 / (h.sum() ** 2 * (n * (h ** 2).sum() / h.sum() ** 2))
    fk = np.fft.rfftfreq(n, 1 / sr)
    nb = int(fmax / df)
    out = np.zeros((len(centers), nb), np.float64)
    xp = np.pad(x, (n, n))
    for a in range(0, len(centers), chunk):
        c = np.asarray(centers[a:a + chunk]) + n - n // 2
        fr = xp[c[:, None] + k[None, :]]
        X = np.fft.rfft(fr * h, axis=1); Xd = np.fft.rfft(fr * dh, axis=1)
        P = np.abs(X) ** 2
        fh = fk[None, :] - np.imag(Xd * np.conj(X)) / (P + 1e-30) * sr / (2 * np.pi)
        idx = np.floor(fh / df).astype(np.int64)
        ok = (idx >= 0) & (idx < nb) & (P > 0)
        rows = np.broadcast_to(np.arange(len(c))[:, None], idx.shape)
        out[a:a + len(c)] = np.bincount((rows * nb + idx)[ok], (P * norm)[ok], minlength=len(c) * nb).reshape(len(c), nb)
    return out


# ---------------------------------------------------------------- 瀑布图渲染（通用：歌 / 振子）
DB_LO, DB_HI = -80.0, -14.0
KURA_FULL = 300.0
TAU_EFF = 0.5                           # 有效频率的平均时间（s）
CMAP_GREY = [(0.0, (0, 0, 0)), (0.3, (0.035, 0.04, 0.045)), (0.6, (0.30, 0.33, 0.35)), (0.85, (0.72, 0.76, 0.79)), (1.0, (0.97, 0.98, 1.0))]
CMAP_MICRO = [(0.0, (0, 0, 0)), (0.3, (0.035, 0.04, 0.045)), (0.6, (0.30, 0.33, 0.35)), (0.8, (0.50, 0.69, 0.75)),
              (0.92, (0.86, 0.86, 0.72)), (1.0, (1.0, 0.95, 0.82))]   # 局部微彩：只有最强的载波带一点冷青→暖白，饱和度低
MARK = {"grey": (200, 206, 210), "micro": (170, 214, 220), "sdr": (200, 206, 210)}
PALETTE = {"name": "grey", "stops": CMAP_GREY}
CMAP_SDR = [(0.0, (0, 0, 0.01)), (0.3, (0.02, 0.05, 0.20)), (0.55, (0.05, 0.30, 0.42)), (0.8, (0.72, 0.68, 0.38)), (1.0, (1.0, 0.97, 0.90))]
PALETTES = {"grey": CMAP_GREY, "micro": CMAP_MICRO, "sdr": CMAP_SDR}


def cmap(v, stops):
    xs = np.array([s[0] for s in stops]); cs = np.array([s[1] for s in stops])
    return np.stack([np.interp(v, xs, cs[:, i]) for i in range(3)], -1)


class Waterfall:
    """rows: (行, 细 bin) 功率 / 计数；f0 + i·df 是细 bin 左沿。显示时用累积和做分数 bin 边的积分 → 列宽不整也不出条纹"""

    def __init__(self, kind, f0, df, nb):
        self.kind, self.f0, self.df, self.nb = kind, f0, df, nb
        self.peak = np.zeros(nb)

    def bin_cols(self, block, lo, hi, ncol):
        cs = np.concatenate([np.zeros((block.shape[0], 1)), np.cumsum(block, 1)], 1)
        e = (np.linspace(lo, hi, ncol + 1) - self.f0) / self.df
        e = np.clip(e, 0, self.nb)
        i0 = np.minimum(np.floor(e).astype(np.int64), self.nb - 1); fr = e - i0
        v = cs[:, i0] * (1 - fr) + cs[:, i0 + 1] * fr
        return np.maximum(np.diff(v, axis=1), 0)

    def level(self, p):
        if self.kind == "song":                  # dBFS → 0..1，范围 DB_LO … DB_HI
            return np.clip((10 * np.log10(p + 1e-14) - DB_LO) / (DB_HI - DB_LO), 0, 1)
        return np.clip(np.log1p(4 * p) / np.log1p(4 * KURA_FULL), 0, 1) ** 0.7       # 振子个数（对数）

    def render(self, rows, lo, hi, w, h, stops, red_f=None, labels=(), title="", s=1.0, tick=100.0, unit="Hz", fmt="{:.0f}"):
        """rows: 最近若干行（最新在最后）；返回 PIL 图"""
        m = int(28 * s)
        x0, x1 = m, w - m - int(46 * s)
        y0 = m + int(26 * s); y1 = y0 + int(0.54 * h)
        sy0 = y1 + int(32 * s); sy1 = sy0 + int(0.17 * h)
        ncol, nrow = x1 - x0, y1 - y0
        self.geom = (x0, x1, y0, y1, lo, hi)
        blk = rows[-nrow:]
        col = self.bin_cols(blk, lo, hi, ncol)[::-1]                       # 最新在上
        v = self.level(col)
        if len(v) < nrow:
            v = np.vstack([v, np.zeros((nrow - len(v), ncol))])
        img = np.zeros((h, w, 3))
        img[y0:y1, x0:x1] = cmap(v, stops)
        fpx = np.linspace(lo, hi, ncol + 1)[:-1] + (hi - lo) / ncol / 2
        if red_f is not None:                                              # 红：跟踪的那根载波
            near = np.exp(-0.5 * ((fpx - red_f) / ((hi - lo) / ncol * 1.6)) ** 2)
            a = (v * near[None, :]) ** 0.8
            img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - a[..., None]) + a[..., None] * S.RED
        # 频谱面板：这一帧的行均值 + 峰值保持
        cur = rows[-8:].mean(0)
        self.peak = np.maximum(self.peak * (0.92 if self.kind == "song" else 0.85), cur)
        box = np.ones(5)                                                   # 面板曲线：5 px 内功率求和（重分配的空列不掉到底）
        tr = self.level(np.convolve(self.bin_cols(cur[None], lo, hi, ncol)[0], box, "same"))
        pk = self.level(np.convolve(self.bin_cols(self.peak[None], lo, hi, ncol)[0], box, "same"))
        im = Image.fromarray(to_u8(img))
        # 抗锯齿：2× 画线再缩
        ss = 2 if s < 2 else 1
        lay = Image.new("L", (w * ss, h * ss), 0); dl = ImageDraw.Draw(lay)
        ty = lambda q: (sy1 - q * (sy1 - sy0)) * ss  # noqa: E731
        pts = [((x0 + i) * ss, ty(q)) for i, q in enumerate(tr)]
        dl.line(pts, fill=235, width=ss)
        for i in range(0, ncol, 2):
            yy = ty(pk[i]); dl.rectangle([(x0 + i) * ss, yy, (x0 + i) * ss + 1, yy + 1], fill=110)
        lay = lay.resize((w, h), Image.LANCZOS)
        base = np.asarray(im, np.float64) / 255
        l = np.asarray(lay, np.float64)[..., None] / 255
        base = np.clip(base + l * np.array([0.9, 0.93, 0.95]), 0, 1)
        im = Image.fromarray(to_u8(base))
        d = ImageDraw.Draw(im)
        fs = ImageFont.truetype(FONT, int(12 * s)); fh = ImageFont.truetype(FONT, int(13 * s))
        gc = (70, 76, 80); lc = (130, 138, 144)
        d.rectangle([x0 - 1, y0 - 1, x1, y1], outline=(48, 52, 56))
        d.rectangle([x0 - 1, sy0, x1, sy1], outline=(48, 52, 56))
        # 频率刻度
        lab_w = len(fmt.format(hi)) * 7.2 * s + 10 * s                     # 标签间距不小于标签宽
        step = next(q * 10.0 ** e for e in range(-3, 5) for q in (1, 2, 5) if q * 10.0 ** e / (hi - lo) * ncol >= lab_w)
        minor = step / 5
        for k_ in range(int(np.ceil(lo / minor)), int(np.floor(hi / minor)) + 1):
            f = k_ * minor
            xx = x0 + (f - lo) / (hi - lo) * ncol
            major = k_ % 5 == 0
            d.line([(xx, y1), (xx, y1 + int((7 if major else 3) * s))], fill=lc)
            if major:
                d.line([(xx, sy0), (xx, sy1)], fill=(28, 31, 33))
                lab = fmt.format(f)
                d.text((xx - len(lab) * 3.4 * s, y1 + int(9 * s)), lab, font=fs, fill=lc)
        d.text((x1 + int(6 * s), y1 + int(9 * s)), unit, font=fs, fill=lc)
        # 时间刻度（右）
        for sec in range(1, int(nrow / ROWS_PER_S) + 1):
            yy = y0 + sec * ROWS_PER_S
            d.line([(x1, yy), (x1 + int(5 * s), yy)], fill=lc)
            d.text((x1 + int(8 * s), yy - int(7 * s)), f"-{sec}s", font=fs, fill=lc)
        # 面板网格
        if self.kind == "song":
            for db in (-20, -30, -40, -50, -60, -70):
                q = (db - DB_LO) / (DB_HI - DB_LO); yy = sy1 - q * (sy1 - sy0)
                d.line([(x0, yy), (x1, yy)], fill=(34, 37, 40))
                d.text((x1 + int(6 * s), yy - int(7 * s)), f"{db}", font=fs, fill=gc)
            d.text((x1 + int(6 * s), sy1 - int(2 * s)), "dBFS", font=fs, fill=gc)
        else:
            for n_ in (1, 10, 100):
                q = (np.log1p(4 * n_) / np.log1p(4 * KURA_FULL)) ** 0.7; yy = sy1 - q * (sy1 - sy0)
                d.line([(x0, yy), (x1, yy)], fill=(34, 37, 40))
                d.text((x1 + int(6 * s), yy - int(7 * s)), f"{n_}", font=fs, fill=gc)
            d.text((x1 + int(6 * s), sy1 - int(2 * s)), "osc", font=fs, fill=gc)
        # 峰值标记
        sm = np.convolve(tr, np.ones(5) / 5, "same")
        cand = [i for i in range(3, ncol - 3) if sm[i] == sm[i - 3:i + 4].max() and sm[i] > 0.25]
        cand.sort(key=lambda i: -sm[i]); pick = []
        for i in cand:
            if all(abs(i - j) > 70 * s for j in pick):
                pick.append(i)
            if len(pick) == 4:
                break
        for i in pick:
            fv = fpx[i]; isred = red_f is not None and abs(fv - red_f) < (hi - lo) / ncol * 4
            colr = (255, 41, 51) if isred else MARK[PALETTE["name"]]
            yy = max(sy1 - tr[i] * (sy1 - sy0), sy0 + 24 * s)
            d.polygon([(x0 + i, yy - 4 * s), (x0 + i - 4 * s, yy - 11 * s), (x0 + i + 4 * s, yy - 11 * s)], fill=colr)
            d.text((x0 + i + 6 * s, yy - 22 * s), fmt.format(fv) if self.kind != "song" else f"{fv:.1f}", font=fs, fill=colr)
        if red_f is not None and lo < red_f < hi:
            xx = x0 + (red_f - lo) / (hi - lo) * ncol
            d.polygon([(xx, y0 - 2 * s), (xx - 5 * s, y0 - 10 * s), (xx + 5 * s, y0 - 10 * s)], fill=(255, 41, 51))
        d.text((x0, m - int(4 * s)), "  ".join(labels), font=fh, fill=(120, 128, 134))
        return im


class SongWF:
    """这首歌 0–39.5 s 的重分配 STFT，每 10.4 ms 一行"""

    def __init__(self, t1=39.5, n=4096, df=0.5, fmax=3000.0):
        x = load_audio(t1 + 0.5)
        self.n, self.df = n, df
        self.r_end = int(t1 * ROWS_PER_S)
        centers = np.round(np.arange(self.r_end) * SR / ROWS_PER_S).astype(np.int64)
        t0 = time.time()
        self.P = reassigned_power(x, SR, centers, n, df, fmax).astype(np.float32)
        print(f"  reassigned STFT {self.P.shape} in {time.time() - t0:.0f} s", flush=True)
        self.wf = Waterfall("song", 0.0, df, self.P.shape[1])
        # 跟踪的载波：最近 1 s 平均功率最大的细 bin（30–1500 Hz，3 Hz 平滑）
        self.track = np.zeros(self.r_end)
        ker = np.ones(6) / 6
        a, b = int(30 / df), int(1500 / df)
        for r in range(0, self.r_end, 4):
            seg = self.P[max(0, r - ROWS_PER_S):r + 1, a:b].mean(0)
            self.track[r:r + 4] = (a + np.argmax(np.convolve(seg, ker, "same")) + 0.5) * df

    @staticmethod
    def view(t, a=PH0, b=PH0 + 4 * PHRASE):
        """调谐漂移：中心 700 → 600 Hz，跨度 1400 → 1150 Hz（匀速，永不停）"""
        u = (t - a) / (b - a)
        return 700 - 100 * u, 1400 - 250 * u

    def frame(self, t, w=W, h=H, stops=None, s=1.0, title="01  RECALL / SPECTRUM", cam=None):
        r = int(round(t * ROWS_PER_S))
        rf = float(self.track[min(r, self.r_end - 1)])
        if cam is not None:
            return cam.shoot(t, rf, lambda ww, hh, ss: self.frame(t, ww, hh, stops, ss, title, None), " Hz", "{:.1f}")
        stops = stops or PALETTE["stops"]
        c, sp = SongWF.view(t)
        lo, hi = c - sp / 2, c + sp / 2
        rf = float(self.track[min(r, self.r_end - 1)])
        im = self.wf.render(self.P[max(0, r - 1200):r + 1].astype(np.float64), lo, hi, w, h, stops, red_f=rf, s=s,
                            labels=["WATERFALL", "Can't Kill Us", f"fs {SR} Hz", f"STFT N={self.n} Hann, freq-reassigned",
                                    f"hop {1000 / ROWS_PER_S:.1f} ms"], tick=50.0)
        cur = self.P[r - 3:r + 1].mean(0)
        i = int(rf / self.df)
        pdb = max(10 * np.log10(cur[max(0, i - 6):i + 7].sum() + 1e-14), -99.9)   # 低音音符间隙 → 本帧此处无能量
        self.hud_lines = None
        lines = [f"t = {t:6.2f} s   center {c:6.1f} Hz   span {sp:6.1f} Hz   {(hi - lo) / ((w - 2 * int(28 * s) - int(46 * s))):.2f} Hz/px",
                        f"tracked carrier (red) {rf:6.1f} Hz   {pdb:6.1f} dBFS   (max 1-s mean power, 30-1500 Hz)",
                 "rows = 10.4 ms, newest on top   trace = this frame   dots = peak hold"]
        if getattr(self, "_nohud", False):
            self.hud_lines, self.hud_title = lines, title
            return im
        return hud(im, lines, title, s)


# ---------------------------------------------------------------- 01 Kuramoto 四视角
class Kuramoto:
    def __init__(self, sn, N=3000, seed=1):
        rng = np.random.default_rng(seed)
        self.N = N
        self.w = np.clip(rng.standard_cauchy(N) * 0.5, -3, 3)
        self.th = rng.uniform(0, 2 * np.pi, N)
        self.rng = np.random.default_rng(seed + 100)
        self.K = 0.8; self.sn = sn
        self.beats = np.array(sn["events"]["beats"])
        self.R = 0.0; self.psi = 0.0
        self.f_lo, self.df, self.nb = -1.6, 0.001, 3600                 # 瞬时频率细 bin（Hz）
        self.rows = []
        self.eff = None; self.kick = np.zeros(N)
        self.wf = Waterfall("kura", self.f_lo, self.df, self.nb)
        ra = np.argsort(np.argsort(np.abs(self.w)))                     # |w| 的秩 → 叶序圆盘半径
        self.rho = np.sqrt((ra + 0.5) / N)
        self.alpha = np.arange(N) * GOLD
        self.acc_s = np.zeros((H, W)); self.accr_s = np.zeros((H, W))
        self.acc_r = np.zeros((H, W)); self.accr_r = np.zeros((H, W))
        an = strong_onsets(sn, PH0, 38.71)
        self.cam_s = Cam(lambda t: 0.07, an, 0.22, 0.55)
        self.cam_r = Cam(lambda t: -0.06, an, -0.20, 0.55)
        self.omega = 0.0

    def step(self, t):
        nb = int(((self.beats > t - 1 / FPS) & (self.beats <= t)).sum())
        self.K += nb * 0.035
        hb = channel(self.sn, "heartbeat", t)
        dt = 1 / (FPS * 4)
        for j in range(4):
            z = np.exp(1j * self.th).mean()
            v = 2.2 * (self.w + 0.6 * hb + self.K * abs(z) * np.sin(np.angle(z) - self.th))
            self.th = self.th + v * dt
            # 有效频率 = 总相位推进（含上一帧的踢）/ 时间，EMA TAU_EFF 秒；锁相的共享同一个值（频率平台）
            dth = v * dt + (self.kick if j == 0 else 0)
            self.eff = dth / dt if self.eff is None else self.eff + (dth / dt - self.eff) * dt / TAU_EFF
            f = self.eff / (2 * np.pi)
            e = (f - self.f_lo) / self.df; i0 = np.floor(e).astype(np.int64); fr = e - i0
            row = np.zeros(self.nb)
            for ii, ww in ((i0, 1 - fr), (i0 + 1, fr)):
                ok = (ii >= 0) & (ii < self.nb)
                row += np.bincount(ii[ok], ww[ok], minlength=self.nb)
            self.rows.append(row.astype(np.float32))
            self.omega = float(np.median(f[np.abs(self.w) < self.K * abs(z)])) if (np.abs(self.w) < self.K * abs(z)).any() else float(np.median(f))
        self.rows = self.rows[-1300:]
        self.kick = self.rng.normal(0, 0.35 * channel(self.sn, "threat", t), self.N)
        self.th = self.th + self.kick                                       # 威胁 = 相位被踢
        z = np.exp(1j * self.th).mean(); self.R = abs(z); self.psi = np.angle(z)
        self.lock = np.abs(self.w) < self.K * self.R
        self.edge = np.abs(np.abs(self.w) - self.K * self.R) < 0.05 + 0.02 * self.K * self.R   # 锁相边界：快要锁上 / 快要滑脱
        self.ys = self.cam_s.step(t); self.yr = self.cam_r.step(t)

    def stats(self, t):
        return [f"t = {t:6.2f} s   N = {self.N}   K = {self.K:.2f}  (+0.035 per beat, K_c = 1)",
                f"R = {self.R:.3f}   locked = {self.lock.mean() * 100:4.1f} %   lock edge |w| = K*R = {self.K * self.R:.2f}"]

    def _dots(self, X, Y, Z, yaw, pitch, dist, wt, red, acc, accr, decay, kv=1.6):
        px, py, k = project(X, Y, Z, yaw, pitch, dist, 1500)
        d = np.zeros((H, W)); dr = np.zeros((H, W))
        for ox, oy, a in ((0, 0, 1.0), (1, 0, .5), (-1, 0, .5), (0, 1, .5), (0, -1, .5)):
            d += splat_aa(px + ox * k, py + oy * k, W, H, wt * k * a)
            dr += splat_aa(px[red] + ox * k[red], py[red] + oy * k[red], W, H, (wt * k * a)[red])
        acc *= decay; acc += d; accr *= decay; accr += dr
        v = 1 - np.exp(-acc / kv); vr = 1 - np.exp(-accr / 1.2)
        img = v[..., None] * S.GREY * (1 - vr[..., None]) + vr[..., None] * S.RED
        return S.glow(np.clip(img, 0, 1), 3, 0.6)

    def scatter(self, t):
        """叶序圆盘：半径 = |w| 的秩（锁相的在中心），每点按自己的相位闪 + 抬起 → 同步从中心往外传"""
        b = ((1 + np.cos(self.th)) / 2) ** 3
        r = 420 * self.rho
        X, Z = r * np.cos(self.alpha), r * np.sin(self.alpha)
        Y = -40 * np.cos(self.th)
        img = self._dots(X, Y, Z, self.ys, 0.80 + 0.07 * np.sin(0.17 * t), 1700 - 6 * (t - PH0),
                         0.7 + 3.0 * b, self.edge, self.acc_s, self.accr_s, 0.62, 1.0)
        im = Image.fromarray(to_u8(img))
        return hud(im, self.stats(t) + ["disc: radius = rank |w_i|, golden angle;  brightness & height = cos(theta_i);  red = lock edge"],
                   "01  RECALL / FIREFLIES")

    def ring(self, t):
        """相位环：角度 = θ_i，半径 = 330 + 70·w_i；锁相的排成一根转动的辐条"""
        r = 270 + 62 * self.w
        X, Z = r * np.cos(self.th), r * np.sin(self.th)
        Y = np.zeros(self.N)
        img = self._dots(X, Y, Z, self.yr, 0.95 + 0.08 * np.sin(0.13 * t + 1), 1650 - 5 * (t - PH0),
                         np.full(self.N, 0.55), self.edge, self.acc_r, self.accr_r, 0.80)
        im = Image.fromarray(to_u8(img))
        return hud(im, self.stats(t) + ["ring: angle = theta_i, radius = 270 + 62 w_i;  trails 0.8/frame;  red = lock edge"],
                   "01  RECALL / PHASE RING")

    def macro(self, t):
        """animatic_v2 的 macro 视角原样搬来（👤2026-10-21：13d 的 26–32 s「绝佳的运动」）：
        环 r = 150 + 32·w_i，画面中心 = 环上 ψ 方向 r = 160 的点，固定 3× —— 镜头跟着同步团走；晃来自威胁踢相位"""
        r = 150 + 32 * self.w
        x, y = W / 2 + r * np.cos(self.th), H / 2 + r * np.sin(self.th)
        cx, cy = W / 2 + 160 * np.cos(self.psi), H / 2 + 160 * np.sin(self.psi)
        mx, my = (x - cx) * 3 + W / 2, (y - cy) * 3 + H / 2
        d = splat_aa(mx, my, W, H); dr = splat_aa(mx[self.lock], my[self.lock], W, H)
        base = S.colorize(np.clip(S.blur(d, 2) * 6, 0, 1) ** 0.5)
        red = np.clip(S.blur(dr, 2) * 6, 0, 1) * self.R
        img = S.glow(np.clip(base * (1 - red[..., None]) + red[..., None] * S.RED, 0, 1), 3, 0.5)
        im = Image.fromarray(to_u8(img))
        return hud(im, self.stats(t) + [f"macro: camera = ring point at psi = {np.degrees(self.psi):6.1f} deg, r = 160, zoom 3x (animatic_v2 law);  red = locked x R"],
                   "01  RECALL / MACRO")

    @staticmethod
    def view(t, a=PH0, b=PH0 + 4 * PHRASE):
        u = (t - a) / (b - a)
        return 0.05 + 0.07 * u, 2.6 - 1.75 * u ** 1.3            # 越来越近地看那根载波：踢动的展开越来越大

    def wf_frame(self, t, w=W, h=H, stops=None, s=1.0, cam=None):
        if cam is not None:
            return cam.shoot(t, self.omega, lambda ww, hh, ss: self.wf_frame(t, ww, hh, stops, ss, None), " Hz", "{:+.3f}")
        stops = stops or PALETTE["stops"]
        c, sp = Kuramoto.view(t)
        lo, hi = c - sp / 2, c + sp / 2
        rows = np.array(self.rows, np.float64)
        im = self.wf.render(rows, lo, hi, w, h, stops, red_f=self.omega, s=s, tick=0.05, unit="Hz", fmt="{:+.2f}",
                            labels=["WATERFALL", "Kuramoto population", f"N = {self.N}", "effective freq <dtheta_i/dt>/2pi, EMA 0.5 s",
                                    f"bin {self.df * 1000:.0f} mHz", f"hop {1000 / ROWS_PER_S:.1f} ms"])
        lines = self.stats(t) + [f"carrier (red) = locked group  Omega/2pi = {self.omega:+.3f} Hz   wiggles = phase slips of drifting oscillators"]
        if getattr(self, "_nohud", False):
            self.hud_lines, self.hud_title = lines, "01  RECALL / OSCILLATOR SPECTRUM"
            return im
        return hud(im, lines, "01  RECALL / OSCILLATOR SPECTRUM", s)


# ---------------------------------------------------------------- 瀑布图的镜头（借 animatic_v2 macro 视角的运动律）
def onset_table(sn, t0=0.0, t1=40.0, pct=75):
    """强起音 + 归一化强度（威胁通道 ±3 帧最大值，前 25 %）"""
    on = np.array(sn["events"]["onsets"]); th = np.array(sn["channels"]["threat"]); fps = sn["fps"]
    st = np.array([th[max(0, int(t * fps) - 1):int(t * fps) + 4].max() for t in on])
    m = (on >= t0) & (on < t1)
    keep = m & (st >= np.percentile(st[m], pct))
    return on[keep], st[keep] / st[keep].max()


def snap(t, on, win=0.4):
    """把计划的推进时刻吸附到附近最强的起音上（镜头按声音点进）"""
    near = on[np.abs(on - t) < win]
    return float(near[0]) if len(near) else t


class WFCam:
    """animatic_v2 的 macro 视角：画面中心 = 环上 ψ 方向的点（同步团所在），固定 3×；
    「移」= 同步团自己在转（Ω + 心跳通道），「晃」= 威胁通道每帧踢相位 + 强起音的 break。
    搬到瀑布图（一台被拍的仪器）上：
      · 缩放 z：乐句里按计划的推进点（吸附到强起音）一次次「进」（3 帧内到位的 70 %，0.35 s 指数缓），
                 每个强起音再加一点（log z += 0.12·强度，2.5 帧进、0.6 s 缓）；
      · 移：放大后画面中心滑向「那个东西」—— 歌 = 跟踪的载波频率，振子 = 锁相群的频率 Ω（EMA 0.25 s），纵向靠近最新的行；
      · 晃：强起音 → 一次衰减的冲击（竖向 6 Hz 阻尼振荡 0.12 s，幅度 ∝ 强度 × z）+ 每帧 ∝ 威胁通道的微抖（种子固定）。
    全在被拍的画面上做（局部放大），不是对整幅画面按拍缩放。仪器按 2–3 倍分辨率渲染再裁，放大后仍是 1:1 的细节。"""

    def __init__(self, sn, owner, steps, zmax=3.2):
        self.sn, self.owner = sn, owner
        self.on, self.st = onset_table(sn)
        self.steps = [(snap(tt, self.on) if dz > 0 else tt, dz) for tt, dz in steps]      # (时刻, Δlog z)；只有「进」吸附到起音
        self.zmax = zmax
        self.f = None
        self.rng = np.random.default_rng(7)
        self.dirs = np.random.default_rng(11).uniform(-1, 1, len(self.on))

    def zoom(self, t):
        lz = 0.0
        for ts, dz in self.steps:
            d = t - ts
            if d > 0:
                lz += dz * (1 - np.exp(-d / 0.35)) if d > 3 / FPS else dz * 0.7 * d / (3 / FPS)
        d = t - self.on; m = d >= 0
        lz += (0.12 * self.st[m] * np.minimum(1, d[m] / (2.5 / FPS)) * np.exp(-d[m] / 0.6)).sum()
        return float(np.clip(np.exp(lz), 1.0, self.zmax))

    def jolt(self, t, z):
        d = t - self.on; m = (d >= 0) & (d < 0.6)
        e = self.st[m] * np.exp(-d[m] / 0.12)
        dy = (10 * e * np.cos(2 * np.pi * 6 * d[m])).sum() * z
        dx = (6 * e * self.dirs[m] * np.cos(2 * np.pi * 6 * d[m] + 1.0)).sum() * z
        th = channel(self.sn, "threat", t)
        dx += self.rng.normal(0, 1.2) * th * z; dy += self.rng.normal(0, 1.2) * th * z
        return dx, dy

    def shoot(self, t, f_target, render, unit, fmt):
        self.f = f_target if self.f is None else self.f + (f_target - self.f) / (0.25 * FPS)
        z = self.zoom(t)
        k = 1 if z < 1.15 else (2 if z < 2.0 else 3)
        o = self.owner
        o._nohud = True
        try:
            big = render(W * k, H * k, float(k))
        finally:
            o._nohud = False
        x0, x1, y0, y1, lo, hi = o.wf.geom
        wz = float(np.clip((z - 1) / 0.8, 0, 1))
        fx = x0 + (self.f - lo) / (hi - lo) * (x1 - x0)
        fy = y0 + 0.24 * (y1 - y0)
        cx = W * k / 2 + (fx - W * k / 2) * wz; cy = H * k / 2 + (fy - H * k / 2) * wz
        dx, dy = self.jolt(t, z)
        cw, ch = W * k / z, H * k / z
        cx -= dx * cw / W; cy -= dy * ch / H
        cx = float(np.clip(cx, cw / 2, W * k - cw / 2)); cy = float(np.clip(cy, ch / 2, H * k - ch / 2))
        im = big.resize((W, H), Image.LANCZOS, box=(cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2))
        if z >= 1.15:                                                    # 放大后 HUD 压在数据上：垫一条半透明黑
            a_ = np.asarray(im, np.float64)
            a_[H - 128:, :760] *= 0.25
            im = Image.fromarray(a_.astype(np.uint8))
        return hud(im, o.hud_lines + [f"cam: zoom x{z:4.2f}  focus {fmt.format(self.f)}{unit}  jolt = onset strength (threat ch.)  [macro law, animatic_v2]"],
                   o.hud_title)


# ---------------------------------------------------------------- 主流程
def ffmpeg_pipe(path, w=W, h=H):
    import imageio_ffmpeg
    return subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16",
                             str(path)], stdin=subprocess.PIPE)


def mux(video, out, start, dur):
    """音轨 = sensorium.decode 的播放时间样本（原曲音频起点 0.042 s 已补零）→ wav → aac。
    v1 用 -ss 直接取原曲，丢了 0.042 s 的起点，音频早 ~22 ms。"""
    import wave
    import imageio_ffmpeg
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import sensorium
    x = sensorium.decode(S.MUSIC, start, dur, 48000, ac=2)
    wav = Path(video).with_suffix(".wav")
    with wave.open(str(wav), "wb") as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(48000); wf.writeframes(np.ascontiguousarray(x, np.int16).tobytes())
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(video), "-i", str(wav),
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
                    "-t", f"{dur:.3f}", str(out)], check=True)
    # 临时视频留在 --tmp 目录里，不删（不在作品目录里删任何东西）


PLAN = [(PH0, "wf_song"), (PH0 + PHRASE, "scatter"), (PH0 + 2 * PHRASE, "macro"), (PH0 + 3 * PHRASE, "wf_kura")]   # v3：26–32 用 macro（👤点名）


def which(t):
    if t < PH0:
        return "string"
    v = PLAN[0][1]
    for a, name in PLAN:
        if t >= a:
            v = name
    return v


def contact(thumbs, path, cols=8):
    tw, th = 320, 180
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * (th + 18)), (12, 12, 12))
    d = ImageDraw.Draw(sheet); f = ImageFont.truetype(FONT, 13)
    for i, (t, name, im) in enumerate(thumbs):
        x, y = (i % cols) * tw, (i // cols) * (th + 18)
        sheet.paste(im.resize((tw, th), Image.LANCZOS), (x, y + 18))
        d.text((x + 4, y + 2), f"{t:5.2f}s  {name}", font=f, fill=(170, 176, 180))
    sheet.save(path, quality=90)
    print("  contact ->", path)


def stills(ver, only=None, palette="grey"):
    sw = SongWF()
    t = 36.5
    jobs = [(PALETTES[palette], None)] if only else [(CMAP_GREY, ""), (CMAP_SDR, "_色")]
    PALETTE.update(name=palette if only else "grey", stops=PALETTES[palette])
    for stops, suffix in jobs:
        sw.wf.peak[:] = 0
        for tt in np.arange(t - 1.0, t, 1 / FPS):        # 峰值保持要有历史
            sw.wf.peak = np.maximum(sw.wf.peak * 0.92, sw.P[int(round(tt * ROWS_PER_S)) - 3:int(round(tt * ROWS_PER_S)) + 1].mean(0))
        im = sw.frame(t, 1920, 1080, stops, 1.5)
        name = only or f"22{'b' if suffix else ''}_瀑布图_高解析_{ver}{suffix}.jpg"
        out = S.MV / "静帧" / name
        if out.exists():
            raise SystemExit(f"不覆盖：{out}")
        im.save(out, quality=94)
        print("  still ->", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ver", default="v1")
    ap.add_argument("--end", type=float, default=T_END)
    ap.add_argument("--start", type=float, default=0.0, help="试渲用：从这里开始出帧（模拟仍从头算）")
    ap.add_argument("--no-material", action="store_true")
    ap.add_argument("--materials", default="wf_song,scatter,macro,ring,wf_kura", help="要单独出的视角素材，逗号分隔")
    ap.add_argument("--palette", default="grey", choices=sorted(PALETTES), help="瀑布图配色")
    ap.add_argument("--still-name", default=None, help="--still 时只出这一张（静帧/ 下的文件名），用 --palette 的配色")
    ap.add_argument("--still", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tmp", default=None, help="无声临时视频放哪（默认系统临时目录/eidolon_opening），不会被删")
    a = ap.parse_args()
    PALETTE.update(name=a.palette, stops=PALETTES[a.palette])
    if a.still:
        stills(a.ver, a.still_name, a.palette); return
    sn = sense()
    out = Path(a.out) if a.out else S.MV / "样片" / f"15_开头_0-38.7_{a.ver}.mp4"
    if out.exists():
        raise SystemExit(f"不覆盖：{out}")
    st = String(sn); km = Kuramoto(sn); sw = SongWF()
    starts = [PH0 + p * PHRASE for p in range(4)]
    cam_song = WFCam(sn, sw, [x for a_ in starts for x in ((a_ + 1.6, np.log(1.35)), (a_ + 3.2, np.log(1.6)), (a_ + PHRASE - 1e-3, -np.log(1.35 * 1.6)))])
    cam_kura = WFCam(sn, km, [x for a_ in starts for x in ((a_ + 1.6, np.log(1.3)), (a_ + 3.2, np.log(1.4)), (a_ + 4.8, np.log(1.5)),
                                                         (a_ + PHRASE - 1e-3, -np.log(1.3 * 1.4 * 1.5)))])
    print(f"  tau = {st.tau * 1000:.2f} ms", flush=True)
    nf = int(np.ceil(a.end * FPS - 1e-9)) + (0 if abs(a.end * FPS - round(a.end * FPS)) < 1e-6 else 0)
    i0 = int(np.floor(a.start * FPS))
    import tempfile
    tmpdir = Path(a.tmp) if a.tmp else Path(tempfile.gettempdir()) / "eidolon_opening"
    tmpdir.mkdir(parents=True, exist_ok=True)
    tmp = tmpdir / f"main_{a.ver}_{int(time.time())}.mp4"
    main_p = ffmpeg_pipe(tmp)
    mats = {}
    if not a.no_material:
        for v in [m for m in a.materials.split(",") if m]:
            p = S.MV / "样片" / f"15_开头素材_{v}_13.1-38.7_{a.ver}.mp4"
            if p.exists():
                raise SystemExit(f"不覆盖：{p}")
            mt = tmpdir / f"{v}_{a.ver}_{int(time.time())}.mp4"
            mats[v] = (ffmpeg_pipe(mt), p, mt)
    thumbs = []; t0 = time.time()
    for i in range(nf):
        t = i / FPS
        if t >= 10.0:
            km.step(t)
        if i < i0:
            if t < PH0:
                st.yaw.step(t)
            continue
        v = which(t)
        frames = {}
        if v == "string":
            frames["string"] = st.frame(t)
        else:
            need = set(mats) if mats else {v}
            need.add(v)
            for name in need:
                frames[name] = (sw.frame(t, cam=cam_song) if name == "wf_song" else km.scatter(t) if name == "scatter"
                                else km.ring(t) if name == "ring" else km.macro(t) if name == "macro" else km.wf_frame(t, cam=cam_kura))
            for name, (p, _, _) in mats.items():
                p.stdin.write(frames[name].tobytes())
        main_p.stdin.write(frames[v].tobytes())
        if i % FPS == 0:
            thumbs.append((t, v, frames[v]))
        if i % 48 == 0:
            print(f"  frame {i}/{nf}  t={t:5.2f}  {v}  {time.time() - t0:.0f}s", flush=True)
    main_p.stdin.close(); main_p.wait()
    mux(tmp, out, a.start if a.start else 0.0, nf / FPS - i0 / FPS if a.end != T_END or i0 else T_END)
    print("->", out)
    for name, (p, path, mt) in mats.items():
        p.stdin.close(); p.wait()
        mux(mt, path, np.ceil(PH0 * FPS) / FPS, (nf - np.ceil(PH0 * FPS)) / FPS)
        print("->", path)
    contact(thumbs, out.with_name(out.stem + "_接触表.jpg"))


if __name__ == "__main__":
    main()
