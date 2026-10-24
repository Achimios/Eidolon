#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🎬Dune-渲染-《静海》预告片引擎（MV_Dune，2026-10-24，Fable）。一切是 t 的纯函数。
元件：
  地形   程序高程场 h(x,y)：多倍频值噪声 + 脊（|noise| 翻折 = 沙丘脊线），世界坐标米；任何尺度都能取样 → A 段从一粒沙对数拉远到整片沙丘。
  等高线 h 的等值线（间距随尺度取对数），线随风场慢漂（§16 船）；底图镜头里改用底图亮度的等值线 + 我们地形的叠加。
  浮雕   低角度日光的兰伯特明暗（天光刚灭：光从地平线来）。
  沙粒   跃移粒子：D 段 68 s 最大一击 → 从沙面起飞，沿风场以临界阻尼弹簧飞向「浪的剖面」目标点（每粒一个），按 x 延迟 → 浪前从左往右掠过。
  读数   测距仪红色读数 = 地形在瞄准点的真高程 / 距离（唯一强调色）。
  字卡   无衬线极大、只名词和数字；击 = 黑帧 1 帧 + 白闪 1 帧。
  底图   Seedream 关键帧（生成/Pxx.jpg，含拆层则 2.5D）+ 等高线重绘；没有底图的镜头全程序。
用法：python tools/mv_dune/render.py still --t 9 --out x.jpg | clip --t0 0 --t1 18 --out 样片/A_吸气_0-18_v1.mp4
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
PROJ = ROOT.parents[1] / "Projekt_Visual"
MV = PROJ / "MV_Dune"
GEN = MV / "生成"
AUDIO = MV / "音床_v2.wav"
CLI = "C:/Users/Administrator/Desktop/Obsidian_V_Synapsis/core/cli.js"
W0, H0 = 1920, 1080
FONT_SANS = "C:/Windows/Fonts/bahnschrift.ttf"
FONT_MONO = "C:/Windows/Fonts/consola.ttf"
RED = (220, 40, 30)

PAL_DUSK = [(10, 8, 7), (84, 56, 30), (190, 140, 84), (238, 214, 170)]
PAL_NIGHT = [(4, 7, 12), (22, 36, 56), (74, 104, 136), (190, 206, 220)]


def ease2(u, a=0.6, t1=3 / 24, t2=9 / 24):
    if u <= 0:
        return 0.0
    return a * (1 - math.exp(-u / t1)) + (1 - a) * (1 - math.exp(-u / t2))


# ---------------------------------------------------------------- 地形（值噪声，确定性）
def _hash2(ix, iy, seed=1):
    h = (ix * 374761393 + iy * 668265263 + seed * 1442695041) & 0xFFFFFFFF
    h = (h ^ (h >> 13)) * 1274126177 & 0xFFFFFFFF
    return ((h ^ (h >> 16)) & 0xFFFFFF) / 0xFFFFFF


def value_noise(x, y, seed=1):
    ix = np.floor(x).astype(np.int64); iy = np.floor(y).astype(np.int64)
    fx = x - ix; fy = y - iy
    fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy)
    a = _hash2(ix, iy, seed); b = _hash2(ix + 1, iy, seed); c = _hash2(ix, iy + 1, seed); d = _hash2(ix + 1, iy + 1, seed)
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def terrain(x, y, px=0.0):
    """高程（米）。x, y 米；px = 每像素多少米 → 带限：波长 < 3 px 的倍频淡出（否则大尺度下亚像素采样混叠成噪声）。
    沙丘：主波长 ~180 m 的脊（翻折噪声），叠小波纹（~2 m）和沙粒尺度（~0.01 m）—— 所以任何尺度都有结构。"""
    def gain(lam):
        if px <= 0:
            return 1.0
        return float(np.clip((lam / px - 40.0) / 40.0, 0.0, 1.0))   # 波长 < 40 px 全去，< 80 px 淡出：小于这个的波纹让等高线看起来像摩尔纹
    h = np.zeros_like(x, dtype=np.float64)
    for i, (lam, amp) in enumerate(((180.0, 14.0), (70.0, 5.0), (28.0, 1.6))):
        g = gain(lam)
        if g <= 0:
            continue
        n = value_noise(x / lam + 7.3 * i, y / lam * 1.7 + 2.1 * i, seed=11 + i)
        h += g * amp * (1 - np.abs(2 * n - 1)) ** 1.6
    for i, (lam, amp) in enumerate(((2.2, 0.05), (0.9, 0.02))):
        g = gain(lam)
        if g <= 0:
            continue
        h += g * amp * np.sin(2 * np.pi * (x / lam + 0.3 * value_noise(x / 30, y / 30, seed=40 + i)) + y / (lam * 6))
    for i, (lam, amp) in enumerate(((0.012, 0.004), (0.004, 0.0015))):
        g = gain(lam)
        if g <= 0:
            continue
        h += g * amp * value_noise(x / lam, y / lam, seed=70 + i)
    return h


# ---------------------------------------------------------------- 数据
class Song:
    def __init__(self):
        ev = json.loads((MV / "事件.json").read_text("utf-8"))
        self.hits = [h["t"] for h in ev["hits"]]
        self.silences = [(s["t0"], s["t1"]) for s in ev["silences"]]
        self.duration = ev["duration"]
        tl = json.loads(subprocess.run(["node", CLI, "timeline", "--json", "--ws", str(MV)], capture_output=True, text=True, encoding="utf-8").stdout)
        self.shots = []
        for pg in tl:
            if pg["page"] == "剪辑":
                continue
            self.shots += json.loads(subprocess.run(["node", CLI, "timeline", pg["page"], "--json", "--ws", str(MV)], capture_output=True, text=True, encoding="utf-8").stdout)["items"]
        self.shots.sort(key=lambda s: s["t0"])

    def shot(self, t):
        for s in self.shots:
            if s["t0"] <= t < s["t1"]:
                return s
        return None

    WHITE_HITS = (40.0, 62.0, 68.0)          # 白闪只给三个最重要的击（ReadScreen：稀有才有分量）
    BLACK_HITS = (30.0, 46.0, 51.0, 55.0, 58.0, 60.0, 78.0)   # 这些击前 1 帧黑；其余只是硬切

    def hit_flash(self, t, fps=24):
        """击：三档 —— 白闪 1 帧 / 黑 1 帧 / 只硬切"""
        for h in self.hits:
            if h in self.WHITE_HITS and 0 <= t - h < 1 / fps:
                return "white"
            if (h in self.WHITE_HITS or h in self.BLACK_HITS) and -1 / fps <= t - h < 0:
                return "black"
        return None

    def in_silence(self, t):
        return any(a <= t < b for a, b in self.silences)


CARDS = [("IT MOVES", 18.0, 21.5), ("EVERY NIGHT", 30.0, 33.5), ("THE SAND", 46.0, 48.4), ("REMEMBERS", 56.5, 58.0),
         ("THE WATER", 71.0, 74.0), ("THE QUIET SEA", 79.0, 86.0), ("2027", 86.0, 89.0)]


class Scene:
    def __init__(self, song):
        self.s = song; self.W, self.H = W0, H0
        self.fonts = {}
        yy, xx = np.mgrid[0:H0, 0:W0]
        self.xx = xx.astype(np.float64); self.yy = yy.astype(np.float64)
        self.plates = {}
        self.wave = None

    def font(self, path, size):
        k = (path, size)
        if k not in self.fonts:
            self.fonts[k] = ImageFont.truetype(path, max(6, int(size)))
        return self.fonts[k]

    def plate(self, pid):
        if pid not in self.plates:
            p = GEN / f"{pid}.jpg"
            self.plates[pid] = np.asarray(Image.open(p).convert("RGB").resize((W0, H0), Image.BILINEAR)).astype(np.float32) / 255 if p.exists() else None
        return self.plates[pid]

    # ------------------------------------------------------------ 地形视图：俯视正交，窗宽 span 米，中心 (cx, cy)，返回 h 和世界坐标
    def terrain_view(self, cx, cy, span, step=2):
        xs = cx + (self.xx[::step, ::step] / W0 - 0.5) * span
        ys = cy + (self.yy[::step, ::step] / H0 - 0.5) * span * H0 / W0
        return terrain(xs, ys, px=span / (W0 / step)), xs, ys

    def shade_contours(self, h, span, t, pal, light_dir=(0.8, 0.3), contour_gain=1.0):
        """浮雕 + 等高线 → (H/2, W/2, 3) float"""
        gy, gx = np.gradient(h)
        px = span / h.shape[1]
        nx, ny = -gx / px, -gy / px
        lam = np.clip(0.55 + 0.45 * (nx * light_dir[0] + ny * light_dir[1]) / np.sqrt(1 + nx * nx + ny * ny), 0, 1)
        # 等高线间距：随尺度取对数（每屏约 14 条）
        rng_h = max(h.max() - h.min(), 1e-6)
        spacing = 10 ** math.floor(math.log10(rng_h / 10 + 1e-12))
        while rng_h / spacing > 16:
            spacing *= 2
        drift = 0.08 * spacing * math.sin(2 * math.pi * t / 11.0)      # 线随「风」慢漂
        frac = ((h + drift) / spacing) % 1.0
        # 线宽随梯度：等高线密处（每像素跨越多少条）线变细变淡，不出摩尔纹
        dens = np.sqrt(gx * gx + gy * gy) / spacing              # 条 / 像素
        k = 14.0 * np.clip(dens * 6, 1.0, 6.0)                   # 密 → 线更窄
        att = np.clip(1.0 - dens * 2.5, 0.0, 1.0)                # 密 → 更淡
        line = np.clip(1 - np.abs(frac - 0.5) * 2 * k, 0, 1) * contour_gain * att
        idx = np.floor((h + drift) / spacing)
        line = np.maximum(line, np.where(idx % 5 == 0, np.clip(1 - np.abs(frac - 0.5) * 2 * k * 0.5, 0, 1) * 0.9 * contour_gain * att, 0))
        P_ = np.array(pal, np.float32) / 255
        base = P_[0] * (1 - lam[..., None]) + P_[1] * lam[..., None] * 0.9 + P_[2] * (lam[..., None] ** 3) * 0.5
        out = base * (1 - line[..., None] * 0.85) + P_[3] * line[..., None] * 0.85
        return out

    def upscale(self, small):
        im = Image.fromarray((np.clip(small, 0, 1) * 255).astype(np.uint8)).resize((W0, H0), Image.BILINEAR)
        return np.asarray(im).astype(np.float32) / 255

    # ------------------------------------------------------------ 浪：粒子
    def wave_profile(self, n, rng):
        """非对称卷浪的外轮廓（屏幕坐标）：迎风面（左）缓、背风面（右）陡、浪顶向前（右）探出卷曲。
        返回轮廓上的目标点 (tx, ty) 和「浪唇」标记。"""
        base_y = H0 * 0.78; x0 = 0.06 * W0; xc = 0.56 * W0; H = 0.50 * H0
        s_ = rng.random(n)
        tx = np.empty(n); ty = np.empty(n); lip = np.zeros(n, bool)
        m1 = s_ < 0.55
        u1 = s_[m1] / 0.55
        tx[m1] = x0 + (xc - x0) * u1
        ty[m1] = base_y - H * (u1 ** 2.2)                         # 迎风面：凸起、越到顶越陡
        m2 = (s_ >= 0.55) & (s_ < 0.80)
        u2 = (s_[m2] - 0.55) / 0.25
        r = 0.16 * H0
        ang = math.pi * 0.5 - u2 * math.pi * 1.15                 # 卷：从顶向前、再卷下去
        tx[m2] = xc + r * np.cos(ang) + r * 0.3
        ty[m2] = (base_y - H + r) - r * np.sin(ang)
        lip[m2] = u2 > 0.45
        m3 = s_ >= 0.80
        u3 = (s_[m3] - 0.80) / 0.20
        tipx = xc + r * math.cos(math.pi * 0.5 - math.pi * 1.15) + r * 0.3
        tipy = (base_y - H + r) - r * math.sin(math.pi * 0.5 - math.pi * 1.15)
        tx[m3] = tipx + (0.76 * W0 - tipx) * u3 + 0.05 * W0 * np.sin(u3 * math.pi)
        ty[m3] = tipy + (base_y - tipy) * (u3 ** 0.7)             # 背风面：陡落
        return tx, ty, lip, m2

    def wave_particles(self, n=400000, seed=3):
        if self.wave is None:
            rng = np.random.default_rng(seed)
            sx = rng.uniform(0, W0, n); sy = H0 * (0.42 + 0.58 * rng.random(n) ** 0.7)
            tx, ty, lip, curl = self.wave_profile(n, rng)
            base_y = H0 * 0.78
            on = (rng.random(n) < 0.7) | curl                       # 70% 在轮廓带，30% 填浪体；卷段不填（卷下面是空的 —— 否则成一个直边竖块）
            ty = np.where(on, ty + rng.normal(0, 7, n), ty + (base_y - ty) * rng.random(n) ** 0.5)
            spray = lip & (rng.random(n) < 0.35)                    # 浪唇被风撕下的沙沫
            delay = (sx / W0) * 1.6 + rng.random(n) * 0.4
            w = 2 * math.pi / (1.6 + 0.6 * rng.random(n))
            self.wave = dict(sx=sx, sy=sy, tx=tx, ty=ty, delay=delay, w=w, b=rng.random(n), lip=lip, spray=spray,
                             crest=(0.56 * W0 + 0.05 * H0, base_y - 0.50 * H0 + 0.16 * H0))
        return self.wave

    def draw_wave(self, out, t, t_hit, pal, settle_from=None):
        """t_hit = 68.0 最大一击；粒子沿弹簧飞向浪轮廓，飞行中绕浪顶有涡旋（卷）；浪唇沙沫顺风撕出"""
        wv = self.wave_particles()
        u = t - t_hit - wv["delay"]
        active = u > 0
        if not active.any():
            return out
        uu = np.clip(u[active], 0, 60)
        e0x = wv["sx"][active] - wv["tx"][active]; e0y = wv["sy"][active] - wv["ty"][active]
        w = wv["w"][active]
        ex = (e0x + (60 + w * e0x) * uu) * np.exp(-w * uu); ey = (e0y + (-220 + w * e0y) * uu) * np.exp(-w * uu)
        x = wv["tx"][active] + ex; y = wv["ty"][active] + ey
        cx, cy = wv["crest"]
        dx = x - cx; dy = y - cy
        rr = np.sqrt(dx * dx + dy * dy) + 1e-6
        ang = 1.6 * np.exp(-rr / (0.22 * H0)) * np.exp(-uu / 1.4) * (1 - np.exp(-uu / 0.3))   # 涡旋：近浪顶、飞行早期转得多
        ca, sa = np.cos(ang), np.sin(ang)
        x = cx + dx * ca - dy * sa; y = cy + dx * sa + dy * ca
        sp = wv["spray"][active]
        drift = np.clip(uu - 0.8, 0, 6)
        x = x + sp * drift * 140.0; y = y + sp * (drift ** 2) * 18.0   # 沙沫：到位后顺风飘、下落
        if settle_from is not None and t > settle_from:
            k = min(1.0, (t - settle_from) / 4.0) ** 1.5
            x = x * (1 - 0.5 * k) + wv["sx"][active] * 0.5 * k
            y = y * (1 - 0.5 * k) + wv["sy"][active] * 0.5 * k
        xi = np.clip(x.astype(int), 0, W0 - 1); yi = np.clip(y.astype(int), 0, H0 - 1)
        wgt = (0.35 + 0.65 * wv["b"][active]) * np.where(sp, np.clip(1 - drift / 5, 0, 1), 1.0)
        img = np.zeros((H0, W0), np.float32); lipimg = np.zeros((H0, W0), np.float32)
        lipw = wgt * wv["lip"][active]
        for dx_, dy_ in ((0, 0), (1, 0), (0, 1), (1, 1)):
            np.add.at(img, (np.clip(yi + dy_, 0, H0 - 1), np.clip(xi + dx_, 0, W0 - 1)), wgt)
            np.add.at(lipimg, (np.clip(yi + dy_, 0, H0 - 1), np.clip(xi + dx_, 0, W0 - 1)), lipw)
        img = np.clip(img / 3.0, 0, 1); lipimg = np.clip(lipimg / 3.0, 0, 1)
        lift = float(active.mean())
        fade = np.clip((self.yy - H0 * 0.42) / (H0 * 0.58), 0, 1)[..., None] * 0.6 * lift
        out = out * (1 - fade)
        sand = np.array((158, 138, 108), np.float32) / 255          # 月光下的沙：暖灰
        cool = np.array((196, 210, 222), np.float32) / 255          # 只有浪唇吃到冷光
        col = sand[None, None, :] * (1 - lipimg[..., None]) + cool[None, None, :] * lipimg[..., None]
        return out * (1 - img[..., None] * 0.95) + col * img[..., None] * 0.95

    # ------------------------------------------------------------ 字卡
    def cards(self, d, t, flash_white):
        for txt, a, b in CARDS:
            if a <= t < b:
                u = t - a
                size = int(H0 * (0.22 if len(txt) < 10 else 0.14))
                f = self.font(FONT_SANS, size)
                tw = d.textlength(txt, font=f)
                alpha = int(255 * min(1.0, u / 0.08))
                col = (10, 10, 10) if flash_white else (236, 232, 224)
                d.text(((W0 - tw) / 2, H0 * 0.5 - size * 0.6), txt, font=f, fill=(*col, alpha))

    def readout(self, d, t, cx, cy, span, x_frac=0.72, y_frac=0.3):
        """测距仪红读数：瞄准点的真高程与距离（世界坐标）"""
        wx = cx + (x_frac - 0.5) * span; wy = cy + (y_frac - 0.5) * span * H0 / W0
        h = float(terrain(np.array([wx]), np.array([wy]))[0])
        dist = math.hypot(wx - cx, wy - cy)
        f = self.font(FONT_MONO, 40)
        d.text((W0 * x_frac + 22, H0 * y_frac - 48), f"h {h:8.3f} m", font=f, fill=(*RED, 240))
        d.text((W0 * x_frac + 22, H0 * y_frac - 4), f"d {dist:8.2f} m", font=f, fill=(*RED, 240))
        d.rectangle([W0 * x_frac - 10, H0 * y_frac - 10, W0 * x_frac + 10, H0 * y_frac + 10], outline=(*RED, 240), width=3)

    # ------------------------------------------------------------ 画
    def draw(self, t):
        s = self.s
        shot = s.shot(t)
        fl = s.hit_flash(t)
        if fl == "black" or (shot and shot["label"].startswith("E01")):
            return Image.new("RGB", (W0, H0), (0, 0, 0))
        night = t >= 38.0
        pal = PAL_NIGHT if night else PAL_DUSK
        label = shot["label"] if shot else ""
        out = None
        if label.startswith("A01"):
            # 对数拉远：span 从 0.08 m（一粒沙）到 2400 m（整片沙丘），18 s；速度跟 drone 包络（前 12 s 慢起）
            u = t / 18.0
            ue = u * u * (3 - 2 * u)                       # 慢入慢出（对数尺度上）
            span = 0.3 * (600 / 0.3) ** ue                 # 0.3 m → 600 m（3.3 个数量级；v2 是 4.5）
            cx = 1200 + 40 * math.sin(2 * math.pi * t / 23); cy = 800 + 25 * math.cos(2 * math.pi * t / 17)
            h, xs, ys = self.terrain_view(cx, cy, span, step=1)
            # 天光：开头稍亮，最后 4 s 转夜
            k = min(1.0, max(0.0, (t - 14) / 4))
            p = [tuple(int(a * (1 - k) + b * k) for a, b in zip(ca, cb)) for ca, cb in zip(PAL_DUSK, PAL_NIGHT)]
            small = self.shade_contours(h, span, t, p, light_dir=(math.cos(0.4 + 0.1 * t), math.sin(0.4 + 0.1 * t)))
            out = self.upscale(small)
            # 最后 3 s：极小人影（P01 底图淡入叠在远景上）
            if t > 15.0:
                pl = self.plate("P16")
                if pl is not None:
                    k2 = (t - 15.0) / 3.0
                    lum = pl.mean(-1, keepdims=True)
                    P_ = np.array(p, np.float32) / 255
                    tinted = P_[0] * (1 - lum) + P_[2] * lum
                    out = out * (1 - 0.85 * k2) + tinted * 0.85 * k2
        elif label.startswith(("D0",)):
            pl = self.plate("P14")
            base = pl if pl is not None else np.zeros((H0, W0, 3), np.float32)
            lum = base.mean(-1)
            # 底图的等值线（亮度）+ 暗化
            P_ = np.array(pal, np.float32) / 255
            frac = (lum * 16) % 1.0
            line = np.clip(1 - np.abs(frac - 0.5) * 2 * 12, 0, 1) * np.clip(lum * 2, 0, 1)
            out = (P_[0] * 0.7 + base * 0.35) * (1 - line[..., None] * 0.6) + P_[3] * line[..., None] * 0.6
            if label.startswith(("D01", "D02")):
                # 62 → 65：悬浮沙粒明显减速进入悬停（位移 = 积分的减速曲线）；65–68 完全静止
                rng = np.random.default_rng(5)
                n = 6000; xs0 = rng.uniform(0, W0, n); ys0 = H0 * (0.3 + 0.7 * rng.random(n) ** 0.6)
                v = 90.0 * (0.5 + rng.random(n))                      # 像素 / 秒的初速（风）
                tau = 1.2
                u = min(max(t - 62.0, 0.0), 3.0)
                disp = v * tau * (1 - math.exp(-u / tau))              # 指数减速 → 停
                xs = (xs0 + disp) % W0; ys = ys0 + 0.15 * disp * np.sin(xs0 / 90)
                img = np.zeros((H0, W0), np.float32); np.add.at(img, (np.clip(ys.astype(int), 0, H0 - 1), xs.astype(int)), 0.8)
                out = out * (1 - img[..., None]) + P_[3] * img[..., None]
            if t >= 68.0:
                out = self.draw_wave(out, t, 68.0, pal, settle_from=74.0)
        elif shot is not None and label.startswith("E02"):
            # 片名由等高线一圈圈画出：文字距离场的等值线随时间向外扩
            from scipy.ndimage import distance_transform_edt
            img = Image.new("L", (W0, H0), 0); d0 = ImageDraw.Draw(img)
            f = self.font(FONT_SANS, int(H0 * 0.2)); txt = "THE QUIET SEA"; tw = d0.textlength(txt, font=f)
            d0.text(((W0 - tw) / 2, H0 * 0.5 - H0 * 0.12), txt, font=f, fill=255)
            m = np.asarray(img) > 128
            dist = distance_transform_edt(~m)
            u = (t - shot["t0"]) / 2.5
            reach = u * 260
            sp = 14.0
            frac = (dist / sp) % 1.0
            rings = np.clip(1 - np.abs(frac - 0.5) * 2 * 9, 0, 1) * (dist < reach) * np.clip(1 - dist / 300, 0, 1)
            P_ = np.array(PAL_NIGHT, np.float32) / 255
            out = np.zeros((H0, W0, 3), np.float32) + P_[0] * 0.3
            out = out * (1 - rings[..., None]) + P_[3] * rings[..., None]
            out = out * (1 - m[..., None] * min(1.0, u)) + P_[3] * m[..., None] * min(1.0, u)
        else:
            # 其它镜头（B / C）：底图等值线重绘；底图按分镜正文里的 Pxx 取
            pid = None
            for cand in ("P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09", "P10"):
                if shot and cand in (shot.get("body") or ""):
                    pid = cand; break
            pl = self.plate(pid) if pid else None
            base = pl if pl is not None else np.zeros((H0, W0, 3), np.float32) + 0.05
            lum = base.mean(-1)
            P_ = np.array(pal, np.float32) / 255
            drift = 0.03 * math.sin(2 * math.pi * t / 9.0)
            frac = ((lum + drift) * 18) % 1.0
            line = np.clip(1 - np.abs(frac - 0.5) * 2 * 12, 0, 1) * np.clip(lum * 2.2, 0, 1)
            out = (P_[0] * 0.7 + base * 0.35) * (1 - line[..., None] * 0.7) + P_[3] * line[..., None] * 0.7
        if out is None:
            out = np.zeros((H0, W0, 3), np.float32)
        if fl == "white":
            out = np.clip(out * 0.3 + 0.75, 0, 1)
        im = Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))
        d = ImageDraw.Draw(im, "RGBA")
        # 读数：B02 / C05 / D02（全静里只有它在跳）
        if label.startswith(("B02", "C05", "D02")):
            self.readout(d, t, 1200 + 3.0 * t, 800 + 1.2 * math.sin(t), 400, x_frac=0.66, y_frac=0.36)
        self.cards(d, t, fl == "white")
        rng = np.random.default_rng(int(t * 24))
        g = rng.normal(0, 2.5, (H0, W0, 1))
        return Image.fromarray(np.clip(np.asarray(im).astype(np.float32) + g, 0, 255).astype(np.uint8))


def contact_sheet(frames, out, cols=4):
    w, h = frames[0][0].size; tw, th = w // 3, h // 3
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (tw * cols, th * rows), (8, 8, 8)); f = ImageFont.truetype(FONT_MONO, 22)
    for i, (im, lab) in enumerate(frames):
        im = im.resize((tw, th), Image.LANCZOS); ImageDraw.Draw(im).text((8, 6), lab, fill=RED, font=f)
        sheet.paste(im, ((i % cols) * tw, (i // cols) * th))
    sheet.save(out, quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["still", "clip", "sheet"])
    ap.add_argument("--t", type=float, default=9.0); ap.add_argument("--t0", type=float, default=0.0); ap.add_argument("--t1", type=float, default=18.0)
    ap.add_argument("--fps", type=int, default=24); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if not out.is_absolute():
        out = MV / out
    if out.exists():
        sys.exit(f"已存在，不覆盖：{out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    sc = Scene(Song())
    if a.mode == "still":
        sc.draw(a.t).save(out, quality=92); print(out); return
    if a.mode == "sheet":
        ts = np.linspace(a.t0, a.t1, 12, endpoint=False)
        contact_sheet([(sc.draw(float(t)), f"t={t:.2f}") for t in ts], out); print(out); return
    import imageio_ffmpeg, time
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    n = int(round((a.t1 - a.t0) * a.fps))
    cmd = [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W0}x{H0}", "-r", str(a.fps), "-i", "-",
           "-ss", f"{a.t0:.3f}", "-t", f"{a.t1 - a.t0:.3f}", "-i", str(AUDIO),
           "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-shortest", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    sheet_idx = set(np.linspace(0, n - 1, 12).round().astype(int).tolist()); sheet = []; t_start = time.time()
    for i in range(n):
        t = a.t0 + i / a.fps
        im = sc.draw(t)
        proc.stdin.write(im.tobytes())
        if i in sheet_idx:
            sheet.append((im, f"t={t:.2f}"))
        if i % 96 == 0:
            el = time.time() - t_start; print(f"{i}/{n}  {el:.0f}s  eta {el / max(i, 1) * (n - i):.0f}s", flush=True)
    proc.stdin.close(); proc.wait()
    contact_sheet(sheet, out.with_name(out.stem + "_接触表.jpg")); print(out)


if __name__ == "__main__":
    main()
