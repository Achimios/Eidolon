#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🎬Piccolo-渲染-2D motion design 版（v2，2026-10-24，Fable）。👤「piccolo 走 2D 风格」。
语法学自 _参考库：Modular（网格量化、生长代替淡入、两段式先快后慢、替换式转场）、FITC 2015 animatic（笔画组装的模块字形、
标签框填实 1 帧、大字 + 极小字没有中间尺度）、信息可视化（虚线二次贝塞尔弧 + 中点箭头、极小暗坐标字、冲击 → 衰减 → 空）。
图形原创；一切都是歌曲时间 t 的纯函数；纯 PIL + numpy，CPU。

画面（黑底 + 看不见的网格 g = H/18）：
  通道时间轴   12 个频带 = 12 行，洞 = 白色细条（真实起音 + 持续时间），左侧极小等宽字写真频率；右移到读头被读 → 变白、留一次 1 帧填实。
  读头         一条红线（唯一的强调色），旁边等宽字 t / beat / bar（真数）。下拍 = 读头顶上的小标签框填实 1 帧。
  他           由「模块笔画」组装：横 / 竖 / 四分之一弧 / 点，落在网格上；每一笔的出生 = 一个真实起音（8 s 后最强的那些，按脚→头）；
               出生时从读头处的洞起一条虚线贝塞尔弧飞到笔画位置（带极小标签 b / Hz / s），弧 1.2 s 内衰减消失；笔画用两段式缓动画出。
  拐杖         一根长笔画，杖尖撑地的摆：拍点把它抬起、阻尼落回（落地那一帧杖尖出一个小点 + θ 读数）。第一次落地 = 第一句人声 28.40 s。
  旋律         小方框标记（f0 → 高度），每 0.25 s 留一枚 [t, Hz] 面包屑随时间轴左移、3 s 内变暗（面包屑 = 时间）。
  底噪         intro 里按 crackle 密度出现的极小白点（寿命 3 帧）；机器接通（8.0 s）后不再画。
  镜头         慢漂 + 两个不互为倍数周期的缓慢变焦 + ±0.5° 转，从不静止。
  8.0 s 事件   行线从左长出来 + 大字 115.28 用笔画拼出，然后对数缩进读头标签。

用法：
  python tools/mv_piccolo/roll2d.py still --t 12.5 --out x.jpg
  python tools/mv_piccolo/roll2d.py sheet --t0 0 --t1 30 --out x_接触表.jpg
  python tools/mv_piccolo/roll2d.py clip  --t0 0 --t1 30 --out 样片/02_纸卷2D_接通_0-30_v2.mp4
输出不覆盖，重名就停。
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
MV = PROJ / "MV_Piccolo"
MUSIC = PROJ / "Assets" / "Piccolo and a Cane[短笛和拐杖] - Electro Swing - 免版税音乐.mp4"
ANALYSIS = MV / "分析_纸卷.json"

BG = (9, 9, 10)
WHITE = (236, 234, 228)
GREY = (120, 118, 112)
DIM = (62, 61, 58)
RED = (214, 52, 38)
FONT_MONO = "C:/Windows/Fonts/consola.ttf"
FONT_MONO_B = "C:/Windows/Fonts/consolab.ttf"

W0, H0 = 1920, 1080
G = H0 / 18                      # 网格
BEAT_PX = 2 * G                  # 一拍 = 2 格
HEAD_X = 7 * G                   # 读头（画面左侧 7 格）
ROW_Y0, ROW_Y1 = 3.2 * G, 8.6 * G  # 通道行区
FLOOR_Y = 16.6 * G               # 地面
T_ON = 8.0                       # 机器接通


# ---------------------------------------------------------------- 缓动
def ease2(u, a=0.6, t1=3 / 24, t2=9 / 24):
    """两段式先快后慢（Modular 实测 τ ≈ 3 帧 / 9 帧）：u 秒 → 0~1"""
    if u <= 0:
        return 0.0
    return a * (1 - math.exp(-u / t1)) + (1 - a) * (1 - math.exp(-u / t2))


def q(v):
    """网格量化"""
    return round(v / G) * G


# ---------------------------------------------------------------- 数据
class Song:
    def __init__(self, path=ANALYSIS):
        d = json.loads(Path(path).read_text("utf-8"))
        self.bpm = d["bpm"]; self.beat = 60.0 / self.bpm
        self.beats = np.array(d["beats"]); self.duration = d["duration"]
        self.bands = d["bands"]; self.nb = len(self.bands)
        self.holes = [h for h in d["holes"] if h["strength"] >= 0.2]
        self.ht = np.array([h["t"] for h in self.holes])
        m = d["melody"]; self.mel_hop = m["hop_s"]
        self.f0 = np.array([np.nan if v is None else v for v in m["f0"]], float)
        self.voiced = np.array(m["voiced"], float)
        fin = self.f0[np.isfinite(self.f0)]
        self.f_lo, self.f_hi = np.percentile(fin, 3), np.percentile(fin, 97)
        self.crack_hop = d["crackle"]["hop_s"]; self.crack = np.array(d["crackle"]["v"])
        self.loud = np.array(d["loud"]["v"])
        self.phrases = d.get("phrases", [])
        low = [h for h in self.holes if h["band"] <= 1]
        sc = np.zeros(4)
        for h in low:
            i = int(np.argmin(np.abs(self.beats - h["t"])))
            if abs(self.beats[i] - h["t"]) < 0.06:
                sc[i % 4] += h["strength"]
        self.down_phase = int(np.argmax(sc))

    def at(self, arr, hop, t):
        i = int(t / hop)
        return float(arr[min(max(i, 0), len(arr) - 1)])

    def melody_at(self, t):
        i = min(max(int(t / self.mel_hop), 0), len(self.f0) - 1)
        return self.f0[i], self.voiced[i]

    def beat_info(self, t):
        """(拍序号, 拍内相位, 距上一拍秒, 是否下拍, 小节号)"""
        i = int(np.searchsorted(self.beats, t, side="right")) - 1
        if i < 0:
            return -1, 0.0, 99.0, False, 0
        nxt = self.beats[i + 1] if i + 1 < len(self.beats) else self.beats[i] + self.beat
        down = (i % 4) == self.down_phase
        bar = (i - self.down_phase) // 4 + 1
        return i, (t - self.beats[i]) / max(nxt - self.beats[i], 1e-6), t - self.beats[i], down, bar


# ---------------------------------------------------------------- 他：模块笔画（网格单位，脚底 (0,0)，y 向上，1 单位 = 半格）
# 笔画类型：bar (x0,y0,x1,y1) / arc (cx,cy,r,a0,a1 度，逆时针) / dot (x,y)
FIGURE = [
    # 腿：右腿直立；左腿交叉到右前（歌舞演员站姿）
    ("bar", (1, 0, 1, 5)), ("bar", (-1.5, 0, 2.5, 5)),
    ("bar", (-2.5, 0, -0.5, 0)),                       # 左脚（横笔）
    ("bar", (0.5, 0, 2.5, 0)),                         # 右脚
    ("bar", (0, 5, 0, 10)),                            # 躯干
    ("bar", (-2.5, 10, 2.5, 10)),                      # 肩
    ("arc", (-2.5, 8, 2, 90, 270)),                    # 左臂：叉腰的弯（半圆）
    ("bar", (2.5, 10, 3.5, 7)),                        # 右上臂
    ("bar", (3.5, 7, 4, 6)),                           # 右前臂到手
    ("arc", (0, 12, 1.5, 0, 90)), ("arc", (0, 12, 1.5, 90, 180)),    # 头：四段四分之一弧
    ("arc", (0, 12, 1.5, 180, 270)), ("arc", (0, 12, 1.5, 270, 360)),
    ("bar", (-2.5, 13.6, 2.5, 13.6)),                  # 帽檐
    ("bar", (-1.5, 13.6, -1.5, 15.2)), ("bar", (1.5, 13.6, 1.5, 15.2)), ("bar", (-1.5, 15.2, 1.5, 15.2)),  # 帽身
]
HAND = (4, 6)
UNIT = G * 0.46
FIG_X = 15.0 * G                  # 他站的位置（网格）
STROKE_W = 0.5 * UNIT             # 笔宽（FITC 式模块字体）


def stroke_order():
    """出生顺序：按最低点从脚到头，眼最后"""
    def low(s):
        k, p = s
        if k == "bar":
            return min(p[1], p[3])
        if k == "arc":
            return p[1] - p[2]
        return 99
    return sorted(range(len(FIGURE)), key=lambda i: low(FIGURE[i]))


# ---------------------------------------------------------------- 场景
class Scene:
    def __init__(self, song: Song, w=W0, h=H0, ss=2):
        self.s = song; self.W, self.H, self.ss = w, h, ss
        self.k = w / W0
        self.fonts = {n: ImageFont.truetype(FONT_MONO, int(n * ss * self.k)) for n in (11, 13, 15, 18)}
        self.fonts_b = {n: ImageFont.truetype(FONT_MONO_B, int(n * ss * self.k)) for n in (13, 96, 150)}
        self._schedule()

    # 出生时刻：T_ON 后最强的起音，间隔 ≥ 0.45 s，和笔画一一对应（脚 → 头）
    def _schedule(self):
        s = self.s
        order = stroke_order()
        cand = sorted([h for h in s.holes if T_ON + 0.4 <= h["t"] <= 26.5], key=lambda h: -h["strength"])
        picked = []
        for h in cand:
            if all(abs(h["t"] - p["t"]) >= 0.45 for p in picked):
                picked.append(h)
            if len(picked) >= len(order):
                break
        picked.sort(key=lambda h: h["t"])
        self.births = {}
        for idx, h in zip(order, picked):
            self.births[idx] = h
        # 拐杖：出现在最后一笔之后、第一句人声落地
        self.cane_t = max(h["t"] for h in picked) + 0.3
        self.voice1 = next((p["start"] for p in s.phrases if p["start"] > self.cane_t), 28.4)

    # 相机
    def cam(self, t):
        z = 1.0 + 0.035 * math.sin(2 * math.pi * t / 23.0) + 0.02 * math.sin(2 * math.pi * t / 9.7)
        dx = 28 * math.sin(2 * math.pi * t / 31.0) + 12 * math.sin(2 * math.pi * t / 7.3)
        dy = 18 * math.cos(2 * math.pi * t / 19.0)
        rot = math.radians(0.5) * math.sin(2 * math.pi * t / 41.0)
        cx, cy = W0 * 0.5, H0 * 0.5
        c, sn = math.cos(rot), math.sin(rot)
        k = self.k * self.ss
        def P(x, y):
            x, y = x - cx, y - cy
            x, y = x * c - y * sn, x * sn + y * c
            return ((x * z + cx + dx) * k, (y * z + cy + dy) * k)
        return P

    # 画线（带笔宽）
    def line(self, d, P, a, b, fill, w):
        d.line([P(*a), P(*b)], fill=fill, width=max(1, int(w * self.k * self.ss)))

    def text(self, d, P, xy, s, fill, n=11, bold=False, anchor="la"):
        d.text(P(*xy), s, font=(self.fonts_b if bold else self.fonts)[n], fill=fill, anchor=anchor)

    def rect(self, d, P, x0, y0, x1, y1, fill=None, outline=None, w=1):
        d.rectangle([*P(x0, y0), *P(x1, y1)], fill=fill, outline=outline, width=max(1, int(w * self.k * self.ss)))

    def bezier_dashed(self, d, P, a, b, fill, bend=0.28, dash=(5, 6), w=1, arrow=True, frac=1.0):
        """虚线二次贝塞尔 + 中点箭头；frac = 画到几成（生长）"""
        ax, ay = a; bx, by = b
        mx, my = (ax + bx) / 2, (ay + by) / 2
        dx, dy = bx - ax, by - ay
        L = math.hypot(dx, dy) + 1e-9
        cx, cy = mx - dy / L * bend * L, my + dx / L * bend * L
        n = max(8, int(L / 4))
        pts = []
        for i in range(n + 1):
            u = i / n * frac
            x = (1 - u) ** 2 * ax + 2 * (1 - u) * u * cx + u * u * bx
            y = (1 - u) ** 2 * ay + 2 * (1 - u) * u * cy + u * u * by
            pts.append((x, y))
        acc = 0.0; on = True; seg = [pts[0]]
        for p0, p1 in zip(pts, pts[1:]):
            acc += math.dist(p0, p1)
            seg.append(p1)
            lim = dash[0] if on else dash[1]
            if acc >= lim:
                if on:
                    d.line([P(*p) for p in seg], fill=fill, width=max(1, int(w * self.k * self.ss)))
                acc = 0.0; on = not on; seg = [p1]
        if on and len(seg) > 1:
            d.line([P(*p) for p in seg], fill=fill, width=max(1, int(w * self.k * self.ss)))
        if arrow and frac >= 0.5:
            u = 0.5
            x = (1 - u) ** 2 * ax + 2 * (1 - u) * u * cx + u * u * bx
            y = (1 - u) ** 2 * ay + 2 * (1 - u) * u * cy + u * u * by
            tx, ty = 2 * (1 - u) * (cx - ax) + 2 * u * (bx - cx), 2 * (1 - u) * (cy - ay) + 2 * u * (by - cy)
            tl = math.hypot(tx, ty) + 1e-9; tx, ty = tx / tl, ty / tl
            nx, ny = -ty, tx
            d.line([P(x - tx * 6 + nx * 4, y - ty * 6 + ny * 4), P(x, y), P(x - tx * 6 - nx * 4, y - ty * 6 - ny * 4)],
                   fill=fill, width=max(1, int(self.k * self.ss)))

    # 笔画几何（屏幕坐标），prog = 画出的比例（0~1）
    def stroke_draw(self, d, P, idx, prog, fill, w=None):
        kind, p = FIGURE[idx]
        w = STROKE_W if w is None else w
        fx, fy = FIG_X, FLOOR_Y
        u = UNIT
        if kind == "bar":
            x0, y0, x1, y1 = p
            a = (fx + x0 * u, fy - y0 * u); b = (fx + x1 * u, fy - y1 * u)
            b2 = (a[0] + (b[0] - a[0]) * prog, a[1] + (b[1] - a[1]) * prog)
            self.line(d, P, a, b2, fill, w)
        elif kind == "arc":
            cx, cy, r, a0, a1 = p
            # 边转边长：起点随 prog 转 45°，长度从 0 长到 (a1-a0)
            span = (a1 - a0) * prog
            start = a0 - 45 * (1 - prog)
            n = max(4, int(span / 6))
            pts = []
            for i in range(n + 1):
                ang = math.radians(start + span * i / n)
                pts.append((fx + (cx + r * math.cos(ang)) * u, fy - (cy + r * math.sin(ang)) * u))
            if len(pts) > 1:
                d.line([P(*q_) for q_ in pts], fill=fill, width=max(1, int(w * self.k * self.ss)), joint="curve")
        else:
            x, y = p
            r = w * 0.6 * prog
            d.ellipse([*P(fx + x * u - r, fy - y * u - r), *P(fx + x * u + r, fy - y * u + r)], fill=fill)

    def stroke_anchor(self, idx):
        kind, p = FIGURE[idx]
        fx, fy, u = FIG_X, FLOOR_Y, UNIT
        if kind == "bar":
            return (fx + p[0] * u, fy - p[1] * u)
        if kind == "arc":
            return (fx + p[0] * u, fy - p[1] * u)
        return (fx + p[0] * u, fy - p[1] * u)

    # 拐杖摆：θ 相对竖直，杖尖撑地，平衡 θ0；拍点抬起（om 负向）后阻尼落回
    def cane(self, t, dt=1 / 240):
        s = self.s
        t0 = self.cane_t
        if t < t0:
            return None
        # 几何：手在 (4,6)，杖尖落地点在手外侧 1.8 单位 → 杖长 = hypot(1.8, 6)，平衡角 = atan2(1.8, 6)
        L = math.hypot(1.8, HAND[1]) * UNIT
        th0 = math.atan2(1.8, HAND[1])
        g_L = 9.81 / 1.0; c = 2.2
        th, om = th0 + 0.35, 0.0                           # 刚出现时是抬着的，落下来 = 第一次落地
        kicks = [b for b in s.beats if t0 <= b <= t]
        kicks = [k for k in kicks if (int(np.searchsorted(s.beats, k)) % 4) == s.down_phase]  # 只在下拍抬
        ki = 0; tt = t0
        landed = []
        prev_above = True
        while tt < t:
            while ki < len(kicks) and kicks[ki] <= tt:
                if kicks[ki] >= self.voice1:          # 第一句人声之前不抬（先落地）
                    om -= 2.4
                ki += 1
            om += (-g_L * math.sin(th - th0) * 3.0 - c * om) * dt
            th += om * dt
            if th < th0:                                # 地面：不能穿地
                if not prev_above or om < -0.3:
                    landed.append(tt)
                th = th0; om = max(om, 0) * -0.15
                prev_above = False
            else:
                prev_above = True
            tt += dt
        return th, th0, L, landed

    # ------------------------------------------------------------ 画
    def draw(self, t):
        s = self.s
        ss, k = self.ss, self.k
        W, H = int(self.W * ss), int(self.H * ss)
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img, "RGBA")
        P = self.cam(t)
        bi, bph, since, down, bar = s.beat_info(t)
        on = t >= T_ON
        u_on = t - T_ON

        # 网格点（坐标纸）
        for gx in range(0, 33):
            for gy in range(0, 19):
                x, y = P(gx * G, gy * G)
                d.point((x, y), fill=(*DIM, 140))

        v = BEAT_PX / s.beat   # 像素 / 秒

        # 通道行：T_ON 后从左长出来（两段式），每行左侧写频率
        rows_prog = ease2(u_on, t1=0.08, t2=0.3) if on else 0.0
        row_h = (ROW_Y1 - ROW_Y0) / s.nb
        for b in range(s.nb):
            y = ROW_Y1 - (b + 0.5) * row_h
            if rows_prog > 0:
                self.line(d, P, (2.2 * G, y), (2.2 * G + rows_prog * (W0 - 2.2 * G + 100), y), (*DIM, 110), 1)
                lab = f"{s.bands[b]:5.0f} Hz" if s.bands[b] < 1000 else f"{s.bands[b] / 1000:4.1f} kHz"
                self.text(d, P, (1.0 * G, y - 8), lab, (*GREY, int(200 * rows_prog)), 11)

        # 洞：细条。机器接通前只画旋律面包屑；接通后画全部频带
        i0 = int(np.searchsorted(s.ht, t - (HEAD_X + 100) / v))
        i1 = int(np.searchsorted(s.ht, t + (W0 - HEAD_X + 100) / v))
        n_read = int(np.searchsorted(s.ht, t))
        if on:
            for h in s.holes[i0:i1]:
                x0 = HEAD_X + (h["t"] - t) * v
                x1 = x0 + max(h["dur"], 0.05) * v
                y = ROW_Y1 - (h["band"] + 0.5) * row_h
                played = h["t"] <= t
                if x1 < 2.8 * G:
                    continue
                hh = 2.5 + 3.5 * h["strength"]
                if played:
                    fade = max(0.0, min(1.0, (x0 - 2.8 * G) / (3.0 * G)))
                    col = (*WHITE, int((255 if h["strength"] > 0.45 else 170) * fade))
                else:
                    col = (*GREY, 90 + int(110 * h["strength"]))
                # 刚被读到：那一个洞填实成红 1 帧（标签框填实），再退回白
                if played and t - h["t"] < 1 / 24:
                    col = (*RED, 255)
                    self.rect(d, P, x0 - 2, y - hh - 2, x1 + 2, y + hh + 2, fill=col)
                self.rect(d, P, x0, y - hh, x1, y + hh, fill=col)

        # 读头：接通前是灰线（未通电），接通后红
        head_col = (*RED, 240) if on else (*DIM, 200)
        self.line(d, P, (HEAD_X, ROW_Y0 - 2.4 * G), (HEAD_X, ROW_Y1 + G * 0.3), head_col, 2 if on else 1)
        # 读头标签：t / beat / bar（真数）；下拍时框填实 1 帧
        lab = f"t {t:7.3f} s   beat {bi + 1 if bi >= 0 else 0:3d}   bar {bar:2d}"
        bx0, by0 = HEAD_X + 8, ROW_Y1 + 0.35 * G
        if on and down and since < 1 / 24:
            self.rect(d, P, bx0 - 4, by0 - 3, bx0 + 262, by0 + 17, fill=(*RED, 255))
            self.text(d, P, (bx0, by0), lab, (*BG, 255), 11)
        else:
            self.rect(d, P, bx0 - 4, by0 - 3, bx0 + 262, by0 + 17, outline=(*RED, 200) if on else (*DIM, 200), w=1)
            self.text(d, P, (bx0, by0), lab, (*RED, 230) if on else (*GREY, 200), 11)

        # 8.0 s 大字：115.28 由笔画拼出（两段式），停 2 拍，然后对数缩进读头标签
        if on and u_on < 2.6:
            if u_on < 2 * s.beat:
                sc = 1.0
                alpha = 255
                pr = ease2(u_on, t1=0.06, t2=0.25)
            else:
                uu = (u_on - 2 * s.beat) / (2.6 - 2 * s.beat)
                sc = math.exp(-2.6 * uu)
                alpha = int(255 * (1 - uu))
                pr = 1.0
            cx, cy = 17 * G, 12.5 * G
            font = ImageFont.truetype(FONT_MONO_B, max(8, int(150 * sc * k * ss)))
            txt = f"{s.bpm:.2f}"
            # 笔画拼出：用裁切模拟逐字出现（每个字符 = 一笔）
            nchar = max(1, int(round(pr * len(txt) + 0.49)))
            x, y = P(cx - 7 * G * sc * (1 - sc) * 0, cy)
            d.text((x, y), txt[:nchar], font=font, fill=(*WHITE, alpha), anchor="lm")
            self.text(d, P, (cx + 2, cy + 85 * sc), "BPM  · MACHINE ON", (*GREY, alpha), 11)

        # 他：每一笔
        born = 0
        for idx, h in self.births.items():
            u = t - h["t"]
            if u < 0:
                continue
            pr = ease2(u, t1=0.07, t2=0.28)
            born += pr > 0.98
            self.stroke_draw(d, P, idx, min(pr, 1.0), (*WHITE, 255))
            # 虚线弧：从洞（读头处、该频带行）飞向笔画锚点，1.2 s 内衰减
            if u < 1.2:
                decay = 1 - u / 1.2
                y = ROW_Y1 - (h["band"] + 0.5) * row_h
                a = (HEAD_X, y); b = self.stroke_anchor(idx)
                self.bezier_dashed(d, P, a, b, (*GREY, int(220 * decay)), frac=min(1.0, u / 0.25))
                # 极小标签：真数（频带 / 强度 / 时刻）
                lx, ly = b[0] + 10, b[1] - 16
                self.text(d, P, (lx, ly), f"b{h['band']:02d} {s.bands[h['band']]:.0f}Hz s{h['strength']:.2f} t{h['t']:.2f}",
                          (*GREY, int(200 * decay)), 11)

        # 拐杖
        cn = self.cane(t)
        if cn:
            th, th0, L, landed = cn
            hx, hy = FIG_X + HAND[0] * UNIT, FLOOR_Y - HAND[1] * UNIT
            tipx, tipy = hx + L * math.sin(th), hy + L * math.cos(th)
            appear = ease2(t - self.cane_t, t1=0.07, t2=0.28)
            tip_d = (hx + (tipx - hx) * appear, hy + (tipy - hy) * appear)
            self.line(d, P, (hx, hy), tip_d, (*WHITE, 255), STROKE_W)
            self.text(d, P, (tipx + 8, tipy - 14), f"θ {th - th0:+.2f} rad", (*GREY, 200), 11)
            # 落地印：最近 0.5 s 内落地 → 杖尖处一个点 + 短横（印记留在地上）
            for tl in landed:
                if t - tl < 0.6:
                    a_ = 1 - (t - tl) / 0.6
                    gx = hx + L * math.sin(th0)
                    self.rect(d, P, gx - 10, FLOOR_Y + 2, gx + 10, FLOOR_Y + 4, fill=(*RED, int(255 * a_)))
            # 地面线：从他脚下向两侧长出（出现在拐杖之后）
        if t >= self.cane_t:
            pr = ease2(t - self.cane_t, t1=0.1, t2=0.4)
            self.line(d, P, (FIG_X - 4 * G * pr, FLOOR_Y + 0.5), (FIG_X + 4 * G * pr, FLOOR_Y + 0.5), (*DIM, 200), 1)

        # 旋律：标记 + 面包屑（随时间轴左移）
        f0, vo = s.melody_at(t)
        def fy(f):
            uu = (math.log2(f) - math.log2(s.f_lo)) / (math.log2(s.f_hi) - math.log2(s.f_lo))
            return ROW_Y0 - 0.5 * G - max(0, min(1, uu)) * 1.7 * G
        mx = HEAD_X
        if np.isfinite(f0):
            my = fy(f0)
            self.rect(d, P, mx - 5, my - 5, mx + 5, my + 5, outline=(*WHITE, 230), w=1)
        # 旋律线：过去 3.3 s 的 f0 折线（每 1/24 s 一点），向左变暗
        pts = []
        for j in range(0, 80):
            tb = t - j / 24
            if tb < 0:
                break
            fb, _ = s.melody_at(tb)
            pts.append(None if not np.isfinite(fb) else (HEAD_X - (t - tb) * v, fy(fb)))
        seg = []
        for j, pt in enumerate(pts + [None]):
            if pt is None:
                if len(seg) > 1:
                    a_ = int(200 * (1 - (j - len(seg) / 2) / 85))
                    d.line([P(*q_) for q_ in seg], fill=(*WHITE, max(40, a_)), width=max(1, int(self.k * self.ss)))
                seg = []
            else:
                seg.append(pt)
        # 面包屑：过去 3.2 s 里每 0.25 s 一枚
        k0 = int(t / 0.25)
        for kk in range(k0 - 13, k0 + 1):
            tb = kk * 0.25
            if tb < 0 or tb > t:
                continue
            fb, _ = s.melody_at(tb)
            if not np.isfinite(fb):
                continue
            age = t - tb
            x = HEAD_X - age * v
            if x < -50:
                continue
            a_ = int(170 * (1 - age / 3.3))
            d.point(P(x, fy(fb)), fill=(*WHITE, a_))
            if kk % 2 == 0:
                self.text(d, P, (x + 3, fy(fb) - 13), f"[{tb:.2f},{fb:.0f}]", (*GREY, a_), 11)

        # intro 底噪：极小白点，密度 = crackle，寿命 3 帧
        if not on:
            cr = s.at(s.crack, s.crack_hop, t)
            seed = int(t * 8)
            rng = np.random.default_rng(seed)
            n_ = int(60 * cr)
            for x, y in zip(rng.uniform(0, W0, n_), rng.uniform(0, H0, n_)):
                d.point(P(x, y), fill=(*WHITE, 150))

        # 页脚：真参数（极小）
        self.text(d, P, (G * 0.5, H0 - G * 0.55), f"PICCOLO AND A CANE   roll 01   {s.bpm:.2f} bpm   holes read {n_read:4d}   {len(self.births)} strokes",
                  (*GREY, 180), 11)

        if ss != 1:
            img = img.resize((int(self.W), int(self.H)), Image.LANCZOS)
        return img


# ---------------------------------------------------------------- 输出
def contact_sheet(frames, out, cols=4):
    w, h = frames[0][0].size
    tw, th = w // 3, h // 3
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (tw * cols, th * rows), BG)
    f = ImageFont.truetype(FONT_MONO, 22)
    for i, (im, lab) in enumerate(frames):
        im = im.resize((tw, th), Image.LANCZOS)
        ImageDraw.Draw(im).text((8, 6), lab, fill=RED, font=f)
        sheet.paste(im, ((i % cols) * tw, (i // cols) * th))
    sheet.save(out, quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["still", "clip", "sheet"])
    ap.add_argument("--t", type=float, default=12.0)
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float, default=30.0)
    ap.add_argument("--w", type=int, default=1920); ap.add_argument("--h", type=int, default=1080)
    ap.add_argument("--ss", type=int, default=2)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if not out.is_absolute():
        out = MV / out
    if out.exists():
        sys.exit(f"已存在，不覆盖：{out}（换 _v2 / _v3）")
    out.parent.mkdir(parents=True, exist_ok=True)
    sc = Scene(Song(), a.w, a.h, a.ss)
    if a.mode == "still":
        sc.draw(a.t).save(out, quality=92); print(out); return
    if a.mode == "sheet":
        ts = np.linspace(a.t0, a.t1, 12, endpoint=False)
        contact_sheet([(sc.draw(float(t)), f"t={t:.2f}") for t in ts], out); print(out); return
    import imageio_ffmpeg, time
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    n = int(round((a.t1 - a.t0) * a.fps))
    cmd = [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{a.w}x{a.h}", "-r", str(a.fps), "-i", "-",
           "-ss", f"{a.t0:.3f}", "-t", f"{a.t1 - a.t0:.3f}", "-i", str(MUSIC),
           "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-shortest", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    sheet_idx = set(np.linspace(0, n - 1, 12).round().astype(int).tolist())
    sheet = []; t_start = time.time()
    for i in range(n):
        t = a.t0 + i / a.fps
        im = sc.draw(t)
        proc.stdin.write(im.tobytes())
        if i in sheet_idx:
            sheet.append((im, f"t={t:.2f}"))
        if i % 48 == 0:
            el = time.time() - t_start
            print(f"{i}/{n}  {el:.0f}s  eta {el / max(i, 1) * (n - i):.0f}s", flush=True)
    proc.stdin.close(); proc.wait()
    contact_sheet(sheet, out.with_name(out.stem + "_接触表.jpg"))
    print(out)


if __name__ == "__main__":
    main()
