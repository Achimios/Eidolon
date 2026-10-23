#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🎬Piccolo-渲染-v3「网格 vs 摆」（2026-10-24，Fable；ReadScreen 第 1 轮审核后重做）。
v2 的问题：上面一块 DAW 卷帘、下面一个小人、中间死黑 = 还是图表；没有 swing；核心冲突（被对齐成格子 vs 摆）没上画面。
v3：
  一个空间   纸卷横贯画面中部，他**站在纸卷上**、洞从他身体穿过；相机沿纸卷走（网格 / 拍线 / 面包屑都是世界坐标，随纸卷左移）。
  拍线       每拍一条竖线（下拍亮、写小节号）以 115.28 BPM 刷过；**每个洞相对最近八分格的偏移 = 一条红刻度**（越偏越亮）—— 这就是 swing。
  swing 读数 读头旁实时写 swing = 反拍落点的中位数（直 = 0.50，三连音 = 0.67；最近 4 小节的真数据）。
  他         contrapposto：重心在右腿、左腿点地、胯歪、帽檐斜、拐杖斜撑；笔画粗细有变化。每拼好一笔起就在**反拍（按真 swing 落点）bob** 一下。
  清屏重长   从第一个下拍起每 4 小节一次：纸卷带 2 帧填白 → 洞从读头向两侧重新长出（密度曲线锯齿呼吸；学 _参考_Short #4 的蓝闪）。
  尺度       0–8 s 只有一条大的旋律线居中（唯一的东西）；8.0 s 通电时它对数缩成纸卷顶上的一行，纸卷同时长出来（替换式转场）。
  其余同 v2（两段式缓动、虚线弧 + 真标签、读头红、第一句人声 28.40 s 拐杖落地）。
用法同 roll2d.py：still / sheet / clip --out ...
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
DIM = (58, 57, 54)
RED = (214, 52, 38)
FONT_MONO = "C:/Windows/Fonts/consola.ttf"
FONT_MONO_B = "C:/Windows/Fonts/consolab.ttf"

W0, H0 = 1920, 1080
G = H0 / 18
BEAT_PX = 2.2 * G
HEAD_X = 6.5 * G
ROW_Y0, ROW_Y1 = 5.6 * G, 11.4 * G   # 纸卷带
T_ON = 7.906                        # 鼓第一下（beat 15）；贝斯 8.434 = bar 1 下拍
FIG_X = 16.5 * G                     # 他站在纸卷上（读头右侧 10 格）
UNIT = (ROW_Y1 - ROW_Y0) / 15.5 * 1.55   # 身高 ≈ 1.55 × 纸卷带高 → 从纸卷底站起，头顶出纸卷


def ease2(u, a=0.6, t1=3 / 24, t2=9 / 24):
    if u <= 0:
        return 0.0
    return a * (1 - math.exp(-u / t1)) + (1 - a) * (1 - math.exp(-u / t2))


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
        self.phrases = d.get("phrases", [])
        low = [h for h in self.holes if h["band"] <= 1]
        sc = np.zeros(4)
        for h in low:
            i = int(np.argmin(np.abs(self.beats - h["t"])))
            if abs(self.beats[i] - h["t"]) < 0.06:
                sc[i % 4] += h["strength"]
        self.down_phase = 16 % 4            # 👤分镜：bar 1 = 贝斯进入的那一拍（beat 16 = 8.434）
        # 每个洞：拍内相位、相对最近八分格的偏移（秒）、是否反拍
        for h in self.holes:
            i, ph, _, _, _ = self.beat_info(h["t"])
            h["phase"] = ph
            grid = round(ph * 2) / 2          # 最近的八分格：0 / 0.5 / 1
            h["off"] = (ph - grid) * self.beat  # 秒，正 = 迟到
            h["offbeat"] = 0.3 < ph < 0.85
        self.downbeats = [b for k, b in enumerate(self.beats) if k % 4 == self.down_phase]

    def melody_at(self, t):
        i = min(max(int(t / self.mel_hop), 0), len(self.f0) - 1)
        return self.f0[i], self.voiced[i]

    def beat_info(self, t):
        i = int(np.searchsorted(self.beats, t, side="right")) - 1
        if i < 0:
            return -1, 0.0, 99.0, False, 0
        nxt = self.beats[i + 1] if i + 1 < len(self.beats) else self.beats[i] + self.beat
        down = (i % 4) == self.down_phase
        bar = (i - self.down_phase) // 4 + 1
        return i, (t - self.beats[i]) / max(nxt - self.beats[i], 1e-6), t - self.beats[i], down, bar

    def swing(self, t, bars=4):
        """最近 bars 小节里反拍洞的相位中位数（直 0.50 / 三连音 0.67）；样本 < 6 → None"""
        i0 = int(np.searchsorted(self.ht, t - bars * 4 * self.beat)); i1 = int(np.searchsorted(self.ht, t))
        ph = [h["phase"] for h in self.holes[i0:i1] if h["offbeat"] and h["strength"] > 0.3]
        return float(np.median(ph)) if len(ph) >= 6 else None


# 他（单位：UNIT；脚底 (0,0)，y 向上）。contrapposto：重心在右腿，胯向右顶，肩向左压，头回正，帽檐斜，拐杖斜撑在右外侧。
# (类型, 参数, 笔宽倍率)
FIGURE = [
    ("bar", (1.2, 0, 1.0, 5.2), 1.0),                  # 右腿（承重，直）
    ("bar", (0.4, 0, 2.2, 0), 0.8),                    # 右脚
    ("bar", (-0.4, 5.0, -2.3, 2.4), 0.9),              # 左大腿（向外）
    ("bar", (-2.3, 2.4, -1.4, 0), 0.9),                # 左小腿（点地）
    ("bar", (-2.4, 0, -0.9, 0), 0.7),                  # 左脚尖
    ("bar", (-0.6, 5.0, 1.4, 5.3), 1.1),               # 胯（向右顶）
    ("bar", (0.4, 5.2, -0.2, 10.2), 1.4),              # 躯干（向左回倾）
    ("bar", (-2.8, 10.6, 2.3, 10.0), 1.0),             # 肩（左高右低）
    ("arc", (-2.6, 8.6, 2.0, 95, 265), 0.8),           # 左臂叉腰（半圆）
    ("bar", (2.3, 10.0, 3.4, 7.6), 0.8),               # 右上臂
    ("bar", (3.4, 7.6, 4.0, 6.0), 0.8),                # 右前臂 → 手
    ("arc", (-0.2, 12.3, 1.5, 0, 90), 0.9), ("arc", (-0.2, 12.3, 1.5, 90, 180), 0.9),
    ("arc", (-0.2, 12.3, 1.5, 180, 270), 0.9), ("arc", (-0.2, 12.3, 1.5, 270, 360), 0.9),   # 头：环
    ("bar", (-2.9, 13.5, 2.6, 14.1), 0.6),             # 帽檐（斜）
    ("bar", (-1.6, 13.65, -1.3, 15.4), 0.9), ("bar", (1.4, 13.95, 1.7, 15.7), 0.9), ("bar", (-1.3, 15.4, 1.7, 15.7), 0.6),  # 帽身（斜）
]
HAND = (4.0, 6.0)
STROKE_W = 0.5 * UNIT


def stroke_order():
    def low(s):
        k, p, _ = s
        return min(p[1], p[3]) if k == "bar" else p[1] - p[2]
    return sorted(range(len(FIGURE)), key=lambda i: low(FIGURE[i]))


class Scene:
    def __init__(self, song: Song, w=W0, h=H0, ss=2):
        self.s = song; self.W, self.H, self.ss = w, h, ss
        self.k = w / W0
        self.fonts = {n: ImageFont.truetype(FONT_MONO, int(n * ss * self.k)) for n in (11, 13, 15, 18, 22)}
        self._schedule()
        self.flashes = [b for b in song.downbeats if b > T_ON + 0.5][::4]   # 每 4 小节一次清屏
        for f in (18.856,):                                                  # 分镜：S05→S06 用清屏重长切入
            if all(abs(f - x) > 0.1 for x in self.flashes):
                self.flashes.append(f)
        self.flashes.sort()

    def _schedule(self):
        s = self.s
        order = stroke_order()
        cand = sorted([h for h in s.holes if 10.6 <= h["t"] <= 26.0], key=lambda h: -h["strength"])   # 分镜：第一笔在 S04（10.516 后）从画右进
        picked = []
        for h in cand:
            if all(abs(h["t"] - p["t"]) >= 0.4 for p in picked):
                picked.append(h)
            if len(picked) >= len(order):
                break
        picked.sort(key=lambda h: h["t"])
        self.births = dict(zip(order, picked))
        self.cane_t = max(h["t"] for h in picked) + 0.3
        self.voice1 = next((p["start"] for p in s.phrases if p["start"] > self.cane_t), 28.4)
        self.first_stroke_t = picked[0]["t"]

    # 分镜（Synapsis 剪辑/00 开头 0-30 的 9 个镜头；t0 t1 变焦 z0→z1 世界中心 (cx,cy) 转角）
    SHOTS = [
        ("S01", 0.000, 4.260, 1.25, 1.42, (HEAD_X - 1.5 * G, 8.5 * G), 0.0),
        ("S02", 4.260, 7.906, 0.82, 0.78, (HEAD_X + 2.5 * G, 8.3 * G), 0.0),
        ("S03", 7.906, 10.516, 0.98, 1.25, (W0 * 0.5, 8.6 * G), 0.0),   # 切前推近 = 溶解是切的前奏
        ("S04", 10.516, 14.686, 2.0, 2.1, (HEAD_X + 3.2 * G, 8.9 * G), 0.0),
        ("S05", 14.686, 18.856, 1.0, 1.06, (HEAD_X + 7.5 * G, 8.4 * G), 0.0),
        ("S06", 18.856, 20.942, 1.7, 1.78, (FIG_X - 0.5 * G, ROW_Y1 - 11.5 * UNIT), -2.5),
        ("S06b", 20.942, 23.026, 2.3, 2.45, (FIG_X - 0.2 * G, ROW_Y1 - 12.3 * UNIT), 3.0),
        ("S07", 23.026, 25.390, 1.0, 1.08, (FIG_X - 1.0 * G, ROW_Y1 - 8.5 * UNIT), 90.0),
        ("S08", 25.390, 28.400, 1.1, 1.28, (FIG_X - 1.5 * G, ROW_Y1 - 7.0 * UNIT), 0.0),
        ("S09", 28.400, 30.000, 2.5, 2.6, (FIG_X + 6.4 * UNIT, ROW_Y1 - 1.2 * UNIT), 0.0),
    ]
    BLACK = [(28.32, 28.40)]   # 黑帧休止

    def shot(self, t):
        for sh in self.SHOTS:
            if sh[1] <= t < sh[2]:
                return sh
        return self.SHOTS[-1] if t >= self.SHOTS[-1][2] else self.SHOTS[0]

    # 相机：每镜独立的变焦 / 中心 / 转角（硬切 = 参数不连续），镜内永远在动（推 + 漂 + 微转）
    def cam(self, t):
        name, t0, t1, z0, z1, (wx, wy), rot_deg = self.shot(t)
        u = (t - t0) / max(t1 - t0, 1e-6)
        z = z0 + (z1 - z0) * u
        z *= 1.0 + 0.012 * math.sin(2 * math.pi * t / 9.7)
        dx = (26 * math.sin(2 * math.pi * t / 31.0) + 10 * math.sin(2 * math.pi * t / 7.3))
        dy = 16 * math.cos(2 * math.pi * t / 19.0)
        rot = math.radians(rot_deg) + math.radians(0.6) * math.sin(2 * math.pi * t / 41.0)
        sx, sy = W0 * 0.5, H0 * 0.5
        c, sn = math.cos(rot), math.sin(rot)
        k = self.k * self.ss
        def P(x, y):
            x, y = (x - wx) * z, (y - wy) * z
            x, y = x * c - y * sn, x * sn + y * c
            return ((x + sx + dx) * k, (y + sy + dy) * k)
        return P

    def line(self, d, P, a, b, fill, w):
        d.line([P(*a), P(*b)], fill=fill, width=max(1, int(w * self.k * self.ss)))

    def text(self, d, P, xy, s, fill, n=11, anchor="la"):
        d.text(P(*xy), s, font=self.fonts[n], fill=fill, anchor=anchor)

    def rect(self, d, P, x0, y0, x1, y1, fill=None, outline=None, w=1):
        """世界矩形 → 屏幕四边形（相机可能转了 90°）"""
        pts = [P(x0, y0), P(x1, y0), P(x1, y1), P(x0, y1)]
        if fill is not None:
            d.polygon(pts, fill=fill)
        if outline is not None:
            d.line(pts + [pts[0]], fill=outline, width=max(1, int(w * self.k * self.ss)))

    def bezier_dashed(self, d, P, a, b, fill, bend=0.28, dash=(5, 6), w=1, frac=1.0):
        ax, ay = a; bx, by = b
        mx, my = (ax + bx) / 2, (ay + by) / 2
        dx, dy = bx - ax, by - ay
        L = math.hypot(dx, dy) + 1e-9
        cx, cy = mx - dy / L * bend * L, my + dx / L * bend * L
        n = max(8, int(L / 4)); pts = []
        for i in range(n + 1):
            u = i / n * frac
            pts.append(((1 - u) ** 2 * ax + 2 * (1 - u) * u * cx + u * u * bx, (1 - u) ** 2 * ay + 2 * (1 - u) * u * cy + u * u * by))
        acc = 0.0; on = True; seg = [pts[0]]
        for p0, p1 in zip(pts, pts[1:]):
            acc += math.dist(p0, p1); seg.append(p1)
            if acc >= (dash[0] if on else dash[1]):
                if on:
                    d.line([P(*p) for p in seg], fill=fill, width=max(1, int(w * self.k * self.ss)))
                acc = 0.0; on = not on; seg = [p1]
        if on and len(seg) > 1:
            d.line([P(*p) for p in seg], fill=fill, width=max(1, int(w * self.k * self.ss)))
        if frac >= 0.5:
            u = 0.5
            x = (1 - u) ** 2 * ax + 2 * (1 - u) * u * cx + u * u * bx; y = (1 - u) ** 2 * ay + 2 * (1 - u) * u * cy + u * u * by
            tx, ty = 2 * (1 - u) * (cx - ax) + 2 * u * (bx - cx), 2 * (1 - u) * (cy - ay) + 2 * u * (by - cy)
            tl = math.hypot(tx, ty) + 1e-9; tx, ty = tx / tl, ty / tl; nx, ny = -ty, tx
            d.line([P(x - tx * 6 + nx * 4, y - ty * 6 + ny * 4), P(x, y), P(x - tx * 6 - nx * 4, y - ty * 6 - ny * 4)],
                   fill=fill, width=max(1, int(self.k * self.ss)))

    # 他的 bob：反拍（按真 swing 落点）下蹲再弹起；幅度随「已拼好的笔画数」长
    def bob(self, t, n_done):
        s = self.s
        if n_done == 0:
            return 0.0, 0.0
        i, ph, _, _, _ = s.beat_info(t)
        sw = s.swing(t) or 0.6
        # 以 ph = sw 为中心的脉冲（宽 0.22 拍），再加同相位的小左右晃
        dph = (ph - sw + 0.5) % 1.0 - 0.5
        pulse = math.exp(-(dph / 0.11) ** 2)
        amp = 0.22 * UNIT * min(1.0, n_done / 8)
        return -amp * pulse, 0.35 * amp * math.sin(2 * math.pi * (ph - sw))   # (dy, dx)

    def fig(self, x, y, off):
        """身体坐标 → 屏幕（世界坐标，他站在纸卷上，纸卷底 = 脚）"""
        return (FIG_X + x * UNIT + off[1], ROW_Y1 - y * UNIT + off[0])

    def stroke_draw(self, d, P, idx, prog, fill, off, w_mul=1.0):
        kind, p, wm = FIGURE[idx]
        w = STROKE_W * wm * w_mul
        if kind == "bar":
            a = self.fig(p[0], p[1], off); b = self.fig(p[2], p[3], off)
            b2 = (a[0] + (b[0] - a[0]) * prog, a[1] + (b[1] - a[1]) * prog)
            self.line(d, P, a, b2, fill, w)
        else:
            cx, cy, r, a0, a1 = p
            span = (a1 - a0) * prog; start = a0 - 45 * (1 - prog)
            n = max(4, int(span / 6)); pts = []
            for i in range(n + 1):
                ang = math.radians(start + span * i / n)
                pts.append(self.fig(cx + r * math.cos(ang), cy + r * math.sin(ang), off))
            if len(pts) > 1:
                d.line([P(*q_) for q_ in pts], fill=fill, width=max(1, int(w * self.k * self.ss)), joint="curve")

    def stroke_anchor(self, idx, off):
        kind, p, _ = FIGURE[idx]
        return self.fig(p[0], p[1], off)

    def cane(self, t, dt=1 / 240):
        s = self.s
        t0 = self.cane_t
        if t < t0:
            return None
        L = math.hypot(2.0, HAND[1]) * UNIT
        th0 = math.atan2(2.0, HAND[1])
        g_L = 9.81 / 1.0; c = 2.2
        th, om = th0 + 0.35, 0.0
        sw = 0.62
        # 踢：第一句人声之后每个反拍（按 swing 落点）抬一下
        kicks = [b + sw * s.beat for b in s.beats if self.voice1 <= b + sw * s.beat <= t]
        ki = 0; tt = t0; landed = []; above = True
        while tt < t:
            while ki < len(kicks) and kicks[ki] <= tt:
                om -= 1.6; ki += 1
            om += (-g_L * math.sin(th - th0) * 3.0 - c * om) * dt
            th += om * dt
            if th < th0:
                if above or om < -0.3:
                    landed.append(tt)
                th = th0; om = max(om, 0) * -0.15; above = False
            else:
                above = True
            tt += dt
        return th, th0, L, landed

    # ------------------------------------------------------------ 画
    def draw(self, t):
        s = self.s
        ss, k = self.ss, self.k
        W, H = int(self.W * ss), int(self.H * ss)
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img, "RGBA")
        if any(a <= t < b for a, b in self.BLACK):
            return img.resize((int(self.W), int(self.H))) if ss != 1 else img
        P = self.cam(t)
        shot_name = self.shot(t)[0]
        bi, bph, since, down, bar = s.beat_info(t)
        on = t >= T_ON
        u_on = t - T_ON
        v = BEAT_PX / s.beat
        def X(tt):   # 世界时间 → 屏幕 x
            return HEAD_X + (tt - t) * v
        row_h = (ROW_Y1 - ROW_Y0) / s.nb

        # 清屏：最近一次 flash 之后的重长进度
        last_flash = max([f for f in self.flashes if f <= t], default=None)
        regrow = ease2(t - last_flash, t1=0.05, t2=0.22) if last_flash is not None else 1.0
        flash_now = last_flash is not None and t - last_flash < 2 / 24

        # 世界网格：随纸卷走（每拍一列、每格一行）
        if on:
            kb0 = int(np.searchsorted(s.beats, t - HEAD_X / v)) - 1
            kb1 = int(np.searchsorted(s.beats, t + (W0 - HEAD_X) / v)) + 1
            for kb in range(max(kb0, 0), min(kb1, len(s.beats))):
                x = X(s.beats[kb])
                isdown = (kb % 4) == s.down_phase
                a_ = int((150 if isdown else 70) * regrow)
                self.line(d, P, (x, ROW_Y0 - 0.8 * G), (x, ROW_Y1 + 0.8 * G), (*GREY, a_) if isdown else (*DIM, a_), 1)
                if isdown:
                    self.text(d, P, (x + 4, ROW_Y0 - 1.3 * G), f"bar {(kb - s.down_phase) // 4 + 1}", (*GREY, a_), 11)
                else:
                    # 八分格：半拍处一根更短的线
                    pass
                xh = X(s.beats[kb] + s.beat / 2)
                self.line(d, P, (xh, ROW_Y1 + 0.2 * G), (xh, ROW_Y1 + 0.5 * G), (*DIM, int(120 * regrow)), 1)

        # 纸卷带：上下边线（通电后长出）
        band_prog = ease2(u_on, t1=0.08, t2=0.3) if on else 0.0
        if band_prog > 0:
            for yy in (ROW_Y0, ROW_Y1):
                self.line(d, P, (HEAD_X - band_prog * (HEAD_X + 100), yy), (HEAD_X + band_prog * (W0 - HEAD_X + 100), yy), (*GREY, 150), 1)
            for b in range(s.nb):
                y = ROW_Y1 - (b + 0.5) * row_h
                lab = f"{s.bands[b]:4.0f}" if s.bands[b] < 1000 else f"{s.bands[b] / 1000:3.1f}k"
                self.text(d, P, (0.6 * G, y - 7), lab, (*DIM, int(255 * band_prog)), 11)
        if flash_now:
            self.rect(d, P, -50, ROW_Y0, W0 + 50, ROW_Y1, fill=(*WHITE, 235))

        # 洞 + swing 刻度
        n_read = int(np.searchsorted(s.ht, t))
        if on and not flash_now:
            i0 = int(np.searchsorted(s.ht, t - (HEAD_X + 100) / v)); i1 = int(np.searchsorted(s.ht, t + (W0 - HEAD_X + 60) / v))
            for h in s.holes[i0:i1]:
                x0 = X(h["t"]); x1 = x0 + max(h["dur"], 0.05) * v
                if x1 < 2.0 * G:
                    continue
                y = ROW_Y1 - (h["band"] + 0.5) * row_h
                played = h["t"] <= t
                # 重长：离读头越远越晚长出
                gr = min(1.0, max(0.0, regrow * 1.6 - abs(x0 - HEAD_X) / (W0 * 0.9)))
                if gr <= 0:
                    continue
                hh = (2.0 + 5.0 * h["strength"]) * gr
                if played:
                    fade = max(0.0, min(1.0, (x0 - 2.0 * G) / (2.5 * G)))
                    col = (*WHITE, int((255 if h["strength"] > 0.45 else 160) * fade * gr))
                else:
                    col = (*GREY, int((70 + 120 * h["strength"]) * gr))
                if played and t - h["t"] < 1 / 24:
                    col = (*RED, 255)
                    self.rect(d, P, x0 - 2, y - hh - 2, x1 + 2, y + hh + 2, fill=col)
                self.rect(d, P, x0, y - hh, x1, y + hh, fill=col)
                # swing 刻度：从最近八分格到洞起点的红短线，越偏越亮
                if abs(h["off"]) > 0.012 and h["strength"] > 0.3:
                    xg = x0 - h["off"] * v
                    a_ = int(min(255, 60 + 1800 * abs(h["off"])) * gr * (1.0 if played else 0.6))
                    self.line(d, P, (xg, y + hh + 3), (x0, y + hh + 3), (*RED, a_), 1.5)
                    self.line(d, P, (xg, y + hh + 1), (xg, y + hh + 5), (*RED, a_), 1)

        # 读头
        head_col = (*RED, 240) if on else (*DIM, 200)
        self.line(d, P, (HEAD_X, ROW_Y0 - 1.6 * G), (HEAD_X, ROW_Y1 + 1.2 * G), head_col, 2 if on else 1)
        sw = s.swing(t)
        lab = f"t {t:7.3f}   beat {bi + 1 if bi >= 0 else 0:3d}   bar {bar:2d}   swing {sw:.2f}" if sw else f"t {t:7.3f}   beat {bi + 1 if bi >= 0 else 0:3d}   bar {bar:2d}   swing  --"
        bx0, by0 = HEAD_X + 8, ROW_Y1 + 1.25 * G
        if on and down and since < 1 / 24:
            self.rect(d, P, bx0 - 4, by0 - 3, bx0 + 330, by0 + 17, fill=(*RED, 255))
            self.text(d, P, (bx0, by0), lab, (*BG, 255), 11)
        else:
            self.rect(d, P, bx0 - 4, by0 - 3, bx0 + 330, by0 + 17, outline=(*RED, 200) if on else (*DIM, 200), w=1)
            self.text(d, P, (bx0, by0), lab, (*RED, 230) if on else (*GREY, 200), 11)

        # 旋律线：0–8 s 大而居中；通电后对数缩到纸卷顶上一行
        # 通电前 0.3 s 就开始缩（溶解是切的前奏）
        sh = min(1.0, ease2(t - (T_ON - 0.3), t1=0.1, t2=0.35)) if t >= T_ON - 0.3 else 0.0
        span_big, span_small = 9.0 * G, 0.9 * G
        yc_big, yc_small = 8.5 * G, ROW_Y0 - 1.0 * G
        span = span_big * (1 - sh) + span_small * sh
        yc = yc_big * (1 - sh) + yc_small * sh
        def fy(f):
            uu = (math.log2(f) - math.log2(s.f_lo)) / (math.log2(s.f_hi) - math.log2(s.f_lo))
            return yc + span / 2 - max(0, min(1, uu)) * span
        pts = []
        for j in range(0, 160):
            tb = t - j / 24
            if tb < 0:
                break
            fb, _ = s.melody_at(tb)
            pts.append(None if not np.isfinite(fb) else (X(tb), fy(fb)))
        seg = []
        lw = (5.0 * (1 - sh) + 1.0 * sh)
        for j, pt in enumerate(pts + [None]):
            if pt is None:
                if len(seg) > 1:
                    a_ = int(230 * (1 - (j - len(seg) / 2) / 170))
                    d.line([P(*q_) for q_ in seg], fill=(*WHITE, max(50, a_)), width=max(1, int(lw * k * ss)))
                seg = []
            else:
                seg.append(pt)
        f0, vo = s.melody_at(t)
        if np.isfinite(f0):
            r = 14 * (1 - sh) + 4 * sh
            self.rect(d, P, HEAD_X - r, fy(f0) - r, HEAD_X + r, fy(f0) + r, outline=(*WHITE, 240), w=1.5)
        k0 = int(t / 0.5)
        for kk in range(k0 - 7, k0 + 1):
            tb = kk * 0.5
            if tb < 0 or tb > t:
                continue
            fb, _ = s.melody_at(tb)
            if not np.isfinite(fb):
                continue
            age = t - tb
            a_ = int(200 * (1 - age / 4.2))
            self.text(d, P, (X(tb) + 4, fy(fb) - 15 * (1 - sh) - 13 * sh), f"[{tb:.1f}s {fb:.0f}Hz]", (*GREY, max(0, a_)), 11)
        if not on:
            self.text(d, P, (HEAD_X + 16, yc_big + span_big / 2 + 0.5 * G), "melody · other stem · f0", (*DIM, 220), 11)

        # 8.0 s 大字：115.28 由字符拼出（两段式），停 2 拍，再对数缩回（S03 里，纸卷带右下）
        if on and u_on < 2.6:
            if u_on < 2 * s.beat:
                sc_ = 1.0; alpha = 255; pr = ease2(u_on, t1=0.06, t2=0.25)
            else:
                uu = (u_on - 2 * s.beat) / (2.6 - 2 * s.beat)
                sc_ = math.exp(-2.6 * uu); alpha = int(255 * (1 - uu)); pr = 1.0
            cx_, cy_ = W0 * 0.5 + 2.5 * G, ROW_Y1 + 2.6 * G
            font = ImageFont.truetype(FONT_MONO_B, max(8, int(130 * sc_ * k * ss)))
            txt = f"{s.bpm:.2f}"
            nchar = max(1, int(round(pr * len(txt) + 0.49)))
            d.text(P(cx_, cy_), txt[:nchar], font=font, fill=(*WHITE, alpha), anchor="lm")
            self.text(d, P, (cx_ + 2, cy_ + 75 * sc_), "BPM · MACHINE ON", (*GREY, alpha), 11)

        # 他：先算 bob
        n_done = sum(1 for h in self.births.values() if t - h["t"] > 0.5)
        off = self.bob(t, n_done)
        for idx, h in self.births.items():
            u = t - h["t"]
            if u < 0:
                continue
            pr = min(1.0, ease2(u, t1=0.07, t2=0.28))
            # 镂空：先画一条背景色粗线，让他压在洞上面
            self.stroke_draw(d, P, idx, pr, (*BG, 255), off, w_mul=2.2)
            self.stroke_draw(d, P, idx, pr, (*WHITE, 255), off)
            if u < 1.2:
                decay = 1 - u / 1.2
                y = ROW_Y1 - (h["band"] + 0.5) * row_h
                a = (HEAD_X, y); b = self.stroke_anchor(idx, off)
                self.bezier_dashed(d, P, a, b, (*GREY, int(220 * decay)), frac=min(1.0, u / 0.25))
                self.text(d, P, (b[0] + 10, b[1] - 16), f"b{h['band']:02d} {s.bands[h['band']]:.0f}Hz  s {h['strength']:.2f}  off {h['off'] * 1000:+.0f}ms",
                          (*GREY, int(200 * decay)), 11)

        cn = self.cane(t)
        if cn:
            th, th0, L, landed = cn
            hx, hy = self.fig(*HAND, off)
            tipx, tipy = hx + L * math.sin(th), hy + L * math.cos(th)
            appear = ease2(t - self.cane_t, t1=0.07, t2=0.28)
            tip_d = (hx + (tipx - hx) * appear, hy + (tipy - hy) * appear)
            self.line(d, P, (hx, hy), tip_d, (*BG, 255), STROKE_W * 2.0)
            self.line(d, P, (hx, hy), tip_d, (*WHITE, 255), STROKE_W * 0.8)
            self.text(d, P, (tipx + 10, tipy - 6), f"θ {th - th0:+.2f} rad", (*GREY, 200), 11)
            for tl in landed:
                if t - tl < 0.5:
                    a_ = 1 - (t - tl) / 0.5
                    gx = FIG_X + HAND[0] * UNIT + L * math.sin(th0)
                    self.rect(d, P, gx - 12, ROW_Y1 + 2, gx + 12, ROW_Y1 + 5, fill=(*RED, int(255 * a_)))

        # intro 底噪
        if not on:
            cr = float(s.crack[min(int(t / s.crack_hop), len(s.crack) - 1)])
            rng = np.random.default_rng(int(t * 8))
            n_ = int(70 * cr)
            for x, y in zip(rng.uniform(0, W0, n_), rng.uniform(0, H0, n_)):
                d.point(P(x, y), fill=(*WHITE, 150))

        # S09：swing 数值放大成大字 1 拍再缩回（字号对数）
        if shot_name == "S09" and sw:
            uu = (t - 28.4) / s.beat
            if uu < 1.6:
                size = int(max(14, 120 * math.exp(-1.6 * max(0, uu - 0.6))) * k * ss)
                f = ImageFont.truetype(FONT_MONO_B, size)
                d.text((W * 0.30, H * 0.30), f"swing {sw:.2f}", font=f, fill=(*WHITE, int(255 * min(1, 1.8 - uu))), anchor="lm")
        # 页脚：屏幕坐标（不随相机）
        d.text((G * 0.5 * k * ss, (H0 - G * 0.55) * k * ss),
               f"PICCOLO AND A CANE   roll 01   {s.bpm:.2f} bpm   holes read {n_read:4d}   strokes {n_done:2d}/{len(self.births)}   {shot_name}",
               font=self.fonts[11], fill=(*GREY, 180))
        if ss != 1:
            img = img.resize((int(self.W), int(self.H)), Image.LANCZOS)
        return img


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
        if i % 96 == 0:
            el = time.time() - t_start
            print(f"{i}/{n}  {el:.0f}s  eta {el / max(i, 1) * (n - i):.0f}s", flush=True)
    proc.stdin.close(); proc.wait()
    contact_sheet(sheet, out.with_name(out.stem + "_接触表.jpg"))
    print(out)


if __name__ == "__main__":
    main()
