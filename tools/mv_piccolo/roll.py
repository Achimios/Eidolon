#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🎬Piccolo-渲染-「纸卷」：活字印刷 / 钢琴纸卷 / 墨 的代码生成画面（MV_Piccolo，2026-10-24，Fable 起草）。
一切都是歌曲时间 t 的函数（无随机状态；噪声按帧号播种），预览帧 = 导出帧。纯 PIL + numpy，CPU。

画面语法（见 MV_Piccolo/Thymos.md「视觉语言」）：
  纸     暖纸底 + 纤维噪声（静态，一次算好）；墨点底噪 = 高频爆裂声的真实密度（分析_纸卷.json 的 crackle）。
  纸卷   横向匀速左移，速度 = 真 BPM（1 拍 = BEAT_PX 像素）；洞 = 分频带起音（低频在下、高频在上），长度 = 能量维持时间。
  读头   固定的一对红色细线（唯一的点缀色）；洞经过读头 = 被播放：墨变实 + 只在这个洞上套印错位一帧。拍点不抖全屏。
  他     经过读头的洞脱离纸卷，沿临界阻尼弹簧飞到轮廓点上（质量守恒：洞飞走的地方纸卷留下空白），从脚往上长成一个拿拐杖的人。
  拐杖   阻尼单摆（真物理），被拍点踢；只有在他成形 ≥ 60% 之后才出现。
  小鸟   短笛 = 一串点 + 两笔翅膀，高度 = 旋律 f0（对数音高），x 领先读头一点；没旋律时淡出并下沉。
  镜头   慢速连续漂移 + 缓慢变焦，不从静止开始也不停在静止（所有坐标过一次相机，不做后期重采样）。

用法：
  python tools/mv_piccolo/roll.py still --t 12.5 --out x.jpg
  python tools/mv_piccolo/roll.py clip --t0 0 --t1 30 --out 样片/01_纸卷_接通_0-30_v1.mp4 [--w 1920 --h 1080] [--ss 2]
输出不覆盖，重名就停。clip 旁边出 <名>_接触表.jpg（12 帧）。
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[2]
PROJ = ROOT.parents[1] / "Projekt_Visual"
MV = PROJ / "MV_Piccolo"
MUSIC = PROJ / "Assets" / "Piccolo and a Cane[短笛和拐杖] - Electro Swing - 免版税音乐.mp4"
ANALYSIS = MV / "分析_纸卷.json"

PAPER = (233, 222, 198)
INK = (30, 26, 24)
RED = (178, 46, 34)
BEAT_PX = 150            # 1 拍走多少像素（1080p 坐标）
HEAD_X = 0.40            # 读头在画面宽度的位置
ROLL_Y0, ROLL_Y1 = 0.22, 0.74   # 纸卷上下边（画面高度比例）
FIG_X, FIG_FOOT_Y = 0.72, 0.84  # 他站的位置（脚底）
FIG_H = 0.52             # 他的身高（画面高度比例）
FONT = "C:/Windows/Fonts/georgia.ttf"
FONT_MONO = "C:/Windows/Fonts/cour.ttf"


# ---------------------------------------------------------------- 数据
class Song:
    def __init__(self, path=ANALYSIS):
        d = json.loads(Path(path).read_text("utf-8"))
        self.bpm = d["bpm"]; self.beat = 60.0 / self.bpm
        self.beats = np.array(d["beats"]); self.duration = d["duration"]
        self.bands = d["bands"]; self.nb = len(self.bands)
        self.holes = [h for h in d["holes"] if h["strength"] >= 0.22]
        self.ht = np.array([h["t"] for h in self.holes])
        m = d["melody"]; self.mel_hop = m["hop_s"]
        self.f0 = np.array([np.nan if v is None else v for v in m["f0"]], float)
        self.voiced = np.array(m["voiced"], float)
        self.crack_hop = d["crackle"]["hop_s"]; self.crack = np.array(d["crackle"]["v"])
        self.loud = np.array(d["loud"]["v"])
        # 下拍相位：低频带（band 0/1）洞最强的拍相位
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

    def beat_phase(self, t):
        """(上一拍序号, 拍内相位 0~1, 距上一拍秒)"""
        i = int(np.searchsorted(self.beats, t, side="right")) - 1
        if i < 0:
            return -1, 0.0, 99.0
        nxt = self.beats[i + 1] if i + 1 < len(self.beats) else self.beats[i] + self.beat
        return i, (t - self.beats[i]) / max(nxt - self.beats[i], 1e-6), t - self.beats[i]


# ---------------------------------------------------------------- 他：轮廓点
def figure_targets(n=190, seed=3):
    """拿拐杖的歌舞演员轮廓（单位：身高 = 1，脚底 (0,0)，y 向上）。笔画 → 沿线均匀取点，带笔宽抖动。
    姿势：重心偏右靠在拐杖上，左腿从前面交叉过去，头是一个环（面具），帽子歪着。
    返回 (n,2) 点，按 y 从低到高排序（先长脚）；以及手的位置（拐杖挂点）。"""
    rng = np.random.default_rng(seed)
    S = []  # (起点, 终点, 笔宽, 权重)
    # 右腿：直立承重；左腿：从前面交叉到右边（脚尖点地）
    S += [((0.04, 0.00), (0.03, 0.25), 0.030, 1.0), ((0.03, 0.25), (0.00, 0.47), 0.030, 1.0)]
    S += [((-0.05, 0.47), (0.06, 0.26), 0.028, 1.0), ((0.06, 0.26), (0.15, 0.02), 0.026, 1.0)]
    S += [((0.00, 0.47), (0.055, 0.78), 0.060, 2.4)]                                                   # 躯干（微倾）
    S += [((-0.07, 0.785), (0.16, 0.775), 0.030, 1.0)]                                                 # 肩
    S += [((-0.07, 0.785), (-0.14, 0.66), 0.022, 0.8), ((-0.14, 0.66), (-0.03, 0.52), 0.020, 0.8)]      # 左臂叉腰
    S += [((0.16, 0.775), (0.22, 0.64), 0.022, 0.8), ((0.22, 0.64), (0.25, 0.53), 0.020, 0.8)]          # 右臂下垂握杖
    # 头：环（面具）—— 12 段折线近似圆
    cx, cy, r = 0.075, 0.86, 0.055
    ring = [(cx + r * math.cos(2 * math.pi * k / 12), cy + r * math.sin(2 * math.pi * k / 12)) for k in range(13)]
    S += [(ring[k], ring[k + 1], 0.018, 1.0) for k in range(12)]
    # 帽：歪的 —— 帽檐、帽身两边、帽顶
    S += [((-0.03, 0.905), (0.19, 0.935), 0.012, 0.7), ((0.02, 0.915), (0.035, 1.0), 0.02, 0.7),
          ((0.135, 0.93), (0.13, 1.015), 0.02, 0.7), ((0.035, 1.0), (0.13, 1.015), 0.012, 0.6)]
    lens = np.array([math.dist(a, b) * w for a, b, _, w in S])
    counts = np.maximum(1, np.round(lens / lens.sum() * n)).astype(int)
    pts = []
    for (a, b, wd, _), c in zip(S, counts):
        u = (np.arange(c) + rng.uniform(0.2, 0.8, c)) / c
        ax, ay = a; bx, by = b
        dx, dy = bx - ax, by - ay
        L = math.hypot(dx, dy) + 1e-9
        nx, ny = -dy / L, dx / L
        off = rng.uniform(-0.5, 0.5, c) * wd
        pts.append(np.c_[ax + dx * u + nx * off, ay + dy * u + ny * off])
    P = np.concatenate(pts)
    P = P[np.argsort(P[:, 1] + rng.uniform(-0.04, 0.04, len(P)))]
    hand = np.array([0.25, 0.53])
    return P, hand


def spring(e0, v0, w, u):
    """临界阻尼弹簧位移：e(u) = (e0 + (v0 + w e0) u) e^{-w u}"""
    return (e0 + (v0 + w * e0) * u) * math.exp(-w * u)


# ---------------------------------------------------------------- 场景
class Scene:
    def __init__(self, song: Song, w=1920, h=1080, ss=2):
        self.s = song; self.W, self.H, self.ss = w, h, ss
        self.paper = self._paper()
        self.targets, self.hand = figure_targets()
        self._assign_particles()
        self.font = ImageFont.truetype(FONT, int(16 * ss * h / 1080))
        self.mono = ImageFont.truetype(FONT_MONO, int(14 * ss * h / 1080))

    # 纸：两层低频噪声 + 纤维（横向拉长的细噪声）
    def _paper(self):
        W, H = self.W * self.ss, self.H * self.ss
        rng = np.random.default_rng(7)
        base = np.zeros((H, W), np.float32)
        for scale, amp in ((64, 0.55), (16, 0.3), (4, 0.15)):
            n = rng.normal(size=(H // scale + 2, W // scale + 2)).astype(np.float32)
            im = Image.fromarray(n, "F").resize((W, H), Image.BICUBIC)
            base += amp * np.asarray(im)
        fib = rng.normal(size=(H, W // 6)).astype(np.float32)
        fib = np.asarray(Image.fromarray(fib, "F").resize((W, H), Image.BILINEAR))
        g = 1 + 0.035 * base + 0.03 * fib
        col = np.stack([np.clip(PAPER[i] * g, 0, 255) for i in range(3)], -1).astype(np.uint8)
        return Image.fromarray(col, "RGB")

    # 哪些洞变成他：经过读头、够强的洞，按时间顺序领取轮廓点（从脚到头）
    def _assign_particles(self):
        s = self.s
        cand = [h for h in s.holes if 8.0 <= h["t"] <= 27.5 and h["strength"] >= 0.3]
        # 均匀抽到目标点数
        n = len(self.targets)
        if len(cand) > n:
            idx = np.linspace(0, len(cand) - 1, n).round().astype(int)
            cand = [cand[i] for i in idx]
        self.particles = []
        for k, h in enumerate(cand):
            self.particles.append({"t": h["t"], "band": h["band"], "dur": h["dur"], "strength": h["strength"], "target": self.targets[k % n],
                                   "w": 2 * math.pi / 1.1})     # ω：约 1.1 s 到位
        self.particle_ts = {(p["t"], p["band"]) for p in self.particles}

    # 相机：坐标变换（1080p 坐标 → 超采样像素）
    def cam(self, t):
        z = 1.045 - 0.045 * min(t / 30.0, 1.0) ** 0.8          # 缓慢拉开（不归零：到 30 s 还在变）
        dx = 10 * math.sin(2 * math.pi * t / 37.0) + 4 * math.sin(2 * math.pi * t / 11.3)
        dy = 7 * math.cos(2 * math.pi * t / 29.0)
        cx, cy = self.W * 0.5, self.H * 0.5
        def P(x, y):
            return ((x - cx) * z + cx + dx) * self.ss, ((y - cy) * z + cy + dy) * self.ss
        return P, z

    def roll_y(self, band, P):
        """频带 → 纸卷上的行（低频在下）"""
        y0, y1 = ROLL_Y0 * self.H, ROLL_Y1 * self.H
        row_h = (y1 - y0) / self.s.nb
        return y1 - (band + 0.5) * row_h, row_h

    def hole_rect(self, h, t, P):
        """洞在屏幕上的矩形（1080p 坐标）：x = 读头 + (洞时间 − t)·速度"""
        v = BEAT_PX / self.s.beat
        x0 = HEAD_X * self.W + (h["t"] - t) * v
        x1 = x0 + max(h["dur"], 0.07) * v
        y, rh = self.roll_y(h["band"], P)
        r = rh * 0.34 * (0.6 + 0.4 * h["strength"])
        return x0, y - r, x1, y + r

    # ------------------------------------------------------------ 画
    def draw(self, t):
        s = self.s
        img = self.paper.copy()
        d = ImageDraw.Draw(img, "RGBA")
        P, z = self.cam(t)
        ss = self.ss
        W, H = self.W, self.H
        bi, bph, since = s.beat_phase(t)

        # 底噪墨点：密度 = crackle；按帧播种
        crack = s.at(s.crack, s.crack_hop, t)
        rng = np.random.default_rng(int(t * 24))
        n_spk = int(900 * crack ** 1.6 * (1 - 0.6 * s.at(s.loud, s.crack_hop, t)))
        if n_spk:
            xs = rng.uniform(0, W * ss, n_spk); ys = rng.uniform(0, H * ss, n_spk)
            rs = rng.uniform(0.4, 1.6, n_spk) * ss
            for x, y, r in zip(xs, ys, rs):
                d.ellipse([x - r, y - r, x + r, y + r], fill=(*INK, 150))

        # 纸卷边：两条很淡的线 + 行的齿孔（真纸卷两侧有牵引孔）
        y0, y1 = ROLL_Y0 * H, ROLL_Y1 * H
        for yy in (y0, y1):
            d.line([P(-50, yy), P(W + 50, yy)], fill=(*INK, 60), width=max(1, int(1 * ss)))
        v = BEAT_PX / s.beat
        # 牵引孔：每半拍一个
        k0 = int((t - (W * HEAD_X + 60) / v) / (s.beat / 2))
        for k in range(k0, k0 + int(W / (BEAT_PX / 2)) + 3):
            tx = k * s.beat / 2
            x = HEAD_X * W + (tx - t) * v
            for yy in (y0 - 14, y1 + 14):
                a, b = P(x - 3, yy - 3), P(x + 3, yy + 3)
                d.ellipse([*a, *b], fill=(*INK, 110))

        # 洞：屏幕内的
        lo, hi = t - (W * (1 - HEAD_X) + 100) / v * -1, 0  # 占位
        i0 = int(np.searchsorted(s.ht, t - (W * HEAD_X + 200) / v))
        i1 = int(np.searchsorted(s.ht, t + (W * (1 - HEAD_X) + 50) / v))
        playing = None
        for h in s.holes[i0:i1]:
            x0, ya, x1, yb = self.hole_rect(h, t, P)
            if x1 < -20 or x0 > W + 20:
                continue
            gone = (h["t"], h["band"]) in self.particle_ts and h["t"] <= t   # 变成他的洞：纸上留下的是空白
            played = h["t"] <= t
            if gone:
                # 空白：比纸稍亮的一块（洞被拿走了，纸也没了 —— 其实是透光）
                d.rounded_rectangle([*P(x0, ya), *P(x1, yb)], radius=(yb - ya) / 2 * ss, fill=(248, 242, 228, 200))
                continue
            alpha = 235 if played else 150 + int(60 * h["strength"])
            d.rounded_rectangle([*P(x0, ya), *P(x1, yb)], radius=(yb - ya) / 2 * ss, fill=(*INK, alpha))
            # 正在被读头读的洞：只在它身上套印错位（红色副本偏 3 px）
            if x0 <= HEAD_X * W <= x1 and t - h["t"] < 0.09:
                off = 3 * (1 - (t - h["t"]) / 0.09)
                d.rounded_rectangle([*P(x0 + off, ya - off * 0.4), *P(x1 + off, yb - off * 0.4)],
                                    radius=(yb - ya) / 2 * ss, fill=(*RED, 120))

        # 读头：一对红细线，拍点上粗 60 ms
        thick = 1.0 + (2.2 if since < 0.06 else 0.0)
        hx = HEAD_X * W
        for dx_ in (-5, 5):
            d.line([P(hx + dx_, y0 - 30), P(hx + dx_, y1 + 30)], fill=(*RED, 210), width=max(1, int(thick * ss)))
        # 下拍：读头顶上一个小红方块盖下来（印章的雏形）
        if bi >= 0 and (bi % 4) == s.down_phase and since < 0.12:
            a = 1 - since / 0.12
            d.rectangle([*P(hx - 9, y0 - 48), *P(hx + 9, y0 - 36)], fill=(*RED, int(230 * a)))

        # 他：飞行中 / 已到达的洞
        fx, fy = FIG_X * W, FIG_FOOT_Y * H
        hgt = FIG_H * H
        arrived = 0
        for p in self.particles:
            if p["t"] > t:
                continue
            u = t - p["t"]
            x0, ya, x1, yb = self.hole_rect(p, p["t"], P)   # 出发时的位置（读头处）
            sx, sy = HEAD_X * W, (ya + yb) / 2
            tx, ty = fx + p["target"][0] * hgt, fy - p["target"][1] * hgt
            e0x, e0y = sx - tx, sy - ty
            # 初速度：顺着纸卷运动方向（向左）再稍微向上，像被弹出来
            v0x, v0y = -v * 0.6, -80.0
            ex = spring(e0x, v0x, p["w"], u); ey = spring(e0y, v0y, p["w"], u)
            x, y = tx + ex, ty + ey
            done = abs(ex) + abs(ey) < 1.5
            arrived += done
            L = max(p["dur"], 0.07) * v
            L = L * (1 - 0.7 * min(u / 1.2, 1)) + 6 * min(u / 1.2, 1)   # 飞着飞着缩成一粒
            r = (yb - ya) / 2 * (1 - 0.35 * min(u / 1.2, 1))
            if done:   # 纸色镂空：让他从纸卷上分离出来（像剪纸贴在上面）
                k = 3.5
                d.rounded_rectangle([*P(x - L / 2 - k, y - r - k), *P(x + L / 2 + k, y + r + k)], radius=(r + k) * ss, fill=(*PAPER, 215))
            d.rounded_rectangle([*P(x - L / 2, y - r), *P(x + L / 2, y + r)], radius=r * ss, fill=(*INK, 235))
        frac = arrived / max(len(self.particles), 1)

        # 拐杖：成形 ≥ 60% 后出现；阻尼单摆，拍点踢
        if frac >= 0.6:
            th = self.cane_angle(t)
            hxp, hyp = fx + self.hand[0] * hgt, fy - self.hand[1] * hgt
            L = self.hand[1] * hgt * 1.0
            tipx, tipy = hxp + L * math.sin(th), hyp + L * math.cos(th)
            a = min(1.0, (frac - 0.6) / 0.25)
            d.line([P(hxp, hyp), P(tipx, tipy)], fill=(*PAPER, int(200 * a)), width=max(1, int(9 * ss)))
            d.line([P(hxp, hyp), P(tipx, tipy)], fill=(*INK, int(235 * a)), width=max(1, int(3 * ss)))
            # 杖头：小弯钩
            d.arc([*P(hxp - 14, hyp - 20), *P(hxp + 14, hyp + 4)], 200, 20, fill=(*INK, int(235 * a)), width=max(1, int(3 * ss)))

        # 小鸟（短笛）
        f0, vo = s.melody_at(t)
        self.draw_bird(d, P, t, f0, vo)

        # 标签：真参数
        lab = f"ROLL No.1   {s.bpm:.2f} BPM   t = {t:6.2f} s   holes played {int(np.searchsorted(s.ht, t))}   figure {frac*100:3.0f}%"
        d.text(P(40, H - 44), lab, font=self.mono, fill=(*INK, 170))
        d.text(P(W - 330, H - 46), "PICCOLO AND A CANE", font=self.font, fill=(*INK, 170))

        if ss != 1:
            img = img.resize((W, H), Image.LANCZOS)
        return img

    def cane_angle(self, t, dt=1 / 240):
        """阻尼单摆 θ'' = −(g/L) sin θ − c θ' + 拍点冲量（低频洞强度）。从拐杖出现时刻起积分（闭区间小，直接数值积）。
        确定性：同一 t 永远同一结果。"""
        s = self.s
        t_start = 20.0
        if t <= t_start:
            return 0.0
        g_L = 9.81 / 0.9; c = 0.9
        th0 = 0.22                      # 平衡位置：杖尖撑在身体外侧（靠着拐杖站）
        th, om = th0, 0.0
        lows = [(h["t"], h["strength"]) for h in s.holes if h["band"] <= 1 and t_start <= h["t"] <= t]
        li = 0
        tt = t_start
        while tt < t:
            while li < len(lows) and lows[li][0] <= tt:
                om += 1.1 * lows[li][1] * (1 if (int(lows[li][0] / s.beat) % 2 == 0) else -0.6)
                li += 1
            om += (-g_L * math.sin(th - th0) - c * om) * dt
            th += om * dt
            tt += dt
        return th

    def draw_bird(self, d, P, t, f0, vo):
        s = self.s
        W, H = self.W, self.H
        y0, y1 = ROLL_Y0 * H, ROLL_Y1 * H
        LO, HI = math.log2(120), math.log2(2500)
        if np.isfinite(f0):
            u = (math.log2(f0) - LO) / (HI - LO)
            y_target = y1 - u * (y1 - y0)
        else:
            y_target = y1 + 40   # 没旋律就往下沉
        # 平滑：对最近 0.25 s 的目标做指数平均（取样 6 点，确定性）
        ys = []
        for k in range(6):
            f, _ = s.melody_at(max(t - k * 0.05, 0))
            if np.isfinite(f):
                uu = (math.log2(f) - LO) / (HI - LO)
                ys.append(y1 - uu * (y1 - y0))
            else:
                ys.append(y_target)
        wts = np.exp(-np.arange(6) / 2.0)
        y = float(np.dot(ys, wts) / wts.sum())
        present = np.isfinite(f0)
        a = int(150 + 85 * min(vo * 6.0, 1.0)) if present else 60
        x = HEAD_X * W + 110 + 30 * math.sin(2 * math.pi * t / 5.1)
        # 翅膀：振翅相位随有声推进（有旋律时拍翅 4 Hz；无旋律时慢 1.2 Hz 滑翔）
        ph = 2 * math.pi * (4.0 if present else 1.2) * t
        wing = 0.65 * math.sin(ph)
        k = 2.6   # 尺寸
        body = [(x - 7 * k, y), (x, y - 2 * k), (x + 8 * k, y), (x + 14 * k, y - 1 * k)]
        for i_, (bx, by) in enumerate(body):
            rr = 2.4 * k * (1.0 if i_ < 3 else 0.6)
            d.ellipse([*P(bx - rr, by - rr), *P(bx + rr, by + rr)], fill=(*INK, a))
        for sgn in (-1, 1):
            d.line([P(x + 1 * k, y), P(x + (-6 + 3 * sgn) * k, y - 16 * k * (1 if sgn > 0 else 0.8) * (0.4 + wing * sgn))],
                   fill=(*INK, a), width=max(1, int(2.2 * self.ss)))


# ---------------------------------------------------------------- 输出
def contact_sheet(frames, out, cols=4):
    w, h = frames[0][0].size
    tw, th = w // 3, h // 3
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (tw * cols, th * rows), PAPER)
    for i, (f, lab) in enumerate(frames):
        im = f.resize((tw, th), Image.LANCZOS)
        ImageDraw.Draw(im).text((8, 6), lab, fill=RED, font=ImageFont.truetype(FONT_MONO, 22))
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
    if a.mode == "sheet":   # 12 帧接触表（不出视频）
        ts = np.linspace(a.t0, a.t1, 12, endpoint=False)
        contact_sheet([(sc.draw(float(t)), f"t={t:.2f}") for t in ts], out); print(out); return
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    n = int(round((a.t1 - a.t0) * a.fps))
    cmd = [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{a.w}x{a.h}", "-r", str(a.fps), "-i", "-",
           "-ss", f"{a.t0:.3f}", "-t", f"{a.t1 - a.t0:.3f}", "-i", str(MUSIC),
           "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-shortest", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    sheet_idx = set(np.linspace(0, n - 1, 12).round().astype(int).tolist())
    sheet = []
    import time
    t_start = time.time()
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
