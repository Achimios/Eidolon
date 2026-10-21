#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🔣几何符号-入口-片头字幕式的几何 / 符号层（拆解 Modular + FITC 2015 Titles 之后学来的 5 个手法）。
⭐判据「符号说真话」：画面上每一个数字、刻度、标签都是真值（时间 = 播放时间，来自 感官.json / 原曲音频）。
冷灰白为主、整幅最多一个红点；运动 = 片头字幕式的精确缓动（先快后慢），镜头从不静止；
拍点只作用在有形的元素上（一个字的笔画、一个标签框、一格字阵），不推全局放大 / 抖动。

手法（拆解见 Projekt_Visual/_参考库/分析_Modular/分析.md、分析_FITC2015_Titles/分析.md）：
  ① 模块字体 + 笔画组装 / 剥落（FITC）：数字由「横 / 竖 / 四分之一环 / 点」笔画拼成（字形是我们自己画的，
     3×5 格、笔宽 1 格）；新数字的笔画逐笔「画出」（条从一端长出，弧边转边长），旧数字的笔画一根根掉落。
  ② 两段式先快后慢缓动（Modular 0:42 实测：前 3 帧每帧 ×0.72，之后每帧 ×0.89）：
     P(t) = a(1−e^(−t/τ1)) + (1−a)(1−e^(−t/τ2))，τ1 = 3 帧、τ2 = 9 帧、a = 0.6，按区间长度归一；终点速度 > 0（不停）。
  ③ 对数螺旋拉远（Modular 的套娃倒数 → 我们改成相似变换群）：第 i 个数字的世界尺度 = Rⁱ、方位 = iφ；
     镜头进度 u(t) 每个真实拍点前进 1（区间内走 ②）→ 第 i 个字的屏幕变换 = R^(i−u)·rot((i−u)φ)。
     ⇒ 每个新数字出生时屏幕位置 / 大小完全相同（自相似，自检逐像素核），旧数字沿对数螺旋缩向消失点。
  ④ 相位环（Modular 的扇形扫出）：四个同心环的扫角 = 真实相位 —— 拍 / 小节（重拍）/ 乐句 / 全曲位置；
     乐句环上 16 个刻度；红点 = 「此刻」在乐句环上的位置（全幅唯一的红）。
  ⑤ 阶梯频谱条（Modular 的分段条）：24 个对数频带的真实能量（原曲 STFT，dB），量化到 8 级台阶；
     起音立刻顶上去，回落每 2 帧掉一级（台阶式缩回，不平滑）。
  ⑥ 标签框填实 1 帧（FITC animatic）：只在真实起音（感官.json onsets）那一帧把「ONSET」框填实。
  ⑦ 字阵 + 行选择条 + 收缩成点（FITC 0:35 + Modular 替换式转场）：狂语段（77.11–115.51 s）每一个真实起音 = 一格
     （142 格），格里的小字形由那个起音的张力（频谱质心）选、粗细由威胁定；吸收段每个起音按特征最近邻
     「认出」一格：选择条扫到那一行、那一格收缩成点、点沿缓动飞进中心的累计环（环的扫角 = 已吸收 / 142）。

  ⑧（B 版，独立，不绑 MV）埃拉托斯特尼筛：13×13 = 1..169，自己的 120 BPM，每个 p ≤ 13 一小节划掉倍数，
     剩下 39 = π(169) 个素数；声音 = 筛的声化（场景自带 audio()）。

两版输出（👤2026-10-22）：A 版绑《Can't Kill Us》（countdown / matrix，读感官.json）→ MV_Cant_Kill_Us/样片/几何符号/；
  B 版独立 motion design（sieve）→ Projekt_Visual/MotionDesign/几何符号/。配色 --palette：cold（冷灰白 + 一个红点）/
  bwr（黑白红，「此刻」整个是红）/ acid（紫黑 + 荧光酸橙）。版本号决定行为（旧版本号复现旧参数，见 build / main）。

用法（Python 用 4_Eidolon/.venv）：
  python tools/geo_symbols.py countdown --ver v5              # 憋住段 192.36–198.76 的拍数倒数（默认提前 1 拍起）
  python tools/geo_symbols.py matrix --ver v3                 # 吸收段 128.36–141.14 的字阵吸收
  python tools/geo_symbols.py sieve --ver v4 --palette bwr    # 独立版：筛（输出到 MotionDesign/）
  python tools/geo_symbols.py countdown --still 195.0         # 只出一帧 png（看构图）
  python tools/geo_symbols.py countdown --scale 0.5 --no-audio # 低分辨率小样
  公共参数：--start / --end（播放时间）、--ver v1、--scale、--ss（超采样倍数）、--out 目录
输出：Projekt_Visual/MV_Cant_Kill_Us/样片/几何符号/21_几何符号_<手法>_<ver>.mp4 + _接触表.jpg + .json（逐帧真值）。
⚠️不覆盖：同名文件存在就停（按版本另起名）。临时无声视频在 --tmp（默认系统临时目录/eidolon_geo），不删。
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stills as S  # noqa: E402

FPS = 24
W, H = 1280, 720
FONT = S.FONT
OUT_DIR = S.MV / "样片" / "几何符号"                  # A 版（绑《Can't Kill Us》）
MD_DIR = S.PROJ / "MotionDesign" / "几何符号"          # B 版（独立 motion design）
BG = (7, 8, 9)
GREY = tuple(int(255 * c) for c in S.GREY)          # 冷灰白（主色）
DIM = tuple(int(255 * 0.30 * c) for c in S.GREY)    # 暗灰（刻度、网格、次要）
MID = tuple(int(255 * 0.55 * c) for c in S.GREY)
RED = tuple(int(255 * c) for c in S.RED)            # 全幅唯一的红
BAR, CELL, FLASH, NOW_RED = (58, 62, 65), (44, 48, 50), GREY, False
# 配色（👤2026-10-22：出两套对比 ① 黑白红 ② 自由发挥）。cold = 本片冷灰白 + 一个红点（v1–v4 用的）。
PALETTES = {
    "cold": dict(BG=BG, GREY=GREY, MID=MID, DIM=DIM, RED=RED, BAR=(58, 62, 65), CELL=(44, 48, 50), FLASH=GREY, NOW_RED=False),
    # 黑白红：纯黑底、纯白主体；红不只是一个点 —— 「此刻」那个元素整个是红的（倒数的前位数字 / 字阵里正在开口的那一格）
    "bwr": dict(BG=(0, 0, 0), GREY=(244, 244, 242), MID=(138, 138, 136), DIM=(62, 62, 61), RED=(232, 24, 36),
                BAR=(70, 70, 69), CELL=(58, 58, 57), FLASH=(244, 244, 242), NOW_RED=True),
    # 酸性：紫黑底 + 偏紫的灰；强调色 = 荧光酸橙（只给「此刻」），选择条 / 标签闪 = 暗洋红
    "acid": dict(BG=(11, 9, 20), GREY=(228, 226, 242), MID=(128, 122, 158), DIM=(56, 50, 82), RED=(196, 255, 18),
                 BAR=(96, 18, 74), CELL=(52, 46, 80), FLASH=(255, 46, 160), NOW_RED=True),
}


def set_palette(name):
    g = globals()
    for k, v in PALETTES[name].items():
        g[k] = v


# ================================================================ ② 缓动
def ease2(u, n_frames=10.0, tau1=3.0, tau2=9.0, a=0.6):
    """两段式先快后慢：u ∈ [0,1] 是区间进度，n_frames = 区间长度（帧）。返回 0→1，终点速度 > 0。
    Modular 0:42+12 实测：帧差前 3 帧每帧 ×0.72（τ1≈3 帧），之后每帧 ×0.89（τ2≈9 帧）。"""
    u = np.clip(np.asarray(u, float), 0, 1)
    t = u * n_frames
    P = a * (1 - np.exp(-t / tau1)) + (1 - a) * (1 - np.exp(-t / tau2))
    PT = a * (1 - np.exp(-n_frames / tau1)) + (1 - a) * (1 - np.exp(-n_frames / tau2))
    return P / PT


# ================================================================ 感官
def sense():
    return json.loads((S.MV / "感官.json").read_text("utf-8"))


def channel(sn, name, t):
    v = sn["channels"][name]
    return float(v[min(max(int(t * sn["fps"]), 0), len(v) - 1)])


def phase_of(t, grid):
    """t 在格子 grid（升序时刻）里的 (序号 k, 相位 0..1)；k = 最后一个 ≤ t 的格点"""
    g = np.asarray(grid)
    k = int(np.searchsorted(g, t, side="right") - 1)
    if k < 0:
        p = g[1] - g[0]
        return -1, float(1 + (t - g[0]) / p) % 1.0
    if k >= len(g) - 1:
        p = g[-1] - g[-2]
        return k, float((t - g[-1]) / p) % 1.0
    return k, float((t - g[k]) / (g[k + 1] - g[k]))


def onset_strength(sn, o):
    """起音强度 = 威胁通道在起音后 ±3 帧内的最大值（同 animatic_v2.onset_strength）"""
    th = sn["channels"]["threat"]; f = int(o * sn["fps"])
    return float(max(th[max(0, f - 1):f + 4]))


def segments():
    return json.loads((S.MV / "情绪大图" / "草案.json").read_text("utf-8"))["段"]


# ================================================================ 画布（超采样 + 相似变换）
class Canvas:
    """在 ss 倍的 RGB 画布上画，最后 LANCZOS 缩到 (w, h)。坐标一律是**输出像素**（浮点）。"""

    def __init__(self, w=W, h=H, ss=3, bg=None):
        self.w, self.h, self.ss = w, h, ss
        bg = BG if bg is None else bg
        self.im = Image.new("RGB", (w * ss, h * ss), bg)
        self.d = ImageDraw.Draw(self.im)
        self._fonts = {}

    def font(self, px):
        k = max(6, int(round(px * self.ss)))
        if k not in self._fonts:
            self._fonts[k] = ImageFont.truetype(FONT, k)
        return self._fonts[k]

    def poly(self, pts, fill):
        s = self.ss
        self.d.polygon([(x * s, y * s) for x, y in pts], fill=fill)

    def line(self, pts, fill, width=1.0):
        s = self.ss
        self.d.line([(x * s, y * s) for x, y in pts], fill=fill, width=max(1, int(round(width * s))))

    def dot(self, x, y, r, fill):
        s = self.ss
        self.d.ellipse([(x - r) * s, (y - r) * s, (x + r) * s, (y + r) * s], fill=fill)

    def rect(self, x0, y0, x1, y1, fill=None, outline=None, width=1.0):
        s = self.ss
        if fill is not None:
            self.d.rectangle([x0 * s, y0 * s, x1 * s, y1 * s], fill=fill)
        if outline is not None:
            self.d.rectangle([x0 * s, y0 * s, x1 * s, y1 * s], outline=outline, width=max(1, int(round(width * s))))

    def text(self, x, y, txt, px, fill, anchor="la"):
        self.d.text((x * self.ss, y * self.ss), txt, font=self.font(px), fill=fill, anchor=anchor)

    def arc_band(self, cx, cy, r_in, r_out, a0, a1, fill, step_deg=2.0):
        """环形扇区（角度：度，屏幕坐标 y 向下 → 顺时针为正，0 = 3 点钟）"""
        if a1 - a0 <= 1e-6 or r_out <= 0:
            return
        n = max(2, int(np.ceil((a1 - a0) / step_deg)) + 1)
        a = np.radians(np.linspace(a0, a1, n))
        outer = [(cx + r_out * np.cos(q), cy + r_out * np.sin(q)) for q in a]
        inner = [(cx + r_in * np.cos(q), cy + r_in * np.sin(q)) for q in a[::-1]]
        self.poly(outer + inner, fill)

    def arc_line(self, cx, cy, r, a0, a1, fill, width=1.0, step_deg=1.5):
        if a1 - a0 <= 1e-6:
            return
        n = max(2, int(np.ceil((a1 - a0) / step_deg)) + 1)
        a = np.radians(np.linspace(a0, a1, n))
        self.line([(cx + r * np.cos(q), cy + r * np.sin(q)) for q in a], fill, width)

    def zoom(self, z, fx, fy):
        """把目前画好的内容绕 (fx, fy)（输出像素）放大 z 倍（之后再画的 HUD 不受影响）"""
        if abs(z - 1) < 1e-6:
            return
        s = self.ss
        W2, H2 = self.im.size
        # 输出 (X,Y) 取输入 (fx + (X − fx)/z, …)
        self.im = self.im.transform((W2, H2), Image.AFFINE, (1 / z, 0, fx * s * (1 - 1 / z), 0, 1 / z, fy * s * (1 - 1 / z)),
                                    resample=Image.BICUBIC, fillcolor=BG)
        self.d = ImageDraw.Draw(self.im)

    def result(self):
        return self.im.resize((self.w, self.h), Image.LANCZOS)


# ================================================================ ① 模块字体（自己画的：3×5 格，笔宽 1 格）
# 笔画：("h", y, x0, x1) 横条（中线 y，左右端 x0..x1）；("v", x, y0, y1) 竖条；
#       ("a", cx, cy, a0, a1) 环（中线半径 1，笔宽 1 → 内径 0.5 外径 1.5），角度顺时针、0 = 3 点钟；("d", x, y) 圆点（半径 0.5）
# 笔画的顺序 = 画出的顺序（自上而下、自左而右，像手写）。
FONT_GLYPHS = {
    "0": [("a", 1.5, 1.5, 180, 360), ("v", 2.5, 1.5, 3.5), ("a", 1.5, 3.5, 0, 180), ("v", 0.5, 1.5, 3.5)],
    "1": [("h", 0.5, 0.0, 1.5), ("v", 1.5, 0.0, 5.0), ("h", 4.5, 0.5, 2.5)],
    "2": [("a", 1.5, 1.5, 180, 450), ("a", 1.5, 3.5, 180, 270), ("v", 0.5, 3.5, 4.0), ("h", 4.5, 0.0, 3.0)],
    "3": [("a", 1.5, 1.5, 180, 450), ("a", 1.5, 3.5, 270, 540)],
    "4": [("v", 0.5, 0.0, 3.0), ("h", 2.5, 0.0, 3.0), ("v", 2.5, 0.0, 5.0)],
    "5": [("h", 0.5, 0.0, 3.0), ("v", 0.5, 0.0, 2.5), ("h", 2.5, 0.5, 1.5), ("a", 1.5, 3.5, 270, 540)],
    "6": [("a", 1.5, 1.5, 180, 315), ("v", 0.5, 1.5, 3.5), ("a", 1.5, 3.5, 0, 360)],
    "7": [("h", 0.5, 0.0, 3.0), ("v", 2.5, 0.5, 2.5), ("a", 1.5, 2.5, 0, 90), ("a", 1.5, 4.5, 180, 270), ("v", 0.5, 4.5, 5.0)],
    # （7 的 v1–v4 版：("v", 1.5, 3.5, 5.0) 直接接在弧的水平切线上，接口有缺口；v5 起改成 S 形两段弧）
    "8": [("a", 1.5, 1.5, 0, 360), ("a", 1.5, 3.5, 0, 360)],
    "9": [("a", 1.5, 1.5, 0, 360), ("v", 2.5, 1.5, 3.5), ("a", 1.5, 3.5, 0, 180)],
    "-": [("h", 2.5, 0.5, 2.5)],
    "/": [("d", 2.5, 0.5), ("d", 1.5, 2.5), ("d", 0.5, 4.5)],
}
GLYPH_W, GLYPH_H, GLYPH_GAP = 3.0, 5.0, 1.0


def stroke_bbox(st):
    k = st[0]
    if k == "h":
        return st[2], st[1] - 0.5, st[3], st[1] + 0.5
    if k == "v":
        return st[1] - 0.5, st[2], st[1] + 0.5, st[3]
    if k == "d":
        return st[1] - 0.5, st[2] - 0.5, st[1] + 0.5, st[2] + 0.5
    a = np.radians(np.linspace(st[3], st[4], 181))                     # 环：按扫过的角度取内外沿的真实外接框
    xs = np.concatenate([st[1] + r * np.cos(a) for r in (0.5, 1.5)])
    ys = np.concatenate([st[2] + r * np.sin(a) for r in (0.5, 1.5)])
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def text_strokes(txt):
    """一串字 → [(笔画, x 偏移)]，字间距 1 格。返回 (笔画列表, 总宽)"""
    out, x = [], 0.0
    for ch in txt:
        for st in FONT_GLYPHS[ch]:
            out.append((st, x))
        x += GLYPH_W + GLYPH_GAP
    return out, x - GLYPH_GAP


def draw_stroke(cv, st, ox, M, prog=1.0, fill=None, spin=0.0, wt=1.0, trim=False):
    """画一笔：M = (平移 x, 平移 y, 缩放 s, 旋转 rad) 把字形格坐标映到屏幕。
    prog ∈ [0,1] = 画出进度：条从起点端长出；环的扫角按 prog 长、同时起点往回转 spin·(1−prog) 度（FITC：边转边长）。
    wt = 笔宽（格）：字形骨架按笔宽 1 设计，wt < 1 时笔画沿中线变细（骨架不变）。"""
    hw = 0.5 * wt
    fill = GREY if fill is None else fill
    if prog <= 0:
        return
    tx, ty, s, rot = M
    c, sn = np.cos(rot), np.sin(rot)

    def P(x, y):
        x = x + ox
        return tx + s * (c * x - sn * y), ty + s * (sn * x + c * y)

    k = st[0]
    if k in ("h", "v"):
        if k == "h":
            y, x0, x1 = st[1], st[2], st[3]
            if trim:                               # 字形外缘的端点跟着笔宽收（wt < 1 时 0 / 3 处不再伸出竖笔外沿）
                x0 = 0.5 - hw if x0 == 0.0 else x0
                x1 = 2.5 + hw if x1 == GLYPH_W else x1
            x1 = x0 + (x1 - x0) * prog
            pts = [P(x0, y - hw), P(x1, y - hw), P(x1, y + hw), P(x0, y + hw)]
        else:
            x, y0, y1 = st[1], st[2], st[3]
            if trim:
                y0 = 0.5 - hw if y0 == 0.0 else y0
                y1 = 4.5 + hw if y1 == GLYPH_H else y1
            y1 = y0 + (y1 - y0) * prog
            pts = [P(x - hw, y0), P(x + hw, y0), P(x + hw, y1), P(x - hw, y1)]
        cv.poly(pts, fill)
    elif k == "d":
        x, y = P(st[1], st[2])
        cv.dot(x, y, hw * s * prog, fill)
    else:
        cx, cy, a0, a1 = st[1:]
        cxs, cys = P(cx, cy)
        off = -spin * (1 - prog)
        cv.arc_band(cxs, cys, (1 - hw) * s, (1 + hw) * s, a0 + off + np.degrees(rot), a0 + off + np.degrees(rot) + (a1 - a0) * prog, fill)


# ================================================================ ③ 对数螺旋拉远
class Spiral:
    """第 i 个元素的世界尺度 Rⁱ、方位 iφ；镜头进度 u(t)：在第 k 个拍区间 [b_k, b_{k+1}) 里 u = k + ease2(相位)。
    层号 r = min(0, i − u + 1)；屏幕变换 T_i(t) = C + R^r·rot(rφ)·v0，尺度 = h0·R^r。
    出生的那一拍 r ≡ 0，之后每拍退一层 —— 对每个 i 都一样（自相似：自检逐像素核）。"""

    def __init__(self, beats, R=2.1, phi_deg=-72.0, v0=(-0.31, 0.03), h0=0.34, center=(0.64, 0.50), w=W, h=H):
        self.b = np.asarray(beats, float)
        self.R, self.phi = R, np.radians(phi_deg)
        self.v0 = np.array([v0[0] * w, v0[1] * h])
        self.h0 = h0 * h
        self.C = np.array([center[0] * w, center[1] * h])

    def u(self, t):
        b = self.b
        k = int(np.searchsorted(b, t, side="right") - 1)
        if k < 0:                                  # 第一拍之前：按第一拍的区间外推（已经在动，镜头不从静止开始）
            p = b[1] - b[0]
            f = (t - b[0]) / p                     # 负
            kk = np.floor(f)
            return float(kk + ease2(f - kk, p * FPS))
        if k >= len(b) - 1:
            p = b[-1] - b[-2]
            f = (t - b[-1]) / p
            kk = np.floor(f)
            return float(len(b) - 1 + kk + ease2(f - kk, p * FPS))
        p = b[k + 1] - b[k]
        return float(k + ease2((t - b[k]) / p, p * FPS))

    def rel(self, i, t):
        """第 i 个元素此刻在第几层：出生（b_i）起就在 0 层（前位）停一拍，下一拍起退到 −1、−2 …
        （v1 是出生时 r = 1、从画外 R 倍大飞进来：笔画还没画完就在画面里乱飞，像散落的方块 → v2 改成原地画出、
        同一拍里上一个数字从同一个前位退走 = Modular 的「替换式转场」）"""
        return min(0.0, i - self.u(t) + 1.0)

    def xform(self, i, t):
        r = self.rel(i, t)
        sc = self.R ** r
        a = r * self.phi
        c, s = np.cos(a), np.sin(a)
        pos = self.C + sc * np.array([c * self.v0[0] - s * self.v0[1], s * self.v0[0] + c * self.v0[1]])
        return pos, sc * self.h0, a, r


# ================================================================ ⑤ 阶梯频谱条
class StepSpectrum:
    """24 个对数频带（40 Hz–12 kHz）的 dB 能量 → 8 级台阶；顶上去立刻，回落每 2 帧掉一级。"""

    def __init__(self, t0, t1, nb=24, steps=8, sr=22050, n=2048):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        import sensorium
        pad = 0.5
        self.t0 = t0 - pad
        x = sensorium.load_audio(S.MUSIC, max(0.0, self.t0), (t1 - t0) + 2 * pad, sr)
        if self.t0 < 0:
            x = np.concatenate([np.zeros(int(-self.t0 * sr)), x])
        self.x, self.sr, self.n, self.nb, self.steps = x, sr, n, nb, steps
        self.edges = np.geomspace(40, 12000, nb + 1)
        f = np.fft.rfftfreq(n, 1 / sr)
        self.bidx = [np.where((f >= self.edges[j]) & (f < self.edges[j + 1]))[0] for j in range(nb)]
        self.win = np.hanning(n)
        self.level = np.zeros(nb, int)
        self.hold = np.zeros(nb, int)
        self.db_lo, self.db_hi = -62.0, -12.0       # 量化范围（dB re 满幅）—— 写进 sidecar

    def bands_db(self, t):
        i = int((t - self.t0) * self.sr) - self.n // 2
        seg = self.x[max(0, i):max(0, i) + self.n]
        if len(seg) < self.n:
            seg = np.pad(seg, (0, self.n - len(seg)))
        X = np.abs(np.fft.rfft(seg * self.win)) / (self.n / 4)
        p = np.array([np.mean(X[ix] ** 2) if len(ix) else 0.0 for ix in self.bidx])
        return 10 * np.log10(p + 1e-12)

    def quantize(self, db):
        q = (db - self.db_lo) / (self.db_hi - self.db_lo)
        return np.clip(np.floor(q * self.steps + 0.5), 0, self.steps).astype(int)

    def step(self, t):
        """每帧调用一次：返回 (台阶级别, 真实 dB)"""
        db = self.bands_db(t)
        q = self.quantize(db)
        up = q >= self.level
        self.level = np.where(up, q, self.level)
        self.hold = np.where(up, 0, self.hold + 1)
        drop = (~up) & (self.hold >= 2)
        self.level = np.where(drop, np.maximum(self.level - 1, q), self.level)
        self.hold = np.where(drop, 0, self.hold)
        return self.level.copy(), db


def draw_spectrum(cv, lv, x0, y_base, bw, step_h, gap, steps, fill=None, frame=None):
    fill = MID if fill is None else fill; frame = DIM if frame is None else frame
    for j, L in enumerate(lv):
        x = x0 + j * (bw + gap)
        cv.line([(x, y_base + 3), (x + bw, y_base + 3)], frame, 1)
        for k in range(int(L)):
            y1 = y_base - k * step_h
            cv.rect(x, y1 - step_h + min(1.2, 0.3 * step_h), x + bw, y1, fill=fill)


# ================================================================ ④ 相位环
def draw_rings(cv, cx, cy, radii, phases, labels, red_ring=2, ticks=16, s=1.0):
    """radii[k] 半径；phases[k] ∈ [0,1]；扫角从 12 点钟顺时针。第 red_ring 个环带 ticks 个刻度 + 红点"""
    for k, (r, ph) in enumerate(zip(radii, phases)):
        cv.arc_line(cx, cy, r, -90, 270, DIM, 1.0)
        band = (2.2 if k < 2 else 1.4) * s
        cv.arc_band(cx, cy, r - band, r + band, -90, -90 + 360 * ph, GREY if k != 3 else MID)
        if k == red_ring:
            for j in range(ticks):
                a = np.radians(-90 + 360 * j / ticks)
                cv.line([(cx + (r + 4 * s) * np.cos(a), cy + (r + 4 * s) * np.sin(a)),
                         (cx + (r + 9 * s) * np.cos(a), cy + (r + 9 * s) * np.sin(a))], MID, 1.0)
            a = np.radians(-90 + 360 * ph)
            cv.dot(cx + r * np.cos(a), cy + r * np.sin(a), 3.2 * s, RED)
        lab = labels[k]
        if lab:                                   # 标签排在环的右侧一列
            ly = cy - 1.5 * radii[-1] * 0.5 + k * 13 * s - 20 * s
            lx = cx + radii[-1] + 26 * s
            cv.text(lx, ly, lab, 9.5 * s, MID, anchor="lm")


# ================================================================ 场景：⑦ 拍数倒数（憋住段）
class Countdown:
    """憋住段：从 192.36（乐句起点）到 198.76（下一乐句 = 冲顶）共 16 拍。第 i 拍出生的数字 = 离冲顶还剩几拍（16 − i），真数。"""
    name = "倒数"

    def __init__(self, sn, t_release=198.76, n=16, scale=1.0, t_end=None, wt=0.62, delay=2.0, spin=45.0, guide=True, trim=True):
        self.wt, self.delay, self.spin, self.guide = wt, delay, spin, guide     # v1/v2：wt 1.0、delay 0、spin 90、无引导线
        self.trim = trim                                                    # v5 起：外缘端点随笔宽收
        beats = np.array(sn["events"]["beats"])
        k1 = int(np.argmin(np.abs(beats - t_release)))
        self.release = float(beats[k1])
        self.beats = beats[k1 - n:k1 + 1]                     # n 个出生拍 + 冲顶那一拍
        self.n = n
        self.sn, self.s = sn, scale
        self.w, self.h = int(W * scale), int(H * scale)
        self.sp = Spiral(self.beats[:-1], w=self.w, h=self.h)
        self.down = np.array(sn["events"]["downbeats"]); self.phr = np.array(sn["events"]["phrases"])
        self.on = np.array(sn["events"]["onsets"])
        rng = np.random.default_rng(7)
        self.drop_order = {}                                   # 每个数字的笔画掉落顺序（固定种子）
        for i in range(n):
            ss, _ = text_strokes(str(n - i))
            self.drop_order[i] = rng.permutation(len(ss))
        self.spec = None
        self.ver_end = wt < 1.0                                # v4 起：最后一拍「1」快画、收缩推迟（v3 的「1」没画完就被收走）
        if self.ver_end:
            self.collapse = (self.beats[-1] - 5.0 / FPS, self.beats[-1] - 1.5 / FPS)
        else:
            self.collapse = (self.beats[-2] + 0.55 * (self.beats[-1] - self.beats[-2]), self.beats[-1] - 2.5 / FPS)
        self.t_end = t_end

    def digit_alpha(self, i, t):
        """返回每一笔的 (画出进度, 是否还在)"""
        born = self.beats[i]
        ss, _ = text_strokes(str(self.n - i))
        out = []
        r = self.sp.rel(i, t)
        for k, _ in enumerate(ss):
            last = (i == self.n - 1) and self.ver_end      # 最后一个数字（1）：只有半拍，要在收缩前画完
            dl, stg, dur = (0.0, 0.5, 4.0) if last else (self.delay, 1.5, 7.0)
            t0 = born + (dl + k * stg) / FPS                # 先让上一个数字从前位退开 delay 帧
            prog = float(ease2((t - t0) * FPS / dur, dur)) if t >= t0 else 0.0
            rank = int(np.where(self.drop_order[i] == k)[0][0])
            alive = r > -2.2 - 1.6 * rank / max(1, len(ss) - 1)   # 退到第 2.2–3.8 层时一根根掉
            out.append((prog, alive))
        return out

    def frame(self, t, cv):
        s = self.s
        sp = self.sp
        u = sp.u(t)
        c0, c1 = self.collapse
        col = float(ease2((t - c0) / (c1 - c0), (c1 - c0) * FPS)) if t > c0 else 0.0   # 最后一拍：全部收缩到红点
        k_on = [o for o in self.on if t - 1.0 / FPS < o <= t]
        info = {}
        # 消失点（螺旋中心）
        C = sp.C
        # 引导线：数字退场走的那条对数螺旋（= 镜头的真实轨迹 C + R^r·rot(rφ)·v0，r ∈ [−6, 0]）
        if self.guide and col < 0.98:
            rr = np.linspace(-6.0, 0.0, 240)
            pts = [tuple(C + (sp.R ** r) * np.array([np.cos(r * sp.phi) * sp.v0[0] - np.sin(r * sp.phi) * sp.v0[1],
                                                       np.sin(r * sp.phi) * sp.v0[0] + np.cos(r * sp.phi) * sp.v0[1]]) * (1 - col) + C * col)
                   for r in rr]
            cv.line(pts, DIM, 1.0)
            for j in range(-6, 1):                                 # 每层一个小刻度点
                q = C + (sp.R ** j) * np.array([np.cos(j * sp.phi) * sp.v0[0] - np.sin(j * sp.phi) * sp.v0[1],
                                                np.sin(j * sp.phi) * sp.v0[0] + np.cos(j * sp.phi) * sp.v0[1]])
                cv.dot(*(q * (1 - col) + C * col), 1.6 * s, MID)
            cv.text(C[0] + 4 * s, C[1] + sp.h0 * 0.62, f"LOG SPIRAL  R {sp.R:.2f}/BEAT  PHI {np.degrees(sp.phi):.0f} DEG", 9.5 * s, DIM)
        # 相位环（以消失点为心）
        kb, pb = phase_of(t, self.sn["events"]["beats"])
        kd, pd = phase_of(t, self.down)
        kp, pp = phase_of(t, self.phr)
        song = t / self.sn["duration"]
        bar_beat = int(np.searchsorted(self.sn["events"]["beats"], t, side="right")) - int(np.searchsorted(self.sn["events"]["beats"], self.down[max(kd, 0)], side="left"))
        radii = [r * self.h for r in (0.034, 0.054, 0.078, 0.104)]
        rs = 1 - col
        labels = [f"BEAT {pb:4.2f}", f"BAR {bar_beat}/4", f"PHRASE {kp + 1}/{len(self.phr)}  {pp:4.2f}",
                  f"SONG {t:6.2f}/{self.sn['duration']:.2f} s"]
        if rs > 0.02:
            draw_rings(cv, C[0], C[1], [r * rs for r in radii], [pb, pd, pp, song], labels if col < 0.05 else [""] * 4, s=s * rs)
        else:
            cv.dot(C[0], C[1], 3.2 * s, RED)
        info.update(beat_phase=round(pb, 4), bar_phase=round(pd, 4), bar_beat=bar_beat, phrase=kp + 1, phrase_phase=round(pp, 4), song=round(song, 5))
        # 数字（老的先画，新的盖上）
        shown = []
        for i in range(self.n):
            if t < self.beats[i] - 1e-9:
                continue
            pos, hpx, ang, r = sp.xform(i, t)
            if r < -6:
                continue
            ss, wid = text_strokes(str(self.n - i))
            unit = hpx / GLYPH_H
            # 字的中心在 pos；收缩时朝消失点拉
            pc = C + (pos - C) * (1 - col)
            unit *= (1 - col)
            if unit < 0.15:
                continue
            M = (pc[0] - wid / 2 * unit, pc[1] - GLYPH_H / 2 * unit, unit, 0.0)     # 字永远正立：只有位置走螺旋
            lum = float(np.clip(1.0 + 0.30 * r, 0.25, 1.0))
            fill = tuple(int(c * lum) for c in GREY)
            if NOW_RED and r > -0.5:                       # 黑白红 / 酸性：「此刻」的数字整个是强调色
                fill = RED
            for (st, xo), (prog, alive) in zip(ss, self.digit_alpha(i, t)):
                if alive:
                    draw_stroke(cv, st, xo, M, prog, fill, spin=self.spin, wt=self.wt, trim=self.trim)
            shown.append(self.n - i)
        info["digits_shown"] = shown
        cur = int(np.searchsorted(self.beats[:-1], t, side="right"))
        remaining = self.n - cur + 1 if cur >= 1 else None    # 当前数字 = 离冲顶还剩几拍
        if cur >= 1:
            remaining = self.n - (cur - 1)
        info["beats_to_release"] = remaining
        info["seconds_to_release"] = round(self.release - t, 3)
        # ⑤ 阶梯频谱（左下）
        lv, db = self.spec.step(t)
        if col < 0.98:
            k = 1 - col
            draw_spectrum(cv, lv, 46 * s, self.h - 74 * s, 7 * s * k, 4.2 * s * k, 3 * s * k, self.spec.steps)
            cv.text(46 * s, self.h - 52 * s, f"SPECTRUM 40 Hz-12 kHz  24 BANDS  {self.spec.db_lo:.0f}..{self.spec.db_hi:.0f} dB  8 STEPS", 9.5 * s, DIM)
        info["spectrum_steps"] = lv.tolist()
        # ⑥ 标签框（右上）：真实起音那一帧填实
        bx0, by0 = self.w - 230 * s, 40 * s
        last = self.on[self.on <= t]
        lo = float(last[-1]) if len(last) else None
        if col < 0.98:
            if k_on:
                cv.rect(bx0, by0, bx0 + 66 * s, by0 + 17 * s, fill=FLASH)
                cv.text(bx0 + 6 * s, by0 + 3.5 * s, "ONSET", 10 * s, BG)
            else:
                cv.rect(bx0, by0, bx0 + 66 * s, by0 + 17 * s, outline=MID, width=1)
                cv.text(bx0 + 6 * s, by0 + 3.5 * s, "ONSET", 10 * s, MID)
            if lo is not None:
                cv.text(bx0 + 76 * s, by0 + 3.5 * s, f"{lo:8.3f} s  THREAT {onset_strength(self.sn, lo):4.2f}", 10 * s, MID)
        info["onset_flash"] = bool(k_on)
        info["last_onset"] = lo
        # HUD 主行（左上）+ 真值
        if col < 0.98:
            rem = f"{remaining:2d}" if remaining is not None else "--"
            cv.text(46 * s, 40 * s, f"BEATS TO RELEASE  {rem}", 15 * s, GREY)
            cv.text(46 * s, 62 * s, f"T-{max(0.0, self.release - t):5.3f} s   RELEASE @ {self.release:.2f} s   150.0 BPM", 10 * s, MID)
            ten = channel(self.sn, "tension", t)
            hun = channel(self.sn, "hunger", t)
            cv.text(46 * s, 78 * s, f"TENSION {ten:4.2f}   HUNGER {hun:4.2f}   CAM u {u:6.3f}  R {sp.R:.2f}/BEAT", 10 * s, DIM)
            info.update(tension=ten, hunger=hun)
        info["cam_u"] = round(u, 4)
        info["collapse"] = round(col, 4)
        return info


# ================================================================ 场景：⑦ 字阵吸收（吸收段）
GLYPH_FAMILY = ["d", "h", "v", "a0", "a1", "a2", "a3", "ring"]     # 张力（频谱质心）从低到高 → 8 种小字形


def draw_mini(cv, kind, cx, cy, u, fill, weight=1.0):
    """格子里的小字形：u = 格边长；weight（来自威胁）= 笔宽倍率"""
    t = 0.16 * u * weight
    r = 0.30 * u
    if kind == "d":
        cv.dot(cx, cy, 0.5 * t + 0.6, fill)
    elif kind == "h":
        cv.rect(cx - r, cy - t / 2, cx + r, cy + t / 2, fill=fill)
    elif kind == "v":
        cv.rect(cx - t / 2, cy - r, cx + t / 2, cy + r, fill=fill)
    elif kind == "ring":
        cv.arc_band(cx, cy, 0.6 * r - t / 3, 0.6 * r + t / 3, 0, 360, fill)      # 环最重 → 收小、收细，不压过别的格
    else:
        q = int(kind[1])
        a0 = 180 + 90 * q
        mid = np.radians(a0 + 45)
        rr = 0.42 * u                                         # 四分之一环：弧中点落在格心
        ccx, ccy = cx - (rr - 0.5 * t) * np.cos(mid) * 0.62, cy - (rr - 0.5 * t) * np.sin(mid) * 0.62
        cv.arc_band(ccx, ccy, rr * 0.62 - t / 2, rr * 0.62 + t / 2, a0, a0 + 90, fill)


class Matrix:
    """狂语段每个真实起音 = 一格（按时间行优先排）；吸收段每个起音按 (张力, 威胁) 最近邻认出一格（不重复）。"""
    name = "字阵吸收"

    def __init__(self, sn, t0, t1, scale=1.0, voices=(77.11, 115.51), cols=18, v2=True, style=2):
        self.sn, self.s = sn, scale
        self.w, self.h = int(W * scale), int(H * scale)
        on = np.array(sn["events"]["onsets"])
        v = on[(on >= voices[0]) & (on < voices[1])]
        self.voices = v
        self.N = len(v)
        self.vt = np.array([channel(sn, "tension", o) for o in v])
        self.vs = np.array([onset_strength(sn, o) for o in v])
        self.kind = [GLYPH_FAMILY[min(7, int(x * 8))] for x in np.clip((self.vt - self.vt.min()) / (np.ptp(self.vt) + 1e-9), 0, 0.999)]
        self.cols = cols
        self.rows = int(np.ceil(self.N / cols))
        a = on[(on >= t0 - 1e-6) & (on < t1)]
        self.absorb_t = a
        # 认出：按出现顺序，每个起音取特征最近的、还没被吸收的格（贪心）
        taken = np.zeros(self.N, bool)
        self.match = []
        for o in a:
            ft, fs = channel(sn, "tension", o), onset_strength(sn, o)
            d = (self.vt - ft) ** 2 + 0.25 * (self.vs - fs) ** 2
            d[taken] = np.inf
            j = int(np.argmin(d)); taken[j] = True
            self.match.append((float(o), j, float(np.sqrt(d[j]))))
        # 版面（v0 静帧草稿：格 0.034H、环 0.17H 在 0.20W —— 环太大、字阵太小，像一张表；v1 起：字阵占右半幅，环收小，加飞行轨迹）
        # v2（style 2，👤主会话判据「没有情绪的精致 = 失败」）：v1 像一张数据表 —— 整片同一个中灰、环是个进度条。
        #   改：字阵暗下去、伸出画外（看不到边的一片声音）；只有被认出的那一格在它的起音上亮 3 帧（它「开口」）；
        #   落进环时环在落点被撞得往里凹（凹深 ∝ 那个声音的威胁，τ 0.18 s 弹回）；吸收过的声音的字形留在环外 = 她的身体长出了别人的笔画。
        self.v2, self.style = v2, (style if v2 else 1)
        st2 = self.style >= 2
        self.u = (0.064 if st2 else 0.052 if v2 else 0.034) * self.h     # 格边长
        self.gx0 = (0.45 if st2 else 0.395 if v2 else 0.40) * self.w; self.gy0 = 0.53 * self.h - self.rows * self.u / 2
        self.core = np.array([(0.215 if st2 else 0.205 if v2 else 0.20) * self.w, 0.53 * self.h])
        self.R = (0.105 if st2 else 0.125 if v2 else 0.17) * self.h
        self.fly = 0.42                                # 点飞行时长（秒）
        self.t0 = t0
        # v3（style 3）：镜头 = 基础速度的慢推 + 乐句边界一次「进」（2.5 帧爬升、τ 0.6 s 缓，不归零）。
        #   v2 只有 9 px/s 的横移，帧差 ~0.4（整段几乎不动）。推只改速度，不跟单个起音缩放。
        ph = np.array(sn["events"]["phrases"])
        self.cam_anchors = ph[(ph > t0 + 0.2) & (ph < t1)]
        self.cam_pos = None

    def cam_zoom(self, t):
        """z(t) = exp(∫ v)，v = 0.010/s + 0.05·Σ kick(t − 锚点)（闭式积分，任意 t 可取）"""
        base = 0.010 * (t - self.t0)
        kick = 0.0
        for a in self.cam_anchors:
            d = t - a
            if d > 0:
                r = 2.5 / FPS; tau = 0.6
                # ∫ min(1, x/r)·e^(−x/τ) dx 的近似：爬升段按线性、其后指数
                kick += tau * (1 - np.exp(-max(0.0, d - r) / tau)) + 0.5 * min(d, r)
        return float(np.exp(base + 0.05 * kick / 0.6))

    def ring_v2(self, cv, t, done, lag, n_in):
        """她的环：r(θ) = R − Σ_k A_k·e^(−age_k/0.18 s)·e^(−(Δθ_k/0.10)²)（落点被撞凹，A_k = (6 + 14·威胁) px），
        扫角 = 已到达 / N（真数）；N 个刻度；到过的声音在环外长出一根径向针（长 ∝ 威胁）。"""
        s = self.s
        cx, cy = self.core
        R = self.R
        th = np.radians(np.linspace(-90, 270, 721))
        r = np.full_like(th, R)
        for idx, (o, j, d) in enumerate(done):
            age = t - o - lag - self.fly
            if 0 <= age < 1.2:
                a = np.radians(-90 + 360 * (idx + 1) / self.N)
                dth = np.angle(np.exp(1j * (th - a)))
                r -= (6 + 14 * self.vs[j]) * s * np.exp(-age / 0.18) * np.exp(-(dth / 0.10) ** 2)
        pts = [(cx + rr * np.cos(q), cy + rr * np.sin(q)) for rr, q in zip(r, th)]
        cv.line(pts, MID, 1.0)
        m = int(round(720 * n_in / self.N))
        if m >= 1:                                                     # 已吸收的那一段：粗
            seg = pts[:m + 1]
            cv.line(seg, GREY, 2.6)
        for q in range(self.N):
            a = np.radians(-90 + 360 * q / self.N)
            ln = 4 if q % 10 == 0 else 1.8
            cv.line([(cx + (R - 5 * s) * np.cos(a), cy + (R - 5 * s) * np.sin(a)),
                     (cx + (R - (5 + ln) * s) * np.cos(a), cy + (R - (5 + ln) * s) * np.sin(a))], DIM, 1.0)
        for idx, (o, j, d) in enumerate(done[:n_in]):                   # 吸收过的声音长在环外：一根径向针，长 ∝ 它的威胁
            a = np.radians(-90 + 360 * (idx + 1) / self.N)               # （试过把小字形贴在环外：20 个挤成一团，像毛边）
            age = t - o - lag - self.fly
            L = (6 + 26 * self.vs[j]) * s * float(ease2(age * FPS / 6.0, 6.0))
            cv.line([(cx + (R + 3 * s) * np.cos(a), cy + (R + 3 * s) * np.sin(a)),
                     (cx + (R + 3 * s + L) * np.cos(a), cy + (R + 3 * s + L) * np.sin(a))], GREY, 1.3)

    def cell_xy(self, j, t):
        r, c = divmod(j, self.cols)
        drift = -9.0 * self.s * (t - self.t0)          # 字阵缓慢左移（镜头横移，永不停）
        return np.array([self.gx0 + (c + 0.5) * self.u + drift, self.gy0 + (r + 0.5) * self.u])

    def frame(self, t, cv):
        s = self.s
        info = {}
        done = [(o, j, d) for (o, j, d) in self.match if o <= t]
        absorbed = {j: o for (o, j, d) in done}
        cur = done[-1] if done else None
        # 选择条：当前被认出的那一行，条从左端以缓动伸到那一格、到格后再缩回（各 ~6 帧）
        if cur is not None:
            o, j, dist = cur
            r, c = divmod(j, self.cols)
            age = t - o
            xy = self.cell_xy(j, t)
            x_left = self.cell_xy(r * self.cols, t)[0] - self.u * 0.5
            grow = float(ease2(age * FPS / 6.0, 6.0))
            shrink = float(ease2((age - 0.25) * FPS / 6.0, 6.0)) if age > 0.25 else 0.0
            if shrink < 1:
                xa = x_left + (xy[0] - x_left) * shrink
                xb = x_left + (xy[0] + self.u * 0.5 - x_left) * grow
                cv.rect(xa, xy[1] - self.u * 0.42, xb, xy[1] + self.u * 0.42, fill=BAR if self.v2 else (38, 41, 43))
        # 格子
        for j in range(self.N):
            xy = self.cell_xy(j, t)
            if xy[0] < -self.u or xy[0] > self.w + self.u:
                continue
            cv.dot(xy[0] - self.u * 0.5, xy[1] - self.u * 0.5, 0.8 * s, DIM)    # 点阵底网（FITC 的坐标纸）
            if j in absorbed:
                age = t - absorbed[j]
                if self.style >= 2 and age < 3.0 / FPS:          # 它「开口」：在自己的起音上亮 3 帧，再收
                    draw_mini(cv, self.kind[j], xy[0], xy[1], self.u, RED if NOW_RED else GREY, 0.7 + 1.1 * self.vs[j])
                    continue
                if self.style >= 2:
                    age -= 3.0 / FPS
                k = float(ease2(age * FPS / 5.0, 5.0))            # 收缩成点（5 帧）
                if k < 1:
                    cv.dot(xy[0], xy[1], (1 - k) * self.u * 0.32 + 1.0 * s, GREY)
                else:
                    cv.rect(xy[0] - 1.2 * s, xy[1] - 1.2 * s, xy[0] + 1.2 * s, xy[1] + 1.2 * s, fill=DIM)   # 空位（记得它在哪）
                continue
            w8 = 0.7 + 1.1 * self.vs[j]
            draw_mini(cv, self.kind[j], xy[0], xy[1], self.u, CELL if self.style >= 2 else MID, w8)
        # 飞行中的点（红 = 最近一个，全幅唯一）→ 累计环
        n_in = 0
        lag = (8.0 if self.style >= 2 else 5.0) / FPS     # 开口 3 帧（v2）+ 收缩 5 帧之后才起飞
        for idx, (o, j, d) in enumerate(done):
            age = t - o - lag
            if age >= self.fly:
                n_in += 1
                continue
            if age < 0:
                continue
            p0 = self.cell_xy(j, o + lag)
            k = float(ease2(age / self.fly, self.fly * FPS))
            # 进环的落点 = 环上第 n 个刻度
            n_target = idx + 1
            a = np.radians(-90 + 360 * n_target / self.N)
            R = self.R
            p1 = self.core + R * np.array([np.cos(a), np.sin(a)])
            p = p0 + (p1 - p0) * k
            is_last = (idx == len(done) - 1)
            if self.v2:                                # 轨迹：从格子到当前位置的细线（FITC 的斜向拖尾），尾端跟着缩
                tail = p0 + (p1 - p0) * float(ease2(max(0.0, age - 0.10) / self.fly, self.fly * FPS))
                cv.line([tuple(tail), tuple(p)], MID, 1.0)
            cv.dot(p[0], p[1], 3.0 * s, RED if is_last else GREY)
        # 累计环：扫角 = 已到达 / N
        cx, cy = self.core
        R = self.R
        if self.style >= 2:
            self.ring_v2(cv, t, done, lag, n_in)
        else:
            cv.arc_line(cx, cy, R, -90, 270, DIM, 1.0)
        for q in range(self.N if self.style < 2 else 0):
            a = np.radians(-90 + 360 * q / self.N)
            ln = 5 if q % 10 == 0 else 2.5
            cv.line([(cx + (R + 4 * s) * np.cos(a), cy + (R + 4 * s) * np.sin(a)),
                     (cx + (R + (4 + ln) * s) * np.cos(a), cy + (R + (4 + ln) * s) * np.sin(a))], DIM, 1.0)
        if self.style < 2:
            cv.arc_band(cx, cy, R - 2.4 * s, R + 2.4 * s, -90, -90 + 360 * n_in / self.N, GREY)
        if self.v2 and self.style < 2:                                    # 到达那一帧：落点刻度拉长成一根亮针（作用在那一个刻度上）
            for idx, (o, j, d) in enumerate(done):
                ta = o + lag + self.fly
                if 0 <= t - ta < 2.0 / FPS:
                    a = np.radians(-90 + 360 * (idx + 1) / self.N)
                    cv.line([(cx + (R - 10 * s) * np.cos(a), cy + (R - 10 * s) * np.sin(a)),
                             (cx + (R + 16 * s) * np.cos(a), cy + (R + 16 * s) * np.sin(a))], GREY, 1.6)
        cv.text(cx, cy - 8 * s, f"{n_in:3d}", 30 * s, GREY, anchor="mm")
        cv.text(cx, cy + 20 * s, f"/ {self.N}", 11 * s, MID, anchor="mm")
        z = 1.0
        if self.style >= 3:
            z = self.cam_zoom(t)
            cv.zoom(z, 0.36 * self.w, 0.53 * self.h)
        info["cam_zoom"] = round(z, 4)
        # HUD（不随镜头）
        cv.text(46 * s, 40 * s, f"ASSIMILATED  {n_in} / {self.N}", 15 * s, GREY)
        cv.text(46 * s, 62 * s, f"VOICES = ONSETS IN 77.11-115.51 s (REMNANT)   ABSORB = ONSETS IN {self.t0:.2f}-   t {t:7.3f} s", 10 * s, MID)
        if cur is not None:
            o, j, dist = cur
            cv.text(46 * s, 78 * s, f"MATCH  onset {o:7.3f} s -> voice #{j:03d} ({self.voices[j]:7.3f} s)   d {dist:5.3f}   tension {self.vt[j]:4.2f}  threat {self.vs[j]:4.2f}", 10 * s, DIM)
        if self.style < 3:
            cv.text(self.gx0, self.gy0 - 14 * s, f"REMNANT VOICES  {self.N}   {self.rows} x {self.cols}   GLYPH = TENSION (8 BINS)   WEIGHT = THREAT", 9.5 * s, DIM)
        else:
            cv.text(46 * s, self.h - 40 * s, f"REMNANT VOICES {self.N}  ({self.rows} x {self.cols})   GLYPH = TENSION (8 BINS)   WEIGHT = THREAT   CAM x{z:5.3f}  (+1 PUSH PER PHRASE)", 9.5 * s, DIM)
        info.update(assimilated=n_in, matched=len(done), current=(None if cur is None else dict(onset=cur[0], voice=cur[1], voice_t=float(self.voices[cur[1]]), dist=round(cur[2], 4))))
        return info


# ================================================================ 独立版（不绑 MV）：⑧ 埃拉托斯特尼筛
def primes_upto(n):
    m = np.ones(n + 1, bool); m[:2] = False
    for p in range(2, int(n ** 0.5) + 1):
        if m[p]:
            m[p * p::p] = False
    return np.nonzero(m)[0]


class Sieve:
    """独立的 motion design（👤2026-10-22：B 版不绑 MV，自己的节奏和主题）。
    主题 = 埃拉托斯特尼筛：13×13 格 = 1..169；对每个 p ≤ √169 = 13 的素数，一小节（4 拍）里从 p² 起按 p 的步长划掉倍数。
    节奏 = 自己定的 120 BPM（拍 0.5 s）：前奏 2 拍 → 6 个小节（p = 2,3,5,7,11,13）→ 尾声 6 拍（剩下的素数按大小依次点亮，π 计数）。
    画面上的每个数都是真的：格里的数字、划掉的笔画数 = 这个数在 p² ≤ n 条件下被几个素数划过（重复劳动也画出来）、
    STRUCK / NEW / REMAINING、π(169) = 39。左边的大数字 = 当前素数 p（模块字体），换 p 时沿对数螺旋退场（手法 ③）。
    声音 = 筛本身的声化：每划一格一个 4 ms 的正弦嗒（音高 = 格所在的行），每小节第一拍一个低音。"""
    name = "筛"

    def __init__(self, scale=1.0, N=169, bpm=120.0, intro_beats=2, outro_beats=6, v=2):
        self.s = scale
        if v >= 2:
            outro_beats = 8                                            # 收尾的 π(N) 停 2 s
        self.w, self.h = int(W * scale), int(H * scale)
        self.N, self.side = N, int(round(N ** 0.5))
        self.beat = 60.0 / bpm
        self.bpm = bpm
        self.P = [int(p) for p in primes_upto(self.side)]               # 2 3 5 7 11 13
        self.t_pass = [intro_beats * self.beat + 4 * self.beat * k for k in range(len(self.P))]
        self.t_outro = self.t_pass[-1] + 4 * self.beat
        self.T = self.t_outro + outro_beats * self.beat
        # 每个划痕事件：(时刻, n, p)。一小节里：第 1 拍 p 亮起；第 1.5–3.75 拍按顺序划（16 分音符量化的错开），最后一拍留白
        self.events = []
        for k, p in enumerate(self.P):
            mult = list(range(p * p, N + 1, p))
            a, b = self.t_pass[k] + 0.5 * self.beat, self.t_pass[k] + 3.5 * self.beat
            for q, n in enumerate(mult):
                self.events.append((a + (b - a) * q / max(1, len(mult)), n, p))
        self.events.sort()
        self.primes = [int(x) for x in primes_upto(N)]
        self.u = 0.046 * self.h
        self.gx0 = (0.52 if v < 4 else 0.55) * self.w; self.gy0 = 0.5 * self.h - self.side * self.u / 2
        # 左边的大数字：每个 p 一个，尾声第 4 拍再来一个 = π(N)（v2 起；v1 尾声只剩 13 慢慢暗下去，没有收尾）
        self.v = v
        self.t_final = self.t_outro + 4 * self.beat
        self.big = [(tp, str(p)) for tp, p in zip(self.t_pass, self.P)] + ([(self.t_final, str(len(self.primes)))] if v >= 2 else [])
        self.sp = Spiral([b[0] for b in self.big], R=2.0, phi_deg=-64.0, v0=(-0.20 if v < 3 else -0.14 if v < 4 else -0.23, -0.02), h0=0.30,
                         center=(0.38 if v < 4 else 0.47, 0.54), w=self.w, h=self.h)   # v4：螺旋中心右移，前位数字不再盖住上一层
        self.cam_anchors = np.array([b[0] for b in self.big])
        self.zk = (0.012, 0.035) if v < 2 else (0.005, 0.022)        # v1 推到 ×1.49，网格顶行被裁 → v2 收到 ~×1.2

    def cell(self, n):
        r, c = divmod(n - 1, self.side)
        return np.array([self.gx0 + (c + 0.5) * self.u, self.gy0 + (r + 0.5) * self.u])

    def cam_zoom(self, t):
        z = self.zk[0] * t
        for a in self.cam_anchors:
            d = t - a
            if d > 0:
                z += self.zk[1] * (1 - np.exp(-d / 0.45)) * min(1.0, d / (2.5 / FPS))
        return float(np.exp(z))

    def frame(self, t, cv):
        s, u = self.s, self.u
        info = {}
        hits = [e for e in self.events if e[0] <= t]
        marks = {}
        last_hit = {}
        for (te, n, p) in hits:
            marks[n] = marks.get(n, 0) + 1
            last_hit[n] = te
        k = int(np.searchsorted(self.t_pass, t, side="right") - 1)
        p_now = self.P[k] if 0 <= k and t < self.t_outro else None
        f_small = 9.0 * s
        # 素数尾声：按大小依次点亮（6 拍里铺完）
        lit = 0
        if t >= self.t_outro:
            span = 4 * self.beat
            lit = int(np.clip(np.floor((t - self.t_outro) / span * len(self.primes)) + 1, 0, len(self.primes)))
        for n in range(1, self.N + 1):
            xy = self.cell(n)
            cv.dot(xy[0] - u / 2, xy[1] - u / 2, 0.7 * s, DIM)
            if n == 1:
                cv.text(xy[0], xy[1], "1", f_small, DIM, anchor="mm")
                continue
            m = marks.get(n, 0)
            if m:
                age = t - last_hit[n]
                g = float(ease2(age * FPS / 5.0, 5.0))
                # 数字收成一道横线（5 帧）；第 2、3 次被划 = 多一道更短的线（重复劳动）
                if g < 1:
                    cv.text(xy[0], xy[1], str(n), f_small * (1 - 0.6 * g), GREY, anchor="mm")
                for q in range(m):
                    L = (0.36 - 0.10 * q) * u * (g if q == m - 1 else 1.0)
                    yy = xy[1] + (q - (m - 1) / 2) * 3.2 * s
                    cv.line([(xy[0] - L, yy), (xy[0] + L, yy)], MID if q < m - 1 or g >= 1 else GREY, 1.4)
                continue
            is_p = n in self.P
            if p_now is not None and n == p_now:
                cv.rect(xy[0] - u * 0.44, xy[1] - u * 0.44, xy[0] + u * 0.44, xy[1] + u * 0.44, fill=RED)
                cv.text(xy[0], xy[1], str(n), f_small * 1.15, BG, anchor="mm")
                continue
            done_p = is_p and t >= self.t_pass[self.P.index(n)]
            if t >= self.t_outro and self.primes.index(n) < lit:
                cv.rect(xy[0] - u * 0.44, xy[1] - u * 0.44, xy[0] + u * 0.44, xy[1] + u * 0.44, outline=GREY, width=1)
                cv.text(xy[0], xy[1], str(n), f_small * 1.1, GREY, anchor="mm")
            else:
                cv.text(xy[0], xy[1], str(n), f_small, GREY if done_p else MID, anchor="mm")
        # 选择条：当前这一划所在的行（FITC 行选择条），左端跟着划痕走
        cur = [e for e in hits if t - e[0] < 3.0 / FPS]
        if cur and t < self.t_outro:
            te, n, p = cur[-1]
            xy = self.cell(n)
            x_left = self.gx0
            cv.line([(x_left, xy[1] + u * 0.5), (xy[0] + u * 0.5, xy[1] + u * 0.5)], FLASH, 1.2)
        # 左：当前 p 的大数字（模块字体）+ 已做过的 p 沿对数螺旋退场
        for i, (tb, txt) in enumerate(self.big):
            if t < tb:
                continue
            pos, hpx, ang, r = self.sp.xform(i, t)
            if r < -5:
                continue
            ss_, wid = text_strokes(txt)
            unit = hpx / GLYPH_H
            if self.v >= 3:                                  # 两位数不许出画（v2 右对齐仍被推出左缘）：前位点右移到 0.24W、
                unit = min(unit, 0.26 * self.w * (self.sp.R ** r) / wid)   # 居中、宽度封顶 0.26W·R^r（算上镜头 ×1.2 推近后左缘仍 > 0.03W）
                M = (pos[0] - wid / 2 * unit, pos[1] - GLYPH_H / 2 * unit, unit, 0.0)
            elif self.v >= 2:                                # v2：宽度封顶 0.26W·R^r，右缘对齐前位点
                unit = min(unit, 0.26 * self.w * (self.sp.R ** r) / wid)
                M = (pos[0] - wid * unit + 1.5 * unit, pos[1] - GLYPH_H / 2 * unit, unit, 0.0)
            else:
                M = (pos[0] - wid / 2 * unit, pos[1] - GLYPH_H / 2 * unit, unit, 0.0)
            lum = float(np.clip(1.0 + 0.32 * r, 0.22, 1.0))
            final = (i == len(self.P))
            fill = RED if (r > -0.5 and (t < self.t_outro or final)) else tuple(int(c * lum) for c in GREY)
            for q, (st, xo) in enumerate(ss_):
                t0 = tb + (2.0 + 1.5 * q) / FPS
                prog = float(ease2((t - t0) * FPS / 7.0, 7.0)) if t >= t0 else 0.0
                draw_stroke(cv, st, xo, M, prog, fill, spin=45.0, wt=0.62, trim=True)
        z = self.cam_zoom(t)
        cv.zoom(z, 0.50 * self.w, 0.52 * self.h)
        # HUD
        struck = len(marks)
        crossings = len(hits)
        remaining = self.N - 1 - struck
        bar = int(t // (4 * self.beat)) + 1
        cv.text(46 * s, 40 * s, "SIEVE OF ERATOSTHENES", 15 * s, GREY)
        cv.text(46 * s, 62 * s, f"N = {self.N} = {self.side}^2    p <= {self.side}    "
                + (f"p = {p_now}   from p^2 = {p_now * p_now}   step {p_now}" if p_now else ("p = -" if t < self.t_outro else f"pi({self.N}) = {lit if lit < len(self.primes) else len(self.primes)}")), 10 * s, MID)
        cv.text(46 * s, 78 * s, f"CROSSINGS {crossings:3d}   STRUCK {struck:3d}   REMAINING {remaining:3d}   (re-crossed {crossings - struck})", 10 * s, DIM)
        cv.text(46 * s, self.h - 40 * s, f"{self.bpm:.0f} BPM   BAR {bar}   BEAT {int(t / self.beat) % 4 + 1}/4   t {t:6.3f} s   CAM x{z:5.3f}", 9.5 * s, DIM)
        info.update(p=p_now, crossings=crossings, struck=struck, remaining=remaining, primes_lit=lit, cam_zoom=round(z, 4))
        return info

    def audio(self, sr=48000):
        """筛的声化：每次划格一个 4 ms 正弦嗒（行 r → 330·2^(−r/12) Hz，往下越低），每个 p 的第一拍一个 55 Hz 低音（120 ms）。"""
        n = int(self.T * sr) + sr
        x = np.zeros(n)
        tt = np.arange(int(0.004 * sr)) / sr
        for (te, m, p) in self.events:
            r = (m - 1) // self.side
            f = 330.0 * 2 ** (-r / 12)
            i = int(te * sr)
            x[i:i + len(tt)] += 0.18 * np.sin(2 * np.pi * f * tt) * np.hanning(len(tt))
        tb = np.arange(int(0.12 * sr)) / sr
        for a in [b[0] for b in self.big]:
            i = int(a * sr)
            x[i:i + len(tb)] += 0.35 * np.sin(2 * np.pi * 55 * tb) * np.exp(-tb / 0.04)
        for j, q in enumerate(self.primes):                  # 尾声：每个素数点亮一声高音（音高 ∝ log q）
            te = self.t_outro + 4 * self.beat * j / len(self.primes)
            i = int(te * sr)
            f = 440 * 2 ** (np.log2(q / 2) / 2)
            x[i:i + len(tt) * 3] += 0.10 * np.sin(2 * np.pi * f * np.arange(len(tt) * 3) / sr) * np.hanning(len(tt) * 3)
        x = np.tanh(x * 1.2) * 0.8
        return np.stack([x, x], 1)


# ================================================================ 输出
def ffmpeg_pipe(path, w, h):
    import imageio_ffmpeg
    return subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "15",
                             str(path)], stdin=subprocess.PIPE)


def contact(thumbs, path, cols=6, tw=400):
    th = int(tw * 9 / 16)
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * (th + 18)), (14, 14, 14))
    d = ImageDraw.Draw(sheet); f = ImageFont.truetype(FONT, 13)
    for i, (t, note, im) in enumerate(thumbs):
        x, y = (i % cols) * tw, (i // cols) * (th + 18)
        sheet.paste(im.resize((tw, th), Image.LANCZOS), (x, y + 18))
        d.text((x + 4, y + 2), f"{t:7.3f}s  {note}", font=f, fill=(170, 176, 180))
    sheet.save(path, quality=90)


def render(scene, t0, t1, out, ss=3, audio=True, tmp=None, sheet_n=24):
    import opening as O                           # mux（原曲 → 播放时间，补了 0.042 s 起点）
    w, h = scene.w, scene.h
    tmpd = Path(tmp) if tmp else Path(__import__("tempfile").gettempdir()) / "eidolon_geo"
    tmpd.mkdir(parents=True, exist_ok=True)
    silent = tmpd / (out.stem + "_无声.mp4")
    p = ffmpeg_pipe(silent if audio else out, w, h)
    nfr = int(round((t1 - t0) * FPS))
    pick = set(np.linspace(0, nfr - 1, sheet_n).astype(int).tolist())
    thumbs, side = [], []
    t_start = time.time()
    for f in range(nfr):
        t = t0 + f / FPS
        cv = Canvas(w, h, ss)
        info = scene.frame(t, cv)
        im = cv.result()
        p.stdin.write(np.asarray(im).tobytes())
        side.append(dict(f=f, t=round(t, 4), **info))
        if f in pick:
            note = info.get("beats_to_release", info.get("assimilated", info.get("p", "")))
            thumbs.append((t, f"{scene.name} {note}", im))
        if f % 24 == 0:
            print(f"  {f}/{nfr}  {time.time() - t_start:5.1f}s", flush=True)
    p.stdin.close(); p.wait()
    if audio and hasattr(scene, "audio"):          # 独立版：场景自己的声音（声化），不读原曲
        import wave
        import imageio_ffmpeg
        x = scene.audio()[int(t0 * 48000):int(t1 * 48000)]
        wav = silent.with_suffix(".wav")
        with wave.open(str(wav), "wb") as wf:
            wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(48000)
            wf.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(silent), "-i", str(wav),
                        "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-t", f"{t1 - t0:.3f}", str(out)], check=True)
    elif audio:
        O.mux(silent, out, t0, t1 - t0)
    contact(thumbs, out.with_name(out.stem + "_接触表.jpg"))
    return side


def build(kind, sn, a):
    if kind == "countdown":
        old = dict(wt=1.0, delay=0.0, spin=90.0, guide=False, trim=False) if a.ver in ("v1", "v2") else dict(trim=False) if a.ver in ("v3", "v4") else {}
        sc = Countdown(sn, t_release=a.release, scale=a.scale, **old)
        return sc
    st = {"v0": 1, "v1": 1, "v2": 2}.get(a.ver, 3)
    return Matrix(sn, a.start, a.end, scale=a.scale, v2=(a.ver != "v0"), style=st)   # v0 = 只出过静帧的初版版面；v1 / v2 / v3+ = style 1 / 2 / 3


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["countdown", "matrix", "sieve"])
    ap.add_argument("--start", type=float, default=None)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--release", type=float, default=198.76, help="countdown：冲顶时刻（吸附到最近的拍）")
    ap.add_argument("--ver", default="v1")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--ss", type=int, default=3)
    ap.add_argument("--still", type=float, default=None, help="只出这一时刻的一帧 png")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tmp", default=None)
    ap.add_argument("--tag", default="", help="文件名附加说明（如 _低清）")
    ap.add_argument("--palette", default="cold", choices=sorted(PALETTES), help="cold 冷灰白+红点 / bwr 黑白红 / acid 酸性荧光")
    a = ap.parse_args()
    set_palette(a.palette)
    if a.palette != "cold" and not a.tag:
        a.tag = {"bwr": "_黑白红", "acid": "_酸性"}[a.palette]
    if a.kind == "sieve":                                           # 独立版：不读感官.json，输出到 MotionDesign/
        sc = Sieve(scale=a.scale, v={"v1": 1, "v2": 2, "v3": 3}.get(a.ver, 4))
        a.start = 0.0 if a.start is None else a.start
        a.end = sc.T if a.end is None else a.end
        outd = Path(a.out) if a.out else MD_DIR
        outd.mkdir(parents=True, exist_ok=True)
        if a.still is not None:
            cv = Canvas(sc.w, sc.h, a.ss); info = sc.frame(a.still, cv)
            out = outd / f"01_几何符号_筛_{a.ver}_静帧_{a.still:.2f}{a.tag}.png"
            if out.exists():
                raise SystemExit(f"不覆盖：{out}")
            cv.result().save(out); print(out); print(info); return
        out = outd / f"01_几何符号_筛_{a.ver}{a.tag}.mp4"
        if out.exists():
            raise SystemExit(f"不覆盖：{out}")
        side = render(sc, a.start, a.end, out, a.ss, not a.no_audio, a.tmp)
        meta = dict(tool="tools/geo_symbols.py", kind="sieve", ver=a.ver, palette=a.palette, fps=FPS, size=[sc.w, sc.h],
                    params=dict(N=sc.N, bpm=sc.bpm, primes_p=sc.P, pass_starts=sc.t_pass, outro=sc.t_outro, T=sc.T,
                                pi_N=len(sc.primes), ease2=dict(tau1_frames=3, tau2_frames=9, a=0.6)),
                    events=[dict(t=round(te, 4), n=n, p=p) for te, n, p in sc.events], frames=side)
        Path(str(out.with_suffix("")) + ".json").write_text(json.dumps(meta, ensure_ascii=False, indent=0, default=float), "utf-8")
        print(out)
        return
    sn = sense()
    if a.kind == "countdown":
        beats = np.array(sn["events"]["beats"])
        rel = float(beats[np.argmin(np.abs(beats - a.release))])
        k = int(np.argmin(np.abs(beats - rel)))
        a.start = a.start if a.start is not None else float(beats[k - 17])     # 提前一拍：剪进来时已经在动
        a.end = a.end if a.end is not None else rel
    else:
        a.start = a.start if a.start is not None else 128.36
        a.end = a.end if a.end is not None else 141.14
    sc = build(a.kind, sn, a)
    if a.kind == "countdown":
        sc.spec = StepSpectrum(a.start, a.end)
    outd = Path(a.out) if a.out else OUT_DIR
    outd.mkdir(parents=True, exist_ok=True)
    label = {"countdown": "倒数", "matrix": "字阵吸收"}[a.kind]
    if a.still is not None:
        if a.kind == "countdown":                                 # 频谱要历史：从 0.5 s 前开始推
            for tt in np.arange(a.still - 0.5, a.still, 1 / FPS):
                sc.spec.step(tt)
        cv = Canvas(sc.w, sc.h, a.ss)
        info = sc.frame(a.still, cv)
        out = outd / f"21_几何符号_{label}_{a.ver}_静帧_{a.still:.2f}{a.tag}.png"
        if out.exists():
            raise SystemExit(f"不覆盖：{out}")
        cv.result().save(out)
        print(out); print(json.dumps(info, ensure_ascii=False, default=float)[:600])
        return
    out = outd / f"21_几何符号_{label}_{a.ver}{a.tag}.mp4"
    if out.exists():
        raise SystemExit(f"不覆盖：{out}")
    t_wall = time.time()
    side = render(sc, a.start, a.end, out, a.ss, not a.no_audio, a.tmp)
    meta = dict(tool="tools/geo_symbols.py", kind=a.kind, ver=a.ver, palette=a.palette, start=a.start, end=a.end, fps=FPS, size=[sc.w, sc.h],
                time="感官.json 播放时间（无偏移）", render_s=round(time.time() - t_wall, 1),
                params=dict(ease2=dict(tau1_frames=3, tau2_frames=9, a=0.6),
                            **({"spiral": dict(R=sc.sp.R, phi_deg=float(np.degrees(sc.sp.phi)), birth_beats=sc.beats[:-1].tolist(),
                                               release=sc.release), "spectrum": dict(bands=24, f=[40, 12000], db=[sc.spec.db_lo, sc.spec.db_hi], steps=8)}
                               if a.kind == "countdown" else
                               {"voices": dict(window=[77.11, 115.51], n=sc.N, rows=sc.rows, cols=sc.cols),
                                "absorb": [dict(onset=o, voice=j, voice_t=float(sc.voices[j]), dist=round(d, 4)) for o, j, d in sc.match]})),
                frames=side)
    Path(str(out.with_suffix("")) + ".json").write_text(json.dumps(meta, ensure_ascii=False, indent=0, default=float), "utf-8")
    print(out)


if __name__ == "__main__":
    main()
