#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🕸感官空间-v2-「开头」：同一个感官空间，换到前奏 38.71–51.51（02 有什么要来了 → 03 吸气，接在 15_开头_0-38.7 后面），
四类层各有自己的视觉语法（👤2026-10-23 对 v1 的反馈：前后层看起来没区别、镜头太静、结尾别冲进）。

层（世界坐标；X = 鱼眼时间，播放头 x = 0，未来不画；Y = 对数频率 30 Hz–10 kHz，高 700；Z = 深度，前 → 后）：
  z = −760  SENSE   感官：4 条通道曲线（θ 威胁最粗 / β 心跳 / η 饥饿虚线 / τ 张力细线），各占一条泳道
  z = −430  EXPECT  期待审计：顶部色带 = 情绪段（等什么 / 忍耐）；尖刺 = 每个事件的意外度（比特）；红 = 显著度 ≥ 阈值；
                    ▼ = 审计 v4 的分歧点（显著度高但剪辑没回应，读 感官_期待审计_v4.md）；细线 = absence
  z = −110  VOICE   人声：一条 f0 线（与频谱共用对数频率轴 → 落在 vocals 频谱的正前方）+ 乐句括号 + 音名格线 + 响度线
  z = 250 / 590 / 930 / 1270  四条 Demucs 分轨频谱（drums / other / bass / vocals）：密纹理（每层 48 条山脊，点溅射）
  远近：雾（亮度随相机深度衰减）+ 线宽随深度变细 + 字号前大后小。

⭐局部不安的抖动 = opening.py 的 Kuramoto 律原样搬来（👤点名的 15_开头素材_wf_kura_v3 @ 24.33–26.12 s）：
  每层 48 条山脊各是一个振子（w ~ Cauchy·0.5，限 ±3；dθ/dt = 2.2·(w + 0.6·β + K·R·sin(ψ − θ))，每帧 4 子步）；
  K 从 10.0 s 起 0.8、每拍 +0.035（与开头同一条 K 曲线 → 接得上）；每帧相位被踢 N(0, σ)：
  开头 σ = 0.35·θ(t) 对所有振子；这里 σ = 0.35·θ(t)·(该层该频带的频谱通量 / 全段 95 百分位)，即「只踢被打中的那几根」→ 局部。
  显示 = 有效频率（EMA 0.5 s）− 它自己 3 s 的慢均值（Hz）× 增益 → 山脊的竖向抖动；锁住的被耦合拉回，靠近锁相边界的久久不平。
镜头：持续环绕 / 漂移（yaw、pitch、dist、lookat 都是慢正弦，叠加不归零的前馈事件移动：提前 2 帧起步，1 − (1 − u)^3.5 快起慢收）；
  结尾不推近：最后一拍（49.78 人声停）之后镜头后退、继续漂。

⭐v3（默认，👤看 v2：「镜头杂乱无章、没有跟踪某个地方；层距太大；要在里面搜寻」）：
  层距压到 ~0.38（相邻 130），雾按近距离重标定；镜头 = 搜寻：目标 = 本段显著度最高的事件（人声起必选，相邻 ≥ 1.1 s），
  起音 → 那一刻通量最大的分轨 × 通量重心频带，人声起 → VOICE 层 f0；事件没到时目标在播放头（感官.json 的已知未来 = 前馈），
  到了随鱼眼往左滑、镜头跟着；找到后 0.55 s「读数」转去 EXPECT 层同一事件的竖刺（只移视线）；每次出发 = 区间里最强的起音（提前 2 帧）；
  准星：灰 = SEARCH x%，红 = FOUND 0.6 s。慢漂叠加不归零；最后后退不推近。v2 的通量 clip(−80, −10) 是 bug（v3 已改），--v 2 保留原样复现。
⭐v5（👤：正面、长焦、层更近；只平移 + 推拉）：camera_v5 —— yaw = pitch = 0、半视场 8°、层距 70；小节首拍一次移动（v4 A），--points 加每拍一点（v4 B）。
v4（A/B/C 节奏试听，--v 4 --mode A|B|C --lowres）见 camera_v4。
用法：python tools/sense_space_v2.py [--v 5|4|3|2] [--points] [--seg0 38.71 --seg1 51.51] [--name …] [--still 秒,秒 --out x.jpg]
输出：MV/样片/信息可视化/<name>.mp4（配乐）+ _接触表.jpg + _镜头速度.png + .json。不覆盖已有文件。
注：v1（tools/sense_space.py）依赖 info_graph；本版自带几个小工具函数，不 import info_graph（别的会话在改它）。
"""
import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import sensorium  # noqa: E402

PROJ = ROOT.parents[1] / "Projekt_Visual"
MV = PROJ / "MV_Cant_Kill_Us"
MUSIC = PROJ / "Assets" / "The Glitch Mob - Can't Kill Us [2020 Edit]_190189282.mp4"
SENSE = MV / "感官.json"
AUDIT_MD = MV / "感官_期待审计_v4.md"
EXPECT_JSON = MV / "情绪大图" / "期待.json"
OUT = MV / "样片" / "信息可视化"
CACHE = MV / "cache" / "info_graph"
W, H, FPS, SS = 1920, 1080, 24, 2
FMONO, FCN = "C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/msyh.ttc"
BG, INK, MID, DIM, FAINT, HOT = (4, 4, 5), (236, 236, 232), (150, 150, 148), (78, 78, 80), (38, 38, 40), (232, 36, 30)

STEMS = [("drums", 250.0), ("other", 590.0), ("bass", 930.0), ("vocals", 1270.0)]
Z_SENSE, Z_EXP, Z_VOICE = -760.0, -430.0, -110.0
X1, TAU, HH = 1700.0, 1.5, 700.0
NB, NR, NC = 96, 48, 300
RIDGE = 30.0
F_LO, F_HI = 30.0, 10000.0
FOCAL = (W / 2) / math.tan(math.radians(25))
SR, HOP = 22050, 0.02
# Kuramoto（opening.py 的律）
K0, K_T0, K_BEAT, KICK, HB_GAIN, V_GAIN, TAU_EFF, TAU_SLOW = 0.8, 10.0, 0.035, 0.35, 0.6, 2.2, 0.5, 3.0
WIG_GAIN, WIG_MAX = 40.0, 18.0          # 世界单位 / Hz；山脊间距 = 700 / 48 ≈ 14.6
LEAD, EASE_P = 2, 3.5
CH = ["heartbeat", "threat", "hunger", "tension"]
SYM = {"heartbeat": "β", "threat": "θ", "hunger": "η", "tension": "τ"}

_F = {}


def font(size, cn=False):
    k = (size, cn)
    if k not in _F:
        _F[k] = ImageFont.truetype(FCN if cn else FMONO, int(round(size * SS)))
    return _F[k]


def text(dr, xy, s, col, size=12, cn=False, anchor="la", knock=None):
    if knock is not None:
        x0, y0, x1, y1 = dr.textbbox(xy, s, font=font(size, cn), anchor=anchor)
        dr.rectangle([x0 - 2 * SS, y0 - 2 * SS, x1 + 2 * SS, y1 + 2 * SS], fill=knock)
    dr.text(xy, s, fill=col, font=font(size, cn), anchor=anchor)


def mix(c, a, bg=BG):
    a = float(min(1.0, max(0.0, a)))
    return tuple(int(round(bg[i] + (c[i] - bg[i]) * a)) for i in range(3))


def ease(u):
    u = min(1.0, max(0.0, u))
    return 1 - (1 - u) ** EASE_P


def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def fmt_t(t):
    return f"{int(t // 60)}:{t % 60:05.2f}"


def yf(f):
    return HH * math.log(max(f, 1e-3) / F_LO) / math.log(F_HI / F_LO)


# ───────────────────────── 数据 ─────────────────────────
def stem_bands(name, t_end):
    """分轨 0 → t_end 的对数频带功率（dB，帧 = 20 ms，帧时间 = 窗口中心）。与 v1 同一算法、同一缓存名。"""
    CACHE.mkdir(parents=True, exist_ok=True)
    cp = CACHE / f"space_{name}_{t_end:.2f}.npy"
    if cp.exists():
        return np.load(cp)
    x = sensorium.load_audio(MV / "分轨" / f"{name}.wav", 0.0, t_end, sr=SR)
    n, hop = 2048, int(HOP * SR)
    frames = (len(x) - n) // hop + 1
    win = np.hanning(n)
    fr = np.fft.rfftfreq(n, 1 / SR)
    edges = np.geomspace(F_LO, F_HI, NB + 1)
    bi = np.clip(np.searchsorted(edges, fr) - 1, -1, NB)
    out = np.zeros((frames + n // (2 * hop), NB))
    for a in range(0, frames, 512):
        idx = np.arange(a, min(frames, a + 512))
        seg = x[idx[:, None] * hop + np.arange(n)[None, :]] * win
        Pw = np.abs(np.fft.rfft(seg, axis=1)) ** 2
        for b in range(NB):
            m = bi == b
            if m.any():
                out[idx + n // (2 * hop), b] = Pw[:, m].sum(1)
    db = 10 * np.log10(out + 1e-12)
    np.save(cp, db)
    return db


def audit_points(t1):
    """感官_期待审计_v4.md 的 A / B 分歧点（只取 ≤ t1 的）"""
    out = []
    if not AUDIT_MD.exists():
        return out
    for line in AUDIT_MD.read_text("utf-8").splitlines():
        m = re.match(r"\|\s*\*\*([AB]\d+)\*\*\s*\|\s*(\d+):(\d+\.\d+)\s*\|\s*([^|]*)\|\s*([^|]*)\|", line)
        if m:
            t = int(m.group(2)) * 60 + float(m.group(3))
            if t <= t1:
                out.append(dict(id=m.group(1), t=t, bits=m.group(4).strip(), ev=m.group(5).strip()))
    return out


def fisheye_x(dt, t_now):
    return -X1 * np.log1p(np.maximum(dt, 0) / TAU) / math.log1p(max(t_now, 1e-3) / TAU)


def fisheye_inv(x, t_now):
    return TAU * ((1 + t_now / TAU) ** (-np.asarray(x) / X1) - 1)


# ───────────────────────── Kuramoto 抖动（opening.py 律，局部踢） ─────────────────────────
def _flux_range(dbs):
    if VER["v"] == 2:
        return -80.0, -10.0
    pool = np.concatenate([d[d > -100].ravel()[::7] for d in dbs])
    return float(np.percentile(pool, 60)), float(np.percentile(pool, 99.7))


def kura_tremor(sense, dbs, t1, seed=1):
    """返回 wig[(layer, frame, ridge)] 世界单位的竖向位移（frame = (t − K_T0)·FPS），以及每帧的 K、R、locked。"""
    fps = sense["fps"]
    th_ch = np.asarray(sense["channels"]["threat"], float)
    hb_ch = np.asarray(sense["channels"]["heartbeat"], float)
    beats = np.asarray(sense["events"]["beats"], float)
    nL = len(dbs)
    # 每层每条山脊的正频谱通量（dB 差，限 −80…−10）→ 24 fps
    flux = []
    for db in dbs:
        # ⚠️v2 用了 v1 的 clip(−80, −10)（以为是 dBFS），但这里的频带电平在 −30…+40 dB → 只有安静频带算得出通量。v3 改成「底噪（四轨 60 百分位）… 99.7 百分位」
        r = np.clip(db, *_flux_range(dbs)).reshape(len(db), NR, NB // NR).mean(2)
        fl = np.r_[np.zeros((1, NR)), np.clip(np.diff(r, axis=0), 0, None)]
        flux.append(fl)
    T = int(round((t1 - K_T0) * FPS)) + 2
    ff = np.zeros((nL, T, NR))
    for f in range(T):
        t = K_T0 + f / FPS
        i0, i1 = int((t - 1 / FPS) / HOP), int(t / HOP) + 1
        for k in range(nL):
            ff[k, f] = flux[k][max(0, i0):i1].max(0) if i1 > i0 else 0
    ref = float(np.percentile(ff[ff > 0], 95))
    rng = np.random.default_rng(seed)
    w = np.clip(rng.standard_cauchy((nL, NR)) * 0.5, -3, 3)
    thv = rng.uniform(0, 2 * np.pi, (nL, NR))
    krng = np.random.default_rng(seed + 100)
    K = K0
    eff = None
    slow = None
    kick = np.zeros((nL, NR))
    wig = np.zeros((nL, T, NR))
    info = np.zeros((T, 3 * nL + 1))
    dt = 1 / (FPS * 4)
    for f in range(T):
        t = K_T0 + f / FPS
        K += int(((beats > t - 1 / FPS) & (beats <= t)).sum()) * K_BEAT
        hb = hb_ch[min(int(t * fps), len(hb_ch) - 1)]
        th = th_ch[min(int(t * fps), len(th_ch) - 1)]
        for j in range(4):
            z = np.exp(1j * thv).mean(1, keepdims=True)
            v = V_GAIN * (w + HB_GAIN * hb + K * np.abs(z) * np.sin(np.angle(z) - thv))
            thv = thv + v * dt
            dth = v * dt + (kick if j == 0 else 0)
            eff = dth / dt if eff is None else eff + (dth / dt - eff) * dt / TAU_EFF
            slow = eff.copy() if slow is None else slow + (eff - slow) * dt / TAU_SLOW
        kick = krng.normal(0, 1, (nL, NR)) * KICK * th * np.clip(ff[:, f] / ref, 0, 1)
        thv = thv + kick
        wig[:, f] = np.clip((eff - slow) / (2 * np.pi) * WIG_GAIN, -WIG_MAX, WIG_MAX)
        z = np.exp(1j * thv).mean(1)
        R = np.abs(z)
        lock = (np.abs(w) < K * R[:, None]).sum(1)
        info[f] = np.r_[K, R, lock, np.zeros(nL)]
    return wig, info, ref


# ───────────────────────── 相机 ─────────────────────────
def cam_basis(st):
    la = st[:3]; yaw, pitch, dist = st[3], st[4], st[5]
    d = np.array([math.cos(pitch) * math.sin(yaw), math.sin(pitch), -math.cos(pitch) * math.cos(yaw)])
    pos = la + dist * d
    f = (la - pos) / np.linalg.norm(la - pos)
    r = np.cross([0, 1, 0], f); r /= np.linalg.norm(r)
    u = np.cross(f, r)
    return pos, f, r, u


def project(Pw, st):
    pos, f, r, u = cam_basis(st)
    q = np.atleast_2d(Pw) - pos
    z = q @ f
    zc = np.maximum(z, 1.0)
    return np.stack([W / 2 + FOCAL * (q @ r) / zc, H / 2 - FOCAL * (q @ u) / zc], -1), z


def drift(t):
    """持续漂移 / 环绕（不归零、不停）：yaw 摆幅 ±0.34 rad，pitch ±0.11，dist ±300，lookat 沿时间轴往播放头滑"""
    return np.array([-820 + 34 * t + 120 * math.sin(0.41 * t + 0.6),
                     300 + 70 * math.sin(0.29 * t + 2.0),
                     240 + 200 * math.sin(0.23 * t + 0.3),
                     0.58 + 0.34 * math.sin(2 * math.pi * t / 15.0 + 2.6),
                     0.13 + 0.11 * math.sin(2 * math.pi * t / 11.0 + 0.9),
                     3400 + 300 * math.sin(2 * math.pi * t / 13.0 + 4.2)])


def camera(seg, T, sense, voice_pts):
    """前馈事件（时刻 = 播放时间 − 段首；提前 LEAD 帧起步）叠加在漂移上，偏移不归零 → 镜头一直在走"""
    s0, s1 = seg
    on = np.asarray(sense["events"]["onsets"], float)
    th = np.asarray(sense["channels"]["threat"], float); fps = sense["fps"]
    st = np.array([th[max(0, int(t * fps) - 1):int(t * fps) + 4].max() for t in on])
    m = (on > s0 + 0.3) & (on < s1 - 0.3)
    thr = np.percentile(st[m], 75)
    ev = []
    last = -9
    for t, s in zip(on[m], st[m]):
        if s >= thr and t - last >= 0.75:
            ev.append((t - s0, "onset", s)); last = t
    moves = []
    sgn = 1
    for t, k, s in ev:                                   # 强起音：横向一甩 + yaw 偏一点，幅度 ∝ 强度
        sgn = -sgn
        moves.append((t, np.array([sgn * 90 * s, 30 * s, sgn * 110 * s, sgn * 0.07 * s, -sgn * 0.025 * s, 0.0]), 0.9, f"onset {t + s0:.2f}"))
    moves.append((45.11 - s0, np.array([130.0, 20.0, -120.0, -0.15, 0.03, -100.0]), 1.5, "03 吸气 (segment)"))
    for t, y in voice_pts:                               # 人声起：转向人声层的 f0 点（不推近）
        moves.append((t - s0, np.array([110.0, (y - 350) * 0.3, -90.0, 0.11, 0.0, -80.0]), 1.2, f"voice_on {t:.2f}"))
    moves.append((49.78 - s0, np.array([-90.0, 20.0, 140.0, -0.12, -0.04, 380.0]), 1.6, "voice_off 49.78 → pull back"))
    moves = [mv for mv in moves if 0 <= mv[0] < (s1 - s0)]
    moves.sort(key=lambda m_: m_[0])
    lead = LEAD / FPS
    out = np.zeros((T, 6))
    for f in range(T):
        t = f / FPS
        x = drift(t)
        for te, d, D, _ in moves:
            x = x + d * ease((t - (te - lead)) / D)
        out[f] = x
    return out, [(te, k) for te, _, _, k in moves]


def screen_speed(cams, refs):
    sp = np.zeros(len(cams))
    prev = None
    for f, st in enumerate(cams):
        q, z = project(refs, st)
        ok = z > 50
        if prev is not None:
            mm = ok & prev[1]
            sp[f] = np.median(np.hypot(*(q[mm] - prev[0][mm]).T)) if mm.any() else 0
        prev = (q, ok)
    return sp


def alignment(sp, events, T):
    """每个事件：事件帧 −LEAD−1 … +4 内速度峰值相对事件帧的偏移（期望 −2 ~ −1：提前起步）"""
    out = []
    for t, k in events:
        fe = int(round(t * FPS))
        a, b = max(1, fe - LEAD - 1), min(T - 1, fe + 4)
        if b < a:
            continue
        out.append((t, k, a + int(np.argmax(sp[a:b + 1])) - fe))
    return out


def speed_plot(sp, events, al, path, s0, title):
    T = len(sp)
    Wp, Hp, m = 1800, 360, 50
    im = Image.new("RGB", (Wp, Hp), (12, 12, 12))
    d = ImageDraw.Draw(im)
    fnt = ImageFont.truetype(FMONO, 14); fcn = ImageFont.truetype(FCN, 14)
    mx = sp.max() + 1e-9
    X = lambda f: m + f / (T - 1) * (Wp - 2 * m)        # noqa: E731
    Y = lambda v: Hp - m - v / mx * (Hp - 2 * m)        # noqa: E731
    for t, k, dd in al:
        d.line([(X(t * FPS), m), (X(t * FPS), Hp - m)], fill=(200, 40, 40) if dd > 0 else (70, 70, 70))
    d.line([(X(f), Y(v)) for f, v in enumerate(sp)], fill=(230, 230, 230), width=2)
    d.line([(m, Y(0)), (Wp - m, Y(0))], fill=(50, 50, 50))
    d.text((m, 12), title, fill=(200, 200, 200), font=fcn)
    d.text((m + 20 + d.textlength(title, font=fcn), 12),
           f"camera screen speed (median px/frame of 112 refs, max {mx:.1f}, min {sp[1:].min():.2f})   grey = event (red = peak after it)", fill=(200, 200, 200), font=fnt)
    for sec in range(int(math.ceil(s0)), int(s0 + T / FPS) + 1):
        x = X((sec - s0) * FPS)
        d.line([(x, Hp - m), (x, Hp - m + 5)], fill=(140, 140, 140))
        d.text((x - 10, Hp - m + 8), f"{sec}", fill=(140, 140, 140), font=fnt)
    im.save(path)


def splat(x, y, w):
    x0 = np.floor(x); y0 = np.floor(y); fx = x - x0; fy = y - y0
    out = np.zeros(H * W)
    for dx, dy, ww in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi = (x0 + dx).astype(np.int64); yi = (y0 + dy).astype(np.int64)
        ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        out += np.bincount(yi[ok] * W + xi[ok], (ww * w)[ok], minlength=H * W)
    return out.reshape(H, W)


FOG = dict(z0=2300.0, L=1500.0)
VER = {"v": 2}


def fog(z):
    """相机深度 → 亮度（雾）：近 1，远处衰减到 0.3"""
    return float(np.clip(np.exp(-max(0.0, z - FOG["z0"]) / FOG["L"]), 0.3, 1.0))


# ───────────────────────── v3：在里面搜寻（👤2026-10-23 看 v2：镜头漫无目的、层距太大） ─────────────────────────
def set_v3():
    """层距压到 v2 的 ~0.38（相邻 130 世界单位），雾按近距离相机重标定"""
    global STEMS, Z_SENSE, Z_EXP, Z_VOICE
    STEMS = [("drums", 100.0), ("other", 230.0), ("bass", 360.0), ("vocals", 490.0)]
    Z_SENSE, Z_EXP, Z_VOICE = -300.0, -170.0, -40.0
    FOG.update(z0=1300.0, L=1000.0)
    VER["v"] = 3


def search_targets(sense, dbs, seg, gap=1.1, n_max=9, min_key=1.0):
    """每个时刻「最值得看」的那一处 = 本段里显著度（无则意外度）最高的事件，人声起必选；相邻 ≥ gap 秒。
    位置：起音 / 动机 → 那一刻正频谱通量最大的分轨层 × 频带；人声起 → VOICE 层该乐句 f0 中位数；absence → EXPECT 层的竖刺顶。
    用感官.json 的已知未来做前馈：事件到来之前，目标就在播放头（x = 0）它将出现的地方。"""
    s0, s1 = seg
    evs = [e for e in sense["expect"]["events"] if s0 + 0.4 <= e["t"] <= s1 - 0.4]
    key = lambda e: (e.get("sal") if e.get("sal") is not None else e["bits"])   # noqa: E731
    pick = [e for e in evs if e["s"] == "voice_on"]
    for e in sorted(evs, key=lambda e: -key(e)):
        if len(pick) >= n_max:
            break
        if e["s"] in ("voice_off",) or key(e) < min_key or any(abs(e["t"] - p["t"]) < gap for p in pick):
            continue
        pick.append(e)
    pick.sort(key=lambda e: e["t"])
    phr = {round(p["start"], 2): p for p in sense["voice"]["phrases"]}
    ys_r = (np.arange(NR) + 0.5) / NR * HH
    out = []
    for e in pick:
        if e["s"] == "voice_on":
            p = phr.get(round(e["t"], 2))
            fz = p["f0_med"] if p else 440.0
            out.append(dict(t=e["t"], y=yf(fz), z=Z_VOICE, kind="voice",
                            lab=f"VOICE ON  {e['bits']:.1f} bit · sal {key(e):.2f} · f0 {fz:.0f} Hz"))
        elif e["s"] == "absence":
            out.append(dict(t=e["t"], y=60 + 34 * min(e["bits"], 12), z=Z_EXP, kind="absence",
                            lab=f"ABSENCE  {e['bits']:.1f} bit (expected onset missing)"))
        else:
            i = int(round(e["t"] / HOP))
            best = (-1, 0, 0)
            for k, db in enumerate(dbs):
                r = np.clip(db[max(0, i - 3):i + 4], *_flux_range(dbs)).reshape(-1, NR, NB // NR).mean(2)
                fl = np.clip(np.diff(r, axis=0), 0, None).sum(0)
                if fl.sum() > best[0]:
                    best = (fl.sum(), k, float((fl * ys_r).sum() / max(fl.sum(), 1e-9)))     # 被打中的分轨 × 通量重心频带
            _, k, yb = best
            fc = F_LO * (F_HI / F_LO) ** (yb / HH)
            out.append(dict(t=e["t"], y=yb + RIDGE * 0.5, z=STEMS[k][1], kind=e["s"],
                            lab=f"{e['s'].upper()}  {e['bits']:.1f} bit · sal {key(e):.2f} · {STEMS[k][0]} {fc:.0f} Hz"))
        out[-1].update(bits=e["bits"], sal=key(e))
    # 找到之后「读数」：0.55 s 后转去 EXPECT 层看这一下有多意外（同一事件的竖刺顶；人声起 → 审计 ▼）—— 只在离下一个目标 ≥ 1.4 s 时
    reads = []
    for i, tg in enumerate(out):
        nxt = out[i + 1]["t"] if i + 1 < len(out) else s1
        if tg["kind"] != "absence" and nxt - tg["t"] >= 1.4:
            y = 60 + 34 * min(tg["bits"], 12) + (26 if tg["kind"] == "voice" else 0)
            reads.append(dict(t=tg["t"] + 0.55, tx=tg["t"], y=y, z=Z_EXP, kind="readout",
                              lab=f"EXPECT  {tg['bits']:.1f} bit · sal {tg['sal']:.2f}" + ("  ▼ no cut answered (audit)" if tg["kind"] == "voice" else "")))
    return sorted(out + reads, key=lambda d: d["t"])


def target_pos(tg, tn):
    """目标此刻在空间里的位置：没到 → 播放头（将要出现的地方）；到了 → 随鱼眼时间往左滑"""
    tx = tg.get("tx", tg["t"])
    x = fisheye_x(tn - tx, tn) if tn >= tx else 0.0
    return np.array([float(x), tg["y"], tg["z"]])


def camera_v3(seg, T, sense, tgs):
    """搜寻：每个目标一次移动 —— 出发 = 上一个目标停留（≥ 0.5 s）之后、本目标前 ≥ 0.45 s 内最强的起音（提前 LEAD 帧起步，
    1 − (1 − u)^3.5：起步最快、到目标时慢慢停住），到达 ≈ 事件时刻；lookat 追的是目标的实时位置（停留时跟着它往左滑）。
    每次换一个斜角（yaw 交替 0.42 / 1.02，贴着层面掠过）、距离 1250–1700。慢漂叠在上面、不归零：yaw 每秒 +0.03，lookat / dist 小正弦。
    最后一个目标停留后：后退 + 继续漂（不推近）。"""
    s0, s1 = seg
    on = np.asarray(sense["events"]["onsets"], float)
    th = np.asarray(sense["channels"]["threat"], float); fps = sense["fps"]
    ost = np.array([th[max(0, int(t * fps) - 1):int(t * fps) + 4].max() for t in on])
    lead = LEAD / FPS
    yaws, pitches = (0.55, 0.82, 0.64, 0.90), (0.10, 0.18, 0.07, 0.15)
    mv = []
    prev_a = -1.0
    for k, tg in enumerate(tgs):
        a = tg["t"] - s0
        lo_, hi_ = max(prev_a + 0.5, 0.0), a - 0.6
        m = (on - s0 >= lo_) & (on - s0 <= hi_)
        d = float(on[m][np.argmax(ost[m])] - s0) if m.any() else max(lo_, a - 1.2)
        if k == 0:
            d = min(d, max(0.0, a - 1.2))
        if tg["kind"] == "readout":
            d = prev_a + 0.2
        d = max(min(d, a - 0.45), prev_a + 0.15, 0.0)
        dist = 1300.0 if tg["kind"] == "voice" else 1500.0 + 200 * (k % 2)
        mv.append(dict(d=d, a=a, tg=tg, yaw=yaws[k % 4], pitch=pitches[k % 4], dist=dist, kind=f"search→{tg['kind']} {tg['t']:.2f}"))
        prev_a = a
    end_t = s1 - s0
    rel = max(prev_a + 0.7, end_t - 1.6) if mv else 0.0
    st0 = np.array([-500.0, 360.0, 60.0, 0.75, 0.16, 2100.0])
    out = np.zeros((T, 6)); info = []
    cur_la, cur_o = st0[:3].copy(), st0[3:].copy()
    k = -1
    start_la, start_o = cur_la.copy(), cur_o.copy()
    released = False
    for f in range(T):
        t = f / FPS
        tn = s0 + t
        while k + 1 < len(mv) and t >= mv[k + 1]["d"] - lead:
            k += 1
            start_la, start_o = cur_la.copy(), cur_o.copy()
        if not released and mv and t >= rel - lead and k == len(mv) - 1:
            released = True
            rel_la, rel_o = cur_la.copy(), cur_o.copy()
        if released:
            u = (t - (rel - lead)) / 1.8
            cur_la = rel_la + (np.array([-260.0, 40.0, -120.0]) ) * ease(u)
            cur_o = rel_o + np.array([-0.15, 0.06, 650.0]) * ease(u)
            info.append(None)
        elif k >= 0:
            m_ = mv[k]
            D = m_["a"] - m_["d"] + 0.3 + lead
            u = (t - (m_["d"] - lead)) / D
            e = ease(u)
            P = target_pos(m_["tg"], tn) + np.array([-240.0, 0.0, 0.0])   # 视线落在目标左边一点：目标在画面偏右，左边是刚听到的（右边是未来，不画）
            cur_la = start_la + (P - start_la) * e
            if m_["tg"]["kind"] != "readout":                    # 读数：只移视线，不换机位（不甩）
                cur_o = start_o + (np.array([m_["yaw"], m_["pitch"], m_["dist"]]) - start_o) * e
            info.append(dict(k=k, u=float(min(u, 1.0)), found=t >= m_["a"], since=t - m_["a"], tg=m_["tg"]))
        else:
            info.append(None)
        dr = np.array([22 * math.sin(0.5 * t), 16 * math.sin(0.37 * t + 1), 30 * math.sin(0.43 * t + 2),
                       0.03 * t + 0.04 * math.sin(0.7 * t), 0.02 * math.sin(0.55 * t), 70 * math.sin(0.3 * t)])
        out[f] = np.r_[cur_la, cur_o] + dr
    evs = [(m_["d"], m_["kind"]) for m_ in mv] + ([(rel, "release / pull back")] if mv else [])
    return out, evs, info


def camera_v4(seg, T, sense, tgs, mode):
    """v4 A/B/C 节奏试验（👤看 v3：「镜头运动和节奏还是对不上」）。诊断 v3：11 次出发里 8 次落在拍内（拍相位 0.2–0.7）——
    「读数」按 +0.55 s 计时出发、出发吸附到切分起音（41.25 / 44.45 是八分音符的「&」），移动间隔 0.85–2.5 s 不成脉冲。
    v4 一律去掉「读数」，出发只落在拍网格上（感官.json beats / downbeats，提前 LEAD 帧）：
      A = 只在小节首拍做一次明确的移动（D 0.9 s），拍内只有慢漂；
      B = A + 其余每拍一个「点」（推近 + yaw 小偏，2 帧起、τ 0.12 s 收回，幅度 ∝ 该拍的威胁峰）；
      C = 移动跟乐句走（每 2 小节 = 半个乐句一次，D 2.4 s），拍点不动镜头，只作用在准星（每拍一跳）和层上的局部抖动。
    目标 = 这次移动之后一个周期里显著度最高的目标（没有 → 下一个）；lookat 追它的实时位置（偏左 240）。"""
    s0, s1 = seg
    beats = np.asarray(sense["events"]["beats"], float)
    dbt = np.asarray(sense["events"]["downbeats"], float)
    th = np.asarray(sense["channels"]["threat"], float); fps = sense["fps"]
    tgs = [t for t in tgs if t["kind"] != "readout"]
    if mode == "C":
        bar = float(np.median(np.diff(dbt)))
        starts = [t for t in dbt if s0 - 1e-3 <= t < s1 - 0.5]
        ph = [t for t in sense["events"]["phrases"] if s0 - bar * 2 < t < s1]
        starts = sorted({round(t, 3) for t in starts if any(abs((t - p) / bar - 2 * round((t - p) / bar / 2)) < 0.1 for p in ph)})
        period, D = 2 * bar, 2.4
    else:
        starts = [float(t) for t in dbt if s0 - 1e-3 <= t < s1 - 0.5]
        period, D = float(np.median(np.diff(dbt))), 0.9
    lead = LEAD / FPS
    yaws, pitches, dists = (0.60, 0.74, 0.66, 0.80), (0.11, 0.16, 0.09, 0.14), (1550.0, 1450.0, 1600.0, 1500.0)   # v3 的甩（51 px/帧）收小
    mv = []
    for k, t0 in enumerate(starts):
        win = [g for g in tgs if t0 <= g["t"] < t0 + period]
        nxt = [g for g in tgs if g["t"] >= t0]
        g = max(win, key=lambda g: g.get("sal", 0) + (10 if g["kind"] == "voice" else 0)) if win else (nxt[0] if nxt else tgs[-1])
        mv.append(dict(d=t0 - s0, tg=g, o=np.array([yaws[k % 4], pitches[k % 4], dists[k % 4]]), kind=f"{mode} move {t0:.2f} → {g['kind']} {g['t']:.2f}"))
    pts = []
    if mode == "B":
        for b in beats:
            if s0 <= b < s1 and np.min(np.abs(dbt - b)) > 0.05:
                s_ = float(th[max(0, int(b * fps) - 1):int(b * fps) + 4].max())
                pts.append((b - s0, s_))
    out = np.zeros((T, 6)); info = []
    g0 = mv[0]["tg"] if mv else tgs[0]                     # 开场就看着第一个目标附近（不让第一次移动变成一次大甩）
    cur_la = target_pos(g0, s0) + np.array([-420.0, 40.0, -60.0])
    cur_o = (mv[0]["o"] if mv else np.array([0.6, 0.12, 1500.0])) + np.array([-0.10, 0.03, 150.0])
    k = -1
    start_la, start_o = cur_la.copy(), cur_o.copy()
    for f in range(T):
        t = f / FPS
        tn = s0 + t
        while k + 1 < len(mv) and t >= mv[k + 1]["d"] - lead:
            k += 1
            start_la, start_o = cur_la.copy(), cur_o.copy()
        if k >= 0:
            m_ = mv[k]
            u = (t - (m_["d"] - lead)) / D
            e = ease(u)
            P = target_pos(m_["tg"], tn) + np.array([-240.0, 0.0, 0.0])
            cur_la = start_la + (P - start_la) * e
            cur_o = start_o + (m_["o"] - start_o) * e
            info.append(dict(k=k, u=float(min(u, 1.0)), found=tn >= m_["tg"]["t"], since=tn - m_["tg"]["t"], tg=m_["tg"]))
        else:
            info.append(None)
        dr = np.array([22 * math.sin(0.5 * t), 16 * math.sin(0.37 * t + 1), 30 * math.sin(0.43 * t + 2),
                       0.03 * t + 0.04 * math.sin(0.7 * t), 0.02 * math.sin(0.55 * t), 70 * math.sin(0.3 * t)])
        kick = np.zeros(6)
        for i, (tb, s_) in enumerate(pts):                  # B：每拍一点（快起快收，回到原处）
            dt_ = t - (tb - lead)
            if 0 <= dt_ < 0.8:
                a_ = min(1.0, dt_ / (2 / FPS)) * math.exp(-max(0.0, dt_ - 2 / FPS) / 0.12) * s_
                kick += a_ * np.array([0.0, 0.0, 0.0, (1 if i % 2 else -1) * 0.025, 0.0, -110.0])
        out[f] = np.r_[cur_la, cur_o] + dr + kick
    evs = [(m_["d"], m_["kind"]) for m_ in mv] + [(tb, "beat point") for tb, _ in pts]
    return out, sorted(evs), info



# ───────────────────────── v5：正面长焦（👤2026-10-23：镜头在正面、长焦、层更近；只平移 + 推拉；不环绕 / 转向 / roll） ─────────────────────────
V5 = dict(half_fov=8.0, spacing=70.0, fogL=380.0, points=False)


def set_v5(points=False):
    """层距 70（v3 的 ~0.54、v2 的 ~0.2）；半视场 8°（≈ 135 mm 等效）；雾 / 线宽 / 字号都按「离最近一层多深」算（长焦下绝对深度几乎一样）"""
    global STEMS, Z_SENSE, Z_EXP, Z_VOICE, FOCAL
    sp = V5["spacing"]
    Z_SENSE, Z_EXP, Z_VOICE = -3 * sp, -2 * sp, -sp
    STEMS = [("drums", 0.0), ("other", sp), ("bass", 2 * sp), ("vocals", 3 * sp)]
    FOCAL = (W / 2) / math.tan(math.radians(V5["half_fov"]))
    FOG.update(z0=0.0, L=V5["fogL"])
    VER.update(v=5, mode="B" if points else "A")
    V5["points"] = points


def dscale(z):
    """线宽 / 点重 / 字号的深度系数。v5：相对最近一层（FOG z0）的深度 → 1 … 0.45；其余版本：旧的绝对深度规则"""
    if VER["v"] == 5:
        return 0.45 + 0.55 * np.exp(-np.maximum(np.asarray(z, float) - FOG["z0"], 0) / FOG["L"])
    return None


def camera_v5(seg, T, sense, tgs, points=False):
    """正面（yaw = pitch = roll = 0），只有 lookat 的 x / y 平移 + dist 推拉。
    节奏 = v4 A：小节首拍（downbeats）一次明确的移动（提前 LEAD 帧起步，1 − (1 − u)^3.5，D 1.2 s），目标 = 这一小节里显著度最高的目标
    （人声优先；没有 → 下一个），画面上目标落在中线偏右（lookat x = 目标 − 0.22 × 画宽）；推拉按目标类型：人声近、起音中、空小节远。
    拍内慢漂：x 每秒 +22 世界单位 + 小正弦，y / dist 小正弦（不归零）。points = v4 B 的「每拍一点」：非首拍的拍上推近 2.5 %、τ 0.12 s 收回，幅度 ∝ 威胁峰。
    最后一个小节首拍之后只往后拉（不推近）。"""
    s0, s1 = seg
    beats = np.asarray(sense["events"]["beats"], float)
    dbt = np.asarray(sense["events"]["downbeats"], float)
    th = np.asarray(sense["channels"]["threat"], float); fps = sense["fps"]
    tgs = [t for t in tgs if t["kind"] != "readout"]
    bar = float(np.median(np.diff(dbt)))
    starts = [float(t) for t in dbt if s0 - 1e-3 <= t < s1 - 0.5]
    lead = LEAD / FPS
    D = 1.2
    tanh = math.tan(math.radians(V5["half_fov"]))
    def dist_for(width):                                   # 画面宽（世界单位）→ 距离
        return width / 2 / tanh
    mv = []
    for k, t0 in enumerate(starts):
        win = [g for g in tgs if t0 <= g["t"] < t0 + bar]
        nxt = [g for g in tgs if g["t"] >= t0]
        if win:
            g = max(win, key=lambda g: g.get("sal", 0) + (10 if g["kind"] == "voice" else 0))
            width = 1250.0 if g["kind"] == "voice" else 1500.0
        else:
            g = nxt[0] if nxt else tgs[-1]
            width = 1850.0
        mv.append(dict(d=t0 - s0, tg=g, width=width, kind=f"bar {t0:.2f} → {g['kind']} {g['t']:.2f}"))
    if mv:                                                 # 结尾不推近：最后一次只拉远
        mv[-1]["width"] = max(mv[-1]["width"], 2100.0, (mv[-2]["width"] if len(mv) > 1 else 0) + 300)
    pts = []
    if points:
        for b in beats:
            if s0 <= b < s1 and np.min(np.abs(dbt - b)) > 0.05:
                pts.append((b - s0, float(th[max(0, int(b * fps) - 1):int(b * fps) + 4].max())))
    out = np.zeros((T, 6)); info = []
    g0 = mv[0]["tg"] if mv else tgs[0]
    w0 = 1900.0
    cur = np.array([target_pos(g0, s0)[0] - 0.22 * w0 - 150, target_pos(g0, s0)[1] + 40, dist_for(w0)])
    k = -1
    start = cur.copy()
    for f in range(T):
        t = f / FPS
        tn = s0 + t
        while k + 1 < len(mv) and t >= mv[k + 1]["d"] - lead:
            k += 1
            start = cur.copy()
        if k >= 0:
            m_ = mv[k]
            u = (t - (m_["d"] - lead)) / D
            e = ease(u)
            P = target_pos(m_["tg"], tn)
            goal = np.array([P[0] - 0.22 * m_["width"], P[1], dist_for(m_["width"])])
            cur = start + (goal - start) * e
            info.append(dict(k=k, u=float(min(u, 1.0)), found=tn >= m_["tg"]["t"], since=tn - m_["tg"]["t"], tg=m_["tg"]))
        else:
            info.append(None)
        x, y, dd = cur
        x += 22 * t + 18 * math.sin(0.5 * t)
        y += 14 * math.sin(0.37 * t + 1)
        dd *= 1 + 0.03 * math.sin(0.3 * t)
        for tb, s_ in pts:
            dt_ = t - (tb - lead)
            if 0 <= dt_ < 0.8:
                dd *= 1 - 0.025 * s_ * min(1.0, dt_ / (2 / FPS)) * math.exp(-max(0.0, dt_ - 2 / FPS) / 0.12)
        out[f] = [x, y, 0.0, 0.0, 0.0, dd]
    evs = [(m_["d"], m_["kind"]) for m_ in mv] + [(tb, "beat point") for tb, _ in pts]
    return out, sorted(evs), info


def draw_reticle(dr, st, inf, tn, pulse=0.0):
    """搜寻准星：四角括号框住目标的实时位置；搜的时候灰、找到的 0.6 s 红、之后白；标签 = 目标的真数据"""
    if inf is None:
        return
    tg = inf["tg"]
    q, z = project(target_pos(tg, tn), st)
    if z[0] < 60:
        return
    x, y = q[0, 0] * SS, q[0, 1] * SS
    found, since = inf["found"], inf["since"]
    if not found:
        r = (26 + 40 * (1 - inf["u"])) * SS
        col = MID
    else:
        r = 22 * SS
        col = HOT if since < 0.6 else INK
    r *= 1 + 0.6 * pulse
    if pulse > 0.5 and col == MID:
        col = INK
    L = r * 0.45
    for sx in (-1, 1):
        for sy in (-1, 1):
            cx, cy = x + sx * r, y + sy * r
            dr.line([(cx, cy), (cx - sx * L, cy)], fill=col, width=SS)
            dr.line([(cx, cy), (cx, cy - sy * L)], fill=col, width=SS)
    tag = ("FOUND  " if found else f"SEARCH {inf['u'] * 100:3.0f}%  ") + tg["lab"]
    text(dr, (x + r + 8 * SS, y - r), tag, col, 13, knock=BG)
    text(dr, (x + r + 8 * SS, y - r + 18 * SS), f"t {fmt_t(tg['t'])}   z {tg['z']:.0f}", DIM if not found else MID, 11, knock=BG)


# ───────────────────────── 渲染 ─────────────────────────
def render(seg, only=None):
    s0, s1 = seg
    T = int(round((s1 - s0) * FPS))
    sense = json.loads(SENSE.read_text("utf-8"))
    fps_s = sense["fps"]
    ch = {k: np.asarray(sense["channels"][k], float) for k in CH}
    ex = sense["expect"]
    evs = [e for e in ex["events"] if e["t"] <= s1]
    sal_all = np.array([e.get("sal") or 0 for e in ex["events"]])
    sal_thr = max(2.0, float(np.percentile(sal_all, 92)))
    absence = np.asarray(ex["absence"], float)
    segs = [s for s in ex["segments"] if s[0] < s1]
    exp_meta = {d["id"]: d for d in json.loads(EXPECT_JSON.read_text("utf-8"))["段"]}
    auds = audit_points(s1)
    vo = sense["voice"]
    f0 = np.array([np.nan if x is None else x for x in vo["f0"]], float)
    loud = np.asarray(vo["loudness"], float)
    phr = [p for p in vo["phrases"] if p["start"] <= s1]
    voice_pts = [(p["start"], yf(p["f0_med"])) for p in phr if p["start"] >= s0]
    dbs = [stem_bands(n, 52.0) for n, _ in STEMS]
    pool = np.concatenate([d[d > -100].ravel()[::7] for d in dbs])
    lo, hi = float(np.percentile(pool, 50)), float(np.percentile(pool, 99.7))
    css = [np.vstack([np.zeros(NB), np.cumsum(10 ** (db / 10), axis=0)]) for db in dbs]
    wig, kinfo, kref = kura_tremor(sense, dbs, s1)
    nL = len(STEMS)
    if VER["v"] == 5:
        tgs = search_targets(sense, dbs, seg)
        cams, cam_ev, cinfo = camera_v5(seg, T, sense, tgs, V5["points"])
    elif VER["v"] == 4:
        tgs = search_targets(sense, dbs, seg)
        cams, cam_ev, cinfo = camera_v4(seg, T, sense, tgs, VER["mode"])
    elif VER["v"] == 3:
        tgs = search_targets(sense, dbs, seg)
        cams, cam_ev, cinfo = camera_v3(seg, T, sense, tgs)
    else:
        cams, cam_ev = camera(seg, T, sense, voice_pts)
        cinfo = [None] * T
    refs = np.array([[x, y, z] for x in np.linspace(-X1, 0, 7) for y in (0, HH)
                     for z in (Z_SENSE, Z_EXP, Z_VOICE) + tuple(z_ for _, z_ in STEMS) + (500.0,)])
    ys_r = (np.arange(NR) + 0.5) / NR * HH
    xf = np.linspace(-X1, 0, NC * 3)
    xs_cols = np.linspace(-X1, 0, NC)
    meta = []
    for f in range(T):
        if only is not None and f not in only:
            continue
        t = f / FPS
        tn = s0 + t
        st = cams[f]
        if VER["v"] == 5:
            FOG["z0"] = st[5] + Z_SENSE - st[2]          # 最近一层（SENSE）离相机的深度
        kf = min(int(round((tn - K_T0) * FPS)), wig.shape[1] - 1)
        # ── 分轨频谱：密纹理（48 条山脊，点溅射）+ Kuramoto 局部抖动
        acc = np.zeros((H, W)); acc_r = np.zeros((H, W))
        tt = tn - fisheye_inv(xs_cols, tn)
        ttf = tn - fisheye_inv(xf, tn)
        hidx = np.clip(np.round((ttf - K_T0) * FPS).astype(int), 0, kf)
        for k, (name, zk) in enumerate(STEMS):
            db = dbs[k]
            edges_t = np.concatenate([[tt[0] + (tt[0] - tt[1]) / 2], (tt[1:] + tt[:-1]) / 2, [tn]])
            ia = np.clip((np.minimum(edges_t[:-1], edges_t[1:]) / HOP).astype(int), 0, len(db) - 1)
            ib = np.clip((np.maximum(edges_t[:-1], edges_t[1:]) / HOP).astype(int) + 1, 1, len(db))
            pw = (css[k][ib] - css[k][ia]) / np.maximum(ib - ia, 1)[:, None]
            pw = pw.reshape(NC, NR, NB // NR).sum(2)
            v = np.clip((10 * np.log10(pw + 1e-12) - lo - 3.0) / (hi - lo), 0, 1)
            vf = np.stack([np.interp(xf, xs_cols, v[:, b]) for b in range(NR)], 1)
            wg = np.where((ttf >= K_T0)[:, None], wig[k][hidx], 0.0)
            X = np.repeat(xf[:, None], NR, 1)
            Y = ys_r[None, :] + vf * RIDGE + wg
            Pw = np.stack([X.ravel(), Y.ravel(), np.full(X.size, zk)], 1)
            q, zc = project(Pw, st)
            ok = zc > 60
            zmed = float(np.median(zc[ok])) if ok.any() else 3000
            wv = (0.16 + 3.0 * vf.ravel() ** 1.3) * (1.6 * dscale(zc) if VER["v"] == 5 else np.clip(1500 / np.maximum(zc, 1), 0.3, 2.0)) * fog(zmed)
            acc += splat(q[ok, 0], q[ok, 1], wv[ok])
            hot = np.clip(np.abs(wg.ravel()) / WIG_MAX - 0.7, 0, None) / 0.3
            if hot.max() > 0.01:
                acc_r += splat(q[ok, 0], q[ok, 1], (hot ** 1.5 * 0.9 * (0.3 + vf.ravel()))[ok])
        img = 1 - np.exp(-acc * 0.85)
        rgb = np.stack([img] * 3, -1) * np.array(INK) / 255.0
        rr = 1 - np.exp(-acc_r * 1.4)
        rgb = rgb * (1 - rr[..., None]) + rr[..., None] * np.array(HOT) / 255.0
        bg = np.array(BG) / 255.0
        rgb = bg + (1 - bg) * rgb
        im = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).resize((W * SS, H * SS), Image.BILINEAR)
        dr = ImageDraw.Draw(im)

        def P2(p):
            q, z = project(np.asarray(p, float), st)
            return q[0] * SS, z[0]

        def poly(pts, col, w=1.0, dash=None):
            q, z = project(np.asarray(pts, float), st)
            if (z < 60).any():
                return
            a = fog(float(np.median(z)))
            c = mix(col, a)
            ww = max(1, int(round(w * SS * (1.5 * float(dscale(np.median(z))) if VER["v"] == 5 else np.clip(1700 / np.median(z), 0.5, 1.6)))))
            q = q * SS
            if dash is None:
                dr.line([tuple(p) for p in q], fill=c, width=ww)
            else:
                on, off = dash
                s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(q, axis=0).T))])
                ph = np.mod(s, on + off) < on
                for i in range(len(q) - 1):
                    if ph[i]:
                        dr.line([tuple(q[i]), tuple(q[i + 1])], fill=c, width=ww)

        def label(p, s, col, size, cn=False, anchor="la", dx=0, dy=0, knock=False):
            q, z = P2(p)
            if z < 60:
                return
            a = fog(z)
            sz = int(round(size * (0.75 + 0.45 * (float(dscale(z)) - 0.45) / 0.55) if VER["v"] == 5 else np.clip(size * 1900 / z, size * 0.75, size * 1.3)))
            text(dr, (q[0] + dx * SS, q[1] + dy * SS), s, mix(col, a), sz, cn, anchor, knock=BG if knock else None)

        # 分轨层：框 + 名字 + 电平 + 振子状态
        i_now = int(tn / HOP)
        for k, (name, zk) in enumerate(STEMS):
            poly([(-X1, 0, zk), (0, 0, zk), (0, HH, zk), (-X1, HH, zk)], FAINT, 1)
            lvl = 10 * np.log10(np.sum(10 ** (dbs[k][max(0, i_now - 10):i_now + 1] / 10), axis=1).mean() + 1e-12)
            Kc, R_, nlock = kinfo[kf, 0], kinfo[kf, 1 + k], kinfo[kf, 1 + nL + k]
            trem = float(np.abs(wig[k][max(0, kf - 6):kf + 1]).max())
            lo5 = 36 * (nL - 1 - k) if VER["v"] == 5 else 0               # 正面看各层的标签会叠在一起 → 竖着错开
            label((0, HH, zk), f"{name.upper()}  {lvl:5.1f} dB", HOT if trem > 0.6 * WIG_MAX else MID, 13, dx=8, dy=-30 - lo5)
            label((0, HH, zk), f"K {Kc:.2f}  R {R_:.2f}  locked {int(nlock)}/{NR}  tremor {trem:4.1f}", DIM, 11, dx=8, dy=-14 - lo5)
        for fz in (100, 1000, 5000):
            label((0, yf(fz), STEMS[0][1]), f"{fz if fz < 1000 else str(fz // 1000) + 'k'}Hz", DIM, 11, dx=6, dy=-6)
        # 播放头：跨所有层的竖平面边
        poly(np.linspace((0, -40, Z_SENSE), (0, -40, STEMS[-1][1]), 12), MID, 1)
        poly(np.linspace((0, HH + 140, Z_SENSE), (0, HH + 140, STEMS[-1][1]), 12), DIM, 1)
        for back in (1, 5, 15, 40):
            if back < tn:
                x = fisheye_x(back, tn)
                poly([(x, -20, Z_SENSE), (x, -40, Z_SENSE)], DIM, 1)
                label((x, -60, Z_SENSE), f"−{back}s", DIM, 11, anchor="ma")

        # ── VOICE：f0 线 + 乐句括号 + 音名格线 + 响度
        for nm, fz in (("C3", 130.81), ("C4", 261.63), ("C5", 523.25), ("C6", 1046.5)):
            y = yf(fz)
            poly(np.linspace((-X1, y, Z_VOICE), (0, y, Z_VOICE), 30), FAINT, 1, dash=(3 * SS, 7 * SS))
            label((-X1, y, Z_VOICE), nm, DIM, 11, dx=-6, anchor="ra")
        nfr = min(int(tn * fps_s) + 1, len(f0))
        ti = np.arange(nfr) / fps_s
        xi = fisheye_x(tn - ti, tn)
        seg_f0 = []
        for i in range(nfr):
            if np.isfinite(f0[i]):
                seg_f0.append((xi[i], yf(f0[i]), Z_VOICE))
            elif seg_f0:
                if len(seg_f0) > 1:
                    poly(seg_f0, INK, 5.0)
                seg_f0 = []
        if len(seg_f0) > 1:
            poly(seg_f0 + [(0.0, seg_f0[-1][1], Z_VOICE)] if tn - ti[-1] < 0.1 else seg_f0, INK, 5.0)
        # 响度：f0 层底部的一条细线（0..1 → y −110…−30）
        sel = np.unique(np.linspace(0, nfr - 1, min(nfr, 700)).astype(int))
        poly(np.stack([xi[sel], -110 + 80 * loud[sel], np.full(len(sel), Z_VOICE)], 1), MID, 0.8)
        label((-X1, -110, Z_VOICE), "loudness", DIM, 11, dx=-6, anchor="ra")
        for k_, p in enumerate(phr):
            if p["start"] > tn:
                continue
            xa = fisheye_x(tn - p["start"], tn)
            xb = fisheye_x(max(0.0, tn - p["end"]), tn)
            yb = yf(p["f0_hi"]) + 46
            poly([(xa, yb - 16, Z_VOICE), (xa, yb, Z_VOICE), (xb, yb, Z_VOICE), (xb, yb - 16, Z_VOICE)], INK, 1.2)
            label(((xa + xb) / 2, yb, Z_VOICE), f"phrase {vo['phrases'].index(p) + 1}  f0 {p['f0_med']:.0f} Hz", INK, 13, anchor="md", dy=-6)
        vf_now = f0[min(int(tn * fps_s), len(f0) - 1)]
        if np.isfinite(vf_now):
            label((0, yf(vf_now), Z_VOICE), f"VOICE  f0 {vf_now:6.1f} Hz", INK, 15, dx=10, dy=-8, knock=True)
        else:
            nxt = [p for p in vo["phrases"] if p["start"] <= tn]
            last = f"last phrase ended {tn - nxt[-1]['end']:.2f}s ago" if nxt else "unvoiced since 0:00"
            label((0, yf(440), Z_VOICE), f"VOICE  f0 —  ({last})", MID, 13, dx=10, dy=-8)

        # ── EXPECT：色带（段）+ 意外度尖刺 + 分歧 ▼ + absence
        YB, YT = HH + 40, HH + 120
        for a_, b_, sid, nm in segs:
            if a_ > tn:
                continue
            xa = fisheye_x(tn - a_, tn); xb = fisheye_x(max(0.0, tn - b_), tn)
            q, z = project(np.array([(xa, YB, Z_EXP), (xb, YB, Z_EXP), (xb, YT, Z_EXP), (xa, YT, Z_EXP)]), st)
            if (z > 60).all():
                cur = a_ <= tn < b_
                dr.polygon([tuple(p) for p in q * SS], fill=mix(MID if cur else DIM, 0.32 * fog(float(z.mean()))),
                           outline=mix(INK if cur else DIM, fog(float(z.mean()))))
                em = exp_meta.get(sid, {})
                if xb - xa > 60 or cur:
                    label((xa, YT, Z_EXP), f"{sid} {nm}", INK if cur else MID, 15 if cur else 12, cn=True, dx=6, dy=4)
                    if cur and em:
                        label((xa, YT, Z_EXP), f"等: {em.get('expecting', '')}   怕: {em.get('fear', '')}   忍: {em.get('tolerance', '')}",
                              MID, 12, cn=True, dx=6, dy=30)
        Y0 = 60.0
        poly(np.linspace((-X1, Y0, Z_EXP), (0, Y0, Z_EXP), 20), DIM, 0.8)
        for e in evs:
            if e["t"] > tn or e["t"] < 0.3:
                continue
            x = fisheye_x(tn - e["t"], tn)
            sal = e.get("sal") or 0
            hi_ = sal >= sal_thr
            hgt = 34 * min(e["bits"], 12)
            col = HOT if hi_ else (INK if e["s"].startswith("voice") else (MID if e["s"] == "onset" else DIM))
            poly([(x, Y0, Z_EXP), (x, Y0 + hgt, Z_EXP)], col, 2.2 if hi_ else (1.4 if e["s"].startswith("voice") else 0.9),
                 dash=(2 * SS, 3 * SS) if e["s"] == "absence" else None)
            if sal > 0.05:
                poly([(x - 5, Y0 + 34 * sal, Z_EXP), (x + 5, Y0 + 34 * sal, Z_EXP)], col, 1.0)    # 横刻 = 显著度
            if (hi_ or e["s"].startswith("voice")) and tn - e["t"] < 6:
                label((x, Y0 + hgt, Z_EXP), f"{e['s']} {e['bits']:.1f} bit" + (f" · sal {sal:.2f}" if hi_ else ""),
                      HOT if hi_ else INK, 12, dx=6, dy=-14)
        nab = min(int(tn * fps_s) + 1, len(absence))
        sel = np.unique(np.linspace(0, nab - 1, min(nab, 600)).astype(int))
        poly(np.stack([fisheye_x(tn - sel / fps_s, tn), Y0 - 50 + 10 * absence[sel], np.full(len(sel), Z_EXP)], 1), DIM, 0.8)
        label((-X1, Y0 - 50, Z_EXP), "absence", DIM, 11, dx=-6, anchor="ra")
        for ap in auds:
            if ap["t"] > tn:
                continue
            x = fisheye_x(tn - ap["t"], tn)
            e_ = min(evs, key=lambda e: abs(e["t"] - ap["t"]))
            ytop = Y0 + 34 * min(e_["bits"], 12) + 26
            q, z = P2((x, ytop, Z_EXP))
            if z > 60:
                s_ = 7 * SS * (1.1 if VER["v"] == 5 else np.clip(1900 / z, 0.7, 1.4))
                col = HOT if ap["id"][0] == "A" else MID
                if ap["id"][0] == "A":
                    dr.polygon([(q[0], q[1] + s_), (q[0] - s_, q[1] - s_), (q[0] + s_, q[1] - s_)], fill=mix(col, fog(z)))
                else:
                    dr.polygon([(q[0], q[1] - s_), (q[0] - s_, q[1] + s_), (q[0] + s_, q[1] + s_)], fill=mix(col, fog(z)))
                if tn - ap["t"] < 8:
                    label((x, ytop, Z_EXP), f"{ap['id']}  {fmt_t(ap['t'])}  {ap['ev']}  {ap['bits']}  — 剪辑没回应（审计 v4）",
                          HOT, 13, cn=True, dx=12, dy=-24)
        label((-X1, YT, Z_EXP), f"EXPECT · surprise = −log₂p (bits)   tick = salience   red ≥ {sal_thr:.2f}   ▼ = audit divergence",
              MID, 12, dx=0, dy=-22)

        # ── SENSE：4 条通道曲线，各一条泳道，各一种线
        tx = np.linspace(-X1, 0, 600)
        tsec = tn - fisheye_inv(tx, tn)
        sty = {"threat": (INK, 2.0, None), "heartbeat": (INK, 1.2, None), "hunger": (MID, 1.2, (5 * SS, 4 * SS)), "tension": (MID, 0.8, None)}
        for k_, cname in enumerate(["threat", "heartbeat", "hunger", "tension"]):
            vals = ch[cname][np.clip((tsec * fps_s).astype(int), 0, len(ch[cname]) - 1)]
            y0 = 30 + k_ * 170
            col, lw, dsh = sty[cname]
            poly(np.stack([tx, np.full(600, y0), np.full(600, Z_SENSE)], 1)[::40], FAINT, 0.8)
            poly(np.stack([tx, y0 + 140 * vals, np.full(600, Z_SENSE)], 1), col, lw, dash=dsh)
            cur = ch[cname][min(int(tn * fps_s), len(ch[cname]) - 1)]
            label((0, y0 + 140 * cur, Z_SENSE), f"{SYM[cname]} {cname}  {cur:.2f}", HOT if (cname == "threat" and cur > 0.6) else INK, 15, dx=10, dy=-10, knock=True)

        pulse = 0.0
        if VER.get("mode") == "C":                        # C：拍点只作用在准星上（每拍一跳，τ 0.12 s）
            bb = np.asarray(sense["events"]["beats"], float)
            d_ = tn - bb[(bb <= tn + LEAD / FPS) & (bb > tn - 0.6)] + LEAD / FPS
            pulse = float(np.sum(np.exp(-np.maximum(d_, 0) / 0.12)))
        draw_reticle(dr, st, cinfo[f], tn, pulse)
        # HUD
        sid_now = next((s for s in segs if s[0] <= tn < s[1]), segs[-1])
        text(dr, (56 * SS, 40 * SS), f"{sid_now[2]}  {sid_now[3]}    SENSORY SPACE · OPENING    t = {tn:6.2f} s", INK, 13, cn=True)
        text(dr, (56 * SS, 62 * SS), f"X = time (fisheye τ={TAU}s, playhead x=0)   Y = log f {F_LO:.0f} Hz – {F_HI / 1000:.0f} kHz   "
             f"Z = SENSE · EXPECT · VOICE · drums · other · bass · vocals", MID, 12)
        text(dr, (56 * SS, 80 * SS), f"tremor = opening Kuramoto law (K {K0} @ {K_T0:.0f}s +{K_BEAT}/beat, kick σ = {KICK}·θ·flux_band/p95), "
             f"shown = eff. freq − 3 s mean", DIM, 12)
        out = im.resize((W, H), Image.LANCZOS)
        meta.append(dict(f=f, t=round(tn, 3), cam=[round(float(v), 3) for v in st]))
        yield f, out, meta[-1]
    render.cams, render.events, render.refs, render.kref = cams, cam_ev, refs, kref


def contact(frames, path, cols=4, tw=640):
    th = int(tw * H / W)
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * (th + 22)), (12, 12, 12))
    d = ImageDraw.Draw(sheet)
    fnt = ImageFont.truetype(FMONO, 15)
    for k, (lab, im) in enumerate(frames):
        x, y = (k % cols) * tw, (k // cols) * (th + 22)
        sheet.paste(im.resize((tw, th), Image.LANCZOS), (x, y + 22))
        d.text((x + 6, y + 3), lab, fill=(200, 200, 200), font=fnt)
    sheet.save(path, quality=88)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seg0", type=float, default=38.71)
    ap.add_argument("--seg1", type=float, default=51.51)
    ap.add_argument("--v", type=int, default=3, choices=(2, 3, 4, 5), help="2 = 漂移环绕 + 层距大；3 = 在里面搜寻 + 层距压缩；4 = 节奏 A/B/C 试验（--mode）")
    ap.add_argument("--mode", default="A", choices=("A", "B", "C"))
    ap.add_argument("--points", action="store_true", help="v5：加 v4 B 的「每拍一点」（非首拍的拍上小推近）")
    ap.add_argument("--lowres", action="store_true", help="960×540（A/B 试听用）")
    ap.add_argument("--name", default=None)
    ap.add_argument("--still", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.v >= 3:
        set_v3()
    if a.v == 4:
        VER.update(v=4, mode=a.mode)
    if a.lowres:
        global W, H, FOCAL
        W, H = 960, 540
        FOCAL = (W / 2) / math.tan(math.radians(25))
    if a.v == 5:
        set_v5(a.points)
    a.name = a.name or (f"24_感官空间_开头_v4_{a.mode}" if a.v == 4 else f"24_感官空间_开头_v5{'_points' if a.points else ''}" if a.v == 5
                        else f"24_感官空间_开头_v{a.v}")
    seg = (a.seg0, a.seg1)
    T = int(round((seg[1] - seg[0]) * FPS))
    if a.still:
        fs = {int(round((float(x) - seg[0]) * FPS)) for x in a.still.split(",")}
        ims = [(f"{m['t']:.2f}s f{f}", im) for f, im, m in render(seg, only=fs)]
        contact(ims, a.out, cols=2, tw=960)
        print("->", a.out)
        return
    vid = OUT / f"{a.name}.mp4"
    if vid.exists():
        sys.exit(f"已存在，不覆盖：{vid}")
    cmd = [ffmpeg(), "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-ss", f"{seg[0]:.3f}", "-t", f"{T / FPS:.3f}", "-i", str(MUSIC), "-map", "0:v", "-map", "1:a",
           "-c:v", "libx264", "-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", str(vid)]
    w = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    picks = set(np.linspace(0, T - 1, 20).round().astype(int).tolist())
    sheet, meta = [], []
    for f, im, m in render(seg):
        w.stdin.write(np.asarray(im).tobytes())
        meta.append(m)
        if f in picks:
            sheet.append((f"{m['t']:.2f}s  f{f}", im))
        if f % 48 == 0:
            print(f"  frame {f}/{T}", flush=True)
    w.stdin.close()
    w.wait()
    contact(sheet, vid.with_name(vid.stem + "_接触表.jpg"))
    sp = screen_speed(render.cams, render.refs)
    al = alignment(sp, render.events, T)
    print("镜头速度峰值 − 事件帧：", {d: sum(1 for *_, x in al if x == d) for d in sorted(set(x for *_, x in al))})
    print(f"屏幕速度 px/帧：min {sp[1:].min():.2f}  median {np.median(sp[1:]):.2f}  max {sp.max():.2f}")
    speed_plot(sp, render.events, al, vid.with_name(vid.stem + "_镜头速度.png"), seg[0], vid.stem)
    vid.with_suffix(".json").write_text(json.dumps(dict(
        tool="tools/sense_space_v2.py", version=VER["v"], seg=seg, stems=STEMS, z_sense=Z_SENSE, z_expect=Z_EXP, z_voice=Z_VOICE, fisheye_tau=TAU,
        bands=[NB, NR, F_LO, F_HI],
        tremor=dict(law="opening.py Kuramoto", K0=K0, K_t0=K_T0, K_per_beat=K_BEAT, kick="N(0, 0.35*threat*clip(flux_band/p95,0,1)) per frame",
                    flux_p95_db=render.kref, v_gain=V_GAIN, hb_gain=HB_GAIN, tau_eff=TAU_EFF, tau_slow=TAU_SLOW, gain_world_per_hz=WIG_GAIN, clip=WIG_MAX),
        camera_events=[[round(t + seg[0], 3), k] for t, k in render.events],
        camera_alignment=[[round(t + seg[0], 3), k, d] for t, k, d in al],
        screen_speed=[round(float(x), 3) for x in sp], frames=meta), ensure_ascii=False, indent=1), "utf-8")
    print("->", vid)


if __name__ == "__main__":
    main()
