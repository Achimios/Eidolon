#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🕸感官空间-正面：从头重做（👤2026-10-23 判 v1–v5 方向废：「颜色没参考图片」「镜头一下前一下后」）。

⭐长相 = 两张参考图（感官_期待审计_v4.jpg、人声_条带_v2.jpg）的条带，原样取色、线型、字号层级、标记方式：
  底 (12,13,15)；左边 86 px 是标签栏（泳道名 / 刻度，和参考图一样，数据不进这一栏）；格线 (34,36,40)；
  （泳道上 → 下：期待 / 感官 / 人声 / other / bass / drums）
  期待 = 段落色块（sense_expect.EXP_COL + 「等: 忍:」）+ 意外度竖刺（STREAM_COL：灰起音 / 蓝人声起 / 紫 absence）+ 编号三角 + absence 紫线；
         审计 v4 的 A 点 = 红 ▼ + 「A5 0:47.19」+ 暗红竖线 (120,30,36)（穿过所有层，各在自己的深度上）；
  人声 = 红条（人声段）+ 灰波形 (120,130,140) + 白响度线 + f0（黄 = 稳，红 = 颤音；C3–C6 音名格）+ 蓝有声概率 / 红颤音 + 乐句括号；
  感官 = 四条曲线（θ 威胁 红粗 / β 心跳 白 / η 饥饿 金虚线 / τ 张力 蓝细）；
  频谱 = 密纹理：分轨的对数频带山脊（灰），被打中的变白；vocals 的频谱放在 f0 泳道正后方，与 f0 共用 80–1100 Hz 对数轴（基频落在自己的谐波脊上）。
层（z = 深度，前 → 后，层间靠近）：voice 0 · vocals 频谱 80 · expect 140 · sense 260 · other 380 · bass 500 · drums 620；
  远近 = 雾（颜色往底色混）+ 线宽 + 字号；正面长焦（半视场 8°）下靠平移时的视差错开（每层自己的播放头竖线错开几十 px）。
X = 播放时间（250 世界单位 / 秒），过去画实、未来只画拍格（已知的拍网格 = 空白的谱纸）。

⭐镜头（👤：正面、长焦、只平移 + 推拉；推拉单调）：
  推拉 = 整段匀速推近（cz 线性，倍率 0.74 → 0.96，即 1.30×），速度符号恒定；
  平移 = 每个小节首拍（感官.json downbeats）提前 LEAD 帧起步，1 − (1 − u)^2.2 快起慢收（D 1.4 s），之间只有慢漂（x 以播放头 0.4 倍速前漂）；
  目标 = 下一小节里最显著的事件（期待 sal，人声起必胜）：起音 → 那一刻通量最大的分轨 × 通量重心频带；人声起 → f0 泳道的乐句 f0 与审计 ▼ 的中点（两者同框）；absence → 紫线；
  竖向每小节最多走 VMAX 世界单位（看得清的上限），远的目标分两小节到。
⭐拍点作用在层上 = opening.py Kuramoto.step 原律：K 从 10 s 起 0.8、每拍 +0.035；每帧相位踢 N(0, 0.35·威胁·clip(通量 / p95, 0, 1))；
  有效频率 EMA 0.5 s − 3 s 慢均值 → 该山脊 / 曲线头部（最近 0.6 s，指数衰减）的竖向位移。锁相边界附近的振子平不下来。
  通量按每条分轨自己的实际电平范围归一（p50 底噪 … p99.7），修掉 v2 的 −10 dB 截断 bug（那会让抖动落在安静频带）。
用法：python tools/sense_space_v3.py [--still 41.0,44.2,47.3,49.2 --out x.jpg] [--name …]
输出：样片/信息可视化/<name>.mp4（配乐）+ _接触表.jpg + _镜头速度.png + .json（+ >30 MB 时 _预览.mp4）。不覆盖已有文件。
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
S0, S1 = 38.71, 51.51

# ── 参考图取色（sense_expect.py / sense_voice.py 的原值） ──
BG = (12, 13, 15)
GRID, GRID_DIM, SEP = (34, 36, 40), (22, 24, 27), (30, 32, 36)
LAB, TICK, TICKLINE, AXLAB = (150, 156, 162), (150, 150, 150), (120, 120, 120), (100, 104, 110)
INK, SUB = (230, 233, 236), (200, 202, 206)
RED, RED_DK, RED_BAR = (255, 42, 52), (120, 30, 36), (200, 60, 70)
WAVE, PHRASE, F0Y, VBLUE = (120, 130, 140), (26, 30, 36), (235, 210, 120), (90, 140, 200)
NOTEGRID, NOTELAB = (40, 44, 50), (110, 116, 122)
ABS_LINE = (50, 40, 60)
STREAM_COL = {"onset": (150, 156, 162), "motif": (235, 200, 90), "voice_on": (90, 150, 230), "voice_off": (150, 190, 240),
              "absence": (180, 110, 230)}
EXP_COL = {"冲击": (150, 40, 46), "John": (150, 118, 40), "自己的声音": (40, 80, 140), "自己": (60, 70, 90),
           "记忆": (70, 90, 100), "无": (44, 46, 50)}
SENSE_STYLE = {"threat": ("θ 威胁", RED, 2.4, None), "heartbeat": ("β 心跳", INK, 1.3, None),
               "hunger": ("η 饥饿", (235, 200, 90), 1.3, (7, 5)), "tension": ("τ 张力", (90, 150, 230), 1.0, None)}
CAP, ABS_ON = 12.0, 4.0                     # src/expect.py

# ── 空间 ──
PPS = 250.0                                  # 世界单位 / 秒
HFOV = math.radians(8.0)
FOC = (W / 2) / math.tan(HFOV)
ZOOM0, ZOOM1 = 0.74, 0.96
CZ0, CZ1 = -FOC / ZOOM0, -FOC / ZOOM1        # z = 0 层的倍率 0.74 → 0.96（匀速推近 1.30×）
FOG_L, FOG_MIN = 1100.0, 0.42
MARGIN = 86                                  # 左侧标签栏（px）

Z = dict(voice=0.0, vspec=80.0, expect=140.0, sense=260.0, other=380.0, bass=500.0, drums=620.0)
# 泳道（世界 Y，向下为正）
Y_MARK, Y_SEG, H_SEG = -46.0, 12.0, 50.0
Y_BITS, H_BITS = 70.0, 120.0
Y_ABS, H_ABS = 198.0, 42.0
Y_TAX = 246.0
Y_SENSE, H_SL = 268.0, 44.0                  # 4 条泳道（紧贴期待下面：她的感官读数挨着她的期待）
Y_VSEC = 470.0
Y_WAVE, H_WAVE = 486.0, 70.0
Y_LOUD, H_LOUD = 564.0, 52.0
Y_F0, H_F0 = 624.0, 180.0
Y_PV, H_PV = 812.0, 50.0
SENSE_ORDER = ["threat", "heartbeat", "hunger", "tension"]
STEM_LANES = [("other", 890.0), ("bass", 1050.0), ("drums", 1210.0)]
H_STEM = 145.0
Y_TOP, Y_BOT = Y_MARK - 70, STEM_LANES[-1][1] + H_STEM + 40     # 顶上留 HUD 的位置
F0_LO, F0_HI = 80.0, 1100.0

# 频谱
SR, HOP = 22050, 0.02
NB, F_LO, F_HI = 96, 30.0, 10000.0
NR = 32                                      # 分轨泳道：每 3 个频带一条山脊
SP_STEM = H_STEM / NR

# Kuramoto（opening.py 的律）
K0, K_T0, K_BEAT, KICK, HB_GAIN, V_GAIN, TAU_EFF, TAU_SLOW = 0.8, 10.0, 0.035, 0.35, 0.6, 2.2, 0.5, 3.0
WIG_SP_PER_HZ, WIG_DEAD_SP, WIG_MAX_SP = 8.0, 0.4, 2.2   # 位移：山脊间距 / Hz；减去 0.4 的死区（未锁振子的慢漂不显示，只显示被踢起来的），上限 2.2 个间距
HEAD_T, HEAD_TAU = 0.6, 0.22                 # 抖的只是头部（最近 0.6 s，指数衰减）
N_SENSE_OSC = 12

# 镜头
LEAD, EASE_P, PAN_D = 2, 2.2, 1.4
FRAME_X = 0.58                               # 小节首拍落在画面 58% 处（播放头在 0.58–0.75 之间走，右边是未来的拍格）
FRAME_Y = 0.46
DRIFT = 0.4 * PPS                            # 拍内慢漂：x 以播放头 0.4 倍速前漂
VMAX = 420.0

_F = {}


def font(size, cn=True):
    k = (round(size * SS), cn)
    if k not in _F:
        _F[k] = ImageFont.truetype(FCN if cn else FMONO, max(6, k[0]))
    return _F[k]


def mix(c, a, bg=BG):
    a = float(min(1.0, max(0.0, a)))
    return tuple(int(round(bg[i] + (c[i] - bg[i]) * a)) for i in range(3))


def ease(u):
    u = min(1.0, max(0.0, u))
    return 1 - (1 - u) ** EASE_P


def fog(z):
    return float(np.clip(math.exp(-max(0.0, z) / FOG_L), FOG_MIN, 1.0))


def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def mmss(t, d=2):
    return f"{int(t // 60)}:{t % 60:0{3 + d}.{d}f}" if d else f"{int(t // 60)}:{int(t % 60):02d}"


def X(t):
    return (t - S0) * PPS


def y_f0(hz, y0=Y_F0, h=H_F0):
    return y0 + h - (math.log2(hz) - math.log2(F0_LO)) / (math.log2(F0_HI) - math.log2(F0_LO)) * h


# ───────────────────────── 数据 ─────────────────────────
def stem_bands(name, t_end=52.0):
    """分轨 0 → t_end 的对数频带功率（dB，帧 = 20 ms）。与 sense_space(_v2) 同一算法、同一缓存名（只读复用）。"""
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


def level_range(db):
    """⚠️按这条分轨自己的实际电平范围归一：p50（底噪）… p99.7"""
    v = db[db > -100].ravel()
    return float(np.percentile(v, 50)), float(np.percentile(v, 99.7))


def band_centers():
    e = np.geomspace(F_LO, F_HI, NB + 1)
    return np.sqrt(e[:-1] * e[1:])


def audit_points():
    out = []
    if not AUDIT_MD.exists():
        return out
    for line in AUDIT_MD.read_text("utf-8").splitlines():
        m = re.match(r"\|\s*\*\*([AB]\d+)\*\*\s*\|\s*(\d+):(\d+\.\d+)\s*\|", line)
        if m:
            out.append(dict(id=m.group(1), t=int(m.group(2)) * 60 + float(m.group(3))))
    return out


class Data:
    def __init__(self):
        self.sense = json.loads(SENSE.read_text("utf-8"))
        self.fps = self.sense["fps"]
        self.ch = {k: np.asarray(v, float) for k, v in self.sense["channels"].items()}
        self.beats = np.asarray(self.sense["events"]["beats"], float)
        self.downbeats = np.asarray(self.sense["events"]["downbeats"], float)
        ex = self.sense["expect"]
        self.events = ex["events"]
        self.absence = np.asarray(ex["absence"], float)
        ep = {}
        if EXPECT_JSON.exists():
            ep = {e["id"]: e for e in json.loads(EXPECT_JSON.read_text("utf-8"))["段"]}
        self.segs = [dict(t0=s[0], t1=s[1], id=s[2], name=s[3], **({"exp": ep[s[2]]["expecting"], "tol": ep[s[2]]["tolerance"]} if s[2] in ep else {}))
                     for s in ex["segments"]]
        v = self.sense["voice"]
        self.loud = np.asarray(v["loudness"], float)
        self.voiced = np.asarray(v["voiced"], float)
        self.vib = np.asarray(v["vibrato"], float)
        self.f0 = np.array([np.nan if x is None else x for x in v["f0"]], float)
        self.phrases, self.vsections, self.breaths = v["phrases"], v["sections"], v["breaths"]
        self.audit = [a for a in audit_points() if a["t"] <= S1 + 0.05]
        # 显著事件编号（本段，sal ≥ 1）
        self.numbered = [e for e in self.events if S0 <= e["t"] <= S1 and (e.get("sal") or 0) >= 1.0 and e["s"] != "voice_off"]
        for i, e in enumerate(self.numbered):
            e["no"] = f"E{i + 1}"
        # 分轨
        self.db = {n: stem_bands(n) for n in ("drums", "bass", "other", "vocals")}
        self.rng_db = {n: level_range(d) for n, d in self.db.items()}
        fc = band_centers()
        self.vbands = np.where((fc >= F0_LO) & (fc <= F0_HI))[0]
        self.vband_y = np.array([y_f0(f) for f in fc[self.vbands]])
        self.lev = {}                                  # 归一电平（帧 × 山脊）
        for n in ("drums", "bass", "other"):
            lo, hi = self.rng_db[n]
            r = np.clip(self.db[n], lo, hi).reshape(len(self.db[n]), NR, NB // NR).mean(2)
            self.lev[n] = np.clip((r - lo) / (hi - lo), 0.0, 1.0)      # ⚠️均值的浮点误差会给出 −1e-16 → ** 1.3 = NaN
        self.flux_ref = {}
        for n in ("drums", "bass", "other"):
            fs = np.clip(np.diff(self.lev[n], axis=0), 0, None).sum(1)
            fs = np.convolve(fs, np.ones(5), "same")
            self.flux_ref[n] = float(np.percentile(fs[fs > 0], 90))
        lo, hi = self.rng_db["vocals"]
        self.lev["vocals"] = np.clip((self.db["vocals"][:, self.vbands] - lo) / (hi - lo), 0.0, 1.0)
        # 人声波形（4 ms 一格 = 1 世界单位）
        a0 = 28.0
        x = sensorium.load_audio(MV / "分轨" / "vocals.wav", a0, S1 + 1.0 - a0, sr=SR)
        n = int(SR / PPS)
        m = len(x) // n
        env = np.abs(x[:m * n]).reshape(m, n).max(1)
        self.wave_t0, self.wave = a0, env / (env.max() + 1e-9)

    def chv(self, name, t):
        a = self.ch[name]
        return float(a[min(max(int(t * self.fps), 0), len(a) - 1)])


# ───────────────────────── Kuramoto（opening.py 律，局部踢） ─────────────────────────
def group_flux(D):
    """每组振子每帧（24 fps，从 K_T0 起）的通量：分轨 = 山脊的正电平差（已按该轨实际电平范围归一，0–1）；感官 = 曲线的正增量"""
    T = int(round((S1 + 0.1 - K_T0) * FPS)) + 1
    out = {}
    for n in ("drums", "bass", "other", "vocals"):
        L = D.lev[n]
        fl = np.r_[np.zeros((1, L.shape[1])), np.clip(np.diff(L, axis=0), 0, None)]
        ff = np.zeros((T, L.shape[1]))
        for f in range(T):
            t = K_T0 + f / FPS
            i0, i1 = int((t - 1 / FPS) / HOP), int(t / HOP) + 1
            ff[f] = fl[max(0, i0):i1].max(0)
        out[n] = ff
    ss = np.zeros((T, 4))
    for j, c in enumerate(SENSE_ORDER):
        a = D.ch[c]
        for f in range(T):
            t = K_T0 + f / FPS
            k = int(t * D.fps)
            ss[f, j] = max(0.0, a[min(k, len(a) - 1)] - a[max(k - 1, 0)])
    out["sense"] = np.repeat(ss, N_SENSE_OSC, axis=1)
    return out, T


def kuramoto(D, seed=1):
    """返回 wig[group] = (帧, 振子) 的位移（单位 = 该组的山脊间距 / 曲线泳道的比例），以及 info（K、各组 R）。
    帧 f ↔ 播放时间 K_T0 + f / FPS。"""
    ff, T = group_flux(D)
    groups = ["drums", "bass", "other", "vocals", "sense"]
    ref = {}
    stem_pool = np.concatenate([ff[g][ff[g] > 0] for g in groups[:4]])
    for g in groups[:4]:
        ref[g] = float(np.percentile(stem_pool, 95))
    ref["sense"] = float(np.percentile(ff["sense"][ff["sense"] > 0], 95))
    rng = np.random.default_rng(seed)
    krng = np.random.default_rng(seed + 100)
    st = {}
    for g in groups:
        n = ff[g].shape[1]
        st[g] = dict(w=np.clip(rng.standard_cauchy(n) * 0.5, -3, 3), th=rng.uniform(0, 2 * np.pi, n),
                     eff=None, slow=None, kick=np.zeros(n))
    K = K0
    wig = {g: np.zeros((T, ff[g].shape[1])) for g in groups}
    info = np.zeros((T, 1 + len(groups)))
    dt = 1 / (FPS * 4)
    for f in range(T):
        t = K_T0 + f / FPS
        K += int(((D.beats > t - 1 / FPS) & (D.beats <= t)).sum()) * K_BEAT
        hb, th = D.chv("heartbeat", t), D.chv("threat", t)
        for gi, g in enumerate(groups):
            s = st[g]
            for j in range(4):
                z = np.exp(1j * s["th"]).mean()
                v = V_GAIN * (s["w"] + HB_GAIN * hb + K * abs(z) * np.sin(np.angle(z) - s["th"]))
                s["th"] = s["th"] + v * dt
                dth = v * dt + (s["kick"] if j == 0 else 0)
                s["eff"] = dth / dt if s["eff"] is None else s["eff"] + (dth / dt - s["eff"]) * dt / TAU_EFF
                s["slow"] = s["eff"].copy() if s["slow"] is None else s["slow"] + (s["eff"] - s["slow"]) * dt / TAU_SLOW
            s["kick"] = krng.normal(0, 1, len(s["w"])) * KICK * th * np.clip(ff[g][f] / ref[g], 0, 1)
            s["th"] = s["th"] + s["kick"]
            raw = (s["eff"] - s["slow"]) / (2 * np.pi) * WIG_SP_PER_HZ
            wig[g][f] = np.sign(raw) * np.clip(np.abs(raw) - WIG_DEAD_SP, 0, WIG_MAX_SP)
            info[f, 1 + gi] = abs(np.exp(1j * s["th"]).mean())
        info[f, 0] = K
    return wig, info, ref


def kframe(t):
    return int(round((t - K_T0) * FPS))


# ───────────────────────── 镜头 ─────────────────────────
def scale_at(z, cz):
    return FOC / (z - cz)


def stem_hit(D, t):
    """t 时刻正通量最大的分轨泳道 × 通量重心 → (层名, 世界 Y)"""
    i = int(round(t / HOP))
    best = (-1.0, "drums", 0.0)
    for n, y0 in STEM_LANES:
        L = D.lev[n][max(0, i - 2):i + 4]
        fl = np.clip(np.diff(L, axis=0), 0, None).sum(0)
        rel = fl.sum() / D.flux_ref[n]                 # 相对这条分轨自己的常态（p90）：鼓总是通量最大，比绝对值会永远选鼓
        if rel > best[0]:
            r = (fl * (np.arange(NR) + 0.5)).sum() / max(fl.sum(), 1e-9)
            best = (float(rel), n, y0 + H_STEM - r * SP_STEM)
    return best[1], best[2]


def bar_targets(D):
    """每个小节首拍：下一小节里最显著的事件（期待 sal；人声起必胜）→ 世界 (y, z)"""
    db = [d for d in D.downbeats if S0 - 1.7 < d < S1]
    out = []
    prev = None
    for k, d in enumerate(db):
        nxt = db[k + 1] if k + 1 < len(db) else d + 1.6
        evs = [e for e in D.events if d - 0.1 <= e["t"] < nxt - 0.1 and e["s"] != "voice_off" and e["t"] <= S1]
        ev = None
        vo = [e for e in evs if e["s"] == "voice_on"]
        if vo:
            ev = vo[0]
        elif evs:
            e = max(evs, key=lambda e: (e.get("sal") or 0))
            if (e.get("sal") or 0) >= 0.3:
                ev = e
        if ev is None:
            out.append(dict(d=d, ev=None, y=prev["y"] if prev else None, z=prev["z"] if prev else None, lab="(hold)"))
        else:
            if ev["s"] == "voice_on":
                p = min(D.phrases, key=lambda p: abs(p["start"] - ev["t"]))
                # 视线落在 f0 与审计 ▼ 的中点：人声出现处 + 「这里剪辑没回应」（A5）同框
                y, z, lab = (y_f0(p["f0_med"]) + Y_MARK) / 2, Z["voice"], f"voice_on {ev['t']:.2f} · f0 {p['f0_med']:.0f} Hz"
            elif ev["s"] == "absence":
                y, z, lab = Y_ABS + H_ABS / 2, Z["expect"], f"absence {ev['t']:.2f}"
            else:
                n, y = stem_hit(D, ev["t"])
                z, lab = Z[n], f"{ev['s']} {ev['t']:.2f} → {n}"
            out.append(dict(d=d, ev=ev, y=y, z=z, lab=lab + f" · sal {(ev.get('sal') or 0):.2f}"))
        prev = out[-1]
    first = next((o for o in out if o["y"] is not None), dict(y=Y_SENSE, z=Z["sense"]))
    for o in out:
        if o["y"] is None:
            o["y"], o["z"] = first["y"], first["z"]
    return out


def camera(D, T):
    """cams[f] = (cx, cy, cz)。推拉：cz 线性（单调）。平移：小节首拍提前 LEAD 帧起步的缓动台阶（不归零）+ x 慢漂。
    返回 cams、moves（每次平移：起步帧、首拍帧、目标）"""
    bt = bar_targets(D)
    t_end = S0 + (T - 1) / FPS
    cz = np.array([CZ0 + (CZ1 - CZ0) * f / (T - 1) for f in range(T)])

    def cz_at(t):
        return CZ0 + (CZ1 - CZ0) * (t - S0) / (t_end - S0)

    def x_need(d, t):
        s = scale_at(Z["voice"], cz_at(t))
        return X(d) + (0.5 - FRAME_X) * W / s - DRIFT * (t - S0)

    def y_need(y, z, t):
        s = scale_at(z, cz_at(t))
        lo = Y_TOP + 0.5 * H / s
        hi = Y_BOT - 0.5 * H / s
        return float(np.clip(y + (0.5 - FRAME_Y) * H / s, lo, hi))

    # 第一小节（段首之前已落下）= 起始构图
    k0 = max(i for i, b in enumerate(bt) if b["d"] <= S0 + 0.1)
    bx = x_need(bt[k0]["d"], S0)
    by = y_need(bt[k0]["y"], bt[k0]["z"], S0)
    moves = []
    px, py = bx, by
    for b in bt[k0 + 1:]:
        fe = int(round((b["d"] - S0) * FPS))
        if fe - LEAD >= T - 2:
            break
        ta = b["d"] + PAN_D
        tx = x_need(b["d"], ta)
        ty = y_need(b["y"], b["z"], ta)
        dy = float(np.clip(ty - py, -VMAX, VMAX))
        moves.append(dict(f_start=fe - LEAD, f_beat=fe, d=b["d"], dx=tx - px, dy=dy, lab=b["lab"]))
        px, py = tx, py + dy
    cams = np.zeros((T, 3))
    for f in range(T):
        t = S0 + f / FPS
        cx = bx + DRIFT * (t - S0)
        cy = by + 6.0 * math.sin(0.9 * (t - S0))           # 竖向极轻的漂（不归零前也不停）
        for m in moves:
            e = ease((f - m["f_start"]) / (PAN_D * FPS))
            cx += m["dx"] * e
            cy += m["dy"] * e
        cams[f] = (cx, cy, cz[f])
    return cams, moves, bt


def speeds(cams):
    """每帧屏幕速度（px/帧）：平移分量（z = 0 层，x/y）与推拉分量（画面角点的径向位移），总 = 参考点中位"""
    T = len(cams)
    pan, push, tot = np.zeros(T), np.zeros(T), np.zeros(T)
    gx, gy = np.meshgrid(np.linspace(0.1, 0.9, 7) * W, np.linspace(0.1, 0.9, 5) * H)
    for f in range(1, T):
        (pcx, pcy, pcz), (cx, cy, cz) = cams[f - 1], cams[f]
        d = []
        for z in (0.0, 300.0, 620.0):                     # 上一帧画面上的参考点（各层）→ 这一帧在哪
            sp, sc = FOC / (z - pcz), FOC / (z - cz)
            Xw, Yw = pcx + (gx - W / 2) / sp, pcy + (gy - H / 2) / sp
            d.append(np.hypot(W / 2 + sc * (Xw - cx) - gx, H / 2 + sc * (Yw - cy) - gy).ravel())
        tot[f] = float(np.median(np.concatenate(d)))
        s1 = FOC / (0 - cz)
        pan[f] = math.hypot(cx - pcx, cy - pcy) * s1
        push[f] = (s1 / (FOC / (0 - pcz)) - 1) * math.hypot(W / 2, H / 2)
    return pan, push, tot


def pan_starts(pan, thr=None):
    """平移起步帧：速度从「慢漂」跃到「移动」的那一帧（阈值 = 慢漂速度的 2 倍）"""
    base = float(np.median(pan[1:]))
    thr = thr or max(2.0 * DRIFT / FPS * 1.0, 1.0)
    out = []
    for f in range(2, len(pan)):
        if pan[f] > thr and pan[f - 1] <= thr:
            out.append(f)
    return out, base


def speed_plot(pan, push, tot, moves, D, path, title):
    T = len(pan)
    Wp, Hp, m = 1800, 520, 60
    im = Image.new("RGB", (Wp, Hp), BG)
    d = ImageDraw.Draw(im)
    fm = ImageFont.truetype(FMONO, 13); fc = ImageFont.truetype(FCN, 15); fs = ImageFont.truetype(FCN, 13)
    Xp = lambda f: m + f / (T - 1) * (Wp - 2 * m)          # noqa: E731
    top1, h1 = 70, 250
    top2, h2 = 370, 90
    mx = max(pan.max(), tot.max()) * 1.05
    for b in D.downbeats:
        f = (b - S0) * FPS
        if 0 <= f < T:
            d.line([(Xp(f), top1), (Xp(f), top2 + h2)], fill=GRID)
            d.text((Xp(f) + 2, top1 - 16), f"{b:.2f}", fill=AXLAB, font=fm)
    for b in D.beats:
        f = (b - S0) * FPS
        if 0 <= f < T and not np.any(np.abs(D.downbeats - b) < 0.02):
            d.line([(Xp(f), top1 + h1 - 8), (Xp(f), top1 + h1)], fill=GRID)
    starts, _ = pan_starts(pan)
    for f in starts:
        d.polygon([(Xp(f) - 5, top1 + h1 + 12), (Xp(f) + 5, top1 + h1 + 12), (Xp(f), top1 + h1 + 2)], fill=RED)
    for g, col, wd in ((tot, WAVE, 1), (pan, INK, 2)):
        d.line([(Xp(f), top1 + h1 - v / mx * h1) for f, v in enumerate(g)], fill=col, width=wd)
    for v in (10, 20, 30, 40):
        if v < mx:
            yy = top1 + h1 - v / mx * h1
            d.line([(m, yy), (Wp - m, yy)], fill=GRID_DIM)
            d.text((m - 26, yy - 7), str(v), fill=AXLAB, font=fm)
    pm = push[1:].max() * 1.3
    d.line([(m, top2 + h2), (Wp - m, top2 + h2)], fill=SEP)
    d.line([(Xp(f), top2 + h2 - v / pm * h2) for f, v in enumerate(push) if f], fill=RED_BAR, width=2)
    d.text((m, top2 - 18), f"推拉（画面角点的径向速度 px/帧，> 0 = 推近）：min {push[1:].min():.3f}  max {push[1:].max():.3f}  —— 整段只推近，不来回",
           fill=LAB, font=fs)
    for sec in range(int(math.ceil(S0)), int(S0 + T / FPS) + 1):
        x = Xp((sec - S0) * FPS)
        d.line([(x, top2 + h2), (x, top2 + h2 + 5)], fill=TICKLINE)
        d.text((x + 2, top2 + h2 + 7), mmss(sec, 0), fill=TICK, font=fm)
    d.text((m, 10), title, fill=INK, font=fc)
    d.text((m, 34), f"白 = 平移速度 px/帧（max {pan.max():.1f}）   灰 = 总屏幕速度（参考点中位，min {tot[1:].min():.2f}  max {tot.max():.1f}）"
                    f"   竖线 = 小节首拍   红 ▲ = 平移起步（{len(starts)} 次，全部在首拍 −{LEAD} 帧）", fill=LAB, font=fs)
    im.save(path)


# ───────────────────────── 渲染 ─────────────────────────
class View:
    def __init__(self, cam):
        self.cx, self.cy, self.cz = cam

    def s(self, z):
        return FOC / (z - self.cz)

    def px(self, Xw, z):
        return (W / 2 + self.s(z) * (np.asarray(Xw) - self.cx)) * SS

    def py(self, Yw, z):
        return (H / 2 + self.s(z) * (np.asarray(Yw) - self.cy)) * SS

    def x_to_t(self, sx_px, z):
        """屏幕 x（px，非 SS）→ 播放时间"""
        return S0 + (self.cx + (sx_px - W / 2) / self.s(z)) / PPS

    def lw(self, base, z):
        return max(1, int(round(base * SS * self.s(z) * (0.55 + 0.45 * fog(z)))))

    def fs(self, size, z):
        return size * self.s(z) * (0.7 + 0.3 * fog(z))


def poly(dr, xs, ys, col, w):
    pts = list(zip(np.asarray(xs).tolist(), np.asarray(ys).tolist()))
    if len(pts) > 1:
        dr.line(pts, fill=col, width=w, joint="curve" if w > 2 else None)


def dashed(dr, xs, ys, col, w, on=7, off=5):
    pts = np.stack([xs, ys], 1)
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    per = (on + off) * SS
    cur = []
    for p, s in zip(pts, seg):
        if (s % per) < on * SS:
            cur.append(tuple(p))
        else:
            if len(cur) > 1:
                dr.line(cur, fill=col, width=w)
            cur = []
    if len(cur) > 1:
        dr.line(cur, fill=col, width=w)


def render_frame(D, cam, f, wig, info, opts=None):
    t = S0 + f / FPS
    v = View(cam)
    im = Image.new("RGB", (W * SS, H * SS), BG)
    dr = ImageDraw.Draw(im)
    kf = kframe(t)

    def vis_t(z):
        return v.x_to_t(MARGIN - 4, z), v.x_to_t(W + 4, z)

    def beat_grid(z, y0, y1):
        ta, tb = vis_t(z)
        f_ = fog(z)
        for b in D.beats[(D.beats >= ta) & (D.beats <= tb)]:
            down = np.any(np.abs(D.downbeats - b) < 0.02)
            c = mix(GRID if down else GRID_DIM, f_)
            xx = float(v.px(X(b), z))
            dr.line([(xx, float(v.py(y0, z))), (xx, float(v.py(y1, z)))], fill=c, width=1)

    def playhead(z, y0, y1):
        xx = float(v.px(X(t), z))
        dr.line([(xx, float(v.py(y0, z))), (xx, float(v.py(y1, z)))], fill=mix(INK, 0.32 * fog(z)), width=max(1, SS // 2 + 1))

    def audit_lines(z, y0, y1):
        for a in D.audit:
            if a["t"] <= t and a["id"].startswith("A"):
                xx = float(v.px(X(a["t"]), z))
                dr.line([(xx, float(v.py(y0, z))), (xx, float(v.py(y1, z)))], fill=mix(RED_DK, fog(z)), width=v.lw(1.0, z))

    def lane_sep(z, y):
        ta, tb = vis_t(z)
        dr.line([(float(v.px(X(ta), z)), float(v.py(y, z))), (float(v.px(X(tb), z)), float(v.py(y, z)))], fill=mix(SEP, fog(z)), width=1)

    def tcols(z, step):
        ta, tb = vis_t(z)
        tt = np.arange(max(ta, 0.0), min(tb, t) + 1e-9, step)
        return tt

    labels = []                                    # (y_px, text, col, size) 左栏

    # ── 分轨频谱（后 → 前） ──
    def ridges(name, z, ys_base, sp, lev, wigs, amp, width_base):
        f_ = fog(z)
        ta, tb = vis_t(z)
        i0, i1 = max(0, int(ta / HOP)), min(len(lev) - 1, int(t / HOP))
        if i1 <= i0 + 1:
            return
        idx = np.arange(i0, i1 + 1)
        tt = idx * HOP
        xs = v.px(X(tt), z)
        age = t - tt
        head = np.where(age < HEAD_T, np.exp(-age / HEAD_TAU), 0.0)
        lw = v.lw(width_base, z)
        order = np.argsort(ys_base)                 # 上（高频）先画，下面的山脊挡住上面的
        ybot_px = None
        for r in order:
            L = lev[idx, r]
            yw = ys_base[r] - L * amp * sp + wigs[r] * sp * head
            ys = v.py(yw, z)
            if ybot_px is None:
                ybot_px = float(v.py(ys_base.max() + sp, z))
            yb = float(v.py(ys_base[r] + 0.6 * sp, z))
            pts = list(zip(xs.tolist(), ys.tolist()))
            dr.polygon(pts + [(float(xs[-1]), yb), (float(xs[0]), yb)], fill=BG)
            # 分块上色：电平 → 灰到白；头部被踢 → 白
            hit = abs(wigs[r]) / WIG_MAX_SP
            n = len(xs)
            for a in range(0, n - 1, 6):
                b = min(n, a + 7)
                lv = float(L[a:b].mean())
                hh = float(head[a:b].mean()) * hit
                c = WAVE if lv < 0.35 else tuple(int(WAVE[i] + (INK[i] - WAVE[i]) * min(1, (lv - 0.35) / 0.4)) for i in range(3))
                c = tuple(int(c[i] + (INK[i] - c[i]) * min(1, hh * 1.6)) for i in range(3))
                a_ = 0.42 + 0.58 * min(1.0, lv * 1.8 + hh)
                dr.line(pts[a:b], fill=mix(c, a_ * f_), width=lw)

    wk = min(max(kf, 0), len(wig["drums"]) - 1)
    for name, y0 in reversed(STEM_LANES):
        z = Z[name]
        beat_grid(z, y0 - 6, y0 + H_STEM + 6)
        ys_base = y0 + H_STEM - (np.arange(NR) + 0.5) * SP_STEM
        ridges(name, z, ys_base, SP_STEM, D.lev[name], wig[name][wk], 3.2, 0.8)
        audit_lines(z, y0 - 6, y0 + H_STEM + 6)
        playhead(z, y0 - 6, y0 + H_STEM + 6)
        lane_sep(z, y0 + H_STEM + 10)
        labels.append((float(v.py(y0 + H_STEM / 2, z)), name, mix(LAB, fog(z)), v.fs(14, z)))
        for hz, lab in ((100, "100"), (1000, "1k"), (5000, "5k")):
            yy = y0 + H_STEM - math.log(hz / F_LO) / math.log(F_HI / F_LO) * H_STEM
            labels.append((float(v.py(yy, z)), lab, mix(AXLAB, fog(z)), v.fs(11, z), "ax"))
    # 底部时间轴（drums 层）
    zt = Z["drums"]; yax = STEM_LANES[-1][1] + H_STEM + 14
    ta, tb = vis_t(zt)
    for sec in range(int(math.ceil(ta)), int(tb) + 1):
        xx = float(v.px(X(sec), zt))
        dr.line([(xx, float(v.py(yax, zt))), (xx, float(v.py(yax + 6, zt)))], fill=mix(TICKLINE, fog(zt)), width=1)
        dr.text((xx + 2 * SS, float(v.py(yax + 8, zt))), mmss(sec, 0), font=font(v.fs(13, zt), False), fill=mix(TICK, fog(zt)))

    # ── 感官：四条曲线 ──
    z = Z["sense"]; f_ = fog(z)
    beat_grid(z, Y_SENSE - 6, Y_SENSE + 4 * H_SL)
    tt = tcols(z, 1 / D.fps)
    for j, c in enumerate(SENSE_ORDER):
        name, col, wd, dash = SENSE_STYLE[c]
        y0 = Y_SENSE + j * H_SL
        lane_sep(z, y0 + H_SL)
        if len(tt) > 1:
            vals = np.array([D.chv(c, x) for x in tt])
            # 头部抖动：12 个振子沿最近 0.6 s 排开，插值
            wv = wig["sense"][wk, j * N_SENSE_OSC:(j + 1) * N_SENSE_OSC]
            pos = t - np.linspace(0, HEAD_T, N_SENSE_OSC)
            age = t - tt
            dsp = np.where(age < HEAD_T, np.interp(-tt, -pos, wv) * np.exp(-age / HEAD_TAU), 0.0)
            yw = y0 + H_SL - 6 - vals * (H_SL - 12) + dsp * 0.16 * H_SL
            xs, ys = v.px(X(tt), z), v.py(yw, z)
            if dash:
                dashed(dr, xs, ys, mix(col, f_), v.lw(wd, z), *dash)
            else:
                poly(dr, xs, ys, mix(col, f_), v.lw(wd, z))
            labels.append((float(v.py(y0 + H_SL / 2, z)), name, mix(LAB, f_), v.fs(14, z)))
            # 数值（播放头处）
            dr.text((float(xs[-1]) + 5 * SS, float(ys[-1]) - 8 * SS * v.s(z)), f"{vals[-1]:.2f}", font=font(v.fs(12, z), False), fill=mix(col, f_ * 0.9))
    audit_lines(z, Y_SENSE - 6, Y_SENSE + 4 * H_SL)
    playhead(z, Y_SENSE - 6, Y_SENSE + 4 * H_SL)

    # ── 期待 ──
    z = Z["expect"]; f_ = fog(z)
    beat_grid(z, Y_BITS, Y_ABS + H_ABS)
    ta, tb = vis_t(z)
    for s in D.segs:                                       # 段落色块（只画到现在）
        a, b = max(s["t0"], ta), min(s["t1"], t)
        if a >= b:
            continue
        c = EXP_COL.get(s.get("exp", "无"), (50, 52, 56))
        x0, x1 = float(v.px(X(a), z)), float(v.px(X(b), z))
        dr.rectangle([x0, float(v.py(Y_SEG, z)), max(x0, x1 - 1), float(v.py(Y_SEG + H_SEG, z))], fill=mix(c, f_))
        if True:
            xl = max(float(v.px(X(s["t0"]), z)), MARGIN * SS) + 3 * SS
            if x1 - xl > 40 * SS:
                dr.text((xl, float(v.py(Y_SEG + 3, z))), f"{s['id']} {s['name']}", font=font(v.fs(14, z)), fill=mix((235, 236, 238), f_))
                if "exp" in s and x1 - xl > 90 * SS:
                    dr.text((xl, float(v.py(Y_SEG + 26, z))), f"等:{s['exp']} 忍:{s['tol']}", font=font(v.fs(13, z)), fill=mix(SUB, f_))
    labels.append((float(v.py(Y_SEG + H_SEG / 2, z)), "段落", mix(LAB, f_), v.fs(14, z)))
    for bb in (2, 4, 8):                                    # 意外度
        yy = Y_BITS + H_BITS - bb / CAP * H_BITS
        dr.line([(float(v.px(X(ta), z)), float(v.py(yy, z))), (float(v.px(X(tb), z)), float(v.py(yy, z)))], fill=mix(GRID, f_), width=1)
        labels.append((float(v.py(yy, z)), str(bb), mix(AXLAB, f_), v.fs(11, z), "ax"))
    labels.append((float(v.py(Y_BITS + H_BITS / 2, z)), "意外度", mix(LAB, f_), v.fs(14, z)))
    for e in D.events:
        if not (ta <= e["t"] <= t):
            continue
        xx = float(v.px(X(e["t"]), z))
        hh = min(1.0, e["bits"] / CAP) * H_BITS
        col = STREAM_COL[e["s"]]
        dr.line([(xx, float(v.py(Y_BITS + H_BITS, z))), (xx, float(v.py(Y_BITS + H_BITS - hh, z)))], fill=mix(col, f_),
                width=v.lw(2.0 if e["s"] != "onset" else 1.0, z))
        if "no" in e:                                       # 编号三角（显著度 ≥ 1）
            yt = float(v.py(Y_BITS + H_BITS - hh - 4, z))
            sz = 5 * SS * v.s(z)
            c2 = RED if (e.get("sal") or 0) >= 2 else SUB
            dr.polygon([(xx - sz, yt - 2 * sz), (xx + sz, yt - 2 * sz), (xx, yt)], fill=mix(c2, f_))
            dr.text((xx + sz + 2 * SS, yt - 2 * sz - 6 * SS), e["no"], font=font(v.fs(12, z), False), fill=mix(c2, f_))
    lane_sep(z, Y_BITS + H_BITS + 4)
    k0, k1 = max(0, int(ta * D.fps)), min(len(D.absence), int(t * D.fps) + 1)   # absence 紫线
    if k1 - k0 > 1:
        kk = np.arange(k0, k1)
        poly(dr, v.px(X(kk / D.fps), z), v.py(Y_ABS + H_ABS - np.minimum(1, D.absence[kk] / CAP) * H_ABS, z),
             mix(STREAM_COL["absence"], f_), v.lw(1.0, z))
    yy = Y_ABS + H_ABS - ABS_ON / CAP * H_ABS
    dr.line([(float(v.px(X(ta), z)), float(v.py(yy, z))), (float(v.px(X(tb), z)), float(v.py(yy, z)))], fill=mix(ABS_LINE, f_), width=1)
    labels.append((float(v.py(Y_ABS + H_ABS / 2, z)), "absence", mix(LAB, f_), v.fs(14, z)))
    for sec in range(int(math.ceil(ta)), int(tb) + 1):     # 时间轴
        xx = float(v.px(X(sec), z))
        dr.line([(xx, float(v.py(Y_TAX, z))), (xx, float(v.py(Y_TAX + 6, z)))], fill=mix(TICKLINE, f_), width=1)
        dr.text((xx + 2 * SS, float(v.py(Y_TAX + 8, z))), mmss(sec, 0), font=font(v.fs(13, z), False), fill=mix(TICK, f_))
    for a in D.audit:                                       # 审计 A 点：红 ▼ + 编号 + 时间
        if not (a["t"] <= t and ta <= a["t"]):
            continue
        xx = float(v.px(X(a["t"]), z))
        col = RED if a["id"].startswith("A") else (255, 160, 40)
        yt = float(v.py(Y_SEG - 2, z)); sz = 6 * SS * v.s(z)
        dr.polygon([(xx - sz, yt - 2 * sz), (xx + sz, yt - 2 * sz), (xx, yt)], fill=mix(col, f_))
        dr.line([(xx, float(v.py(Y_MARK + 2, z))), (xx, yt - 2 * sz)], fill=mix(col, f_), width=v.lw(1, z))
        dr.text((xx + 3 * SS, float(v.py(Y_MARK - 4, z))), f"{a['id']} {mmss(a['t'])}", font=font(v.fs(14, z)), fill=mix(col, f_))
    audit_lines(z, Y_SEG + H_SEG, Y_ABS + H_ABS)
    playhead(z, Y_SEG, Y_ABS + H_ABS)

    # ── vocals 频谱（f0 泳道正后方，共用对数轴） ──
    z = Z["vspec"]
    beat_grid(z, Y_F0, Y_F0 + H_F0)
    sp_v = float(np.median(np.abs(np.diff(D.vband_y))))
    ridges("vocals", z, D.vband_y, sp_v, D.lev["vocals"] * 0.8, wig["vocals"][wk], 2.0, 0.7)
    labels.append((float(v.py(Y_F0 + H_F0 - 12, z)), "vocals 频谱", mix(LAB, fog(z) * 0.8), v.fs(12, z)))

    # ── 人声 ──
    z = Z["voice"]; f_ = 1.0
    ta, tb = vis_t(z)
    beat_grid(z, Y_WAVE, Y_LOUD + H_LOUD)
    beat_grid(z, Y_PV, Y_PV + H_PV)
    for s0_, s1_ in D.vsections:                            # 红条 = 人声段
        a, b = max(s0_, ta), min(s1_, t)
        if a < b:
            dr.rectangle([float(v.px(X(a), z)), float(v.py(Y_VSEC, z)), float(v.px(X(b), z)), float(v.py(Y_VSEC + 8, z))], fill=RED_BAR)
    for p in D.phrases:                                     # 乐句底色（波形 / 响度 / 有声泳道）+ f0 泳道的乐句括号
        a, b = max(p["start"], ta), min(p["end"], t)
        if a >= b:
            continue
        x0, x1 = float(v.px(X(a), z)), float(v.px(X(b), z))
        for ya, yb in ((Y_WAVE, Y_LOUD + H_LOUD), (Y_PV, Y_PV + H_PV)):
            dr.rectangle([x0, float(v.py(ya, z)), x1, float(v.py(yb, z))], fill=PHRASE)
        ylo, yhi = y_f0(p["f0_lo"]) + 14, y_f0(p["f0_hi"]) - 14
        bw = v.lw(1.3, z); tk = 6 * SS * v.s(z)
        xa, xb = float(v.px(X(p["start"]), z)), float(v.px(X(min(p["end"], t)), z))
        Ya, Yb = float(v.py(yhi, z)), float(v.py(ylo, z))
        dr.line([(xa + tk, Ya), (xa, Ya), (xa, Yb), (xa + tk, Yb)], fill=INK, width=bw)
        if p["end"] <= t:
            dr.line([(xb - tk, Ya), (xb, Ya), (xb, Yb), (xb - tk, Yb)], fill=INK, width=bw)
        dr.text((xa + 3 * SS, Ya - 18 * SS * v.s(z)), f"phrase {p['start']:.2f}  f0 {p['f0_med']:.0f} Hz", font=font(v.fs(13, z), False), fill=SUB)
    # 波形
    i0 = max(0, int((ta - D.wave_t0) * PPS)); i1 = min(len(D.wave), int((t - D.wave_t0) * PPS))
    if i1 > i0:
        ii = np.arange(i0, i1)
        xs = v.px(X(D.wave_t0 + ii / PPS), z)
        yc = Y_WAVE + H_WAVE / 2
        amp = D.wave[ii] * H_WAVE / 2
        yt_, yb_ = v.py(yc - amp, z), v.py(yc + amp, z)
        step = max(1, int(1 / (v.s(z) * SS) * 2))
        for k in range(0, len(ii), step):
            dr.line([(float(xs[k]), float(yt_[k])), (float(xs[k]), float(yb_[k]))], fill=WAVE, width=max(1, SS))
        poly(dr, xs, v.py(np.full(len(ii), yc), z), WAVE, 1)
    labels.append((float(v.py(Y_WAVE + H_WAVE / 2, z)), "wave", LAB, v.fs(14, z)))
    tt = tcols(z, 1 / D.fps)
    kk = np.clip((tt * D.fps).astype(int), 0, len(D.loud) - 1)
    if len(tt) > 1:                                         # 响度
        poly(dr, v.px(X(tt), z), v.py(Y_LOUD + H_LOUD - D.loud[kk] * H_LOUD, z), INK, v.lw(1.0, z))
    labels.append((float(v.py(Y_LOUD + H_LOUD / 2, z)), "loudness", LAB, v.fs(14, z)))
    lane_sep(z, Y_LOUD + H_LOUD + 5)
    for midi, lab in ((48, "C3"), (60, "C4"), (72, "C5"), (84, "C6")):   # f0 音名格
        hz = 440 * 2 ** ((midi - 69) / 12)
        yy = y_f0(hz)
        dr.line([(float(v.px(X(ta), z)), float(v.py(yy, z))), (float(v.px(X(tb), z)), float(v.py(yy, z)))], fill=NOTEGRID, width=1)
        labels.append((float(v.py(yy, z)), lab, NOTELAB, v.fs(12, z), "ax"))
    labels.append((float(v.py(Y_F0 + H_F0 / 2, z)), "f0", LAB, v.fs(14, z)))
    if len(tt) > 1:                                         # f0：黄 = 稳，红 = 颤音（线段连着有声的相邻帧）
        f0v = D.f0[kk]
        ok = np.isfinite(f0v)
        xs = v.px(X(tt), z)
        ys = np.array([float(v.py(y_f0(h), z)) if np.isfinite(h) else np.nan for h in f0v])
        r = 2.2 * SS * v.s(z)
        for i in np.where(ok)[0]:
            vb = D.vib[kk[i]]
            c = (255, 42 + int(150 * (1 - vb)), 52 + int(150 * (1 - vb))) if vb > 0 else F0Y
            if i > 0 and ok[i - 1]:
                dr.line([(float(xs[i - 1]), ys[i - 1]), (float(xs[i]), ys[i])], fill=c, width=v.lw(1.4, z))
            dr.ellipse([float(xs[i]) - r, ys[i] - r, float(xs[i]) + r, ys[i] + r], fill=c)
        if ok[-1]:
            dr.text((float(xs[-1]) + 8 * SS, ys[-1] - 9 * SS), f"f0 {f0v[-1]:.1f} Hz", font=font(v.fs(13, z), False), fill=INK)
    lane_sep(z, Y_F0 + H_F0 + 5)
    if len(tt) > 1:                                         # 有声概率 / 颤音
        xs = v.px(X(tt), z)
        poly(dr, xs, v.py(Y_PV + H_PV - D.voiced[kk] * H_PV, z), VBLUE, v.lw(1.0, z))
        poly(dr, xs, v.py(Y_PV + H_PV - D.vib[kk] * H_PV, z), RED, v.lw(1.0, z))
    labels.append((float(v.py(Y_PV + H_PV / 2, z)), "voiced/vib", LAB, v.fs(14, z)))
    for b0, b1 in D.breaths:
        if ta <= (b0 + b1) / 2 <= t:
            xx = float(v.px(X((b0 + b1) / 2), z))
            dr.line([(xx, float(v.py(Y_WAVE, z))), (xx, float(v.py(Y_PV + H_PV, z)))], fill=RED, width=1)
    audit_lines(z, Y_VSEC, Y_PV + H_PV)
    playhead(z, Y_VSEC, Y_PV + H_PV)

    # ── 左栏（参考图的标签列）──
    dr.rectangle([0, 0, MARGIN * SS, H * SS], fill=BG)
    for it in labels:
        yy, s_, col, sz = it[:4]
        if -20 * SS < yy < (H + 20) * SS:
            if len(it) > 4:
                dr.text(((MARGIN - 6) * SS, yy), s_, font=font(sz, False), fill=col, anchor="rm")
            else:
                dr.text((8 * SS, yy), s_, font=font(sz), fill=col, anchor="lm")
    # ── HUD（参考图的标题行） ──
    seg = next((s for s in D.segs if s["t0"] <= t < s["t1"]), None)
    K = info[min(max(kf, 0), len(info) - 1), 0]
    dr.rectangle([0, 0, W * SS, 46 * SS], fill=mix(BG, 1.0))                # 标题带（参考图的标题行），数据从它下面过
    dr.line([(0, 46 * SS), (W * SS, 46 * SS)], fill=SEP, width=1)
    dr.text((10 * SS, 8 * SS), f"MV_Cant_Kill_Us  感官空间  {mmss(t)}" + (f"   {seg['id']} {seg['name']}" if seg else ""), font=font(16), fill=INK)
    dr.text((10 * SS, 30 * SS), f"前 → 后：voice 0 · vocals 频谱 80 · 期待 140 · 感官 260 · other 380 · bass 500 · drums 620    "
                                f"抖动 = Kuramoto  K {K:.2f}  （每帧相位踢 N(0, 0.35·θ·通量)）    倍率 {FOC / (0 - cam[2]):.3f}",
            font=font(12), fill=(140, 146, 152))
    return im.resize((W, H), Image.LANCZOS)


def contact(frames, path, cols=4, tw=640):
    th = int(tw * H / W)
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * (th + 22)), BG)
    d = ImageDraw.Draw(sheet)
    fnt = ImageFont.truetype(FMONO, 15)
    for k, (lab, im) in enumerate(frames):
        x, y = (k % cols) * tw, (k // cols) * (th + 22)
        sheet.paste(im.resize((tw, th), Image.LANCZOS), (x, y + 22))
        d.text((x + 6, y + 3), lab, fill=(200, 200, 200), font=fnt)
    sheet.save(path, quality=88)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="25_感官空间_正面_v1")
    ap.add_argument("--still", default=None, help="逗号分隔的播放时间，只出静帧")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    T = int(round((S1 - S0) * FPS))
    print("数据 …", flush=True)
    D = Data()
    print("Kuramoto …", flush=True)
    wig, info, ref = kuramoto(D)
    cams, moves, bt = camera(D, T)
    if a.still:
        ims = []
        for x in a.still.split(","):
            f = int(round((float(x) - S0) * FPS))
            ims.append((f"{S0 + f / FPS:.2f}s f{f}", render_frame(D, cams[f], f, wig, info)))
        if len(ims) == 1:
            ims[0][1].save(a.out, quality=92)
        else:
            contact(ims, a.out, cols=2, tw=960)
        print("->", a.out)
        return
    vid = OUT / f"{a.name}.mp4"
    if vid.exists():
        sys.exit(f"已存在，不覆盖：{vid}")
    cmd = [ffmpeg(), "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-ss", f"{S0:.3f}", "-t", f"{T / FPS:.3f}", "-i", str(MUSIC), "-map", "0:v", "-map", "1:a",
           "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", str(vid)]
    w = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    picks = set(np.linspace(0, T - 1, 20).round().astype(int).tolist())
    sheet = []
    for f in range(T):
        im = render_frame(D, cams[f], f, wig, info)
        w.stdin.write(np.asarray(im).tobytes())
        if f in picks:
            sheet.append((f"{S0 + f / FPS:.2f}s  f{f}", im))
        if f % 48 == 0:
            print(f"  frame {f}/{T}", flush=True)
    w.stdin.close()
    w.wait()
    contact(sheet, vid.with_name(vid.stem + "_接触表.jpg"))
    pan, push, tot = speeds(cams)
    starts, _ = pan_starts(pan)
    speed_plot(pan, push, tot, moves, D, vid.with_name(vid.stem + "_镜头速度.png"), vid.stem)
    mb = vid.stat().st_size / 1e6
    if mb > 30:
        pv = vid.with_name(vid.stem + "_预览.mp4")
        subprocess.run([ffmpeg(), "-loglevel", "error", "-n", "-i", str(vid), "-c:v", "libx264", "-crf", "28", "-preset", "slow",
                        "-vf", "scale=1280:-2", "-c:a", "aac", "-b:a", "128k", str(pv)], check=True)
    vid.with_suffix(".json").write_text(json.dumps(dict(
        tool="tools/sense_space_v3.py", seg=[S0, S1], fps=FPS, size_mb=round(mb, 1),
        palette="感官_期待审计_v4.jpg + 人声_条带_v2.jpg（sense_expect / sense_voice 原值）",
        layers_z=Z, pps=PPS, hfov_deg=math.degrees(HFOV), zoom=[ZOOM0, ZOOM1],
        tremor=dict(law="opening.py Kuramoto.step", K0=K0, K_t0=K_T0, K_per_beat=K_BEAT,
                    kick="N(0, 0.35*threat*clip(flux/p95,0,1)) per frame; flux = positive diff of level normalised to each stem's p50..p99.7",
                    flux_p95=ref, tau_eff=TAU_EFF, tau_slow=TAU_SLOW, gain_spacing_per_hz=WIG_SP_PER_HZ, dead_spacing=WIG_DEAD_SP, clip_spacing=WIG_MAX_SP, head_s=HEAD_T),
        camera=dict(push="cz linear, monotonic", lead_frames=LEAD, ease=f"1-(1-u)^{EASE_P}", pan_s=PAN_D, drift=DRIFT, vmax=VMAX,
                    moves=[dict(f_start=m["f_start"], f_downbeat=m["f_beat"], downbeat=m["d"], dx=round(m["dx"], 1), dy=round(m["dy"], 1), target=m["lab"]) for m in moves],
                    pan_starts_detected=starts),
        speed=dict(pan_max=round(float(pan.max()), 2), total_min=round(float(tot[1:].min()), 3), total_max=round(float(tot.max()), 2),
                   push_min=round(float(push[1:].min()), 4), push_max=round(float(push[1:].max()), 4)),
        cams=[[round(float(x), 2) for x in c] for c in cams]), ensure_ascii=False, indent=1), "utf-8")
    print(f"平移起步帧 {starts}；首拍帧 {[m['f_beat'] for m in moves]}")
    print(f"速度 px/帧：平移 max {pan.max():.1f}；总 min {tot[1:].min():.2f} max {tot.max():.1f}；推拉 {push[1:].min():.3f}…{push[1:].max():.3f}")
    print("->", vid, f"{mb:.1f} MB")


if __name__ == "__main__":
    main()
