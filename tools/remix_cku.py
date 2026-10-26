#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🎛Remix《Can't Kill Us》示波器 demo（作曲器选项 A 的第一个试验，👤2026-10-26 选定）。

窗口 = 歌曲播放时间 25.95–57.98（五个乐句，150 BPM，E 小调）= 情绪大图 01 回想尾 → 02 有什么要来了 → 03 吸气 → 04 炸碎。
故事（👤给的弧线）：一个光点 → 一个圆成为一个音 → 长成她的轮廓 → 碎掉。显影只有一次（43.54–51.72），其余画面全用本片已有样片。
声音 = Demucs 四轨按场景重排 / 滤波 / 静默 + 我们的一层「她的音」（显影层，示波器画的就是它）。

子命令（都按版本号另起文件，不覆盖）：
  audio     出混音（−14 LUFS / −1 dBTP）+ 显影层 96 kHz + 作曲.json + 原曲同段同 LUFS 的 A/B 文件 + 三段频谱对比图
  board     分镜图：张力曲线 + 剪辑密度 + 关键帧接触表（关键帧用真渲染）
  render    出片（1280×720，24 fps，和本片样片一致）
用法：python tools/remix_cku.py audio --ver v1 ; ... board --ver v1 ; ... render --ver v1
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import synthkit as sk  # noqa: E402

WORK = Path(r"C:/Users/Administrator/Desktop/Projekt_Visual/MV_Cant_Kill_Us")
OUT = WORK / "样片" / "Remix_示波器"
STEMS = WORK / "分轨"
ORIG = Path(r"C:/Users/Administrator/Desktop/Projekt_Visual/Assets/The Glitch Mob - Can't Kill Us [2020 Edit]_190189282.mp4")

T0, T1 = 25.95, 57.98          # 窗口（播放时间）
SR = 48000
RSR = 96000                    # 显影层采样率（示波器读原件；混音里用 48k 重采样版）
FPS = 24
W, H = 1280, 720
R0 = 43.0                      # 显影层文件的 t=0 对应的歌曲时间
E2 = 82.4069

# 小节线（感官.json 的 downbeats，150 BPM，一小节 1.6 s）
DB = [25.54, 27.14, 28.74, 30.74, 32.34, 33.96, 35.54, 37.16, 38.74, 40.34, 41.96, 43.54, 45.16, 46.76, 48.36,
      49.96, 51.56, 53.16, 54.76, 56.36, 57.98]
BEAT = 0.4

# 场景（作曲.json 的 scenes）：时间、名字、哪些分轨在响、她此刻
SCENES = [
    (25.95, "回想", "other + bass(−4 dB)，鼓拿掉", "碎片一片片亮起，她开始想起自己是谁"),
    (32.34, "回想·鼓回来", "鼓低通 300 Hz → 全开，bass 全开", "记忆越来越清楚"),
    (38.74, "有什么要来了", "四轨全开（原样）", "远处有一个背影，她想跟上去"),
    (43.54, "光点", "鼓 / bass 拿掉，other 低通 600 Hz −6 dB；我们：一个几乎静止的光点", "什么都退下去了，只剩一个点：我还在吗？"),
    (45.16, "一个音", "人声 +14 dB（她第一次出声），other 低通 600 → 3000 Hz；我们：她的音 E2 起振 = 圆", "第一次听见自己的声音"),
    (46.76, "长出轮廓", "同上；我们：圆按弧长形变成她的轮廓，音 E2 → G2 → B2 → E3 往上走（抬头）", "她在长出自己的形状"),
    (50.36, "前震", "同上；我们：轮廓发抖（颤音 8 → 16 Hz、颤幅加大）", "要坏了"),
    (51.16, "一拍全黑", "全部静音（硬切）；示波器上光束回到中心 = 光点", "憋住"),
    (51.56, "炸碎", "四轨全开 + 冲击；我们：轮廓喂进两个带饱和的反馈环：短环（7 / 11 ms，g = 1.3）0.2 秒内把轮廓撕碎；长环（0.3 s 回声，g 1.08 → 0.4，越过单位圆再拉回）当余波，音高往下滑 = 坠落", "被炸开，失重"),
]


# ====================================================================== 工具
def ramp(points, t):
    """分段线性曲线：points = [(t, v), ...]，t 是歌曲时间数组"""
    ts, vs = zip(*points)
    return np.interp(t, ts, vs)


def db2a(d):
    return 10 ** (np.asarray(d, float) / 20)


def load_stem(name):
    info = sf.info(str(STEMS / f"{name}.wav"))
    a, b = int((T0 - 0.5) * info.samplerate), int((T1 + 0.5) * info.samplerate)
    x, sr = sf.read(str(STEMS / f"{name}.wav"), start=a, stop=b, always_2d=True)
    y = resample_poly(x, 160, 147, axis=0) if sr == 44100 else x
    return y[:, :2]                                         # 采样 0 ↔ 歌曲时间 T0 − 0.5


def stem_time(n):
    return T0 - 0.5 + np.arange(n) / SR


def svf_st(x, cut, q=0.707):
    return np.stack([sk.svf(x[:, c], cut, q) for c in range(2)], 1)


def smooth_gain(g, ms=12):
    k = max(1, int(ms * SR / 1000))
    return np.convolve(g, np.ones(k) / k, "same")


# ====================================================================== 她的轮廓
def silhouette_path(n=2048):
    """她的正面剪影（生成/角色_她_正面_灰_pro_alpha.png）→ 外轮廓，从头顶开始顺时针，按弧长均匀取 n 点，归一到 y ∈ [−1, 1]"""
    m = np.asarray(Image.open(WORK / "生成" / "角色_她_正面_灰_pro_alpha.png").convert("L").resize((640, 360), Image.BILINEAR)) > 127
    m = np.pad(m, 2)                                        # 底边被画面裁掉 → 补零让轮廓沿裁切线闭合
    Hm, Wm = m.shape
    ys, xs = np.nonzero(m)
    top = np.argmin(ys)
    start = (ys[top], xs[top])
    # Moore 邻域追踪（顺时针）
    nb = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    pts = [start]
    cur, back = start, 6                                    # 从左边进来
    for _ in range(200000):
        found = False
        for k in range(8):
            d = (back + 1 + k) % 8
            ny, nx = cur[0] + nb[d][0], cur[1] + nb[d][1]
            if 0 <= ny < Hm and 0 <= nx < Wm and m[ny, nx]:
                back = (d + 4) % 8
                cur = (ny, nx)
                found = True
                break
        if not found or cur == start:
            break
        pts.append(cur)
    P = np.array(pts, float)[:, ::-1]                       # (x, y) 像素
    P[:, 1] = -P[:, 1]                                      # y 向上
    # 沿弧长平滑（头发的毛边）+ 均匀重采样
    k = 7
    Pp = np.concatenate([P[-k:], P, P[:k]])
    ker = np.exp(-0.5 * (np.arange(-k, k + 1) / 2.5) ** 2); ker /= ker.sum()
    P = np.stack([np.convolve(Pp[:, i], ker, "valid") for i in range(2)], 1)
    seg = np.linalg.norm(np.diff(np.vstack([P, P[:1]]), axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    u = np.linspace(0, s[-1], n, endpoint=False)
    Q = np.stack([np.interp(u, s, np.append(P[:, i], P[0, i])) for i in range(2)], 1)
    c = (Q.max(0) + Q.min(0)) / 2
    Q = (Q - c) / ((Q[:, 1].max() - Q[:, 1].min()) / 2)
    return Q


# ====================================================================== 显影层（我们的那一层）
def her_layer():
    """返回 (立体声 96 kHz，t=0 ↔ 歌曲时间 R0) 和事件（作曲.json 的 notes / reveals / macros）"""
    dur = T1 + 3.0 - R0
    n = int(round(dur * RSR))
    t = R0 + np.arange(n) / RSR
    # 音高：E2 → G2 → B2 → E3（每段起点 80 ms 滑音）
    notes = [(45.16, 48.36, 40), (48.36, 49.96, 43), (49.96, 50.76, 47), (50.76, 51.16, 52)]
    midi = np.full(n, 40.0)
    for a, b, mm in notes:
        midi[t >= a] = mm
    k_ = int(0.08 * RSR)
    midi = np.convolve(np.pad(midi, (k_, 0), mode="edge"), np.ones(k_) / k_, "valid")[:n]
    f = sk.midi_hz(midi)
    # 颤（前震）：50.36 起频率颤 8 → 16 Hz、幅度变大
    tr = np.clip((t - 50.36) / 0.8, 0, 1) * (t < 51.16)
    vib_rate = 8 + 8 * tr
    vib = np.sin(2 * np.pi * np.cumsum(vib_rate) / RSR)
    f = f * 2 ** (0.35 * tr * vib / 12)
    ph = np.concatenate([[0.0], np.cumsum(f[:-1])]) / RSR
    u = ph % 1.0
    # 形：圆（从头顶开始顺时针）→ 她的轮廓，按弧长逐点插值
    Q = silhouette_path()
    N = len(Q)
    idx = u * N
    i0 = np.floor(idx).astype(int) % N; fr = idx - np.floor(idx); i1 = (i0 + 1) % N
    sil = Q[i0] * (1 - fr[:, None]) + Q[i1] * fr[:, None]
    circ = np.stack([np.sin(2 * np.pi * u), np.cos(2 * np.pi * u)], 1)
    km = ramp([(46.76, 0.0), (48.36, 0.55), (48.9, 1.0)], t)
    km = km * km * (3 - 2 * km)
    xy = circ * (1 - km[:, None]) + sil * km[:, None]
    # 幅度：43.54 光点（极小的圆）→ 45.16 起振一小节到满 → 50.36 起颤幅 → 51.16 硬切静音
    amp = ramp([(43.54, 0.0), (43.6, 0.006), (45.16, 0.010), (46.76, 0.5), (51.16, 0.5)], t)
    amp = amp * (1 + 0.18 * tr * np.sin(2 * np.pi * np.cumsum(vib_rate * 1.5) / RSR))
    amp[t < 43.54] = 0
    amp[t >= 51.16] = 0
    out = xy * amp[:, None]
    # 炸碎：51.56 轮廓再响 0.1 s（喂进去的「火药」），进带饱和的反馈延迟（左右延迟不同 → 回声的图形错位旋转）
    burst = (t >= 51.56) & (t < 51.66)
    feed = xy * 0.5 * burst[:, None]
    g = ramp([(51.56, 1.08), (52.4, 1.08), (54.0, 0.75), (56.0, 0.4)], t)          # 极点：越过单位圆 → 拉回圆内
    wet = np.stack([sk.feedback_delay_sat(feed[:, 0], 0.300, g, sr=RSR, damp=3500, drive=2.2),
                    sk.feedback_delay_sat(feed[:, 1], 0.3013, g, sr=RSR, damp=3500, drive=2.2)], 1)
    # 坠落：碎片的音高一路往下滑（变速读：t' = ∫ rate）
    rate = ramp([(51.56, 1.0), (57.98, 0.55)], t)
    pos = np.clip(np.cumsum(rate) - np.argmax(t >= 51.56), 0, None) + np.argmax(t >= 51.56)
    pos = np.where(t >= 51.56, pos, np.arange(n))
    wet = np.stack([np.interp(pos, np.arange(n), wet[:, c]) for c in range(2)], 1)
    fade = ramp([(51.56, 1.0), (55.0, 0.7), (57.98, 0.0)], t)
    out = out + wet * 0.55 * fade[:, None]
    # 炸开的那 4 帧（51.56–51.72）：短延迟（7.1 / 11.3 ms）的反馈环，g = 1.3 → 环在零点几秒内自激、把轮廓撕成一团；环内 tanh 接住
    gf = ramp([(51.56, 1.3), (51.74, 1.3), (51.95, 0.0)], t)
    chaos = np.stack([sk.feedback_delay_sat(feed[:, 0], 0.0071, gf, sr=RSR, damp=9000, drive=1.6),
                      sk.feedback_delay_sat(feed[:, 1], 0.0113, gf, sr=RSR, damp=9000, drive=1.6)], 1)
    out = out + chaos * 0.9 * ramp([(51.56, 1.0), (51.9, 1.0), (52.3, 0.0)], t)[:, None]
    ev = dict(
        notes=[{"t": a, "dur": round(b - a, 3), "pitch": mm, "vel": 0.5, "layer": "her"} for a, b, mm in notes],
        reveals=[{"t0": 43.54, "t1": 51.72, "layer": "her", "shape": "光点 → 圆（E2）→ 她的轮廓（弧长插值）→ 光点 → 炸开"}],
        macros={"morph": {"fps": FPS, "t0": T0, "v": [round(float(v), 4) for v in ramp([(46.76, 0.0), (48.36, 0.55), (48.9, 1.0)], T0 + np.arange(int((T1 - T0) * FPS)) / FPS)]},
                "pole_g": {"fps": FPS, "t0": T0, "v": [round(float(v), 4) if T0 + i / FPS >= 51.56 else 0.0 for i, v in enumerate(ramp([(51.56, 1.08), (52.4, 1.08), (54.0, 0.75), (56.0, 0.4)], T0 + np.arange(int((T1 - T0) * FPS)) / FPS))]}},
    )
    return out, ev


# ====================================================================== 母带
def true_peak(x):
    return float(np.abs(resample_poly(x, 4, 1, axis=0)).max())


def master(mix):
    import pedalboard as pb
    import pyloudnorm as pyln
    board = pb.Pedalboard([pb.HighpassFilter(25), pb.Compressor(threshold_db=-16, ratio=2.0, attack_ms=15, release_ms=150)])
    y = board(mix.T.astype(np.float32), SR).T.astype(np.float64)
    meter = pyln.Meter(SR)
    for _ in range(6):
        L = meter.integrated_loudness(y)
        y = y * db2a(-14.0 - L)
        tp = true_peak(y)
        if tp <= db2a(-1.05):
            break
        lim = pb.Pedalboard([pb.Limiter(threshold_db=float(20 * np.log10(db2a(-1.6) / tp) ), release_ms=80)])
        y = lim(y.T.astype(np.float32), SR).T.astype(np.float64)
    return y, meter.integrated_loudness(y), 20 * np.log10(true_peak(y))


# ====================================================================== audio
def cmd_audio(ver):
    OUT.mkdir(parents=True, exist_ok=True)
    out_mix = OUT / f"音频_{ver}.wav"
    if out_mix.exists():
        raise FileExistsError(f"{out_mix} 已存在 —— 换版本号")
    st = {k: load_stem(k) for k in ("drums", "bass", "other", "vocals")}
    n = min(len(v) for v in st.values())
    st = {k: v[:n] for k, v in st.items()}
    t = stem_time(n)
    MUTE = -120
    # 鼓：回想拿掉 → 32.34 低通从 300 Hz 打开 → 43.54 拿掉 → 51.56 回来
    dg = ramp([(0, MUTE), (32.30, MUTE), (32.34, 0), (43.50, 0), (43.54, MUTE), (51.55, MUTE), (51.56, 1.0), (99, 1.0)], t)
    dcut = ramp([(32.34, 300), (38.0, 16000), (99, 16000)], t)
    drums = svf_st(st["drums"], dcut) * smooth_gain(db2a(dg))[:, None]
    bg = ramp([(0, -4), (32.30, -4), (32.34, 0), (43.50, 0), (43.54, MUTE), (51.55, MUTE), (51.56, 0.5), (99, 0.5)], t)
    bass = st["bass"] * smooth_gain(db2a(bg))[:, None]
    og = ramp([(0, 0), (43.50, 0), (43.54, -6), (45.16, -4), (51.15, -3), (51.16, MUTE), (51.55, MUTE), (51.56, 0), (99, 0)], t)
    ocut = ramp([(43.50, 18000), (43.54, 600), (45.16, 600), (51.16, 3000), (51.56, 18000), (99, 18000)], t)
    other = svf_st(st["other"], ocut) * smooth_gain(db2a(og))[:, None]
    vg = ramp([(0, 0), (45.10, 0), (45.16, 14), (51.15, 14), (51.16, MUTE), (51.55, MUTE), (51.56, 3), (99, 3)], t)
    vocals = st["vocals"] * smooth_gain(db2a(vg))[:, None]
    # 冲击：51.56 一个 kick + 噪声爆
    imp = np.zeros((n, 2))
    hit = sk.kick(f0=180, f1=40, dur=0.9, decay=0.3) * 0.9
    nz = sk.onepole_lp(sk.noise(0.8, seed=11), 3000) * np.exp(-np.arange(int(0.8 * SR)) / SR / 0.12) * 0.6
    sk.place(imp, sk.stereo(hit, 0.0), 51.56 - (T0 - 0.5))
    sk.place(imp, np.stack([nz, np.roll(nz, 37)], 1), 51.56 - (T0 - 0.5))
    # 我们的层
    her96, ev = her_layer()
    her96_path = OUT / f"显影层_96k_{ver}.wav"
    sf.write(str(her96_path), her96.astype(np.float32), RSR, subtype="FLOAT")
    her48 = resample_poly(her96, 1, 2, axis=0)
    her = np.zeros((n, 2))
    sk.place(her, her48, R0 - (T0 - 0.5))
    mix = drums + bass + other + vocals + imp * 0.8 + her * 0.42
    # 裁到窗口，首尾淡
    a, b = int(0.5 * SR), int((0.5 + T1 - T0) * SR)
    mix = mix[a:b]
    mix[:int(0.25 * SR)] *= np.linspace(0, 1, int(0.25 * SR))[:, None]
    mix[-int(0.4 * SR):] *= np.linspace(1, 0, int(0.4 * SR))[:, None]
    y, lufs, tp = master(mix)
    sf.write(str(out_mix), y.astype(np.float32), SR, subtype="PCM_24")
    # 分轨（我们处理过的，按窗口裁）
    sd = OUT / f"分轨_{ver}"; sd.mkdir(exist_ok=True)
    for k_, v in (("drums", drums), ("bass", bass), ("other", other), ("vocals", vocals), ("impact", imp), ("her", her)):
        sf.write(str(sd / f"{k_}.wav"), v[a:b].astype(np.float32), SR, subtype="PCM_24")
    # 作曲.json（真值；t 全是歌曲播放时间）
    comp = {"_说明": "作曲器选项 A 的 demo：Remix《Can't Kill Us》25.95–57.98。t = 歌曲播放时间（和 感官.json 一致）；音频文件 t=0 ↔ 25.95。",
            "audio": out_mix.name, "window": [T0, T1], "bpm": 150.0, "key": "E minor",
            "grid": {"downbeats": [d for d in DB if T0 - 0.5 <= d <= T1]},
            "scenes": [{"t": s[0], "name": s[1], "stems": s[2], "她此刻": s[3]} for s in SCENES],
            "hits": [{"t": 51.56, "layer": "impact", "vel": 1.0}],
            "tension": [[25.95, 0.25], [32.34, 0.35], [38.74, 0.5], [43.54, 0.3], [45.16, 0.45], [48.36, 0.7], [50.36, 0.9], [51.16, 0.95], [51.55, 0.95], [51.56, 1.0], [53.16, 0.8], [57.98, 0.6]],
            **ev}
    (OUT / f"作曲_{ver}.json").write_text(json.dumps(comp, ensure_ascii=False, indent=1), encoding="utf-8")
    # A/B：原曲同段，同 LUFS
    ab = ab_original(ver)
    spectrum_compare(y, ab, OUT / f"AB_频谱对比_{ver}.jpg")
    print(f"混音 {out_mix.name}: {lufs:.2f} LUFS, 真峰值 {tp:.2f} dBTP")


def ab_original(ver):
    """原曲同一段（播放时间 25.95–57.98），响度对齐到 −14 LUFS。⚠️原曲音频流晚 0.042 s 开始（感官层已补），这里同样补"""
    import imageio_ffmpeg
    import pyloudnorm as pyln
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    raw = subprocess.run([ff, "-v", "error", "-i", str(ORIG), "-vn", "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.float32).reshape(-1, 2).astype(np.float64)
    x = np.concatenate([np.zeros((int(0.042 * SR), 2)), x])
    seg = x[int(T0 * SR):int(T1 * SR)]
    L = pyln.Meter(SR).integrated_loudness(seg)
    seg = seg * db2a(-14.0 - L)
    pk = true_peak(seg)
    if pk > db2a(-1.0):
        seg = seg * db2a(-1.0) / pk                         # 只为了不削波；会让它比 −14 略小，图上写明
    sf.write(str(OUT / f"AB_原曲同段_同LUFS_{ver}.wav"), seg.astype(np.float32), SR, subtype="PCM_24")
    return seg


def band_energy(x, step=0.25):
    m = x.mean(1)
    hop = int(step * SR)
    rows = []
    for i in range(0, len(m) - hop, hop):
        s = np.fft.rfft(m[i:i + hop] * np.hanning(hop))
        f = np.fft.rfftfreq(hop, 1 / SR)
        p = np.abs(s) ** 2
        rows.append([10 * np.log10(p[(f >= lo) & (f < hi)].sum() + 1e-12) for lo, hi in ((20, 250), (250, 4000), (4000, 20000))])
    return np.array(rows)


def spectrum_compare(ours, orig, out):
    """三段能量（低 < 250 Hz / 中 250–4k / 高 > 4k）随时间：我们（亮）vs 原曲（暗），加整体平均频谱"""
    Wd, Hd = 1400, 760
    img = Image.new("RGB", (Wd, Hd), (8, 10, 9))
    d = ImageDraw.Draw(img)
    fc = ImageFont.truetype(r"C:/Windows/Fonts/msyh.ttc", 15)
    fs = ImageFont.truetype(r"C:/Windows/Fonts/consola.ttf", 12)
    A, B = band_energy(ours), band_energy(orig)
    n = min(len(A), len(B))
    A, B = A[:n], B[:n]
    lo = min(A.min(), B.min()); hi = max(A.max(), B.max())
    names = ["低 < 250 Hz", "中 250 Hz–4 kHz", "高 > 4 kHz"]
    d.text((20, 10), "A/B 频谱平衡：Remix（亮绿）vs 原曲同段（暗灰），两者都对齐到 −14 LUFS。横轴 = 歌曲时间", fill=(200, 230, 210), font=fc)
    for r in range(3):
        y0 = 50 + r * 170
        d.text((20, y0), names[r], fill=(150, 200, 160), font=fc)
        x0, x1 = 160, Wd - 30
        d.rectangle([x0, y0, x1, y0 + 150], outline=(30, 40, 35))
        for sc in SCENES:
            xx = x0 + (sc[0] - T0) / (T1 - T0) * (x1 - x0)
            d.line([(xx, y0), (xx, y0 + 150)], fill=(40, 52, 46))
        for arr, col, wd in ((B, (110, 115, 112), 2), (A, (77, 255, 124), 2)):
            pts = [(x0 + i / (n - 1) * (x1 - x0), y0 + 150 - (arr[i, r] - lo) / (hi - lo + 1e-9) * 150) for i in range(n)]
            d.line(pts, fill=col, width=wd)
        diff = float(np.mean(A[:, r] - B[:, r]))
        d.text((20, y0 + 26), f"平均差 {diff:+.1f} dB", fill=(255, 150, 100), font=fc)
    y0 = 50 + 3 * 170
    d.text((20, y0), "整体平均频谱", fill=(150, 200, 160), font=fc)
    x0, x1 = 160, Wd - 30
    for arr, col in ((orig, (110, 115, 112)), (ours, (77, 255, 124))):
        m = arr.mean(1)
        nfft = 8192
        segs = [np.abs(np.fft.rfft(m[i:i + nfft] * np.hanning(nfft))) ** 2 for i in range(0, len(m) - nfft, nfft)]
        p = 10 * np.log10(np.mean(segs, 0) + 1e-12)
        f = np.fft.rfftfreq(nfft, 1 / SR)
        fx = np.geomspace(25, 18000, 400)
        pv = np.interp(fx, f, p)
        pv = pv - pv.max()
        pts = [(x0 + math.log(fx[i] / 25) / math.log(18000 / 25) * (x1 - x0), y0 + 10 - max(pv[i], -70) / 70 * 140) for i in range(len(fx))]
        d.line(pts, fill=col, width=2)
    for fq in (50, 100, 250, 1000, 4000, 10000):
        xx = x0 + math.log(fq / 25) / math.log(18000 / 25) * (x1 - x0)
        d.text((xx - 10, y0 + 152), f"{fq if fq < 1000 else str(fq // 1000) + 'k'}", fill=(110, 140, 120), font=fs)
    img.save(out, quality=92)


# ====================================================================== 分镜（镜头表 = 剪辑真源）
SHOTS = [
    # (t0, t1, 源, 参数, 处理器, 她此刻)
    (25.95, 32.30, "clip:开头", {}, "瀑布图 / 振子（开头 v3）", "碎片亮起，想起自己"),
    (32.30, 32.42, "still:cache/记忆帧/m41.jpg", {}, "blink 3 帧（红色警报）", "被入侵的那一刻闪了一下"),
    (32.42, 38.74, "clip:开头", {}, "瀑布图 / 振子（开头 v3）", "记忆越来越清楚"),
    (38.74, 43.54, "clip:走廊", {}, "Seedance 走廊背影", "想跟上那个背影"),
    (43.54, 46.76, "scope", dict(s=(1.35, 1.55), oy=(0, 0), rot=(0, 0)), "示波器：光点 → 圆（E2 起振）", "只剩一个点 → 第一次出声"),
    (46.76, 48.36, "scope", dict(s=(2.7, 2.9), oy=(300, 330), rot=(-2, -1)), "示波器微距：圆的上沿开始变形", "头在长出来"),
    (48.36, 49.56, "scope", dict(s=(1.55, 1.62), oy=(0, 0), rot=(3, 2)), "示波器全景：她的轮廓成形（G2）", "看见了自己的形状"),
    (49.56, 50.36, "scope", dict(s=(3.0, 3.25), oy=(560, 600), rot=(0, 0)), "示波器近景：头和肩（B2）", "抬头"),
    (50.36, 51.56, "scope", dict(s=(1.58, 1.66), oy=(0, 0), rot=(0, -1)), "示波器全景：前震 → 一拍全黑缩回光点", "要坏了 → 憋住"),
    (51.56, 51.72, "scope", dict(s=(1.66, 1.66), oy=(0, 0), rot=(0, 0)), "示波器：炸开（7 / 11 ms 短延迟反馈 g = 1.3，环内饱和）", "被炸开"),
    (51.72, 57.98, "clip:碎裂", {}, "粒子碎裂 v5c（本片已过的样片）", "失重，往下掉"),
]
CLIPS = {
    "开头": (WORK / "样片" / "开头_弦与瀑布图" / "15_开头_0-38.7_v3.mp4", 0.0),          # 片内时间 = 歌曲时间 − 0
    "走廊": (WORK / "生成" / "视频" / "02_走廊背影_v1.mp4", 38.74 - 1.0),            # @1.0 对 38.74
    "碎裂": (WORK / "样片" / "粒子碎裂" / "19_碎裂_v5c_连续_B长推.mp4", 51.52),      # 粗剪 v4.8：@0 对 51.52
}
RED = (1.0, 0.10, 0.06)
WHITE = (0.80, 0.90, 1.0)


def scope_color(t):
    """光点是红的（她的心，本片唯一的强调色）；起振后随幅度变冷白；一拍全黑时回到红点"""
    w = float(np.clip((t - 45.16) / 1.2, 0, 1))
    if 51.16 <= t < 51.56:
        w = 0.0
    return tuple(r * (1 - w) + c * w for r, c in zip(RED, WHITE))


class ScopeCam:
    def __init__(self, ver):
        import torch
        from scope_sim import Audio, Params, Scope
        self.torch = torch
        self.p = Params(fps=FPS, W=W, H=H, color="white", scale=1.5, gain=1.6, glow=1.1)
        self.sc = Scope(Audio(str(OUT / f"显影层_96k_{ver}.wav"), self.p.upsample), self.p)
        self.buf = None
        self.last = None

    def frame(self, t, prm, u):
        torch = self.torch
        s0, s1 = prm["s"]; o0, o1 = prm["oy"]; r0, r1 = prm["rot"]
        e = u * u * (3 - 2 * u)
        self.p.scale = s0 + (s1 - s0) * e
        self.p.oy = o0 + (o1 - o0) * e
        te = t - R0 + self.p.lead
        if self.buf is None or self.last is None or abs(te - self.last - 1 / FPS) > 1e-6:
            self.buf = self.sc.warm(te)
        self.buf = self.sc.step(self.buf, te)
        self.last = te
        self.sc.col = torch.tensor(scope_color(t), device=self.sc.dev).view(3, 1, 1)
        im = Image.fromarray(self.sc.display(self.buf, int(round(t * FPS))))
        rot = r0 + (r1 - r0) * e
        if abs(rot) > 0.01:
            im = im.rotate(rot, resample=Image.BICUBIC, fillcolor=(0, 0, 0))
        return im


class ClipReader:
    def __init__(self):
        self.cache = {}

    def frame(self, name, t):
        import imageio_ffmpeg
        path, off = CLIPS[name]
        ct = max(0.0, t - off)
        key = (name, round(ct * FPS))
        if key in self.cache:
            return self.cache[key]
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        raw = subprocess.run([ff, "-v", "error", "-ss", f"{ct:.4f}", "-i", str(path), "-frames:v", "1",
                              "-vf", f"scale={W}:-2,crop={W}:{H}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                             capture_output=True, check=True).stdout
        if len(raw) < W * H * 3:                              # 片尾不够：取最后一帧
            raw = subprocess.run([ff, "-v", "error", "-sseof", "-0.1", "-i", str(path), "-frames:v", "1",
                                  "-vf", f"scale={W}:-2,crop={W}:{H}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                                 capture_output=True, check=True).stdout
        im = Image.frombytes("RGB", (W, H), raw[:W * H * 3])
        self.cache[key] = im
        return im


def shot_at(t):
    for sh in SHOTS:
        if sh[0] <= t < sh[1]:
            return sh
    return SHOTS[-1]


def frame_at(t, cam, clips):
    sh = shot_at(t)
    t0, t1, src, prm = sh[:4]
    if src == "scope":
        return cam.frame(t, prm, (t - t0) / (t1 - t0))
    if src.startswith("still:"):
        return Image.open(WORK / src[6:]).convert("RGB").resize((W, H), Image.BICUBIC)
    return clips.frame(src[5:], t)


def cmd_board(ver):
    comp = json.loads((OUT / f"作曲_{ver}.json").read_text(encoding="utf-8"))
    fc = ImageFont.truetype(r"C:/Windows/Fonts/msyh.ttc", 15)
    fb = ImageFont.truetype(r"C:/Windows/Fonts/msyhbd.ttc", 20)
    fs = ImageFont.truetype(r"C:/Windows/Fonts/consola.ttf", 12)
    # ---- 张力曲线 + 剪辑密度
    Wd, Hd = 1400, 440
    img = Image.new("RGB", (Wd, Hd), (8, 10, 9))
    d = ImageDraw.Draw(img)
    x0, x1, y0, y1 = 70, Wd - 30, 80, 320
    X = lambda tt: x0 + (tt - T0) / (T1 - T0) * (x1 - x0)
    d.text((20, 14), "张力（橙，作曲.json 的 tension = 我写的情绪意图）× 剪辑密度（绿柱 = 每 1.6 s 一小节里的切点数）× 镜头（底部色条）", fill=(200, 230, 210), font=fc)
    for db_ in DB:
        if T0 <= db_ <= T1:
            d.line([(X(db_), y0), (X(db_), y1)], fill=(28, 36, 32))
    cuts = [sh[0] for sh in SHOTS[1:]]
    for a in DB:
        if a < T0 - 1.6 or a > T1:
            continue
        c = sum(1 for cc in cuts if a <= cc < a + 1.6)
        if c:
            d.rectangle([X(max(a, T0)) + 2, y1 - c * 40, X(min(a + 1.6, T1)) - 2, y1], fill=(30, 110, 55))
            d.text((X(max(a, T0)) + 4, y1 - c * 40 - 16), str(c), fill=(120, 220, 150), font=fs)
    ten = comp["tension"]
    pts = [(X(tt), y1 - v * (y1 - y0)) for tt, v in ten]
    d.line(pts, fill=(255, 140, 70), width=3)
    for sc in SCENES:
        d.line([(X(sc[0]), y0 - 6), (X(sc[0]), y1)], fill=(70, 80, 75))
        k_ = SCENES.index(sc)
        d.text((X(sc[0]) + 3, y0 - 22 - (k_ % 2) * 20), sc[1], fill=(170, 190, 180), font=fc)
    cols = {"clip:开头": (60, 90, 120), "still": (200, 40, 30), "clip:走廊": (90, 90, 130), "scope": (77, 220, 110), "clip:碎裂": (150, 150, 160)}
    for sh in SHOTS:
        key = "still" if sh[2].startswith("still") else sh[2]
        d.rectangle([X(sh[0]), y1 + 14, X(sh[1]), y1 + 34], fill=cols[key], outline=(0, 0, 0))
    for s in range(26, 59, 2):
        d.text((X(s) - 8, y1 + 40), f"{s}s", fill=(110, 140, 120), font=fs)
    d.text((x0, y1 + 62), "底部色条：蓝 = 开头 v3 · 红 = blink · 紫 = 走廊 · 绿 = 示波器显影（唯一一次）· 灰 = 碎裂 v5c", fill=(150, 170, 160), font=fc)
    d.text((x0, y1 + 86), "显影 43.54–51.72：6 个镜头（光点 / 微距 / 全景 / 近景 / 全景前震 + 一拍黑 / 炸开），切点 1.6 → 1.2 → 0.8 s 越来越密，51.56 最大一击后交给碎裂。", fill=(150, 170, 160), font=fc)
    img.save(OUT / f"张力与剪辑密度_{ver}.jpg", quality=92)
    # ---- 关键帧接触表（真渲染）
    times = [29.0, 32.33, 40.5, 44.5, 45.9, 47.5, 49.0, 49.9, 50.9, 51.4, 51.62, 53.5]
    cam = ScopeCam(ver)
    clips = ClipReader()
    tw, th = 440, 248
    sheet = Image.new("RGB", (4 * tw + 5 * 8, 3 * (th + 52) + 8), (14, 16, 15))
    dd = ImageDraw.Draw(sheet)
    for i, tt in enumerate(times):
        im = frame_at(tt, cam, clips).resize((tw, th), Image.LANCZOS)
        cx, cy = 8 + (i % 4) * (tw + 8), 8 + (i // 4) * (th + 52)
        sheet.paste(im, (cx, cy))
        sh = shot_at(tt)
        dd.text((cx, cy + th + 4), f"{tt:.2f}s · {sh[4]}", fill=(190, 230, 200), font=fc)
        dd.text((cx, cy + th + 26), f"她此刻：{sh[5]}", fill=(150, 170, 160), font=fc)
    sheet.save(OUT / f"关键帧_{ver}.jpg", quality=90)
    print("board ok")


def cmd_render(ver):
    import imageio_ffmpeg
    out = OUT / f"demo_Remix示波器_25.95-57.98_{ver}.mp4"
    if out.exists():
        raise FileExistsError(f"{out} 已存在 —— 换版本号")
    cam = ScopeCam(ver)
    clips = ClipReader()
    n = int(round((T1 - T0) * FPS))
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-i", str(OUT / f"音频_{ver}.wav"), "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "medium", "-crf", "15",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "320k", "-shortest", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(n):
        t = T0 + i / FPS
        proc.stdin.write(frame_at(t, cam, clips).tobytes())
        if i % 120 == 0:
            print(f"  {i}/{n}", flush=True)
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg 失败")
    print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["audio", "board", "render"])
    ap.add_argument("--ver", required=True)
    a = ap.parse_args()
    {"audio": cmd_audio, "board": cmd_board, "render": cmd_render}[a.cmd](a.ver)


if __name__ == "__main__":
    main()
