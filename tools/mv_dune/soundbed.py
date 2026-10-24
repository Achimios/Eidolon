#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🎬Dune-音床-预告片音床合成（MV_Dune，2026-10-24，Fable）。零版权、不下载：numpy 合成 90 s。
结构照 Thymos.md：A 吸气（风 + 极低频 drone）→ B 量（每 4 s 一击 braam）→ C 记（喉音 / 人声进，密度升）→ D 浪（一击 → 3 s 全静 → 最大一击）→ E 题（低频拖尾）。
同时写 事件.json：hits（击，秒 + 强度）、silences（[起, 止]）、sections、drone 包络 —— 分镜锚这些事件，不锚秒数。
声音元件：
  drone  sub 32 Hz + 48 Hz（五度）+ 缓慢拍频，低通
  braam  100 ms 起 → 1.8 s 衰减：锯齿 55 Hz + 82 Hz 过 tanh 失真，再加 8 kHz 以下的噪声冲击 + 简易 Schroeder 混响
  throat 60–70 Hz 脉冲串（glottal）过两个共振峰带通（F1 500 / F2 1400，缓慢滑动）= Tuvan 式泛音感
  wind   粉噪 → 带通中心随「风场」漂（0.3–2 kHz）
  riser  D 段静默前 6 s 的白噪上扫
用法：python tools/mv_dune/soundbed.py --out MV_Dune/音床_v1.wav
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MV = ROOT.parents[1] / "Projekt_Visual" / "MV_Dune"
SR = 48000
DUR = 90.0

SECTIONS = [("A 吸气", 0.0, 18.0), ("B 量", 18.0, 40.0), ("C 记", 40.0, 62.0), ("D 浪", 62.0, 78.0), ("E 题", 78.0, 90.0)]
# v2（ReadScreen 数值审计后）：击的响度 B → C → D 一路上行；68 s 全片峰值（≈ −3 dBFS）
HITS = [(18.0, 0.30), (22.0, 0.32), (26.0, 0.34), (30.0, 0.38), (34.0, 0.40), (38.0, 0.45),
        (40.0, 0.48), (43.0, 0.50), (46.0, 0.55), (48.5, 0.58), (51.0, 0.62), (53.0, 0.65), (55.0, 0.70), (56.5, 0.72), (58.0, 0.76), (59.0, 0.78), (60.0, 0.82),
        (62.0, 0.90),                     # D：一击（撕图）
        (68.0, 1.00),                     # 静默 65–68 之后最大一击 = 峰值
        (78.0, 0.45)]                     # E：题
SILENCES = [(65.0, 68.0)]
THROAT = [(40.0, 62.0)]


def env_ad(n, a, d, sr=SR):
    t = np.arange(n) / sr
    return np.where(t < a, t / a, np.exp(-(t - a) / d))


def lowpass(x, fc, sr=SR):
    rc = 1 / (2 * math.pi * fc); a = (1 / sr) / (rc + 1 / sr)
    y = np.empty_like(x); acc = 0.0
    for i in range(len(x)):          # 单极，够用（90 s × 48k = 4.3M 样本，Python 循环慢 → 用 lfilter 替代）
        acc += a * (x[i] - acc); y[i] = acc
    return y


def onepole(x, fc, sr=SR):
    from scipy.signal import lfilter
    rc = 1 / (2 * math.pi * fc); a = (1 / sr) / (rc + 1 / sr)
    return lfilter([a], [1, -(1 - a)], x)


def bandpass(x, lo, hi, sr=SR, order=2):
    from scipy.signal import butter, sosfilt
    sos = butter(order, [lo, hi], btype="band", fs=sr, output="sos")
    return sosfilt(sos, x)


def reverb(x, sr=SR, decay=0.6):
    """Schroeder：4 梳状 + 2 全通"""
    from scipy.signal import lfilter
    out = np.zeros_like(x)
    for d in (1557, 1617, 1491, 1422):
        b = np.zeros(d + 1); b[0] = 1
        a = np.zeros(d + 1); a[0] = 1; a[d] = -decay
        out += lfilter(b, a, x)
    out /= 4
    for d in (225, 556):
        g = 0.5
        b = np.zeros(d + 1); b[0] = -g; b[d] = 1
        a = np.zeros(d + 1); a[0] = 1; a[d] = -g
        out = lfilter(b, a, out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(MV / "音床_v2.wav"))
    a = ap.parse_args()
    import soundfile as sf
    n = int(DUR * SR); t = np.arange(n) / SR
    rng = np.random.default_rng(7)
    mix = np.zeros(n)

    # drone：A 段起、全程，E 段拖尾；静默段挖掉
    f1, f2 = 32.0, 48.0
    beat = 0.5 + 0.5 * np.sin(2 * np.pi * 0.07 * t)
    drone = 0.6 * np.sin(2 * np.pi * f1 * t) + 0.35 * np.sin(2 * np.pi * f2 * t + 0.4) * beat + 0.15 * np.sin(2 * np.pi * 2 * f1 * t)
    dr_env = np.clip(t / 12.0, 0, 1) * np.where(t < 78, 1.0, np.exp(-(t - 78) / 5.0))
    for s0, s1 in SILENCES:
        dr_env *= np.where((t >= s0) & (t < s1), 0.0, 1.0)
    # 击前抽气（pull-back）：每一击前 0.8 s 床退 9 dB，击后 0.4 s 回来
    pull = np.ones(n)
    for ht, amp in HITS:
        i0 = int((ht - 0.8) * SR); i1 = int(ht * SR); i2 = int((ht + 0.4) * SR)
        ramp = np.linspace(1.0, 10 ** (-12 / 20), max(i1 - i0, 1))
        pull[i0:i1] = np.minimum(pull[i0:i1], ramp)
        back = np.linspace(10 ** (-12 / 20), 1.0, max(i2 - i1, 1))
        pull[i1:i2] = np.minimum(pull[i1:i2], back)
    mix += 0.30 * drone * dr_env * pull

    # wind：粉噪带通，中心频率漂
    white = rng.normal(size=n)
    pink = onepole(white, 400)
    center = 600 + 500 * np.sin(2 * np.pi * 0.05 * t) + 300 * np.sin(2 * np.pi * 0.013 * t + 1)
    # 分段带通（每 2 s 一段，中心变）
    wind = np.zeros(n)
    seg = int(2 * SR)
    for i in range(0, n, seg):
        c = max(float(center[i]), 150.0); wind[i:i + seg] = bandpass(pink[i:i + seg], max(80, c * 0.5), c * 1.6)
    w_env = 0.25 + 0.15 * np.sin(2 * np.pi * 0.11 * t)
    w_env *= np.where(t < 18, 1.0, 0.55) * np.where((t >= 62) & (t < 78), 0.3, 1.0)
    for s0, s1 in SILENCES:
        w_env *= np.where((t >= s0) & (t < s1), 0.0, 1.0)
    mix += 0.22 * wind / (np.abs(wind).max() + 1e-9) * w_env * pull
    # 中低频实体（100–400 Hz）：类铜管长音 —— 锯齿 110 Hz + 165 Hz 过低通，B 段起、随击升
    body_env = np.clip((t - 18) / 6, 0, 1) * np.where(t < 78, 1.0, np.exp(-(t - 78) / 3)) * (0.5 + 0.5 * np.clip((t - 18) / 44, 0, 1))
    for s0, s1 in SILENCES:
        body_env *= np.where((t >= s0) & (t < s1), 0.0, 1.0)
    saw = (2 * ((t * 110) % 1) - 1) * 0.6 + (2 * ((t * 164.8) % 1) - 1) * 0.4
    body = bandpass(np.tanh(1.5 * saw), 90, 420)
    mix += 0.16 * body / (np.abs(body).max() + 1e-9) * body_env * pull
    # 沙粒质感层（2–8 kHz）：颗粒噪声，包络随风纹（0.6–1.4 Hz 起伏）
    grain = bandpass(rng.normal(size=n), 2000, 8000)
    g_env = (0.5 + 0.5 * np.sin(2 * np.pi * 0.9 * t + np.sin(2 * np.pi * 0.11 * t) * 2)) ** 2
    g_env *= np.where(t < 18, 0.6, 1.0) * np.where((t >= 62) & (t < 78), 0.5, 1.0)
    for s0, s1 in SILENCES:
        g_env *= np.where((t >= s0) & (t < s1), 0.0, 1.0)
    mix += 0.10 * grain / (np.abs(grain).max() + 1e-9) * g_env * pull

    # braam hits
    for ht, amp in HITS:
        L = int(2.4 * SR); i0 = int(ht * SR); tt = np.arange(L) / SR
        saw = (2 * ((tt * 55) % 1) - 1) * 0.6 + (2 * ((tt * 82.4) % 1) - 1) * 0.4
        body = np.tanh(2.5 * saw) * env_ad(L, 0.08, 0.7)
        noise = bandpass(rng.normal(size=L), 80, 9000) * env_ad(L, 0.008, 0.22) * 0.9
        mid = bandpass((2 * ((tt * 110) % 1) - 1), 100, 500) * env_ad(L, 0.03, 0.9) * 0.8   # 击里的中低频实体
        hit = onepole(body + noise + mid, 3500)
        hit = 0.7 * hit + 0.5 * reverb(hit, decay=0.72)
        hit = hit / (np.abs(hit).max() + 1e-9)
        mix[i0:i0 + L] += amp * 4.2 * hit[:min(L, n - i0)]

    # throat：脉冲串 + 两个共振峰，C 段
    for s0, s1 in THROAT:
        L = int((s1 - s0) * SR); i0 = int(s0 * SR); tt = np.arange(L) / SR
        f0 = 64 + 4 * np.sin(2 * np.pi * 0.21 * tt)
        phase = np.cumsum(f0) / SR
        pulses = np.exp(-((phase % 1.0) / 0.08) ** 2)           # 窄脉冲
        F1 = 480 + 120 * np.sin(2 * np.pi * 0.09 * tt); F2 = 1300 + 500 * np.sin(2 * np.pi * 0.04 * tt + 2)
        thr = np.zeros(L); seg = int(0.5 * SR)
        for i in range(0, L, seg):
            thr[i:i + seg] = bandpass(pulses[i:i + seg], F1[i] * 0.8, F1[i] * 1.25) + 0.5 * bandpass(pulses[i:i + seg], F2[i] * 0.85, F2[i] * 1.18)
        env = np.clip((tt) / 4.0, 0, 1) * np.clip((s1 - s0 - tt) / 2.0, 0, 1) * (0.5 + 0.5 * np.clip(tt / (s1 - s0), 0, 1))
        thr = thr / (np.abs(thr).max() + 1e-9)
        mix[i0:i0 + L] += 0.45 * (0.6 * thr + 0.4 * reverb(thr, decay=0.6)) * env

    # riser：静默前 6 s
    for s0, s1 in SILENCES:
        L = int(6 * SR); i0 = int((s0 - 6) * SR); tt = np.arange(L) / SR
        ris = np.zeros(L); seg = int(0.25 * SR)
        for i in range(0, L, seg):
            c = 200 * 2 ** (4 * (i / L))
            ris[i:i + seg] = bandpass(rng.normal(size=min(seg, L - i)), c * 0.7, c * 1.4)
        mix[i0:i0 + L] += 0.35 * ris / (np.abs(ris).max() + 1e-9) * (tt / 6) ** 2

    # 静默：硬切到 0（预告片的「全静」）
    for s0, s1 in SILENCES:
        mix[int(s0 * SR):int(s1 * SR)] = 0.0
    # 响度整形：软限幅，让 68 s 峰值 ≈ −3 dBFS，床不被抬平
    mix = np.tanh(mix / (np.abs(mix).max() + 1e-9) * 1.15) * 0.86   # 轻限幅：68 s 峰 ≈ −3 dBFS
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), np.c_[mix, mix].astype(np.float32), SR)
    ev = {"source": out.name, "duration": DUR, "sections": [{"name": s_, "t0": a_, "t1": b_} for s_, a_, b_ in SECTIONS],
          "hits": [{"t": h, "amp": am} for h, am in HITS], "silences": [{"t0": a_, "t1": b_} for a_, b_ in SILENCES],
          "throat": [{"t0": a_, "t1": b_} for a_, b_ in THROAT]}
    (out.parent / "事件.json").write_text(json.dumps(ev, ensure_ascii=False, indent=1), "utf-8")
    print(out, "hits", len(HITS))


if __name__ == "__main__":
    main()
