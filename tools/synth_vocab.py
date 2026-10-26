#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🎛合成器词表：15 个词，每个一段示例音频（无损 wav）+ 示波器四格（scope_sim）+ 左右声道波形 / 频谱小图（频谱算听的版本）+ 带声音的示波器短片。

输出到 Projekt_Visual/样片_示波器/词表/<版本>/；页面 -给人类的解释/合成器词表.html 读这些文件。
⭐你听到的 = 示波器画出来的（同一个 wav）。两种接法，每个词写明：
  - 「真立体声」：左声道 = x，右声道 = y（失谐、齐奏、延迟、琶音器）。
  - 「延迟嵌入」：右声道 = 左声道晚 τ（τ = 基音周期的 1/4）。正弦 → 圆；任何稳定的音 → 一个闭环；乱的声音 → 乱结构。
    听感上只是极轻微的左右错位（几毫秒）。
示波器的「放大」（scale）按每个例子的幅度单独调 = 示波器面板上的 V/div 旋钮，页面上写了。

用法：
  python tools/synth_vocab.py --ver v1            # 全部
  python tools/synth_vocab.py --ver v1 --only 05   # 只做一个（调试）
  python tools/synth_vocab.py --ver v1 --no-video  # 只出 wav + 图
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import synthkit as sk  # noqa: E402

SR = sk.SR
OUT_BASE = Path(r"C:/Users/Administrator/Desktop/Projekt_Visual/样片_示波器/词表")
A2, A1 = 110.0, 55.0
BPM = 120.0
BEAT = 60.0 / BPM


def envmul(y, env):
    """包络比音短时补 0（包络结束 = 静音）"""
    e = np.zeros(len(y)); k = min(len(y), len(env)); e[:k] = env[:k]
    return y * e


def embed(m, f0, frac=0.25):
    """延迟嵌入：右 = 左晚 τ（τ = 周期·frac）。返回立体声和 τ（毫秒）"""
    d = int(round(SR / f0 * frac))
    r = np.concatenate([np.zeros(d), m[:-d]])
    return np.stack([m, r], 1), 1000.0 * d / SR


# ====================================================================== 15 个例子
# 每个返回 dict(st=立体声, mode=接法说明, frames=[(t, hud), ...] 四格, f0=画频谱时标的基音)

def ex_osc():
    dur = 4.0
    n = int(round(dur * SR))
    m = sk.osc("sine", A2, dur) * sk.adsr(n - int(0.6 * SR), 0.4, 0.1, 1.0, 0.6)[:n]
    st, tau = embed(m, A2)
    return dict(st=st, mode=f"延迟嵌入 τ = {tau:.2f} ms", f0=A2,
                frames=[(0.15, "起振 · 110 Hz"), (0.6, "正弦 110 Hz"), (2.0, "每秒画 110 圈"), (3.7, "释放")])


def ex_wave():
    seg = 1.25
    parts = []
    for kind in ("sine", "tri", "square", "saw"):
        n = int(seg * SR)
        e = sk.adsr(n - int(0.08 * SR), 0.01, 0.05, 1.0, 0.08)[:n]
        g = {"sine": 1.0, "tri": 1.0, "square": 0.55, "saw": 0.65}[kind]
        parts.append(sk.osc(kind, A2, seg) * e * g)
    m = np.concatenate(parts)
    st, tau = embed(m, A2)
    return dict(st=st, mode=f"延迟嵌入 τ = {tau:.2f} ms", f0=A2,
                frames=[(0.6, "正弦 sine"), (1.85, "三角 triangle"), (3.1, "方波 square"), (4.35, "锯齿 saw")])


def ex_harm():
    dur = 5.0
    n = int(round(dur * SR))
    t = np.arange(n) / SR
    amps = []
    for k in range(1, 9):
        on = (k - 1) * 0.5                                     # 每 0.5 秒加一个泛音
        ramp = np.clip((t - on) / 0.15, 0, 1)
        amps.append(ramp / k)
    m = sk.additive(A2, amps, dur)
    m *= sk.adsr(n - int(0.4 * SR), 0.05, 0.1, 1.0, 0.4)[:n]
    st, tau = embed(m, A2)
    return dict(st=st, mode=f"延迟嵌入 τ = {tau:.2f} ms", f0=A2,
                frames=[(0.4, "只有第 1 泛音（基音）"), (1.3, "1–3 次泛音"), (2.4, "1–5 次"), (4.2, "1–8 次，振幅 1/k")])


def ex_adsr():
    out = np.zeros(int(5.0 * SR))
    # 拨弦：A 快、D 短、S = 0；铺底：A 慢、S 高、R 长
    n1 = int(0.9 * SR)
    pl = sk.osc("saw", A2 * 2, 1.0)
    pl = envmul(sk.svf(pl, 2400, 0.8), sk.adsr(n1, 0.002, 0.25, 0.0, 0.1))
    sk.place(out, pl * 0.45, 0.1)
    n2 = int(2.3 * SR)
    pad = sk.osc("saw", A2, 3.6) * 0.5 + sk.osc("saw", A2 * 1.004, 3.6) * 0.5
    pad = sk.svf(pad, 1200, 0.7)
    env = sk.adsr(n2, 1.0, 0.4, 0.6, 1.2)
    sk.place(out, pad[:len(env)] * env, 1.3)
    st, tau = embed(out, A2)
    return dict(st=st, mode=f"延迟嵌入 τ = {tau:.2f} ms（按 110 Hz）", f0=A2,
                frames=[(0.14, "拨弦：Attack 2 ms，刚起"), (0.3, "衰减：图形缩成一点，因为声音在变小"), (2.6, "铺底：Sustain 60%"), (4.0, "铺底：Release 1.2 s")])


def ex_filter():
    dur = 6.0
    n = int(round(dur * SR))
    t = np.arange(n) / SR
    cut = 120 * (40 ** (0.5 - 0.5 * np.cos(2 * np.pi * t / 6.0)))     # 120 Hz → 4800 Hz → 120 Hz
    x = sk.osc("saw", A1, dur) * 0.6 + sk.osc("saw", A1 * 1.003, dur) * 0.4
    pre = sk.svf(x, cut, 0.707)                                 # 两级串联 = 24 dB/倍频程（和 Moog 梯形一样陡）
    y = sk.svf(pre, cut, 0.8)
    y2 = sk.svf(pre, cut, 6.0)                                   # 后半：第二级共振拉高
    y = np.where(t < 3.0, y, y2)
    y *= sk.adsr(n - int(0.3 * SR), 0.02, 0.1, 1.0, 0.3)[:n]
    st, tau = embed(y, A1)
    return dict(st=st, mode=f"延迟嵌入 τ = {tau:.2f} ms（55 Hz）", f0=A1,
                frames=[(0.3, f"截止 {cut[int(0.3*SR)]:.0f} Hz · 共振低"), (1.6, f"截止 {cut[int(1.6*SR)]:.0f} Hz"),
                        (3.3, f"截止 {cut[int(3.3*SR)]:.0f} Hz · 共振高"), (5.4, f"截止 {cut[int(5.4*SR)]:.0f} Hz · 共振高")],
                curve=("截止频率", t, cut))


def ex_lfo():
    dur = 6.0
    n = int(round(dur * SR))
    t = np.arange(n) / SR
    rate = np.where(t < 3.0, 0.5, 6.0)                          # 前半 0.5 Hz 慢呼吸，后半 6 Hz 颤
    ph = np.cumsum(rate) / SR
    l = np.sin(2 * np.pi * ph)
    f = A2 * 2 ** (np.where(t < 3.0, 0.0, 0.25) * l / 12)       # 后半：音高颤 ±25 音分（颤音）
    x = sk.osc("tri", f, dur)
    amp = np.where(t < 3.0, 0.55 + 0.45 * l, 0.85 + 0.15 * l)  # 前半：音量慢起伏（颤幅）
    y = x * amp * sk.adsr(n - int(0.3 * SR), 0.05, 0.1, 1.0, 0.3)[:n]
    st, tau = embed(y, A2)
    return dict(st=st, mode=f"延迟嵌入 τ = {tau:.2f} ms", f0=A2,
                frames=[(0.5, "LFO 0.5 Hz → 音量：最大"), (1.5, "同一个 LFO：最小"), (3.6, "LFO 6 Hz → 音高（颤音）"), (5.0, "颤音 ±25 音分")],
                curve=("LFO", t, l))


def ex_mod():
    # 同一段贝斯，第一小节无调制；第二小节每个音一个包络同时推截止频率（滤波包络）
    out = np.zeros(int(4.2 * SR))
    notes = [A1, A1, A1 * 2 ** (3 / 12), A1 * 2 ** (-2 / 12)] * 2
    for i, f in enumerate(notes):
        t0 = i * BEAT * 1.0
        d = BEAT * 0.9
        n = int(d * SR)
        x = sk.osc("saw", f, d + 0.1) * 0.7 + sk.osc("square", f * 1.002, d + 0.1) * 0.3
        if i < 4:
            y = sk.svf(x, 500, 0.9)
        else:
            tt = np.arange(len(x)) / SR
            y = sk.svf(x, 180 + 5200 * np.exp(-tt / 0.09), 3.0)       # 包络 → 截止（调制）
        y = envmul(y, sk.adsr(n, 0.004, 0.2, 0.7, 0.08))
        sk.place(out, y, 0.05 + t0)
    st, tau = embed(out, A1)
    return dict(st=st, mode=f"延迟嵌入 τ = {tau:.2f} ms（55 Hz）", f0=A1,
                frames=[(0.1, "无调制：截止固定 500 Hz"), (1.2, "无调制"), (2.08, "包络 → 截止：刚起音 5.4 kHz"), (2.4, "包络落下：截止回到 ~200 Hz")])


def ex_fm():
    dur = 6.0
    n = int(round(dur * SR))
    t = np.arange(n) / SR
    idx = np.where(t < 3.0, 5.0 * t / 3.0, 0.0)
    out = sk.fm(A2, 1.0, idx, dur) * np.where(t < 3.0, 1.0, 0.0)
    # 后半：比例 3.5 的钟声（非整数比 → 不和谐的泛音），index 随时间落下
    bell = np.zeros(n)
    for k, (st_, f) in enumerate(((3.1, A2 * 2), (3.9, A2 * 2 * 2 ** (3 / 12)), (4.7, A2 * 2 * 2 ** (7 / 12)))):
        d = 1.6
        tt = np.arange(int(d * SR)) / SR
        b = sk.fm(f, 3.5, 4.0 * np.exp(-tt / 0.35), d) * np.exp(-tt / 0.6)
        sk.place(bell, b, st_)
    env = sk.adsr(int(2.9 * SR), 0.02, 0.1, 1.0, 0.1)
    out[:len(env)] *= env
    y = out + bell * 0.7
    st, tau = embed(y, A2)
    return dict(st=st, mode=f"延迟嵌入 τ = {tau:.2f} ms（按 110 Hz；钟声是 220 Hz 起）", f0=A2,
                frames=[(0.25, f"比例 1:1 · 深度 {5*0.25/3:.1f}"), (1.4, f"深度 {5*1.4/3:.1f}"), (2.8, f"深度 {5*2.8/3:.1f}"), (3.2, "比例 1:3.5 · 钟")],
                curve=("FM 深度 index", t, idx))


def ex_detune():
    dur = 6.0
    n = int(round(dur * SR))
    t = np.arange(n) / SR
    det = np.where(t < 3.0, 0.5, 3.0)                            # 差 0.5 Hz → 每 2 秒转一圈；差 3 Hz → 一秒 3 圈
    l = sk.osc("sine", A2, dur) * 0.9 + sk.osc("saw", A2, dur) * 0.1
    r = sk.osc("sine", A2 + det, dur) * 0.9 + sk.osc("saw", A2 + det, dur) * 0.1
    e = sk.adsr(n - int(0.3 * SR), 0.05, 0.1, 1.0, 0.3)[:n]
    st = np.stack([l * e, r * e], 1)
    return dict(st=st, mode="真立体声：左 110 Hz，右 110 + Δ Hz", f0=A2,
                frames=[(0.5, "Δ 0.5 Hz：相位差 90°"), (1.0, "相位差 180°（斜线）"), (1.5, "相位差 270°"), (3.6, "Δ 3 Hz：转得快 = 拍得快")])


def ex_unison():
    dur = 6.0
    n = int(round(dur * SR))
    t = np.arange(n) / SR
    chord = [A2, A2 * 2 ** (3 / 12), A2 * 2 ** (7 / 12)]
    one = sum(sk.stereo(sk.osc("saw", f, dur), 0.0) for f in chord) / 3
    many = sum(sk.unison("saw", f, dur, voices=7, cents=22, width=1.0, seed=i) for i, f in enumerate(chord)) / 3
    x = np.where((t < 2.8)[:, None], one, many)
    lp = np.stack([sk.svf(x[:, c], 3500, 0.7) for c in range(2)], 1)
    e = sk.adsr(n - int(0.4 * SR), 0.03, 0.1, 1.0, 0.4)[:n]
    st = lp * e[:, None]
    return dict(st=st, mode="真立体声（齐奏声部左右铺开）", f0=A2,
                frames=[(1.0, "1 个声部 / 音（居中 = 斜线）"), (2.5, "1 个声部"), (3.4, "7 个声部 ±22 音分"), (5.0, "7 声部：宽、厚、一团")])


def ex_dist():
    dur = 6.0
    n = int(round(dur * SR))
    t = np.arange(n) / SR
    amt = 0.3 + 11.7 * (t / dur) ** 1.6                          # 推进 tanh 的增益 0.3 → 12
    x = sk.osc("sine", A1, dur) * 0.8 + sk.osc("sine", A1 * 2, dur) * 0.2
    y = sk.drive(x, amt)
    y = sk.svf(y, 7000, 0.7)
    y *= sk.adsr(n - int(0.3 * SR), 0.05, 0.1, 1.0, 0.3)[:n]
    st, tau = embed(y, A1)
    fr = []
    for tt in (0.4, 2.5, 4.2, 5.6):
        fr.append((tt, f"推入增益 ×{amt[int(tt*SR)]:.1f}"))
    return dict(st=st, mode=f"延迟嵌入 τ = {tau:.2f} ms（55 Hz）", f0=A1, frames=fr, curve=("推入增益", t, amt))


def ex_space():
    # 前半：乒乓延迟（一个拨弦，回声左右交替，g = 0.55）；后半：同样的拨弦进混响
    def pluck(f, d=0.5):
        x = sk.osc("saw", f, d) * 0.6 + sk.osc("tri", f * 2, d) * 0.4
        tt = np.arange(len(x)) / SR
        return sk.svf(x, 600 + 3000 * np.exp(-tt / 0.06), 1.2) * np.exp(-tt / 0.12)

    a = pluck(A2 * 2)
    pp = sk.pingpong(a, BEAT * 0.75, 0.55, n_rep=8)
    out = np.zeros((int(7.0 * SR), 2))
    sk.place(out, pp, 0.05)
    dry = np.zeros((int(3.6 * SR), 2))
    for k, f in enumerate((A2 * 2, A2 * 2 * 2 ** (3 / 12), A2 * 2 * 2 ** (7 / 12))):
        sk.place(dry, sk.stereo(pluck(f), 0.0), 0.05 + k * 0.25)
    wet = sk.reverb(dry, size=0.88, damp=0.3, wet=0.9)
    sk.place(out, wet, 3.4)
    return dict(st=out, mode="真立体声（乒乓：回声一左一右；混响：左右不一样的衰减）", f0=A2 * 2,
                frames=[(0.12, "干声：居中 = 斜线"), (0.47, "第 1 次回声：只在左 = 横线"), (0.85, "第 2 次：只在右 = 竖线"), (4.05, "混响：左右不相关 → 一团云")])


def ex_side():
    bars = 2
    dur = bars * 4 * BEAT + 0.6
    n = int(round(dur * SR))
    t = np.arange(n) / SR
    f = A2 * 2
    l = sk.osc("sine", f, dur) + 0.12 * sk.osc("sine", f * 2, dur)
    r = sk.osc("sine", f * 1.0012, dur, phase0=0.25) + 0.12 * sk.osc("sine", f * 2 * 1.0012, dur, phase0=0.5)   # 差 0.26 Hz：椭圆约 4 秒转一圈
    pad = np.stack([l, r], 1)
    duck = np.ones(n)
    kk = np.zeros(n)
    for b in range(bars * 4):
        tb = b * BEAT
        sk.place(kk, sk.kick(dur=0.2, decay=0.05), tb)          # 短 kick：尾巴不和圆叠成乱线
        if b >= 4:                                              # 第二小节开侧链：kick 来时 pad 被压到 5%，150 ms 恢复
            i = int(tb * SR)
            rr = np.arange(n - i) / SR
            duck[i:] = np.minimum(duck[i:], 1 - 0.95 * np.exp(-rr / 0.15) * (rr < BEAT))
    e = sk.adsr(n - int(0.5 * SR), 0.02, 0.1, 1.0, 0.5)[:n]
    st = pad * (duck * e)[:, None] * 0.6 + kk[:, None] * 0.7
    return dict(st=st, mode="真立体声：pad = 左右两个微失谐的正弦（慢转的椭圆），kick 居中（斜线）", f0=f,
                frames=[(0.02, "没侧链：kick 和圆叠加 → 缠成一团（糊）"), (0.4, "没侧链：两拍之间，圆满"), (2.03, "侧链：圆让开，只剩 kick 的斜线"), (2.4, "圆长回来（150 ms）")],
                curve=("pad 音量（侧链）", t, duck))


def ex_arp():
    # 左声道：持续的 A1 低音；右声道：琶音器把按住的和弦 Am(A C E A) 按 16 分音符往上弹
    # 左右两个音的频率比 = 音程 → 李萨如图：比例越简单越稳（八度 1:2、五度 2:3），平均律所以图形慢慢转
    dur = 5.0
    n = int(round(dur * SR))
    t = np.arange(n) / SR
    l = sk.osc("sine", A1 * 2, dur) * 0.8
    step = BEAT / 4
    seq = [0, 3, 7, 12, 7, 3]                                   # 半音：A C E A' E C
    r = np.zeros(n)
    k = 0
    while k * step < dur - 0.2:
        semis = seq[k % len(seq)]
        f = A1 * 2 * 2 ** (semis / 12)
        d = step * 0.95
        x = sk.osc("sine", f, d + 0.05) * 0.85 + sk.osc("tri", f, d + 0.05) * 0.15
        x = envmul(x, sk.adsr(int(d * SR), 0.003, 0.06, 0.7, 0.04))
        sk.place(r, x, k * step)
        k += 1
    e = sk.adsr(n - int(0.2 * SR), 0.02, 0.1, 1.0, 0.2)[:n]
    st = np.stack([l * e, r * e], 1)
    names = {0: "1:1 同度", 3: "≈5:6 小三度", 7: "≈2:3 五度", 12: "1:2 八度"}
    fr = []
    for kk_ in (0, 1, 2, 3):
        tt = 4 * len(seq) * step * 0 + kk_ * step + step * 0.6 + len(seq) * step   # 第二轮，每个音的中间
        fr.append((tt, f"右 = {['A', 'C', 'E', "A'"][kk_]} · {names[seq[kk_]]}"))
    return dict(st=st, mode="真立体声：左 = 低音 A (110 Hz)，右 = 琶音", f0=A2, frames=fr)


SEQ_STEPS = [  # (半音, 波形)；None = 空步。A 小调五声：A C D E G
    (0, "sine"), (12, "tri"), (7, "sine"), (None, None), (3, "square"), (5, "tri"), (7, "saw"), (None, None),
    (0, "sine"), (10, "tri"), (7, "square"), (5, "sine"), (3, "tri"), (None, None), (12, "saw"), (7, "sine")]
SEQ_NAMES = {0: "A", 3: "C", 5: "D", 7: "E", 10: "G", 12: "A'"}


def ex_seq():
    # 16 步音序器：一排格子循环两遍；每格一个音高 + 一种波形（空格 = 不弹）。单声道旋律，延迟嵌入画成闭环
    step = BEAT / 2                                             # 八分音符一步（0.25 s），每一步的形状看得清
    loops = 2
    dur = len(SEQ_STEPS) * step * loops + 0.5
    n = int(round(dur * SR))
    m = np.zeros(n)
    gains = {"sine": 1.0, "tri": 1.0, "square": 0.5, "saw": 0.6}
    for L in range(loops):
        for k, (semi, kind) in enumerate(SEQ_STEPS):
            if semi is None:
                continue
            f = A2 * 2 ** (semi / 12)
            d = step * 0.92
            x = sk.osc(kind, f, d)
            x = sk.svf(x, 3000, 0.7) if kind in ("square", "saw") else x
            x = envmul(x * gains[kind], sk.adsr(int(d * SR) - 400, 0.004, 0.08, 0.8, 0.008))
            sk.place(m, x, (L * len(SEQ_STEPS) + k) * step)
    st, tau = embed(m, A2 * 2 ** (7 / 12))                     # τ = E3 周期的 1/4：音域中间，A–A' 每个音的相位差都在 60°–120°，全是闭环
    fr = []
    for k in (0, 1, 4, 6):
        semi, kind = SEQ_STEPS[k]
        fr.append((k * step + step * 0.45, f"第 {k + 1} 步：{SEQ_NAMES[semi]} · {kind}"))
    return dict(st=st, mode=f"延迟嵌入 τ = {tau:.2f} ms（按 E3 165 Hz）", f0=A2, frames=fr, grid=SEQ_STEPS)


TERMS = [
    ("01", "oscillator", "振荡器", "Oscillator", ex_osc,
     "发声的源头：一个每秒重复几百次的形状。重复的次数就是音高。",
     "一个干净、没表情的低音 A。", "一个圆。每秒被重画 110 次，所以看起来不动。"),
    ("02", "waveform", "波形", "Waveform", ex_wave,
     "振荡器重复的那个形状。四种基本形：正弦、三角、方波、锯齿，越尖的越亮、越吵。",
     "同一个音高，四次换「衣服」：圆润 → 木 → 空心的电子味 → 最亮的嗡嗡。", "圆 → 菱形 → 四个角的光点（方波只停在两个值上）→ 带一条快跳暗线的斜三角。"),
    ("03", "harmonics", "泛音 / 音色", "Harmonics / Timbre", ex_harm,
     "任何声音 = 基音 + 一串整数倍的泛音。泛音的多少和配比就是「音色」：为什么钢琴和小号弹同一个音听得出不同。",
     "音高一直没变，但声音一层层变亮，最后接近锯齿波。", "圆上长出越来越多的小圈。泛音越多，形状越复杂。"),
    ("04", "adsr", "ADSR 包络", "ADSR Envelope", ex_adsr,
     "一个音的音量怎么随时间走：Attack 起多快 / Decay 落多快 / Sustain 按住时停在多高 / Release 松开后多久消失。",
     "前面一下拨弦（快起、立刻落）；后面一个铺底（慢慢浮起、按住、松开后慢慢退）。", "形状大小跟着音量：拨弦是一闪而过的亮团，铺底是慢慢长大又慢慢缩回的环。"),
    ("05", "filter", "滤波器（截止 / 共振）", "Filter (Cutoff / Resonance)", ex_filter,
     "把一部分频率削掉。低通 = 只放低频过去；截止频率 = 从哪开始削；共振 = 在截止那里鼓起一个尖峰，会「啸」。",
     "锯齿低音被「捂住 → 打开 → 再捂住」（24 dB/倍频程低通）；后半共振拉高，开合时有一声哨音在扫。", "截止低：近乎一个光滑的环；打开：边缘出现锯齿和尖刺；共振高：环上缠了一圈高频的小波纹。频谱图里橙线（截止）以上的泛音变暗，共振时橙线附近亮一条。"),
    ("06", "lfo", "LFO", "LFO (Low-Frequency Oscillator)", ex_lfo,
     "一个慢到听不见的振荡器（一般 < 20 Hz），不发声，只用来推别的旋钮来回动。",
     "前半：音量每 2 秒呼吸一次；后半：音高每秒颤 6 次（颤音）。", "前半：同一个形状慢慢胀缩；后半：形状在原地抖。"),
    ("07", "modulation", "调制", "Modulation", ex_mod,
     "让一个信号去推另一个参数。LFO、包络、键盘力度都可以是「源」；音高、截止、声像都可以是「目标」。连线 = 调制矩阵。",
     "同一段贝斯弹两遍：第一遍截止固定（闷）；第二遍每个音的包络去推截止（每下都「哇」地张嘴）。", "第一遍形状一直很圆；第二遍每个音刚起来时满是尖刺，然后迅速收成圆。"),
    ("08", "fm", "FM（调频）", "FM Synthesis", ex_fm,
     "用一个振荡器去快速推另一个振荡器的相位。两个音的频率比定「哪些泛音」，推的深度定「多亮」。整数比 → 乐音；非整数比 → 钟、金属。",
     "前半：一个音被慢慢「拧亮」，从长笛变成铜管；后半：三下钟声（比例 3.5）。", "圆被拧成越来越多瓣的花（深度越大瓣越多）；钟声是不闭合、一直在转的花纹。"),
    ("09", "detune", "失谐", "Detune", ex_detune,
     "两个振荡器的音高故意差一点点。差出来的那一点会变成「拍」：一种慢慢起伏的摇摆感。",
     "前半：声音像在慢慢转（每 2 秒一次）；后半差得多，变成很快的颤动。", "左右差一点 → 李萨如椭圆在圆和斜线之间慢慢翻转。转一圈的时间 = 差的那几 Hz 的倒数。"),
    ("10", "unison", "齐奏", "Unison (Supersaw)", ex_unison,
     "同一个音用好几个略微失谐的振荡器同时弹，再左右铺开。trance、EDM 那种又宽又厚的「墙」就是它。",
     "前 2.8 秒：一个声部弹 A 小三和弦（窄、薄）；后半：每个音 7 个声部（突然变宽、变厚、在流动）。", "一个声部、居中 = 一条斜线；齐奏后左右不再相等，斜线炸成一团在翻滚的云。"),
    ("11", "distortion", "失真 / 饱和", "Distortion / Saturation", ex_dist,
     "把声音推得太猛，波形的顶被压平。轻的叫饱和（温暖、变肥），重的叫失真（毛、凶）。",
     "一个圆润的低音，越推越粗、越凶，最后像吉他音箱。", "圆慢慢被压成圆角方块：顶被削平的地方，光束停留得久，所以四个角最亮。"),
    ("12", "delay-reverb", "延迟 / 混响", "Delay / Reverb", ex_space,
     "延迟 = 回声，一下一下清楚地重复；混响 = 成千上万次回声糊在一起，变成「空间」。反馈 g 决定回声多久消失：g < 1 会散，g → 1 永远不停（这就是 IIR 的极点）。",
     "前半：一下拨弦，回声左右左右地弹走、越来越暗；后半：三个音进了一个大厅。", "干声在中间是斜线；回声只在一边时变成横线 / 竖线；混响让左右各走各的，变成一团云。"),
    ("13", "sidechain", "侧链", "Sidechain", ex_side,
     "让 A 的音量被 B 控制：kick 一打，铺底就被压下去再弹回来。舞曲的「抽吸感」就是它。",
     "第一小节：kick 和一个持续的长音叠在一起；第二小节：每打一下 kick，长音就让开再回来，整体在呼吸。", "长音是一个慢慢转的圆 / 椭圆；开了侧链后，每次 kick 圆缩成一个点（只剩 kick 的一道斜线），150 ms 后再长回来。"),
    ("14", "arpeggiator", "琶音器", "Arpeggiator", ex_arp,
     "你按住一个和弦，它自动把和弦里的音按节奏一个一个弹出来（往上、往下、来回）。",
     "左耳一个持续的低音 A，右耳琶音 A–C–E–A'–E–C 按 16 分音符跑。", "左右两个音高的比例 = 音程，画出来是李萨如图：八度 1:2 是 ∞ 字，五度 2:3 是三瓣。越协和的音程图越简单。"),
    ("15", "sequencer", "音序器", "Step Sequencer", ex_seq,
     "一排格子（常见 16 步）循环播放：每一格放不放音、放哪个音。很多电子乐就是几排格子叠出来的。",
     "16 格的旋律循环两遍：每一格一个音（A 小调五声），有的格换一种波形，空格就停一下。", "一步一个干净的闭环：音高变 → 环的倾斜和圈数变；波形变 → 圆 / 菱形 / 方 / 锯齿换着来；空格 = 只剩余辉。"),
]


# ====================================================================== 图
FONT_CN = r"C:/Windows/Fonts/msyh.ttc"
FONT_MONO = r"C:/Windows/Fonts/consola.ttf"
GREEN = (60, 255, 110)
DIM = (60, 90, 70)


def font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def scope_scale(st):
    pk = float(np.percentile(np.abs(st), 99.7))
    return 0.80 / max(pk, 1e-4)


def strip(wav, frames, scale, out, W=560, H=420):
    from scope_sim import Params, render_frame
    p = Params(W=W, H=H, scale=scale)
    tiles = []
    fm_ = font(FONT_CN, 17)
    fs = font(FONT_MONO, 13)
    for t, hud in frames:
        img = Image.fromarray(render_frame(str(wav), t, p))
        d = ImageDraw.Draw(img)
        d.text((12, 10), f"t = {t:5.2f} s", fill=(150, 200, 160), font=fs)
        d.text((12, H - 30), hud, fill=(190, 240, 200), font=fm_)
        tiles.append(img)
    gap = 6
    canvas = Image.new("RGB", (len(tiles) * W + (len(tiles) - 1) * gap, H), (8, 10, 9))
    for i, im in enumerate(tiles):
        canvas.paste(im, (i * (W + gap), 0))
    canvas.save(out, quality=92)


def spectro_plot(st, lis, f0, out, curve=None, W=1200, H=470):
    """上：左 / 右声道各一条波形（同一段时间、同一个比例，最响处的 2 个周期）——示波器的 x 和 y 分开看；
    下：对数频谱图（30 Hz–12 kHz），横轴 = 时间，算的是「听的版本」lis：
    延迟嵌入的例子只用真实声音（左声道）；真立体声把左右两个声道的能量相加（不先混成单声道，免得相位抵消）。
    v2 的坑：用左右平均算频谱，延迟嵌入时右 = 左晚 1/4 周期，平均后第 2、6、10… 次泛音互相抵消，图上少了这几条线。"""
    n = len(lis)
    img = Image.new("RGB", (W, H), (8, 10, 9))
    d = ImageDraw.Draw(img)
    fs = font(FONT_MONO, 12)
    fc = font(FONT_CN, 13)
    RED = (255, 110, 140)
    # ---- 左右声道波形
    mono = lis.mean(1)
    rms = np.convolve(mono ** 2, np.ones(2048) / 2048, "same")
    c0 = int(np.argmax(rms))
    L = int(2 * SR / f0)
    i0 = max(0, min(c0 - L // 2, len(st) - L))
    seg = st[i0:i0 + L]
    pk = np.abs(seg).max() + 1e-9                               # 左右共用一个比例：谁大谁小看得出来
    rows_y = ((8, "左 L → 示波器 x", GREEN, 0), (58, "右 R → 示波器 y", RED, 1))
    for y0, lab, col, ch in rows_y:
        mid = y0 + 22
        d.line([(50, mid), (390, mid)], fill=DIM)
        if len(seg) > 2:
            xs = 50 + np.arange(len(seg)) / (len(seg) - 1) * 340
            ys = mid - seg[:, ch] / pk * 20
            d.line(list(zip(xs, ys)), fill=col, width=2)
        d.text((400, mid - 9), lab, fill=col, font=fc)
    d.text((560, 14), f"上两条：同一时刻的左右声道，最响处的 2 个周期（{f0:.0f} Hz），同一个比例", fill=(150, 200, 160), font=fc)
    d.text((560, 40), "两条一样 → 斜线；右比左晚 1/4 周期 → 圆 / 闭环；各走各的 → 一团", fill=(150, 200, 160), font=fc)
    d.text((560, 66), "下：频谱图 = 听到的声音（横轴时间，纵轴对数频率，亮度 = dB，0 到 −60）", fill=(150, 200, 160), font=fc)
    d.text((560, 88), "一条横亮线 = 一个泛音；亮线怎么排 = 音色", fill=(150, 200, 160), font=fc)
    # ---- 频谱图
    top, bot = 118, H - 22
    nfft, hop = 8192, 256
    win = np.hanning(nfft)
    nfr = max(1, (n - nfft) // hop)
    cols = W - 60
    idx = np.linspace(0, nfr - 1, cols).astype(int)
    fr = np.fft.rfftfreq(nfft, 1 / SR)
    fmin, fmax = 30.0, 12000.0
    rows = bot - top
    fy = fmin * (fmax / fmin) ** (1 - np.arange(rows) / (rows - 1))
    bins = np.clip(np.searchsorted(fr, fy), 1, len(fr) - 1)
    chans = [lis[:, 0]] if np.allclose(lis[:, 0], lis[:, 1]) else [lis[:, 0], lis[:, 1]]
    spec = np.zeros((rows, cols))
    for c, k in enumerate(idx):
        pw = 0.0
        for x in chans:
            seg_ = x[k * hop:k * hop + nfft]
            if len(seg_) < nfft:
                seg_ = np.pad(seg_, (0, nfft - len(seg_)))
            pw = pw + np.abs(np.fft.rfft(seg_ * win)) ** 2
        spec[:, c] = np.sqrt(pw / len(chans))[bins]
    db = 20 * np.log10(spec + 1e-9)
    db = np.clip((db - db.max() + 60) / 60, 0, 1)              # 色标 = dB，0 到 −60 dB 线性，下限固定
    rgb = np.stack([db ** 3 * 160, db ** 1.4 * 255, db ** 2.4 * 170], -1).astype(np.uint8)
    img.paste(Image.fromarray(rgb), (50, top))
    for f in (50, 100, 200, 500, 1000, 2000, 5000, 10000):
        y = top + int((1 - math.log(f / fmin) / math.log(fmax / fmin)) * (rows - 1))
        d.line([(46, y), (50, y)], fill=DIM)
        d.text((4, y - 7), f"{f if f < 1000 else str(f // 1000) + 'k'}", fill=(110, 140, 120), font=fs)
    dur = n / SR
    for s_ in range(int(dur) + 1):
        x = 50 + int(s_ / dur * cols)
        d.line([(x, bot), (x, bot + 4)], fill=DIM)
        d.text((x - 3, bot + 5), f"{s_}s", fill=(110, 140, 120), font=fs)
    if curve is not None:
        name, t, v = curve
        v = np.asarray(v, float)
        if name in ("截止频率",):
            vy = np.clip(1 - np.log(np.clip(v, fmin, fmax) / fmin) / math.log(fmax / fmin), 0, 1) * (rows - 1) + top
        else:
            lo, hi = v.min(), v.max()
            vy = top + (1 - (v - lo) / (hi - lo + 1e-9)) * (rows - 1) * 0.9 + rows * 0.05
        xs = 50 + t / dur * cols
        k = max(1, len(t) // 1500)
        d.line(list(zip(xs[::k], vy[::k])), fill=(255, 120, 60), width=2)
        d.text((W - 260, top + 6), f"橙线 = {name}（真值）", fill=(255, 150, 100), font=fc)
    img.save(out, quality=92)


# ====================================================================== 主程序
def build(term, outdir, video=True):
    num, slug, cn, en, fn, *_ = term
    r = fn()
    st = sk.finish(r["st"])
    wav = outdir / f"{num}_{slug}.wav"
    if wav.exists():
        raise FileExistsError(f"{wav} 已存在 —— 换版本号")
    sf.write(str(wav), st.astype(np.float32), SR, subtype="PCM_24")
    embedded = r["mode"].startswith("延迟嵌入")
    lis = np.stack([st[:, 0], st[:, 0]], 1) if embedded else st      # 听的版本：延迟嵌入的只要真实声音（左声道）
    sf.write(str(outdir / f"{num}_{slug}_listen.wav"), lis.astype(np.float32), SR, subtype="PCM_24")
    scale = scope_scale(st)
    strip(wav, r["frames"], scale, outdir / f"{num}_{slug}_scope.jpg")
    spectro_plot(st, lis, r["f0"], outdir / f"{num}_{slug}_spec.jpg", r.get("curve"))
    if video:
        from scope_sim import Params, render_clip
        p = Params(W=960, H=720, scale=scale)
        render_clip(str(wav), outdir / f"{num}_{slug}_scope.mp4", 0.0, len(st) / SR, p, duty=0.9, log=lambda s: None)
    return dict(num=num, slug=slug, cn=cn, en=en, mode=r["mode"], scale=round(scale, 3), embedded=embedded,
                dur=round(len(st) / SR, 2), frames=[[t, h] for t, h in r["frames"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ver", required=True)
    ap.add_argument("--only", default=None)
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()
    outdir = OUT_BASE / a.ver
    outdir.mkdir(parents=True, exist_ok=True)
    meta = []
    for term in TERMS:
        if a.only and term[0] != a.only:
            continue
        print(f"{term[0]} {term[2]} …", flush=True)
        meta.append(build(term, outdir, video=not a.no_video))
    jp = outdir / "词表.json"
    if a.only and jp.exists():                                   # 只重做一个：合并进已有的索引
        old = {m_["num"]: m_ for m_ in json.loads(jp.read_text(encoding="utf-8"))}
        old.update({m_["num"]: m_ for m_ in meta})
        meta = [old[k] for k in sorted(old)]
    if not a.only or jp.exists():
        jp.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print("done", outdir)


if __name__ == "__main__":
    main()
