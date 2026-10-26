# -*- coding: utf-8 -*-
""">v<🎛合成器积木（numpy + scipy，确定性）：振荡器 / 包络 / 滤波 / LFO / FM / 失谐齐奏 / 失真 / 延迟 / 混响 / 侧链 / 鼓。

给「合成器词表」和以后的作曲器共用。每个函数都是纯函数：同样的参数 → 同样的采样（噪声都带 seed）。
约定：采样率 SR（默认 48 kHz）；单声道 = 1 维 float64；立体声 = (n, 2)。音高用 Hz 或 MIDI（midi_hz）。
不做的事：不实时、不读 MIDI 设备 —— 这一层只是「声音的数学」，乐器外壳以后另写（见 _设计/设计_作曲器.md）。
"""
import math

import numpy as np
from scipy.signal import lfilter

SR = 48000


def midi_hz(m):
    return 440.0 * 2.0 ** ((np.asarray(m, float) - 69.0) / 12.0)


def tgrid(dur, sr=SR):
    return np.arange(int(round(dur * sr))) / sr


def as_curve(v, n):
    """常数或数组 → 长度 n 的数组"""
    a = np.asarray(v, float)
    return np.full(n, float(a)) if a.ndim == 0 else a[:n]


# ------------------------------------------------------------------ 振荡器
def phase_of(freq, n, sr=SR, phase0=0.0):
    """频率曲线 → 相位（周期数，未取模）。频率可随时间变（滑音、颤音、FM 都走这里）"""
    f = as_curve(freq, n)
    return phase0 + np.concatenate(([0.0], np.cumsum(f[:-1]))) / sr, f / sr


def _polyblep(t, dt):
    """带限修正：只在相位跳变附近一个采样内起作用（去掉锯齿 / 方波的混叠）"""
    out = np.zeros_like(t)
    dt = np.maximum(dt, 1e-9)
    m = t < dt
    x = t[m] / dt[m]
    out[m] = x + x - x * x - 1.0
    m = t > 1.0 - dt
    x = (t[m] - 1.0) / dt[m]
    out[m] = x * x + x + x + 1.0
    return out


def osc(kind, freq, dur, sr=SR, phase0=0.0, pw=0.5):
    """kind: sine / saw / square / tri。saw、square 用 PolyBLEP 带限；tri 用奇次谐波相加（到 Nyquist）"""
    n = int(round(dur * sr))
    ph, dt = phase_of(freq, n, sr, phase0)
    if kind == "sine":
        return np.sin(2 * np.pi * ph)
    t = ph % 1.0
    if kind == "saw":
        return 2.0 * t - 1.0 - _polyblep(t, dt)
    if kind == "square":
        t2 = (t + (1 - pw)) % 1.0
        return (2.0 * t - 1.0 - _polyblep(t, dt)) - (2.0 * t2 - 1.0 - _polyblep(t2, dt))
    if kind == "tri":
        f = as_curve(freq, n)
        out = np.zeros(n)
        k = 1
        while k * f.min() < sr / 2 and k < 400:
            amp = (8 / np.pi ** 2) * ((-1) ** ((k - 1) // 2)) / k ** 2
            out += amp * np.sin(2 * np.pi * k * ph) * (k * f < sr / 2)
            k += 2
        return out
    raise ValueError(kind)


def additive(freq, amps, dur, sr=SR, phase0=0.0):
    """泛音相加：amps[k-1] = 第 k 次泛音的幅度（可以是随时间变的数组）"""
    n = int(round(dur * sr))
    ph, _ = phase_of(freq, n, sr, phase0)
    f = as_curve(freq, n)
    out = np.zeros(n)
    for k, a in enumerate(amps, start=1):
        out += as_curve(a, n) * np.sin(2 * np.pi * k * ph) * (k * f < sr / 2)
    return out


def fm(fc, ratio, index, dur, sr=SR, feedback=0.0):
    """两算子 FM（实际是相位调制，DX7 的做法）：载波相位 += index · sin(调制器相位)。index 可随时间变"""
    n = int(round(dur * sr))
    phc, _ = phase_of(fc, n, sr)
    phm, _ = phase_of(as_curve(fc, n) * ratio, n, sr)
    return np.sin(2 * np.pi * phc + as_curve(index, n) * np.sin(2 * np.pi * phm))


def noise(dur, sr=SR, seed=0):
    return np.random.default_rng(seed).uniform(-1, 1, int(round(dur * sr)))


# ------------------------------------------------------------------ 包络 / LFO
def adsr(n_on, a, d, s, r, sr=SR, curve=4.0):
    """按住 n_on 个采样；返回长度 n_on + r·sr 的包络。起音线性，衰减 / 释放按指数形（curve 越大越像真的乐器）"""
    na, nd, nr = max(1, int(a * sr)), max(1, int(d * sr)), max(1, int(r * sr))
    env = np.empty(n_on + nr)
    i = np.arange(n_on)
    att = np.minimum(i / na, 1.0)
    dec = s + (1 - s) * np.exp(-curve * np.clip(i - na, 0, None) / nd)
    env[:n_on] = np.where(i < na, att, dec)
    last = env[n_on - 1]
    env[n_on:] = last * np.exp(-curve * np.arange(nr) / nr)
    return env


def lfo(rate, n, sr=SR, shape="sine", phase0=0.0):
    ph = phase0 + np.arange(n) * rate / sr
    if shape == "sine":
        return np.sin(2 * np.pi * ph)
    if shape == "tri":
        return 1 - 4 * np.abs((ph % 1.0) - 0.5)
    if shape == "square":
        return np.where((ph % 1.0) < 0.5, 1.0, -1.0)
    raise ValueError(shape)


# ------------------------------------------------------------------ 滤波
def svf(x, cutoff, q=0.707, mode="lp", sr=SR):
    """TPT 状态变量滤波器（Zavalishin），截止频率可随时间变；q 越大共振越强（在截止处鼓起一个峰）"""
    n = len(x)
    fc = np.clip(as_curve(cutoff, n), 20, sr * 0.45)
    g = np.tan(np.pi * fc / sr)
    k = 1.0 / max(q, 0.05)
    a1 = 1.0 / (1.0 + g * (g + k))
    a2 = g * a1
    a3 = g * a2
    lp = np.empty(n); bp = np.empty(n); hp = np.empty(n)
    ic1 = ic2 = 0.0
    xs = x.tolist(); a1l = a1.tolist(); a2l = a2.tolist(); a3l = a3.tolist()
    for i in range(n):
        v3 = xs[i] - ic2
        v1 = a1l[i] * ic1 + a2l[i] * v3
        v2 = ic2 + a2l[i] * ic1 + a3l[i] * v3
        ic1 = 2 * v1 - ic1
        ic2 = 2 * v2 - ic2
        lp[i] = v2; bp[i] = v1
    if mode == "lp":
        return lp
    if mode == "bp":
        return bp
    hp = x - k * bp - lp
    return hp


def onepole_lp(x, fc, sr=SR):
    a = math.exp(-2 * math.pi * fc / sr)
    return lfilter([1 - a], [1, -a], x)


def onepole_hp(x, fc, sr=SR):
    return x - onepole_lp(x, fc, sr)


# ------------------------------------------------------------------ 失真
def drive(x, amount):
    """tanh 饱和：amount 是推进去的增益；输出按 tanh(amount) 归一，所以响度大致不变、形状变方"""
    a = as_curve(amount, len(x))
    return np.tanh(a * x) / np.tanh(np.maximum(a, 1e-6))


# ------------------------------------------------------------------ 空间
def stereo(m, pan=0.0):
    """等功率声像：pan -1 左 … +1 右（可以是曲线）"""
    p = as_curve(pan, len(m))
    th = (p + 1) * np.pi / 4
    return np.stack([m * np.cos(th), m * np.sin(th)], 1)


def feedback_delay(x, t, g, sr=SR, damp=6000.0):
    """y[n] = x[n] + g·LP(y[n-D])：一个 IIR 梳状滤波器。极点在 z^D = g 上 —— g < 1 回声消散，g → 1 永远不停"""
    D = int(round(t * sr))
    a = math.exp(-2 * math.pi * damp / sr)
    # 环路里加一阶低通 (1-a)/(1-a z^-1)：分母 (1 - a z^-1) - g(1-a) z^-D
    den = np.zeros(D + 1); den[0] = 1.0; den[1] -= a; den[D] -= g * (1 - a)
    num = np.zeros(2); num[0] = 1.0; num[1] = -a
    return lfilter(num, den, x)


def pingpong(m, t, g, n_rep=10, sr=SR, damp=5000.0):
    """乒乓延迟（写成 FIR 展开：第 k 次回声 = g^k·低通^k，左右交替）。返回立体声，干声在中间"""
    D = int(round(t * sr))
    out = np.zeros((len(m) + D * n_rep, 2))
    out[:len(m)] += np.stack([m, m], 1) * 0.707
    e = m.copy()
    for k in range(1, n_rep + 1):
        e = onepole_lp(e, damp, sr) * g
        ch = 0 if k % 2 else 1
        out[k * D:k * D + len(m), ch] += e
    return out


_COMB = (1557, 1617, 1491, 1422, 1277, 1356, 1188, 1116)   # Freeverb 的梳状延迟（44.1 kHz 采样数）
_AP = (225, 556, 441, 341)


def reverb(x_st, size=0.84, damp=0.25, wet=0.3, sr=SR, spread=23, predelay=0.012):
    """Schroeder / Freeverb：8 个带阻尼的 IIR 梳状 + 4 个全通。x_st 立体声 → 立体声（干 + 湿）"""
    sc = sr / 44100.0
    pd = int(predelay * sr)
    outs = []
    for ch in range(2):
        x = np.concatenate([np.zeros(pd), x_st[:, ch]]) * 0.015
        acc = np.zeros(len(x))
        for c in _COMB:
            D = int((c + spread * ch) * sc)
            den = np.zeros(D + 1); den[0] = 1.0; den[1] = -damp; den[D] -= size * (1 - damp)
            acc += lfilter([1.0, -damp], den, x)
        for a in _AP:
            D = int((a + spread * ch) * sc)
            b = np.zeros(D + 1); b[0] = -0.5; b[D] = 1.0
            den = np.zeros(D + 1); den[0] = 1.0; den[D] = -0.5
            acc = lfilter(b, den, acc)
        outs.append(acc[:len(x_st)])
    w = np.stack(outs, 1)
    return x_st * (1 - wet * 0.5) + w * wet * 3.0


# ------------------------------------------------------------------ 齐奏
def unison(kind, freq, dur, voices=7, cents=18.0, width=1.0, sr=SR, seed=1):
    """齐奏：voices 个振荡器，音高均匀铺在 ±cents 音分，随机初相，左右铺开。返回立体声"""
    rng = np.random.default_rng(seed)
    n = int(round(dur * sr))
    out = np.zeros((n, 2))
    for i in range(voices):
        d = 0.0 if voices == 1 else (i / (voices - 1) * 2 - 1)
        f = as_curve(freq, n) * 2 ** (d * cents / 1200)
        v = osc(kind, f, dur, sr, phase0=rng.random())
        out += stereo(v, d * width)
    return out / math.sqrt(voices)


# ------------------------------------------------------------------ 鼓
def kick(sr=SR, f0=150.0, f1=45.0, dur=0.45, click=0.3, decay=0.16):
    t = tgrid(dur, sr)
    f = f1 + (f0 - f1) * np.exp(-t / 0.03)
    body = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t / decay)
    c = noise(0.004, sr, 7)
    body[:len(c)] += click * c * np.linspace(1, 0, len(c))
    return np.tanh(1.6 * body) / np.tanh(1.6)


def hat(sr=SR, dur=0.06, seed=3, open_=False):
    d = 0.32 if open_ else dur
    t = tgrid(d, sr)
    h = onepole_hp(noise(d, sr, seed), 7000, sr)
    return h * np.exp(-t / (0.09 if open_ else 0.018)) * 0.8


def clap(sr=SR, seed=5):
    t = tgrid(0.35, sr)
    nz = svf(noise(0.35, sr, seed), 1400, 1.2, "bp", sr)
    env = np.exp(-t / 0.07)
    for k, dt_ in enumerate((0.0, 0.011, 0.023)):          # 三下快速拍手叠在一起
        env = np.maximum(env, np.where(t >= dt_, np.exp(-(t - dt_) / 0.006), 0) * (k == 2 or 0.8))
    return nz * env * 2.2


def place(buf, s, at, sr=SR, gain=1.0):
    """把单声道 / 立体声 s 加到 buf 的 at 秒处（越界截掉）"""
    i = int(round(at * sr))
    if i >= len(buf):
        return buf
    m = min(len(s), len(buf) - i)
    if buf.ndim == 2 and s.ndim == 1:
        buf[i:i + m] += gain * s[:m, None]
    else:
        buf[i:i + m] += gain * s[:m]
    return buf


# ------------------------------------------------------------------ 收尾
def finish(st, rms_db=-18.0, peak_db=-1.0, fade=0.02, sr=SR):
    """立体声收尾：去直流 → 按 RMS 定响度 → 峰值超过就整体压下来（不削波）→ 首尾淡入淡出"""
    st = st - st.mean(0, keepdims=True)
    rms = math.sqrt(float(np.mean(st ** 2)) + 1e-12)
    st = st * (10 ** (rms_db / 20) / rms)
    pk = float(np.abs(st).max())
    lim = 10 ** (peak_db / 20)
    if pk > lim:
        st = st * (lim / pk)
    nf = int(fade * sr)
    ramp = np.linspace(0, 1, nf)
    st[:nf] *= ramp[:, None]
    st[-nf:] *= ramp[::-1, None]
    return st


def feedback_delay_sat(x, t, g, sr=SR, damp=5000.0, drive=1.0):
    """带饱和的反馈延迟：y[n] = x[n] + tanh(drive · g[n] · LP(y[n-D])) / drive。
    ⭐限幅在环里（tanh），所以 g ≥ 1 时也有界：极点越过单位圆 = 反馈失稳、自激，但幅度被 tanh 接住（≤ 1/drive + |x|）。
    g 可以是曲线（情绪 → 极点位置）。按 D 个采样一块向量化（第 k 块只依赖第 k-1 块）。x 单声道。"""
    D = max(1, int(round(t * sr)))
    n = len(x)
    gc = as_curve(g, n)
    a = math.exp(-2 * math.pi * damp / sr)
    y = np.zeros(n)
    zi = np.zeros(1)
    prev = np.zeros(D)
    for k in range(0, n, D):
        m = min(D, n - k)
        lp, zi = lfilter([1 - a], [1, -a], prev[:m], zi=zi)
        y[k:k + m] = x[k:k + m] + np.tanh(drive * gc[k:k + m] * lp) / drive
        prev = np.zeros(D)
        prev[:m] = y[k:k + m]
    return y


def feedback_delay_sat_xy(xy, t, g, sr=SR, damp=5000.0, drive=1.0):
    """立体声（XY）版：左右各自的延迟 t = (tL, tR)，饱和是**径向**的 —— 对向量 (L, R) 的模长做 tanh、方向不变：
    v = g·(LP(yL[n-DL]), LP(yR[n-DR]))，v ← v · tanh(drive·|v|) / (drive·|v|)，y = x + v。
    示波器上饱和边界是一个圆（分声道 tanh 会是正方形）。g ≥ 1 也有界（|v| ≤ 1/drive）。按 min(DL, DR) 分块向量化。"""
    tl, tr = (t, t) if np.isscalar(t) else t
    DL, DR = max(1, int(round(tl * sr))), max(1, int(round(tr * sr)))
    B = min(DL, DR)
    n = len(xy)
    gc = as_curve(g, n)
    a = math.exp(-2 * math.pi * damp / sr)
    y = np.zeros((n, 2))
    zl, zr = np.zeros(1), np.zeros(1)
    for k in range(0, n, B):
        m = min(B, n - k)
        idx = np.arange(k, k + m)
        dl = np.where(idx - DL >= 0, y[np.clip(idx - DL, 0, None), 0], 0.0)
        dr = np.where(idx - DR >= 0, y[np.clip(idx - DR, 0, None), 1], 0.0)
        fl, zl = lfilter([1 - a], [1, -a], dl, zi=zl)
        fr, zr = lfilter([1 - a], [1, -a], dr, zi=zr)
        v = gc[k:k + m, None] * np.stack([fl, fr], 1)
        r = np.linalg.norm(v, axis=1) * drive
        sc = np.where(r > 1e-12, np.tanh(r) / np.maximum(r, 1e-12), 1.0)
        y[k:k + m] = xy[k:k + m] + v * sc[:, None]
    return y
