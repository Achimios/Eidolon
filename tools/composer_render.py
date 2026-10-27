#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🎛作曲器-渲染侧：读 HTML 作曲器导出的 作曲_vN.json（真值）→ 无损混音 + 每层分轨 + 报告。

页面（tools/composer_web/）里的声音只是预览；这里按**同一条信号链**用 numpy / scipy 离线算（确定性，采样级），
参数含义和页面 js/engine.js 一一对应（改一边要改另一边，见文末「对照表」）：
  音符 → synthVoice（齐奏 / 失谐 / FM / 两级低通 + 滤波包络 / LFO / 滑音 bend / ADSR，力度推音量和亮度）
  打击 → drumHit（kick / snare / clap / hat / ohat / perc / tom / impact，全是现算的）
  录音 / 分轨 → clips（file 相对作品文件夹；voice 层乘 rawGainDb；按场景切开，场景里不响的那段不放）
  每层总线：失真 tanh → 音量（gainDb + 音色响度修正 trim）→ 侧链闪避（kick 触发，深度 = sidechain 有效值）→ 声像 → 送混响 / 延迟
  母线：低通（cutoff / reso 旋钮）→ tanh（drive 旋钮）→ 混响返回（space）+ 延迟返回（反馈 g = pole_g，环内饱和）→ 母带（−14 LUFS / 真峰值 ≤ −1 dBTP）
  旋钮有效值 = v × (a + b × 张力)（macro_rules；张力为空 = 0.5）。
输出（不覆盖，已存在就报错）：<作品>/作曲/渲染_vN/ 混音.wav（48 kHz / 24 bit）、分轨/<层名>.wav、报告.json

用法：
  python tools/composer_render.py <作品文件夹>/作曲/作曲_v3.json            # → 作曲/渲染_v3/
  python tools/composer_render.py <json> --out <目录> --no-master            # 不做母带（给后续混音用）
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import synthkit as sk  # noqa: E402

SR = 48000
TAIL = 3.0

# ---- 和 js/engine.js 的 PRESETS 同一份（页面导出的 layers[].params 优先；这里只在缺字段时兜底）
PRESETS = {
    "bass":  dict(octave=2, wave="sawtooth", voices=1, detune=0, fmRatio=0, fmIndex=0, cutoff=520, reso=0.25, envAmt=0.5, attack=0.003, decay=0.25, sustain=0.5, release=0.08, lfoRate=5, lfoDepth=0, lfoTarget="cutoff", drive=0.25, delaySend=0, reverbSend=0.03, duck=True, pan=0),
    "pad":   dict(octave=4, wave="sawtooth", voices=5, detune=18, fmRatio=0, fmIndex=0, cutoff=1800, reso=0.1, envAmt=0.1, attack=0.6, decay=0.5, sustain=0.8, release=1.2, lfoRate=0.25, lfoDepth=0.15, lfoTarget="cutoff", drive=0, delaySend=0.1, reverbSend=0.45, duck=True, pan=0),
    "lead":  dict(octave=4, wave="square", voices=2, detune=8, fmRatio=0, fmIndex=0, cutoff=3000, reso=0.25, envAmt=0.35, attack=0.005, decay=0.2, sustain=0.7, release=0.15, lfoRate=5.5, lfoDepth=0, lfoTarget="pitch", drive=0.1, delaySend=0.25, reverbSend=0.2, duck=False, pan=0),
    "pluck": dict(octave=4, wave="triangle", voices=1, detune=0, fmRatio=0, fmIndex=0, cutoff=2500, reso=0.15, envAmt=0.6, attack=0.002, decay=0.35, sustain=0, release=0.2, lfoRate=5, lfoDepth=0, lfoTarget="pitch", drive=0, delaySend=0.3, reverbSend=0.25, duck=False, pan=0),
    "bell":  dict(octave=5, wave="sine", voices=1, detune=0, fmRatio=3.5, fmIndex=2.5, cutoff=9000, reso=0, envAmt=0, attack=0.002, decay=1.2, sustain=0, release=1.0, lfoRate=5, lfoDepth=0, lfoTarget="pitch", drive=0, delaySend=0.15, reverbSend=0.35, duck=False, pan=0),
    "arp":   dict(octave=4, wave="sawtooth", voices=1, detune=0, fmRatio=0, fmIndex=0, cutoff=1800, reso=0.3, envAmt=0.5, attack=0.002, decay=0.12, sustain=0.2, release=0.05, lfoRate=5, lfoDepth=0, lfoTarget="pitch", drive=0.1, delaySend=0.25, reverbSend=0.15, duck=True, pan=0),
    "sing":  dict(octave=4, wave="triangle", voices=2, detune=6, fmRatio=1, fmIndex=0.4, cutoff=2600, reso=0.1, envAmt=0.2, attack=0.02, decay=0.3, sustain=0.8, release=0.2, lfoRate=5.5, lfoDepth=0, lfoTarget="pitch", drive=0, delaySend=0.2, reverbSend=0.3, duck=False, pan=0),
}
MACRO_DEFAULTS = dict(tension=0.5, cutoff=1, reso=0.1, drive=0, space=0.25, pole_g=0.35, sidechain=0.4)
WAVE = {"sawtooth": "saw", "square": "square", "triangle": "tri", "sine": "sine"}


def db2a(d):
    return 10 ** (np.asarray(d, float) / 20)


def midi_hz(m):
    return 440.0 * 2 ** ((np.asarray(m, float) - 69) / 12)


# ====================================================================== 一个音（= engine.synthVoice）
def exp_seg(n, v0, v1, tau):
    """从 v0 指数逼近 v1（时间常数 tau 秒）= WebAudio setTargetAtTime"""
    return v1 + (v0 - v1) * np.exp(-np.arange(n) / SR / max(tau, 1e-4))


def adsr_curve(n_total, n_on, peak, attack, decay, sustain, release, start=0.0):
    """attack 线性到 peak → 指数逼近 peak·sustain（τ = decay/3）→ 松开后指数逼近 0（τ = release/4）"""
    na = max(1, int(attack * SR))
    e = np.empty(n_total)
    e[:na] = np.linspace(start, peak, na, endpoint=False)
    e[na:] = exp_seg(n_total - na, peak, peak * sustain, decay / 3)
    if n_on < n_total:
        e[n_on:] = exp_seg(n_total - n_on, e[n_on - 1] if n_on > 0 else 0.0, 0.0, release / 4)
    return e


def biquad_lp_tv(x, fc, q):
    """时变两极低通（= WebAudio BiquadFilter lowpass，Q 是线性值）。用 TPT SVF 的低通输出实现：k = 1/Q"""
    return sk.svf(x, fc, q, "lp", SR)


def synth_voice(p, pitch, vel, dur, bend=None):
    """返回 (立体声, 长度)。dur = 按住多久（秒）"""
    f = float(midi_hz(pitch))
    amp = 0.18 + 0.82 * vel ** 1.5
    bright = 2 ** ((vel - 0.7) * 1.5)
    rel = max(0.01, p["release"])
    total = dur + rel * 1.6 + 0.05
    n = int(total * SR)
    n_on = min(n, int(max(0.005, dur) * SR))
    t = np.arange(n) / SR
    cents = np.zeros(n)
    if bend and bend.get("cents") and len(bend["cents"]) > 1:
        bc = np.asarray(bend["cents"], float)
        cents += np.interp(t, np.arange(len(bc)) * bend.get("dt", 0.02), bc)
    lfo = np.sin(2 * np.pi * p["lfoRate"] * t) if p.get("lfoDepth", 0) > 0 else None
    if lfo is not None and p["lfoTarget"] == "pitch":
        cents += lfo * p["lfoDepth"] * 60
    nv = max(1, int(p["voices"]))
    sig = np.zeros((n, 2))
    for v in range(nv):
        det = (v / (nv - 1) - 0.5) * 2 * p["detune"] if nv > 1 else 0.0
        fv = f * 2 ** ((cents + det) / 1200)
        if p.get("fmIndex", 0) > 0 and p.get("fmRatio", 0) > 0:      # 频率调制：偏移 = index · f · ratio（= 页面的 mg.gain）
            fv = fv + p["fmIndex"] * f * p["fmRatio"] * np.sin(2 * np.pi * f * p["fmRatio"] * t)
        o = sk.osc(WAVE[p["wave"]], fv, total, SR, phase0=(v * 0.37) % 1.0)[:n]
        pan = (v / (nv - 1) - 0.5) * 1.2 if nv > 1 else 0.0
        sig += sk.stereo(o, float(np.clip(pan, -1, 1))) * (1 / math.sqrt(nv)) * (1.0 if nv > 1 else math.sqrt(2))
    if lfo is not None and p["lfoTarget"] == "amp":
        sig *= (1 + lfo * p["lfoDepth"] * 0.5)[:, None]
    # 滤波包络：base → peak（attack 线性）→ base + (peak − base)·sustain（τ = decay/3）
    fmax = min(18000, SR * 0.45)
    base = float(np.clip(p["cutoff"] * bright, 40, fmax))
    peak = float(np.clip(base * 2 ** (p["envAmt"] * 4), 40, fmax))
    na = max(1, int(max(0.003, p["attack"]) * SR))
    fc = np.empty(n)
    fc[:na] = np.linspace(base, peak, na, endpoint=False)
    fc[na:] = exp_seg(n - na, peak, base + (peak - base) * p["sustain"], max(0.01, p["decay"] / 3))
    if lfo is not None and p["lfoTarget"] == "cutoff":
        fc = fc * 2 ** (lfo * p["lfoDepth"] * 1800 / 1200)
    fc = np.clip(fc, 20, fmax)
    q1 = 0.5 + p["reso"] * 14
    out = np.stack([biquad_lp_tv(biquad_lp_tv(sig[:, c], fc, q1), fc, 0.707) for c in range(2)], 1)
    env = adsr_curve(n, n_on, amp, max(0.002, p["attack"]), max(0.03, p["decay"]), p["sustain"], rel)
    return out * env[:, None]


# ====================================================================== 鼓（= engine.drumHit）
def _env(n, peak, dec, t0=0):
    e = np.zeros(n)
    i0 = int(t0 * SR)
    na = max(1, int(0.002 * SR))
    if i0 >= n:
        return e
    e[i0:i0 + na] = np.linspace(0, peak, na)[:max(0, min(na, n - i0))]
    k = i0 + na
    if k < n:
        e[k:] = peak * np.exp(-np.arange(n - k) / SR / (dec / 4))
    return e


def _sweep(f0, f1, tf, n):
    t = np.arange(n) / SR
    f = f0 * (f1 / f0) ** np.clip(t / tf, 0, 1) if f1 else np.full(n, f0)
    return np.cumsum(f) / SR


def drum(sound, vel, seed=0):
    a = 0.25 + 0.75 * vel ** 1.3
    rng = np.random.default_rng(seed)
    nz = lambda d: rng.uniform(-1, 1, int(d * SR))
    if sound == "kick":
        n = int(0.6 * SR)
        x = np.sin(2 * np.pi * _sweep(150, 45, 0.11, n)) * _env(n, 1, 0.45)
        c = sk.onepole_hp(sk.onepole_hp(nz(0.01), 2000), 2000)
        x[:len(c)] += c * _env(len(c), 0.3, 0.01)
    elif sound == "snare":
        n = int(0.3 * SR)
        x = sk.svf(nz(0.3), 1800, 0.7, "bp") * _env(n, 0.8, 0.18) * 1.6
        tri = (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * _sweep(185, 160, 0.05, int(0.2 * SR))))
        x[:len(tri)] += tri * _env(len(tri), 0.5, 0.1)
    elif sound == "clap":
        n = int(0.4 * SR)
        e = np.zeros(n)
        tt = np.arange(n) / SR
        for dt in (0.0, 0.012, 0.024):
            m = tt >= dt
            e[m] = np.maximum(e[m], 0.1 + 0.8 * np.exp(-(tt[m] - dt) / 0.004))
        m = tt >= 0.034
        e[m] = 0.7 * np.exp(-(tt[m] - 0.034) / 0.05)
        x = sk.svf(nz(0.4), 1100, 1.2, "bp") * e * 2.0
    elif sound in ("hat", "ohat"):
        dec = 0.045 if sound == "hat" else 0.32
        d = dec * 2 + 0.05
        x = sk.onepole_hp(sk.onepole_hp(nz(d), 7500), 7500) * _env(int(d * SR), 0.45, dec)
    elif sound == "perc":
        n = int(0.06 * SR)
        sq = np.sign(np.sin(2 * np.pi * 820 * np.arange(n) / SR))
        x = sk.svf(sq, 1700, 0.7, "bp") * _env(n, 0.5, 0.03)
    elif sound == "tom":
        n = int(0.6 * SR)
        x = np.sin(2 * np.pi * _sweep(120, 78, 0.25, n)) * _env(n, 0.9, 0.35)
    elif sound == "impact":
        n = int(2.2 * SR)
        x = np.sin(2 * np.pi * _sweep(60, 28, 1.4, n)) * _env(n, 1, 1.6)
        x[:int(2 * SR)] += sk.onepole_lp(sk.onepole_lp(nz(2), 500), 500) * _env(int(2 * SR), 0.6, 1.4) * 2.5
    else:
        raise ValueError(f"不认识的鼓：{sound}")
    return sk.stereo(x * a, 0.0) * math.sqrt(2)


# ====================================================================== 音色响度修正（= engine.measureTrims）
_TRIMS = {}


def trim_for(preset):
    if preset in ("drums", "voice", "stem") or preset not in PRESETS:
        return 0.0
    if preset not in _TRIMS:
        p = PRESETS[preset]
        x = synth_voice(p, 12 * (p["octave"] + 1) + 4, 0.8, 1.2)[:int(1.2 * SR)]
        db = 10 * math.log10(float(np.mean(x ** 2)) + 1e-12)
        _TRIMS[preset] = float(np.clip(-20 - db, -12, 18))
    return _TRIMS[preset]


# ====================================================================== 旋钮
class Macros:
    def __init__(self, J, n, t0):
        self.t = t0 + np.arange(n) / SR
        defaults = {**MACRO_DEFAULTS, **(J.get("macro_defaults") or {})}
        ten = J.get("tension") or []
        tension = np.interp(self.t, [p[0] for p in ten], [p[1] for p in ten]) if ten else np.full(n, defaults["tension"])
        rules = J.get("macro_rules")
        self.eff = {"tension": tension}
        for k in ("cutoff", "reso", "drive", "space", "pole_g", "sidechain"):
            m = (J.get("macros") or {}).get(k)
            if m:
                tv = m["t0"] + np.arange(len(m["v"])) / m["fps"]
                v = np.interp(self.t, tv, m["v"])
            else:
                v = np.full(n, float(defaults[k]))
            if rules and k in rules:
                v = v * (rules[k]["a"] + rules[k]["b"] * tension)
            self.eff[k] = v

    def at(self, k, t):
        return float(np.interp(t, self.t, self.eff[k]))


# ====================================================================== 场景
def audible(J, layer, t):
    """层在 t 进不进成片：mute + 场景（= 页面 state.js 的 inMixAt）。
    音符 / 打击不用它：页面导出时已算好每个事件的 audible 字段，直接读（逻辑只有一份，在页面）。
    这里只剩两处用：clips（一段录音可能跨场景，页面不导出逐段结果）和没有 audible 字段的旧导出（v1.2 及以前，会打日志）。"""
    if layer.get("mute"):
        return False
    cur = None
    for sc in J.get("scenes") or []:
        if sc["t"] <= t + 1e-6:
            cur = sc
    return cur is None or layer["id"] in cur["layers"]


def scene_cuts(J, a, z):
    return [a] + [sc["t"] for sc in (J.get("scenes") or []) if a < sc["t"] < z] + [z]


# ====================================================================== 混响 / 延迟（= engine 的 _ir / dly）
def ir(sec=2.8):
    """和页面一样的冲激响应：LCG 噪声 × 一阶低通 × exp(−6.5·t/T)，前 10 ms 淡入"""
    n = int(SR * sec)
    out = np.zeros((n, 2))
    for ch in range(2):
        s = 777 + ch * 31
        w = np.empty(n)
        for i in range(n):                       # 同一个 LCG（页面的 Math 版）
            s = (s * 1103515245 + 12345) & 0x7FFFFFFF
            w[i] = s / 0x3FFFFFFF - 1
        lp = sk.onepole_lp(w, -math.log(0.65) * SR / (2 * math.pi))
        i = np.arange(n)
        out[:, ch] = lp * np.exp(-i / n * 6.5) * np.minimum(1.0, i / (SR * 0.01))
    return out


# ====================================================================== 主程序
def render(J, work_dir, master=True, log=print):
    t0, t1 = J["window"]
    n = int((t1 - t0 + TAIL) * SR)
    M = Macros(J, n, t0)
    spb = 60.0 / J["bpm"]
    layers = {L["id"]: L for L in J["layers"]}
    bus = {lid: np.zeros((n, 2)) for lid in layers}

    def put(lid, sig, t):
        i = int(round((t - t0) * SR))
        if i >= n or i + len(sig) <= 0:
            return
        a = max(0, -i)
        m = min(len(sig), n - i)
        bus[lid][i + a:i + m] += sig[a:m]

    def heard(ev, L):
        if "audible" in ev:
            return bool(ev["audible"])
        if not old_export:
            old_export.append(1)
            log("⚠ 旧导出：notes / hits 没有 audible 字段，按 mute + 场景现算（页面 v1.3 起导出这个字段）")
        return audible(J, L, ev["t"])

    old_export = []
    # 音符
    for k, nt in enumerate(J.get("notes") or []):
        L = layers.get(nt["layer"])
        if not L or not heard(nt, L):
            continue
        if L["kind"] == "synth":
            p = {**PRESETS.get(L.get("preset"), PRESETS["lead"]), **L["params"]}
            bend = None
        elif L["kind"] == "voice" and L["params"].get("synthOn", True):
            p = PRESETS[L["params"].get("synthPreset", "sing")]
            bend = nt.get("bend") if L["params"].get("bendOn", True) else None
        else:
            continue
        put(L["id"], synth_voice(p, nt["pitch"], nt["vel"], max(0.03, nt["dur"]), bend), nt["t"])
    # 打击 + 侧链触发点
    kicks = []
    for k, h in enumerate(J.get("hits") or []):
        L = layers.get(h["layer"])
        if not L or not heard(h, L):
            continue
        put(L["id"], drum(h["sound"], h.get("vel", 0.8), seed=k), h["t"])
        if h["sound"] == "kick":
            kicks.append(h["t"])
    # 录音 / 分轨
    for c in J.get("clips") or []:
        if not c.get("active", True) or not c.get("file"):
            continue
        L = layers.get(c["layer"])
        if not L:
            continue
        x, sr = sf.read(str(Path(work_dir) / c["file"]), always_2d=True)
        x = x[:, :2] if x.shape[1] >= 2 else np.repeat(x, 2, axis=1)
        if sr != SR:
            g = math.gcd(sr, SR)
            x = resample_poly(x, SR // g, sr // g, axis=0)
        off = float(c.get("offset", 0.0) or (c["src"].get("offset", 0.0) if isinstance(c.get("src"), dict) else 0.0))
        gain = float(db2a(L["params"].get("rawGainDb", 0))) if L["kind"] == "voice" else 1.0
        if gain < 0.01:
            continue
        for a, z in zip(scene_cuts(J, c["t"], c["t"] + c["dur"])[:-1], scene_cuts(J, c["t"], c["t"] + c["dur"])[1:]):
            if not audible(J, L, a + 1e-6):
                continue
            i0 = int((a - c["t"] + off) * SR)
            put(L["id"], x[i0:i0 + int((z - a) * SR)] * gain, a)
    # 每层总线 → 母线 + 送出
    total = np.zeros((n, 2)); rev_send = np.zeros((n, 2)); dly_send = np.zeros((n, 2))
    duck = np.ones(n)
    tt = t0 + np.arange(n) / SR
    for tk in kicks:                                         # 侧链：kick 时压到 1 − depth，+20 ms 起按 τ 0.09 s 弹回
        depth = M.at("sidechain", tk)
        if depth <= 0.001:
            continue
        i = int((tk - t0) * SR)
        r = (np.arange(n - i) / SR) - 0.02
        duck[i:] = np.minimum(duck[i:], 1 - depth * np.where(r < 0, 1.0, np.exp(-np.clip(r, 0, None) / 0.09)))
    stems = {}
    for lid, x in bus.items():
        L = layers[lid]
        p = L["params"]
        drv = p.get("drive", 0) or 0
        k_ = 1 + drv * 18
        y = np.tanh(k_ * x) / math.tanh(k_)                  # 页面 tanhCurve(1 + drive·18)（归一到 ±1）
        g = db2a((L.get("gain_db", 0) or 0) + trim_for(L.get("preset") if L["kind"] == "synth" else L["kind"])) / (1 + drv * 1.2)
        y = y * g
        if p.get("duck"):
            y = y * duck[:, None]
        pan = float(np.clip(p.get("pan", 0) or 0, -1, 1))
        if pan:
            th = (pan + 1) * np.pi / 4
            y = y * np.array([math.cos(th), math.sin(th)]) * math.sqrt(2)
        stems[L.get("name") or lid] = y
        total += y
        rev_send += y * (p.get("reverbSend", 0) or 0)
        dly_send += y * (p.get("delaySend", 0) or 0)
    # 延迟：时间 = 0.75 拍（页面 setDelayTime(spb·0.75)），环里低通 4.5 kHz + tanh，反馈 = pole_g，返回 0.6
    dt = float(np.clip(spb * 0.75, 0.01, 3.9))
    g_curve = M.eff["pole_g"]
    dly = np.stack([sk.feedback_delay_sat(dly_send[:, c], dt, g_curve, sr=SR, damp=4500, drive=1.5) - dly_send[:, c] for c in range(2)], 1)
    total += dly * 0.6
    # 混响：卷积，返回 = space × 1.2（随时间）
    if np.any(rev_send):
        R = ir()
        wet = np.stack([fftconvolve(rev_send[:, c], R[:, c])[:n] for c in range(2)], 1)
        total += wet * (M.eff["space"] * 1.2)[:, None]
    # 母线：低通（两级，Q1 = 0.7 + reso·12）→ tanh（1 + drive·14）+ 补偿
    fc = 60 * 2 ** (M.eff["cutoff"] * 8.4)
    if float(np.min(fc)) < 18000:
        q1 = 0.7 + M.eff["reso"] * 12
        total = np.stack([sk.svf(sk.svf(total[:, c], fc, q1), fc, 0.707) for c in range(2)], 1)
    kd = 1 + M.eff["drive"] * 14
    total = np.tanh(kd[:, None] * total) / np.tanh(kd)[:, None] / (1 + M.eff["drive"] * 1.6)[:, None]
    rep = {"window": [t0, t1], "tail_s": TAIL, "sr": SR, "notes": len(J.get("notes") or []), "hits": len(J.get("hits") or []),
           "clips": len(J.get("clips") or []), "trims_db": {k: round(v, 2) for k, v in _TRIMS.items()}, "delay_s": round(dt, 4)}
    if master:
        sys.path.insert(0, str(ROOT / "tools"))
        from remix_cku import master as _master, true_peak
        total, lufs, tp = _master(total)
        rep.update(lufs=round(lufs, 2), true_peak_dbtp=round(tp, 2))
    rep["clipped_samples"] = int((np.abs(total) >= 0.999).sum())
    return total, stems, rep


def main():
    ap = argparse.ArgumentParser(description="作曲.json → 无损混音 + 分轨")
    ap.add_argument("json")
    ap.add_argument("--out", default=None, help="默认：同目录下 渲染_<版本>/")
    ap.add_argument("--no-master", action="store_true")
    a = ap.parse_args()
    jp = Path(a.json)
    J = json.loads(jp.read_text(encoding="utf-8"))
    work = jp.parent.parent if jp.parent.name == "作曲" else jp.parent
    ver = jp.stem.split("_")[-1]
    out = Path(a.out) if a.out else jp.parent / f"渲染_{ver}"
    if out.exists():
        raise FileExistsError(f"{out} 已存在 —— 不覆盖，换版本号或 --out")
    mix, stems, rep = render(J, work, master=not a.no_master)
    (out / "分轨").mkdir(parents=True)
    sf.write(str(out / "混音.wav"), mix.astype(np.float32), SR, subtype="PCM_24")
    for name, y in stems.items():
        safe = "".join(ch if ch not in '\\/:*?"<>|' else "_" for ch in name)
        sf.write(str(out / "分轨" / f"{safe}.wav"), y.astype(np.float32), SR, subtype="PCM_24")
    rep["source"] = jp.name
    (out / "报告.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False))


if __name__ == "__main__":
    main()

# 对照表（页面 js/engine.js ↔ 这里）
#   synthVoice: amp = 0.18 + 0.82·vel^1.5；bright = 2^((vel − 0.7)·1.5)；f1 Q = 0.5 + reso·14、f2 Q = 0.707；peak = base·2^(envAmt·4)
#               LFO pitch 深度·60 音分 / cutoff 深度·1800 音分 / amp 深度·0.5；齐奏声像 ±0.6；FM 频偏 = index·f·ratio
#   drumHit:    各鼓的频率 / 衰减 / 带通照抄；力度 a = 0.25 + 0.75·vel^1.3
#   层总线:      tanh(1 + drive·18)；音量 / (1 + drive·1.2)；trim = clamp(−20 − 一个音的 RMS dB, −12, 18)
#   母线:        cutoff → 60·2^(v·8.4) Hz；Q1 = 0.7 + reso·12；tanh(1 + drive·14)、补偿 1/(1 + drive·1.6)；混响返回 space·1.2；延迟 0.75 拍、反馈 pole_g、返回 0.6
#   差别:        页面的母线限幅器（DynamicsCompressor）这里换成正式母带（−14 LUFS / 真峰值 ≤ −1 dBTP）；延迟环的 tanh 归一方式略不同（这里 tanh(1.5·g·x)/1.5）
