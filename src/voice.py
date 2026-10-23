# -*- coding: utf-8 -*-
""">v<👂感官层-人声-从人声分轨（Demucs vocals.wav）算声学特征 → 感官.json 的 `voice` 节点（2026-10-23）。
通用：任何歌的人声分轨都能喂；不做歌词识别 / 转写，只要声学特征。只依赖 numpy。

逐帧（和 channels 同 fps、同长度）：
  loudness       响度 0~1（40 ms RMS 的 dB，50 ms 平滑；0 = 比全曲人声峰值低 40 dB）
  voiced         有声概率 0~1（CMNDF 最低谷值 d 的 logistic：d=0.2 → 0.5，d≤0.1 → >0.9；再乘响度门）
                 （先试过 pYIN 的 Beta(2,18) 阈值先验 CDF：d=0.1 只给 0.42，真人声段有声率被压到一半，2026-10-23 换掉）
  f0             基频 Hz；只在有声帧给，其余 null（YIN + 抛物线插值 + 八度跳变修正 + 5 帧中值）
  vibrato        颤音强度 0~1（f0 在 0.6 s 窗里拟合 二次趋势 + 4–8 Hz 正弦，= 正弦解释的方差比 × 深度饱和）
  vibrato_cents  颤音半幅（音分）；vibrato_hz 颤音频率 —— 只在拟合可信（强度 > 0）的帧给，其余 null
区段（秒）：
  phrases   乐句：有声段之间的空隙 < 0.18 s 并成一句（辅音 / 滑音的短断不算断句）；start / end = 首 / 末有声帧中心
  breaths   换气：相邻两句之间 0.18–1.0 s 的空隙；更长的是休止
  sections  人声段：乐句之间空隙 < 3 s 并成一段（「女声在哪几段」）
"""
import numpy as np

SR = 16000
HOP = 160                 # 10 ms 分析帧
W = 640                   # YIN 积分窗 40 ms
FMIN, FMAX = 65.0, 1100.0
TAU_MAX = int(SR / FMIN)  # 246
TAU_MIN = int(SR / FMAX)  # 14
ABS_THRESH = 0.15         # YIN 绝对阈值：第一个低于它的谷 = 周期
P_MID, P_WIDTH = 0.2, 0.04  # 有声概率的 logistic
FLOOR_DB = 40.0           # 比人声峰值（99.5 百分位）低 40 dB 以下 = 无声
MERGE_GAP = 0.18          # 乐句内短断
BREATH_MAX = 1.0          # 换气上限
SECTION_GAP = 3.0         # 人声段
VIB_WIN = 0.6             # 颤音拟合窗（秒）
VIB_BAND = (4.0, 8.0)


def frame_rms_db(x, hop=HOP, win=W):
    """中心对齐：第 k 帧的中心 = k·hop 样本"""
    xp = np.pad(x, (win // 2, win))
    n = len(x) // hop + 1
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    p = (xp[idx] ** 2).mean(1)
    p = np.convolve(np.pad(p, 2, mode="edge"), np.ones(5) / 5, "valid")    # 50 ms 平滑（功率域）
    return 10 * np.log10(p + 1e-12)


def yin(x, hop=HOP, chunk=1500):
    """→ (tau 估计（样本，浮点；未找到 = 0）, 最低谷 CMNDF 值, cmndf 矩阵 (帧, TAU_MAX+1))
    第 k 帧的积分窗 = [k·hop − W/2, k·hop + W/2)，和 rms 的中心对齐。"""
    xp = np.pad(x, (W // 2, W + TAU_MAX))
    n = len(x) // hop + 1
    nfft = 1 << int(np.ceil(np.log2(W + TAU_MAX + W)))
    taus = np.arange(TAU_MAX + 1)
    tau_out = np.zeros(n); dmin = np.ones(n); cm_all = np.ones((n, TAU_MAX + 1), np.float32)
    for c0 in range(0, n, chunk):
        ks = np.arange(c0, min(n, c0 + chunk))
        seg = xp[np.arange(W + TAU_MAX)[None, :] + hop * ks[:, None]]
        a = seg[:, :W]
        r = np.fft.irfft(np.conj(np.fft.rfft(a, nfft)) * np.fft.rfft(seg, nfft), nfft)[:, :TAU_MAX + 1]
        cs = np.concatenate([np.zeros((len(ks), 1)), np.cumsum(seg ** 2, 1)], 1)
        e0 = cs[:, W][:, None]
        et = cs[:, taus + W] - cs[:, taus]
        d = np.maximum(e0 + et - 2 * r, 0)
        d[:, 0] = 0
        cum = np.cumsum(d[:, 1:], 1)
        cm = np.ones_like(d)
        cm[:, 1:] = d[:, 1:] * taus[1:] / (cum + 1e-12)
        cm[e0[:, 0] < 1e-10] = 1.0
        cm_all[ks] = cm
        # 局部谷（只在 [TAU_MIN, TAU_MAX)）
        v = cm[:, TAU_MIN:TAU_MAX]
        trough = np.zeros_like(v, bool)
        trough[:, 1:-1] = (v[:, 1:-1] < v[:, :-2]) & (v[:, 1:-1] <= v[:, 2:])
        tv = np.where(trough, v, np.inf)
        gmin = tv.min(1)
        below = trough & (v < ABS_THRESH)
        first = np.where(below.any(1), below.argmax(1), tv.argmin(1))
        # YIN step 5 的精神：第一个过阈值的谷，往后滑到这个谷的底
        tau_out[ks] = first + TAU_MIN
        dmin[ks] = np.where(np.isfinite(gmin), gmin, 1.0)
    # 抛物线插值
    t = tau_out.astype(int)
    ok = (t > 0) & (t < TAU_MAX)
    i = np.nonzero(ok)[0]
    y0, y1, y2 = cm_all[i, t[i] - 1], cm_all[i, t[i]], cm_all[i, t[i] + 1]
    den = y0 - 2 * y1 + y2
    off = np.where(np.abs(den) > 1e-9, 0.5 * (y0 - y2) / np.where(np.abs(den) > 1e-9, den, 1), 0)
    tau_f = tau_out.astype(float)
    tau_f[i] += np.clip(off, -0.5, 0.5)
    return tau_f, dmin, cm_all


def _runs(mask):
    """True 段 → [(a, b)) 帧下标"""
    m = np.r_[False, mask, False].astype(int)
    d = np.diff(m)
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def _trough_near(cm_row, tau, rel=0.12):
    """cm_row 在 tau·(1±rel) 里的最低点（抛物线插值）→ (tau, 值)；越界返回 (None, 1)"""
    lo, hi = max(TAU_MIN, int(tau * (1 - rel))), min(TAU_MAX - 1, int(np.ceil(tau * (1 + rel))))
    if hi - lo < 2:
        return None, 1.0
    j = lo + int(np.argmin(cm_row[lo:hi + 1]))
    if 0 < j < TAU_MAX:
        y0, y1, y2 = cm_row[j - 1], cm_row[j], cm_row[j + 1]
        den = y0 - 2 * y1 + y2
        return j + (0.5 * (y0 - y2) / den if abs(den) > 1e-9 else 0.0), float(y1)
    return float(j), float(cm_row[j])


def octave_fix(f0, voiced, cm, win=31, passes=2):
    """八度跳变：和 0.3 s 内有声帧的 log2 中值差 ≈ ±1 八度的帧，改选 2τ / τ/2 附近的谷（谷要够深 < 0.5）"""
    f = f0.copy()
    lf = np.full(len(f), np.nan)
    for _ in range(passes):
        lf[:] = np.nan
        lf[voiced] = np.log2(f[voiced])
        h = win // 2
        for k in np.nonzero(voiced)[0]:
            seg = lf[max(0, k - h):k + h + 1]
            seg = seg[np.isfinite(seg)]
            if len(seg) < 5:
                continue
            dev = lf[k] - np.median(seg)
            for octv, mul in ((1, 2.0), (-1, 0.5)):      # 高了一个八度 → 周期 ×2；低了 → ÷2
                if abs(dev - octv) < 0.3:
                    tau = SR / f[k] * mul
                    t2, v2 = _trough_near(cm[k], tau)
                    if t2 is not None and v2 < 0.5:
                        f[k] = SR / t2
    return f


def median_fill(f0, voiced, n=5):
    """有声段内的 log 域中值（段边缘窗口收窄，不跨段）"""
    out = f0.copy()
    h = n // 2
    for a, b in _runs(voiced):
        lf = np.log2(f0[a:b])
        for k in range(b - a):
            out[a + k] = 2 ** np.median(lf[max(0, k - h):k + h + 1])
    return out


def vibrato(f0, voiced, fr=100.0, win=VIB_WIN, step=5):
    """每 step 帧一个窗：cents = 二次趋势 + A·sin(2πft+φ)；f 扫 3.5–9 Hz 取解释方差最大者。
    → strength, depth_cents, rate_hz（逐分析帧；没拟合 = 0 / nan）"""
    n = len(f0)
    S = np.zeros(n); D = np.full(n, np.nan); R = np.full(n, np.nan)
    L = int(round(win * fr))
    freqs = np.arange(3.5, 9.0001, 0.05)
    for a, b in _runs(voiced):
        if b - a < int(0.4 * fr):
            continue
        c = 1200 * np.log2(f0[a:b] / 440.0)
        Lw = min(L, b - a)
        tt = (np.arange(Lw) - Lw / 2) / fr
        P = np.stack([np.ones(Lw), tt, tt ** 2], 1)               # 趋势
        Cs = np.cos(2 * np.pi * freqs[:, None] * tt[None]); Sn = np.sin(2 * np.pi * freqs[:, None] * tt[None])
        Ppinv = np.linalg.pinv(P)
        # 先把趋势投影掉，再在残差上拟合正弦（正弦也减去它在趋势上的投影 = 联合最小二乘）
        Cr = Cs - (P @ (Ppinv @ Cs.T)).T
        Sr = Sn - (P @ (Ppinv @ Sn.T)).T
        cc, cs_, ss = (Cr * Cr).sum(1), (Cr * Sr).sum(1), (Sr * Sr).sum(1)
        det = cc * ss - cs_ ** 2 + 1e-12
        starts = list(range(0, b - a - Lw + 1, step))
        if starts[-1] != b - a - Lw:
            starts.append(b - a - Lw)
        vals = []
        for s0 in starts:
            y = c[s0:s0 + Lw]
            r = y - P @ (Ppinv @ y)
            tot = (r * r).sum() + 1e-9
            yc, ys = Cr @ r, Sr @ r
            A = (ss * yc - cs_ * ys) / det; B = (cc * ys - cs_ * yc) / det
            expl = (A * yc + B * ys) / tot
            j = int(np.argmax(expl))
            depth = float(np.hypot(A[j], B[j]))
            rate = float(freqs[j])
            r2 = float(expl[j])
            st = r2 * min(1.0, depth / 40.0) if (VIB_BAND[0] <= rate <= VIB_BAND[1] and r2 > 0.5 and depth > 8) else 0.0
            vals.append((a + s0 + Lw // 2, st, depth, rate))
        centers = np.array([v[0] for v in vals]); sv = np.array([v[1] for v in vals])
        dv = np.array([v[2] for v in vals]); rv = np.array([v[3] for v in vals])
        ks = np.arange(a, b)
        S[a:b] = np.interp(ks, centers, sv)
        near = np.abs(ks[:, None] - centers[None]).argmin(1)
        good = sv[near] > 0
        D[a:b] = np.where(good, dv[near], np.nan); R[a:b] = np.where(good, rv[near], np.nan)
    return S, D, R


def _segments(mask, fr, merge, min_len=0.2):          # < 0.2 s 的孤立有声多半是串音的拨弦（新歌前奏）
    segs = []
    for a, b in _runs(mask):
        if segs and (a - segs[-1][1]) / fr < merge:
            segs[-1] = (segs[-1][0], b)
        else:
            segs.append((a, b))
    return [(a, b) for a, b in segs if (b - a) / fr >= min_len]


def analyze_voice(x, sr=SR, fps=24, n_out=None):
    """>v<👂感官层-人声-分析：单声道人声（sr=16 kHz，样本 i ↔ 播放时间 i/sr）→ voice 节点 dict"""
    assert sr == SR, "先重采样到 16 kHz（sensorium.decode(path, sr=16000)）"
    fr = sr / HOP
    db = frame_rms_db(x)
    ref = float(np.percentile(db, 99.5))
    loud = np.clip((db - (ref - FLOOR_DB)) / FLOOR_DB, 0, 1)
    tau, dmin, cm = yin(x)
    n = min(len(db), len(tau))
    db, loud, tau, dmin, cm = db[:n], loud[:n], tau[:n], dmin[:n], cm[:n]
    p_period = 1 / (1 + np.exp((dmin - P_MID) / P_WIDTH))
    gate = np.clip((db - (ref - FLOOR_DB)) / 6.0, 0, 1)  # 门：底线以上 6 dB 内渐入
    pv = p_period * gate
    # 迟滞：0.5 进、0.3 出；短于 50 ms 的有声段扔掉
    v = np.zeros(n, bool); on = False
    for k in range(n):
        on = pv[k] > (0.3 if on else 0.5)
        v[k] = on
    for a, b in _runs(v):
        if b - a < 5:
            v[a:b] = False
    f0 = np.where(v & (tau > 0), sr / np.maximum(tau, 1e-9), np.nan)
    v &= np.isfinite(f0)
    f0 = octave_fix(f0, v, cm)
    # 颤音在中值之前算（5 帧中值会把 5.5 Hz 的颤音削掉 ~5%）；YIN 的 40 ms 积分窗 = 盒式平均，深度 ÷ sinc(f·W) 补回
    vs, vd, vr = vibrato(np.where(v, f0, 1.0), v, fr)
    vd = vd / np.sinc(np.nan_to_num(vr, nan=0.0) * W / sr)
    f0 = median_fill(f0, v)
    # 区段
    ph = _segments(v, fr, MERGE_GAP)
    phrases = []
    for a, b in ph:
        ff = f0[a:b][v[a:b]]
        tail = vs[max(a, b - int(0.5 * fr)):b]
        phrases.append({"start": round(float(a / fr), 3), "end": round(float((b - 1) / fr), 3),
                        "f0_med": round(float(np.median(ff)), 1),
                        "f0_lo": round(float(np.percentile(ff, 10)), 1), "f0_hi": round(float(np.percentile(ff, 90)), 1),
                        "peak": round(float(loud[a:b].max()), 3),
                        "tail_vibrato": round(float(tail.mean()), 3)})
    breaths = [[p["end"], q["start"]] for p, q in zip(phrases, phrases[1:]) if q["start"] - p["end"] <= BREATH_MAX]
    sections = []
    for p in phrases:
        if sections and p["start"] - sections[-1][1] < SECTION_GAP:
            sections[-1][1] = p["end"]
        else:
            sections.append([p["start"], p["end"]])
    # 重采样到视频帧：第 i 帧 = i/fps 秒；分析帧 k 的中心 = k/fr 秒
    if n_out is None:
        n_out = int(len(x) / sr * fps)
    t = np.arange(n_out) / fps
    tk = np.arange(n) / fr
    k_near = np.clip(np.round(t * fr).astype(int), 0, n - 1)
    vo = v[k_near]

    def nul(a, nd):
        return [round(float(z), nd) if np.isfinite(z) else None for z in a]
    return {
        "source": None,
        "method": "yin(40ms)+logistic_voicing+octave_fix+median5; vibrato=lsq trend+sine 3.5-9Hz on 0.6s",
        "loudness_ref_db": round(ref, 2), "floor_db": FLOOR_DB,
        "loudness": np.round(np.interp(t, tk, loud), 3).tolist(),
        "voiced": np.round(np.interp(t, tk, pv), 3).tolist(),
        "f0": nul(np.where(vo, f0[k_near], np.nan), 1),
        "vibrato": np.round(np.where(vo, np.interp(t, tk, vs), 0.0), 3).tolist(),
        "vibrato_cents": nul(np.where(vo, vd[k_near], np.nan), 1),
        "vibrato_hz": nul(np.where(vo, vr[k_near], np.nan), 2),
        "phrases": phrases, "breaths": breaths, "sections": sections,
        # 分析帧（100 Hz）的结果，给诊断 / 自检，不写进 json
        "_frames": {"fr": fr, "db": db, "loud": loud, "pv": pv, "voiced": v, "f0": np.where(v, f0, np.nan), "vib": vs},
    }
