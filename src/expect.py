# -*- coding: utf-8 -*-
""">v<👂感官层-期待-「她」在每个情绪段落里期待什么、哪里被意外到（2026-10-23）。
⚠️👤「不要太数学化，核心还是跟着 Thymos 和 emotion flow。数学是 building block 和审计视野等作用。」
所以这里只有两块积木，不做导演：
  surprisal  意外度（比特）= 这个事件在「她此刻所在的情绪段落」里有多罕见（香农：−log2 p；哲学 §15）
  salience   显著度 = 情绪段落给的乘子和门（她在听哪一路 / 能忍多少意外）× 意外度
同样的鼓点，在「憋住」段是意外，在 drop 段是日常 —— 所以 p 不是全曲一个分布，而是段落上下文。

意外度怎么算（每一路事件流单独算，只用这一刻之前听到的 —— 期待只能从已经听到的东西里长出来）：
  记忆权重  同一段落里听过的：全记（w = 1；没有段落时用 τ_slide 的滑动窗口）
            之前段落里听过的：几秒内淡掉（w = exp(−Δt / τ_out)）→ 刚换段时她还带着上一段的期待，然后重新适应
  发生      λ = (Σw + α_n) / (有效时长 + α_n / λ_全曲)；这一拍里预期来几个 p = min(1, λ·拍长)，bits_rate = −log2 p
  大小      p = (Σ w·[m_j ≥ m] + 1 + α_m·全曲尾概率(m)) / (Σw + 1 + α_m)（秩，自己算一个），bits_mag = −log2 p（「这么大或更大」有多罕见）
  缺席      该来的没来：P(这段空隙里一个都没有) = exp(−λ·gap)，bits = λ·gap / ln2（鼓停、憋住）
只依赖 numpy。
"""
import math

import numpy as np

LN2 = math.log(2)
TAU_OUT = 3.0        # 换段后上一段的期待几秒内淡掉
TAU_SLIDE = 12.8     # 没有情绪段落时的滑动记忆（= 2 个乐句）
ALPHA_N = 0.5        # 发生率的先验：半个事件的「全曲平均」
ALPHA_M = 2.0        # 大小分布的先验：两个事件的「全曲分布」
CAP = 12.0           # 比特上限（防 log 0；12 比特 ≈ 1/4096，已经是「从没见过」）
ABS_ON = 4.0         # 缺席超过 4 比特（≈ 1/16）才算一次「该来的没来」


def seg_index(ts, segments):
    """每个时刻属于哪一段（segments = [(start, end), ...] 按时间排序；None = 全曲一段）"""
    ts = np.asarray(ts, float)
    if not segments:
        return np.zeros(len(ts), int)
    starts = np.array([s for s, _ in segments], float)
    return np.clip(np.searchsorted(starts, ts, side="right") - 1, 0, len(starts) - 1)


def _mem(t, seg_start, tau_in, tau_out):
    """在时刻 t 的记忆：返回 (权重函数, 有效时长)。tau_in=None = 段内全记。"""
    def w(tj, same):
        dt = t - tj
        win = np.ones_like(dt) if tau_in is None else np.exp(-dt / tau_in)
        return np.where(same, win, np.exp(-dt / tau_out))
    inside = max(0.0, t - seg_start)
    T_in = inside if tau_in is None else tau_in * (1 - math.exp(-inside / tau_in))
    T_out = tau_out * (math.exp(-inside / tau_out) - math.exp(-max(t, inside) / tau_out)) if seg_start > 0 else 0.0
    return w, T_in + max(0.0, T_out)


def _prior(ts, mags):
    ts = np.asarray(ts, float)
    span = (ts.max() - ts.min()) if len(ts) > 1 else 1.0
    lam_g = max(len(ts), 1) / max(span, 1.0)
    ms = np.sort(np.asarray(mags, float))

    def tail(m):                                   # 全曲里 ≥ m 的比例（经验，含自己）
        return (len(ms) - np.searchsorted(ms, m - 1e-12)) / max(len(ms), 1)
    return lam_g, tail


def surprisal(ts, mags, segments=None, beat=0.4, use_rate=True, use_mag=True,
              tau_in=None, tau_out=TAU_OUT, alpha_n=ALPHA_N, alpha_m=ALPHA_M):
    """一路事件流的意外度。ts 秒（升序），mags 大小（越大越强）。
    segments=None → 滑动窗口（tau_in 默认 TAU_SLIDE）。返回 dict(bits, rate, mag, lam)，各为 (n,) 数组。"""
    ts = np.asarray(ts, float); mags = np.asarray(mags, float)
    if segments is None and tau_in is None:
        tau_in = TAU_SLIDE
    seg = seg_index(ts, segments)
    starts = [s for s, _ in segments] if segments else [0.0]
    lam_g, tail = _prior(ts, mags)
    n = len(ts)
    br, bm, lam = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        t = ts[i]
        w, T = _mem(t, starts[seg[i]], tau_in, tau_out)
        past = ts < t
        wj = w(ts[past], seg[past] == seg[i]) if past.any() else np.zeros(0)
        N = wj.sum()
        lam[i] = (N + alpha_n) / (T + alpha_n / lam_g)
        if use_rate:
            p = min(1.0, lam[i] * beat)          # 这一拍里预期来几个；≥ 1 个 = 不意外（泊松的 1−e^−x 会让每拍一个的流永远剩 0.66 比特）
            br[i] = -math.log2(max(p, 2 ** -CAP))
        if use_mag:
            ge = (mags[past] >= mags[i] - 1e-12)
            # 秩：「这么大或更大」在听过的里面占多少；自己算一个（听过 N 个里最大的 ≈ log2(N+1) 比特，不会因为全曲也罕见而重复计）
            p = ((wj * ge).sum() + 1 + alpha_m * tail(mags[i])) / (N + 1 + alpha_m)
            bm[i] = -math.log2(min(1.0, max(p, 2 ** -CAP)))
    return {"bits": np.minimum(br + bm, CAP), "rate": br, "mag": bm, "lam": lam}


def absence(ts, frame_t, segments=None, tau_in=None, tau_out=TAU_OUT, alpha_n=ALPHA_N, until=None):
    """每帧的「缺席」比特：从上一个事件到现在一个都没来，按此刻的期待（λ）有多罕见。
    第一个事件之前、until（默认最后一个事件）之后 = 0（还没开始 / 歌已经结束，不算「该来没来」）。"""
    ts = np.asarray(ts, float); ft = np.asarray(frame_t, float)
    if segments is None and tau_in is None:
        tau_in = TAU_SLIDE
    out = np.zeros(len(ft))
    if len(ts) == 0:
        return out
    until = ts[-1] if until is None else until
    lam_g, _ = _prior(ts, np.zeros(len(ts)))
    fseg = seg_index(ft, segments); eseg = seg_index(ts, segments)
    starts = [s for s, _ in segments] if segments else [0.0]
    for k, t in enumerate(ft):
        if t <= ts[0] or t > until:
            continue
        j = np.searchsorted(ts, t, side="left")          # ts[:j] < t
        gap = t - ts[j - 1]
        w, T = _mem(t, starts[fseg[k]], tau_in, tau_out)
        N = w(ts[:j], eseg[:j] == fseg[k]).sum()
        lam = (N + alpha_n) / (T + alpha_n / lam_g)
        out[k] = min(CAP, lam * gap / LN2)
    return out


def absence_events(curve, fps, on=ABS_ON):
    """缺席曲线 → 事件：每段连续空隙里第一次超过 on 比特的那一刻（「等太久了」被感到的时刻）；bits = 这段空隙里的最大值"""
    ev = []
    k, n = 0, len(curve)
    while k < n:
        if curve[k] >= on:
            j = k
            while j + 1 < n and curve[j + 1] >= 0.5 * curve[j] and curve[j + 1] > 0:   # 来了一个 → 曲线塌回去 = 这段空隙结束
                j += 1
            ev.append((k / fps, float(curve[k:j + 1].max()), (j - k + 1) / fps))
            k = j + 1
        else:
            k += 1
    return ev


# 情绪的门：tolerance 越低，越小的意外就能过门（全黑里一点动静都大）
GATE = {"低": 0.5, "中": 1.5, "高": 3.0}
UNATTENDED = 0.3     # 她没在听的那一路：还能听见，但弱


def salience(bits, stream, seg, expect):
    """显著度 = 乘子（她在听哪一路：expect[seg]["attend"][stream]，没列的 = UNATTENDED）
              × 门（logistic，中心 = tolerance 对应的比特数，宽 0.4 比特）× 意外度。
    expect = 期待.json 的「段」列表，和 segments 同顺序。"""
    e = expect[seg]
    a = e.get("attend", {}).get(stream, UNATTENDED)
    th = GATE[e.get("tolerance", "中")]
    g = 1 / (1 + math.exp(-(bits - th) / 0.4))
    return a * g * bits


def envelope(ts, vals, n, fps, decay=0.18):
    """事件 → 每帧曲线：冲到事件值，按 decay 秒指数衰减（同 threat 的做法）；同帧取大"""
    out = np.zeros(n)
    hit = np.zeros(n)
    for t, v in zip(ts, vals):
        k = int(round(t * fps))
        if 0 <= k < n:
            hit[k] = max(hit[k], v)
    d = math.exp(-1 / (fps * decay)); lvl = 0.0
    for k in range(n):
        lvl = max(lvl * d, hit[k]); out[k] = lvl
    return out
