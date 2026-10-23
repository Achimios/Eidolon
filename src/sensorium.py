# -*- coding: utf-8 -*-
""">v<👂感官层-入口-把一段音频拆成生物能「感觉到」的几路刺激（👤2026-10-19：感知 → 解构 → 构建）。
规则（Kuramoto 的 K、逻辑斯蒂的 r、沙堆加沙……）读这些通道，而不是读手写的时间线参数曲线。
以后给 NPC 用：输入从音乐换成游戏事件，通道语义不变。

通道（每帧一个值，0~1）：
  heartbeat  心跳   —— 低频（< 120 Hz）包络：贝斯 / 底鼓
  threat     威胁   —— 起音强度（频谱通量）的冲击，冲上去再按指数衰减
  hunger     饥饿   —— 慢响度的反面：音乐越空越饿（breakdown 里最高）
  tension    张力   —— 频谱质心（越亮越紧）
  rhythm     节律   —— 拍子相位 0→1（每拍一圈），由起音自相关估 BPM
事件：onsets（威胁峰值的时刻，秒）、beats（拍点，秒）。
只依赖 numpy；音频用 imageio-ffmpeg 解码。
"""
import json
import subprocess
from pathlib import Path

import numpy as np

SR = 11025
HOP = SR // 100            # 10 ms 一个分析帧
NFFT = 1024                # 心跳 / 饥饿 / 张力：长窗，频率分辨率够看 120 Hz 以下
NON = 256                  # 起音：短窗（23ms）。长窗会在起音进入窗口前沿时就报，整体早 ~30ms（2026-10-19 自检抓到）
T0 = NFFT / 2 / SR         # 分析帧的时间 = 窗口中心，不是窗口起点
T_ON = NON / 2 / SR


_LEAD = {}


def audio_lead(path):
    """音频流第一个包的播放时间（秒）。原曲 mp4 的音频从 2016/48000 = 0.042 s 才开始（视频从 0），
    ffmpeg 从头解码时第一个样本就是那一刻 —— 不补零的话 样本 i ↔ i/sr + 0.042（2026-10-21 抓到：感官.json 整体早 42 ms）。
    按 (路径, mtime) 缓存。"""
    import imageio_ffmpeg
    p = Path(path).resolve()
    key = (str(p), p.stat().st_mtime)
    if key not in _LEAD:
        r = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-nostats", "-loglevel", "error", "-i", str(p),
                            "-map", "0:a:0", "-c", "copy", "-frames:a", "1", "-f", "framecrc", "-"],
                           capture_output=True, check=True, text=True).stdout
        tb, lead = None, 0.0
        for line in r.splitlines():
            if line.startswith("#tb 0:"):
                n, d = line.split(":", 1)[1].strip().split("/")
                tb = int(n) / int(d)
            elif line and not line.startswith("#") and tb:
                lead = max(0.0, int(line.split(",")[2]) * tb)   # 第 3 列 = pts（输出时间轴，已减去文件 start_time = 播放时间）
                break
        _LEAD[key] = lead
    return _LEAD[key]


def decode(path, start=0.0, dur=None, sr=SR, ac=1, fmt="s16le"):
    """解码成 (n,) 或 (n, ac)，样本 i ↔ 播放时间 start + i/sr。
    start 早于音频起点时，前面补 (起点 − start)·sr 个零（ffmpeg 的 -ss 在起点之后是准的，之前不会补）。"""
    import imageio_ffmpeg
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-ss", str(start)]
    if dur:
        cmd += ["-t", str(dur)]
    cmd += ["-i", str(path), "-vn", "-ac", str(ac), "-ar", str(sr), "-f", fmt, "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    x = np.frombuffer(raw, {"s16le": np.int16, "f32le": np.float32}[fmt])
    x = x.reshape(-1, ac) if ac > 1 else x
    pad = int(round(max(0.0, audio_lead(path) - start) * sr))
    if pad:
        x = np.concatenate([np.zeros((pad,) + x.shape[1:], x.dtype), x])
    if dur:
        x = x[:int(round(dur * sr))]
    return x


def load_audio(path, start=0.0, dur=None, sr=SR):
    """单声道 float64，样本 i ↔ 播放时间 start + i/sr"""
    return decode(path, start, dur, sr).astype(np.float64) / 32768


def _norm(v, lo=1, hi=99):
    a, b = np.percentile(v, lo), np.percentile(v, hi)
    return np.clip((v - a) / (b - a + 1e-12), 0, 1)


def _smooth(v, n):
    if n <= 1:
        return v
    k = np.ones(n) / n
    return np.convolve(np.pad(v, (n // 2, n - 1 - n // 2), mode="edge"), k, mode="valid")


def _stft_mag(x, n=1024, hop=HOP):
    frames = max(1, (len(x) - n) // hop + 1)
    idx = np.arange(n)[None, :] + hop * np.arange(frames)[:, None]
    return np.abs(np.fft.rfft(x[idx] * np.hanning(n), axis=1))


def estimate_bpm(flux, frame_rate, lo=70, hi=180):
    """起音包络的自相关取峰；结果折到 [lo, hi)"""
    f = flux - flux.mean()
    ac = np.correlate(f, f, mode="full")[len(f) - 1:]
    lags = np.arange(len(ac))
    ok = (lags >= frame_rate * 60 / hi) & (lags <= frame_rate * 60 / lo)
    lag = lags[ok][np.argmax(ac[ok])]
    # 抛物线插值把峰位估到亚帧
    if 0 < lag < len(ac) - 1:
        y0, y1, y2 = ac[lag - 1], ac[lag], ac[lag + 1]
        lag = lag + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12)
    return 60 * frame_rate / lag


def pick_peaks(v, frame_rate, thresh=0.3, min_gap=0.08):
    gap = int(min_gap * frame_rate)
    m = (v[1:-1] > v[:-2]) & (v[1:-1] >= v[2:]) & (v[1:-1] > thresh)
    idx = np.nonzero(m)[0] + 1
    out = []
    for i in idx:
        if not out or i - out[-1] >= gap:
            out.append(i)
        elif v[i] > v[out[-1]]:
            out[-1] = i
    return np.array(out, int)


def analyze(x, sr=SR, fps=24):
    """>v<👂感官层-分析-音频 → 各通道（按 fps 重采样）+ 事件"""
    fr = sr / HOP                                # 分析帧率 100
    S = _stft_mag(x)
    freqs = np.fft.rfftfreq(1024, 1 / sr)
    # 心跳：< 120 Hz 能量
    hb = np.sqrt((S[:, freqs < 120] ** 2).sum(1))
    # 威胁：半波整流的频谱通量（对数谱），冲击后指数衰减
    L = np.log1p(_stft_mag(x, n=NON) * 10)
    flux = np.r_[0, np.maximum(L[1:] - L[:-1], 0).sum(1)]
    flux_n = _norm(flux - _smooth(flux, int(fr * 0.5)), 5, 99.5)
    onset_idx = pick_peaks(flux_n, fr, 0.35)
    threat = np.zeros_like(flux_n)
    decay = np.exp(-1 / (fr * 0.18))            # 180 ms 衰减
    lvl = 0.0
    hits = set(onset_idx.tolist())
    for i in range(len(threat)):
        lvl *= decay
        if i in hits:
            lvl = max(lvl, flux_n[i])
        threat[i] = lvl
    # 饥饿：慢响度（2 秒窗）的反面
    rms = np.sqrt((S ** 2).sum(1))
    ldb = 20 * np.log10(_smooth(rms, int(fr * 2)) + 1e-9)
    live = ldb > ldb.max() - 30               # 只在有音乐的部分里定范围：片尾全静会把范围撑大，整首就压平了（2026-10-19 条带图看出来的）
    lo, hi = np.percentile(ldb[live], [5, 95]) if live.any() else (ldb.min(), ldb.max())
    loud = np.clip((ldb - lo) / (hi - lo + 1e-12), 0, 1)
    hunger = 1 - loud
    # 张力：频谱质心
    cent = (S * freqs).sum(1) / (S.sum(1) + 1e-9)
    tension = _norm(_smooth(cent, int(fr * 0.5)))
    # 节律：BPM + 拍点（从第一个强起音起按 BPM 铺格，再逐拍吸附到附近起音）
    bpm = estimate_bpm(flux_n, fr)
    period = fr * 60 / bpm
    start = onset_idx[0] if len(onset_idx) else 0
    beats = []
    b = float(start)
    while b < len(flux_n):
        w = int(period * 0.12)
        lo, hi = max(0, int(b) - w), min(len(flux_n), int(b) + w + 1)
        j = lo + int(np.argmax(flux_n[lo:hi]))
        if flux_n[j] > 0.2:
            b = 0.7 * b + 0.3 * j               # 轻吸附，别被单个杂音拽走
        beats.append(b)
        b += period
    beats = np.array(beats)
    phase = np.zeros(len(flux_n))
    for k in range(len(beats) - 1):
        a, c = int(beats[k]), int(beats[k + 1])
        phase[a:c] = np.linspace(0, 1, c - a, endpoint=False)
    # 重采样到视频帧率
    n_out = int(len(x) / sr * fps)
    t_out = np.arange(n_out) / fps
    t_big = np.arange(len(S)) / fr + T0         # 长窗通道的时间轴
    t_on = np.arange(len(flux_n)) / fr + T_ON   # 起音通道的时间轴
    out = {k: np.round(np.interp(t_out, t_big, v), 3) for k, v in
           {"heartbeat": _norm(_smooth(hb, 3)), "hunger": hunger, "tension": tension}.items()}
    out["threat"] = np.round(np.interp(t_out, t_on, threat), 3)
    # 相位不能线性插值穿过 1→0 的跳变：按帧取最近的分析帧
    out["rhythm"] = np.round(phase[np.clip(np.round((t_out - T_ON) * fr).astype(int), 0, len(phase) - 1)], 3)
    out = {k: out[k] for k in ("heartbeat", "threat", "hunger", "tension", "rhythm")}
    return {"fps": fps, "duration": round(len(x) / sr, 3), "bpm": round(float(bpm), 2),
            "channels": {k: v.tolist() for k, v in out.items()},
            "events": {"onsets": np.round(onset_idx / fr + T_ON, 3).tolist(), "beats": np.round(beats / fr + T_ON, 3).tolist()}}


TAIL = ("voice", "expect")                     # 放在 events 后面的整节点（顺序固定）


def write(data, path, source=None):
    data = dict(data)
    if source:
        data = {"source": source, **data}
    head = {k: v for k, v in data.items() if k not in ("channels", "events") + TAIL}
    # 一个通道一行：文件小一点，人扫一眼也能看
    lines = ["{"] + [f' {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)},' for k, v in head.items()]
    lines.append(' "channels": {')
    chs = list(data["channels"].items())
    lines += [f'  {json.dumps(k)}: {json.dumps(v)}' + ("," if i < len(chs) - 1 else "") for i, (k, v) in enumerate(chs)]
    lines.append(" },")
    lines.append(' "events": {')
    evs = list(data["events"].items())
    lines += [f'  {json.dumps(k)}: {json.dumps(v)}' + ("," if i < len(evs) - 1 else "") for i, (k, v) in enumerate(evs)]
    for node in TAIL:                                # 人声 voice（voice.py）、期待 expect（expect.py），2026-10-23：一个字段一行，按 TAIL 顺序放最后
        if node not in data:
            continue
        lines.append(" },")
        lines.append(f' "{node}": {{')
        vs = list(data[node].items())
        lines += [f'  {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}' + ("," if i < len(vs) - 1 else "") for i, (k, v) in enumerate(vs)]
    lines += [" }", "}"]
    Path(path).write_text("\n".join(lines) + "\n", "utf-8", newline="\n")
