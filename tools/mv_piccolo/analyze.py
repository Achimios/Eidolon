#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🎬Piccolo-分析-把歌拆成「纸卷上的洞」和「短笛的飞行轨迹」（MV_Piccolo，2026-10-24）。
符号说真话：每个洞 = 一个真实的音事件（分频带起音峰），洞的长度 = 该频带能量维持的时间；
小鸟的高度 = 旋律 f0（pyin，限 500–3000 Hz，短笛音域）。拍点 / 下拍优先读 感官.json（ReadScreen 的感受器），没有就 librosa。

输出 MV_Piccolo/分析_纸卷.json：
  bpm, beats[], downbeats[]（秒，播放时间）
  bands: 频带中心 Hz 列表（低→高）
  holes: [{t, dur, band, strength}]   洞
  melody: {hop_s, f0[], voiced[]}      每帧 f0（Hz / null）与有声概率
  crackle: {hop_s, v[]}               高频底噪密度 0~1（intro 的唱片噪）
  loud: {hop_s, v[]}                  响度 0~1
用法：.venv/Scripts/python tools/mv_piccolo/analyze.py [--music 路径] [--out 路径]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
PROJ = ROOT.parents[1] / "Projekt_Visual"
MV = PROJ / "MV_Piccolo"
MUSIC = PROJ / "Assets" / "Piccolo and a Cane[短笛和拐杖] - Electro Swing - 免版税音乐.mp4"

SR = 22050
HOP = 512
N_BANDS = 12


def band_onsets(y, sr):
    """分频带起音 → 洞。每个频带：onset_strength 峰 → 洞起点；能量回落到峰值 35% 以下 → 洞结束。"""
    import librosa
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP)) ** 2
    mel = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=N_BANDS, fmin=40, fmax=8000)
    E = mel @ S                                        # (band, frame) 能量
    centers = librosa.mel_frequencies(n_mels=N_BANDS + 2, fmin=40, fmax=8000)[1:-1]
    logE = np.log1p(E * 1e3)
    flux = np.maximum(0, np.diff(logE, axis=1, prepend=logE[:, :1]))
    t = librosa.frames_to_time(np.arange(E.shape[1]), sr=sr, hop_length=HOP)
    holes = []
    for b in range(N_BANDS):
        f = flux[b]
        f = f / (np.percentile(f, 99) + 1e-9)
        # 能量门：这个频带此刻的能量相对它全曲的峰值（intro 里贝斯带的微弱起伏不该成洞）
        gate = np.sqrt(np.clip(E[b] / (np.percentile(E[b], 99.5) + 1e-12), 0, 1))
        f = f * gate
        pk = librosa.util.peak_pick(f, pre_max=3, post_max=3, pre_avg=8, post_avg=8, delta=0.18, wait=4)
        e = E[b]
        for i in pk:
            peak = e[i:i + 6].max() if i + 6 <= len(e) else e[i]
            j = i + 1
            while j < len(e) and e[j] > 0.35 * peak and j - i < int(2.0 * sr / HOP):
                j += 1
            holes.append({"t": round(float(t[i]), 4), "dur": round(float(max(t[j] - t[i], 0.06)), 4),
                          "band": b, "strength": round(float(min(f[i], 1.0)), 3)})
    holes.sort(key=lambda h: h["t"])
    return holes, [round(float(c), 1) for c in centers], t


def melody(y, sr):
    """旋律线（小鸟的高度）：从 Demucs 的 other 分轨（短笛 / 单簧管 / 铜管都在这轨）取谐波成分 → pyin 120–2500 Hz。
    实测这首的 intro 主旋律基频在 170 Hz 附近（不是短笛的高音区），所以下限放到 120。"""
    import librosa
    H = librosa.effects.harmonic(y, margin=3.0)
    f0, vf, vp = librosa.pyin(H, fmin=120, fmax=2500, sr=sr, hop_length=HOP * 2, frame_length=2048)
    # 响度门：这一段谐波能量太小就不算有旋律
    S = np.abs(librosa.stft(H, n_fft=2048, hop_length=HOP * 2))
    fr = librosa.fft_frequencies(sr=sr, n_fft=2048)
    hi = S[(fr > 120) & (fr < 2500)].sum(0)
    gate = np.clip(hi / (np.percentile(hi, 95) + 1e-9), 0, 1)
    v = vp * np.sqrt(gate)
    f0 = np.where(np.isfinite(f0), f0, np.nan)
    # 5 帧中值去抖
    pad = np.pad(f0, 2, mode="edge")
    med = np.array([np.nanmedian(pad[i:i + 5]) if np.isfinite(pad[i:i + 5]).any() else np.nan for i in range(len(f0))])
    return {"hop_s": HOP * 2 / sr,
            "f0": [None if not np.isfinite(x) else round(float(x), 1) for x in med],
            "voiced": [round(float(x), 3) for x in v]}


def envelopes(y, sr):
    import librosa
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))
    fr = librosa.fft_frequencies(sr=sr, n_fft=2048)
    hi = S[fr > 6000]
    # 底噪 = 高频里「不稳定」的部分：帧间差的均值（持续的高频乐器差小，爆裂声差大）
    crack = np.abs(np.diff(hi, axis=1, prepend=hi[:, :1])).mean(0)
    crack = crack / (np.percentile(crack, 98) + 1e-9)
    rms = librosa.feature.rms(S=S)[0]
    db = 20 * np.log10(rms + 1e-9)
    loud = np.clip((db - (db.max() - 40)) / 40, 0, 1)
    return {"hop_s": HOP / sr, "v": [round(float(x), 3) for x in np.clip(crack, 0, 1)]}, \
           {"hop_s": HOP / sr, "v": [round(float(x), 3) for x in loud]}


def beats_from_sense(path):
    if not path.exists():
        return None
    d = json.loads(path.read_text("utf-8"))
    b = d.get("events", {}).get("beats") or d.get("beats")
    db = d.get("events", {}).get("downbeats") or d.get("downbeats")
    if not b:
        return None
    return {"bpm": d.get("bpm"), "beats": b, "downbeats": db or [], "source": str(path.name)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--music", default=str(MUSIC))
    ap.add_argument("--out", default=str(MV / "分析_纸卷.json"))
    a = ap.parse_args()
    import librosa
    import sensorium as Sn
    y = Sn.load_audio(a.music, sr=SR).astype(np.float32)
    holes, centers, _ = band_onsets(y, SR)
    bt = beats_from_sense(MV / "感官.json")
    if bt is None:
        tempo, beats = librosa.beat.beat_track(y=y, sr=SR, units="time", hop_length=HOP)
        bpm = float(np.atleast_1d(tempo)[0])
        # 下拍：4/4，用低频起音最强的相位
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))
        fr = librosa.fft_frequencies(sr=SR, n_fft=2048)
        low = S[fr < 150].sum(0)
        tl = librosa.frames_to_time(np.arange(len(low)), sr=SR, hop_length=HOP)
        lowat = np.interp(beats, tl, low)
        ph = int(np.argmax([lowat[k::4].mean() for k in range(4)]))
        bt = {"bpm": bpm, "beats": [round(float(b), 4) for b in beats],
              "downbeats": [round(float(b), 4) for b in beats[ph::4]], "source": "librosa"}
    other = MV / "分轨" / "other.wav"
    mel = melody(Sn.load_audio(other, sr=SR).astype(np.float32) if other.exists() else y, SR)
    crack, loud = envelopes(y, SR)
    out = {"music": Path(a.music).name, "duration": round(len(y) / SR, 3), **bt, "bands": centers,
           "holes": holes, "melody": mel, "crackle": crack, "loud": loud}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False), "utf-8")
    print(f"holes {len(holes)}  beats {len(bt['beats'])}  bpm {bt['bpm']:.2f}  src {bt['source']}  -> {a.out}")


if __name__ == "__main__":
    main()
