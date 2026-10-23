#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<👂感官层-人声-对照：同一条人声分轨上 YIN（旧，numpy）vs torchcrepe（crepe_raw = 单用；crepe = + YIN 细化，默认）vs librosa pYIN，没有真值时怎么选默认（2026-10-23）。
用法：python tools/voice_compare.py <作品文件夹> [<作品文件夹> ...] [--crepe-model full|tiny]
指标（两两比，只在双方都判有声的帧上算音高）：
  med_cents   |音分差| 中位
  gross50     |差| > 50 音分的帧比例
  octave      差 ≈ ±1200 / ±1902 / ±2400 音分（八度 / 十二度 / 两个八度）的帧比例
  voicing     响度门以上的帧里，两者有声判断一致的比例；jaccard = 有声帧集合的交 / 并
仲裁（⚠️ YIN 和 pYIN 同源 —— 它们一致不算交叉验证）：YIN 和 crepe 差一个八度左右的帧，用一个跟两者都无关的谐波判据定谁对：
  score(f) = mean log|X(h·f)| − mean log|X((h+½)·f)|，h = 1..8（128 ms Hann 窗的幅度谱）。
  真 f0 的整数倍全落在谐波上、半整数倍全落在谐波之间 → 分数最高；高八度的候选，半整数倍正好是奇次谐波；低八度的候选，一半「谐波」位置是空的。
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import sensorium as S  # noqa: E402
import voice as V  # noqa: E402


def cmp(fa, va, fb, vb, gate):
    both = va & vb & np.isfinite(fa) & np.isfinite(fb)
    ac = np.abs(1200 * np.log2(fa[both] / fb[both]))
    octv = (np.abs(ac - 1200) < 150) | (np.abs(ac - 1902) < 150) | (np.abs(ac - 2400) < 150)
    return {"frames": int(both.sum()), "med_cents": round(float(np.median(ac)), 1) if len(ac) else None,
            "gross50": round(float((ac > 50).mean()), 3) if len(ac) else None,
            "octave": round(float(octv.mean()), 3) if len(ac) else None,
            "voicing": round(float((va == vb)[gate].mean()), 3),
            "jaccard": round(float((va & vb).sum() / max(1, (va | vb).sum())), 3)}


def harmonic_score(x, k, f, sr=V.SR, n=2048):
    c = k * V.HOP
    seg = np.pad(x, n)[c + n - n // 2:c + n + n // 2] * np.hanning(n)
    mag = np.log(np.abs(np.fft.rfft(seg)) + 1e-9)
    bins = np.fft.rfftfreq(n, 1 / sr)
    hs = np.arange(1, 9)
    hs = hs[(hs + 0.5) * f < sr * 0.45]
    if len(hs) < 3:
        return np.nan
    return float(np.interp(hs * f, bins, mag).mean() - np.interp((hs + 0.5) * f, bins, mag).mean())


def arbitrate(x, fa, fb, both, step=3):
    """fa / fb 在 both 帧上差一个八度左右的帧 → (a 赢的比例, b 赢的比例, 帧数)"""
    ac = np.abs(1200 * np.log2(fa / fb))
    dis = np.nonzero(both & ((np.abs(ac - 1200) < 150) | (np.abs(ac - 2400) < 150) | (np.abs(ac - 1902) < 150)))[0][::step]
    wa = wb = 0
    for k in dis:
        sa, sb = harmonic_score(x, k, fa[k]), harmonic_score(x, k, fb[k])
        if np.isfinite(sa) and np.isfinite(sb):
            wa += sa > sb; wb += sb > sa
    tot = max(1, wa + wb)
    return round(wa / tot, 3), round(wb / tot, 3), int(wa + wb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mv", nargs="+")
    ap.add_argument("--crepe-model", default="full")
    a = ap.parse_args()
    import librosa
    for mv in a.mv:
        mv = Path(mv)
        x = S.decode(mv / "分轨" / "vocals.wav", sr=V.SR, fmt="f32le").astype(np.float64)
        Fy = V.analyze_voice(x, f0_method="yin")["_frames"]
        Fr = V.analyze_voice(x, f0_method="crepe_raw", crepe_model=a.crepe_model)["_frames"]
        Fc = V.analyze_voice(x, f0_method="crepe", crepe_model=a.crepe_model)["_frames"]
        n = min(len(Fy["f0"]), len(Fc["f0"]), len(Fr["f0"]))
        fp, vp, _ = librosa.pyin(x, fmin=V.FMIN, fmax=V.FMAX, sr=V.SR, frame_length=1024, hop_length=V.HOP, center=True)
        n = min(n, len(fp))
        gate = Fy["loud"][:n] > 0
        y, vy = Fy["f0"][:n], Fy["voiced"][:n]
        c, vc = Fc["f0"][:n], Fc["voiced"][:n]
        r, vr = Fr["f0"][:n], Fr["voiced"][:n]
        p, vpp = fp[:n], vp[:n] & gate
        print(f"== {mv.name}  ({n} 帧 @100 Hz；门以上 {gate.sum()} 帧)")
        print(f"  有声比例（门以上）：yin {vy[gate].mean():.3f}  crepe {vc[gate].mean():.3f}  pyin {vpp[gate].mean():.3f}")
        bo = vy & vr & np.isfinite(y) & np.isfinite(r)
        wy0, wr0, m0 = arbitrate(x, y, r, bo)
        print(f"  八度分歧仲裁 yin vs crepe_raw（{m0} 帧）：yin 对 {wy0 * 100:.1f}%  crepe_raw 对 {wr0 * 100:.1f}%")
        for name, (fa, va, fb, vb) in {"yin       vs pyin": (y, vy, p, vpp), "crepe_raw vs pyin": (r, vr, p, vpp),
                                       "crepe     vs pyin": (c, vc, p, vpp), "yin       vs crepe": (y, vy, c, vc),
                                       "crepe_raw vs crepe": (r, vr, c, vc)}.items():
            print(f"  {name}: {cmp(fa, va, fb, vb, gate)}")
        both = vy & vc & np.isfinite(y) & np.isfinite(c)
        wy, wc, m = arbitrate(x, y, c, both)
        print(f"  八度分歧仲裁 yin vs crepe（谐波判据，{m} 帧抽样）：yin 对 {wy * 100:.1f}%  crepe 对 {wc * 100:.1f}%")
        bp = vc & vpp & np.isfinite(c) & np.isfinite(p)
        wc2, wp2, m2 = arbitrate(x, c, p, bp)
        print(f"  八度分歧仲裁 crepe vs pyin（{m2} 帧）：crepe 对 {wc2 * 100:.1f}%  pyin 对 {wp2 * 100:.1f}%")


if __name__ == "__main__":
    main()
