#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<👂感官层-节拍-beat_this（CPJKU）整首歌的 beat / downbeat → 节拍.json（和 感官.json 并排，不改它的格式）。
要在本仓库的 .venv 里跑（装了 torch + beat_this）：
  .venv/Scripts/python.exe tools/beats.py [音频或视频] [--out 节拍.json] [--grid 13.11 --bars 4]
默认：Can't Kill Us → Projekt_Visual/MV_Cant_Kill_Us/节拍.json
字段：source, model, bpm, beats[], downbeats[]（秒）
交叉校验（「必须相等」的量）：给 --grid 起点，就按 bpm 推 G(k)=起点+k·(bars 小节) 的乐句网格，
数一数有多少格点 ±30ms 内有检测到的 downbeat。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJ = ROOT.parents[1] / "Projekt_Visual"
MV = PROJ / "MV_Cant_Kill_Us"
MUSIC = PROJ / "Assets" / "The Glitch Mob - Can't Kill Us [2020 Edit]_190189282.mp4"


def decode_mono(path, sr=22050):
    """样本 i ↔ 播放时间 i/sr（走 sensorium.decode：补上原曲音频晚开始的 0.042 s）"""
    sys.path.insert(0, str(ROOT / "src"))
    import sensorium as S
    return S.decode(path, sr=sr, fmt="f32le").copy()


def bpm_of(beats):
    ibi = np.diff(beats)
    return float(60.0 / np.median(ibi)) if len(ibi) else 0.0


def grid_check(downbeats, start, bpm, bars=4, tol=0.030):
    """乐句网格 G(k)=start+k·bars·4拍 → (命中数, 格点数, 每格最近 downbeat 误差列表)"""
    period = bars * 4 * 60.0 / bpm
    end = downbeats[-1] if len(downbeats) else start
    G = start + period * np.arange(int((end - start) / period) + 1)
    err = np.array([downbeats[np.argmin(np.abs(downbeats - g))] - g for g in G]) if len(downbeats) else np.full(len(G), np.inf)
    return int(np.sum(np.abs(err) <= tol)), len(G), G, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?", default=str(MUSIC))
    ap.add_argument("--out", default=str(MV / "节拍.json"))
    ap.add_argument("--ckpt", default="final0")
    ap.add_argument("--dbn", action="store_true", help="用 madmom DBN 后处理（要另装 madmom）")
    ap.add_argument("--grid", type=float, default=None, help="乐句网格起点（秒），给了就做交叉校验")
    ap.add_argument("--bars", type=int, default=4)
    a = ap.parse_args()
    import torch
    from beat_this.inference import Audio2Beats
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    x = decode_mono(a.src)
    beats, downbeats = Audio2Beats(checkpoint_path=a.ckpt, device=dev, dbn=a.dbn)(x, 22050)
    beats, downbeats = np.asarray(beats, float), np.asarray(downbeats, float)
    bpm = bpm_of(beats)
    d = {"source": Path(a.src).name, "model": f"beat_this:{a.ckpt}" + (":dbn" if a.dbn else ""), "bpm": round(bpm, 2),
         "beats": [round(float(t), 4) for t in beats], "downbeats": [round(float(t), 4) for t in downbeats]}
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"beat_this on {dev}: {time.time() - t0:.1f}s  bpm={bpm:.2f}  beats={len(beats)}  downbeats={len(downbeats)}  -> {out}")
    if a.grid is not None:
        hit, n, G, err = grid_check(downbeats, a.grid, bpm, a.bars)
        print(f"grid {a.grid}+{a.bars}小节·k：{hit}/{n} 格点 ±30ms 内有 downbeat；|误差| 中位 {np.median(np.abs(err)) * 1000:.1f}ms")


if __name__ == "__main__":
    main()
