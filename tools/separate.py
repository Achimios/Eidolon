#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<👂感官层-分轨-Demucs（htdemucs）把混好的歌拆成 鼓 / 贝斯 / 人声 / 其他 四轨（👤2026-10-19：没有原片分轨也要 demodulate）。
要在本仓库的 .venv 里跑（装了 torch + demucs）：
  .venv/Scripts/python.exe tools/separate.py [音频或视频] [--out 目录]
默认：Can't Kill Us → Projekt_Visual/MV_Cant_Kill_Us/分轨/{drums,bass,vocals,other}.wav（44.1kHz 立体声 float）
读写都不经过 torchaudio（新版 torchaudio 的读写依赖变了），用 ffmpeg 解码、soundfile 写。
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJ = ROOT.parents[1] / "Projekt_Visual"
MV = PROJ / "MV_Cant_Kill_Us"
MUSIC = PROJ / "Assets" / "The Glitch Mob - Can't Kill Us [2020 Edit]_190189282.mp4"


def decode(path, sr=44100):
    """(2, n)，样本 i ↔ 播放时间 i/sr（走 sensorium.decode：补上原曲音频晚开始的 0.042 s；
    2026-10-21 之前的分轨从 0.042 s 起，已在文件前补零对齐，旧件留 .bak_2026-10-21_offset）"""
    sys.path.insert(0, str(ROOT / "src"))
    import sensorium as S
    return S.decode(path, sr=sr, ac=2, fmt="f32le").T.copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?", default=str(MUSIC))
    ap.add_argument("--out", default=str(MV / "分轨"))
    ap.add_argument("--model", default="htdemucs")
    a = ap.parse_args()
    import torch
    import soundfile as sf
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    model = get_model(a.model); model.eval()
    x = decode(a.src, model.samplerate)
    wav = torch.from_numpy(x)
    ref = wav.mean(0)
    mu, sd = ref.mean(), ref.std()
    with torch.no_grad():
        out = apply_model(model, ((wav - mu) / sd)[None], device=dev, split=True, overlap=0.25, progress=False)[0]
    out = out * sd + mu
    od = Path(a.out); od.mkdir(parents=True, exist_ok=True)
    for name, src in zip(model.sources, out):
        sf.write(od / f"{name}.wav", src.cpu().numpy().T, model.samplerate, subtype="FLOAT")
    # 交叉校验：四轨加起来应该 ≈ 原混音（「必须相等」的量）
    rec = out.sum(0).cpu().numpy()
    err = np.sqrt(np.mean((rec - x) ** 2)) / (np.sqrt(np.mean(x ** 2)) + 1e-12)
    print(f"{a.model} on {dev}: {time.time() - t0:.1f}s  sources={list(model.sources)}  residual(sum vs mix)={err * 100:.2f}%  -> {od}")


if __name__ == "__main__":
    main()
