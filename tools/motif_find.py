#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<👂感官层-动机检索：用户指一段声音（如 1:02+15f ~ 1:03+16f 的「抽搐音」）→ 在全曲（和每个分轨）里找它还在哪出现。
做法：对数梅尔谱（每轨各算），拿这段当模板滑动，算余弦相似度；峰值 = 同一个动机再次出现。输出时间、相似度、哪一轨最像，
外加一张图：模板 + 每次出现的谱（人眼核对）。同一元素、不同进展的画面要按这里的「第几次出现」排。
用法：python tools/motif_find.py --t0 62.625 --t1 63.667 [--name 抽搐] [--top 12]
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

MV = Path("C:/Users/Administrator/Desktop/Projekt_Visual/MV_Cant_Kill_Us")
SR, NFFT, HOP, NMEL = 22050, 2048, 256, 96


def mel_fb(sr, nfft, n):
    f = np.linspace(0, sr / 2, nfft // 2 + 1)
    mel = lambda h: 2595 * np.log10(1 + h / 700)
    imel = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = imel(np.linspace(mel(30), mel(sr / 2), n + 2))
    fb = np.zeros((n, len(f)))
    for i in range(n):
        l, c, r = pts[i:i + 3]
        fb[i] = np.clip(np.minimum((f - l) / (c - l), (r - f) / (r - c)), 0, None)
    return fb


def logmel(x, fb):
    n = 1 + (len(x) - NFFT) // HOP
    idx = np.arange(NFFT)[None, :] + HOP * np.arange(n)[:, None]
    S = np.abs(np.fft.rfft(x[idx] * np.hanning(NFFT), axis=1)) ** 2
    return np.log(fb @ S.T + 1e-8)                       # (mel, frames)


def load(path):
    x, sr = sf.read(path); x = x.mean(1) if x.ndim > 1 else x
    if sr != SR:
        t = np.arange(0, len(x) / sr, 1 / SR); x = np.interp(t, np.arange(len(x)) / sr, x)
    return x


def match(M, a, b):
    """模板 = M[:, a:b]；对每个起点算均值归一化后的余弦相似度"""
    T = M[:, a:b]; T = (T - T.mean()) / (T.std() + 1e-9); L = T.shape[1]
    out = np.full(M.shape[1] - L, -1.0)
    for s in range(M.shape[1] - L):
        W = M[:, s:s + L]; sd = W.std()
        if sd > 1e-6:
            out[s] = ((W - W.mean()) / sd * T).mean()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0", type=float, required=True)
    ap.add_argument("--t1", type=float, required=True)
    ap.add_argument("--name", default="动机")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--thr", type=float, default=0.45)
    ap.add_argument("--tracks", default="", help="只在这些分轨上比（逗号分隔，如 bass,other）；鼓每个乐句都重复同一型，混进来会误报（👤 2026-10-21 纠正：1:37 内只有 1:03、1:15）")
    a = ap.parse_args()
    fb = mel_fb(SR, NFFT, NMEL)
    tracks = {"mix": None}
    for f in glob.glob(str(MV / "分轨" / "**" / "*.wav"), recursive=True):
        tracks[os.path.basename(f)[:-4]] = f
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import sensorium as S   # 样本 i ↔ 播放时间（补上原曲音频晚开始的 0.042 s；分轨文件 2026-10-21 起也是播放时间）
    mix = S.decode(MV.parent / "Assets" / "The Glitch Mob - Can't Kill Us [2020 Edit]_190189282.mp4", sr=SR, fmt="f32le")
    mels = {"mix": logmel(mix.astype(np.float64), fb)}
    for k, f in tracks.items():
        if f:
            mels[k] = logmel(load(f), fb)
    fa, fb_ = int(a.t0 * SR / HOP), int(a.t1 * SR / HOP)
    # 模板在哪一轨能量最大 → 主轨
    energy = {k: float(np.exp(M[:, fa:fb_]).mean()) for k, M in mels.items() if k != "mix"}
    use = [t for t in a.tracks.split(",") if t] or None
    main_tr = max((k for k in energy if not use or k in use), key=energy.get)
    sims = {k: match(M, fa, fb_) for k, M in mels.items() if (use is None or k in use)}
    s = np.min([sims[k] for k in sims], 0) if use else np.maximum(sims["mix"], sims[main_tr])   # 指定分轨时要求每一轨都像
    L = fb_ - fa; hits = []
    order = np.argsort(-s)
    for i in order:
        if s[i] < a.thr or len(hits) >= a.top:
            break
        if all(abs(i - j) > L for j, _ in hits):
            hits.append((i, s[i]))
    hits.sort()
    res = {"name": a.name, "template": [a.t0, a.t1], "main_track": main_tr, "track_energy": energy,
           "hits": [{"t": round(i * HOP / SR, 3), "sim": round(float(v), 3),
                     "best_track": max(sims, key=lambda k: sims[k][i])} for i, v in hits]}
    outj = MV / f"动机_{a.name}.json"
    outj.write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    # 核对图：每次出现的主轨谱
    M = mels[main_tr]; lo, hi = np.percentile(M, [5, 99.5])
    cw = L * 3; ch = NMEL * 2
    img = Image.new("RGB", (cw * 4, (ch + 24) * (-(-len(hits) // 4))), (8, 9, 10))
    d = ImageDraw.Draw(img); fnt = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 14)
    for n, (i, v) in enumerate(hits):
        P = np.clip((M[::-1, i:i + L] - lo) / (hi - lo), 0, 1)
        tile = Image.fromarray((P * 255).astype(np.uint8)).resize((cw, ch))
        r, c = divmod(n, 4); img.paste(tile.convert("RGB"), (c * cw, r * (ch + 24) + 24))
        t = i * HOP / SR
        d.text((c * cw + 4, r * (ch + 24) + 4), f"{int(t // 60)}:{t % 60:05.2f}  sim {v:.2f}", font=fnt, fill=(232, 44, 52) if abs(t - a.t0) < 0.05 else (200, 205, 210))
    img.save(MV / f"动机_{a.name}.png")
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
