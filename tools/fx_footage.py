#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🧪原片后处理-入口-把数学规则直接作用在剪辑表的画面上（👤2026-10-20：静帧里的效果还没应用，一些画面应用后处理）。
第一个：breakdown 的 DCT —— 她的记忆被压缩，系数个数跟着「饥饿」走：越空越只剩一束光。
  python tools/fx_footage.py dct [--from 153.91] [--to 173.11]
画面来源 = 剪辑表.txt（和快速预览同一份），先按帧号切出原片，再逐帧做效果，配原曲。
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stills as S  # noqa: E402
import edl as E  # noqa: E402
import animatic_intro as A  # noqa: E402

W, H, FPS = 1280, 720, 24
F = ImageFont.truetype(S.FONT, 15)


def frames_from_edl(t0, t1):
    """按剪辑表取 [t0, t1) 的画面，1280×720，逐帧产出 (t, RGB float)"""
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    rows = E.read_edl()
    for k, (t, src, speed) in enumerate(rows):
        nxt = rows[k + 1][0] if k + 1 < len(rows) else E.END
        a, b = max(t, t0), min(nxt, t1)
        if b <= a:
            continue
        f0, f1 = int(round(a * FPS)), int(round(b * FPS))
        video, at, still = E.resolve(src)
        if still is not None:
            img = np.asarray(still.resize((W, H)), np.float64) / 255
            for f in range(f0, f1):
                yield f / FPS, img
            continue
        start = at + (a - t) * speed
        raw = subprocess.run([ff, "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(video), "-vf",
                              f"setpts=PTS/{speed},scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps={FPS}",
                              "-frames:v", str(f1 - f0), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
        arr = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)
        for j in range(f1 - f0):
            yield (f0 + j) / FPS, arr[min(j, len(arr) - 1)].astype(np.float64) / 255


def dct_mats():
    def m(n):
        k = np.arange(n)[:, None]; i = np.arange(n)[None, :]
        M = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.sqrt(2 / n); M[0] /= np.sqrt(2)
        return M
    return m(H // 2), m(W // 2)


def fx_dct(img, frac, Mh, Mw):
    """整幅 DCT，只留最大的 frac 比例系数。亮度和色度分开：色度留得更少（记忆里先褪的是颜色）"""
    small = np.asarray(Image.fromarray((img * 255).astype(np.uint8)).resize((W // 2, H // 2), Image.BILINEAR), np.float64) / 255
    Y = small @ np.array([0.299, 0.587, 0.114]); Cb = small[..., 2] - Y; Cr = small[..., 0] - Y
    out = []
    for ch, f in ((Y, frac), (Cb, frac * 0.3), (Cr, frac * 0.3)):
        C = Mh @ ch @ Mw.T
        n = max(1, int(C.size * f))
        th = np.partition(np.abs(C).ravel(), -n)[-n]
        out.append(Mh.T @ np.where(np.abs(C) >= th, C, 0) @ Mw)
    Y, Cb, Cr = out
    rgb = np.stack([Y + Cr, Y - 0.194 * Cb - 0.509 * Cr, Y + Cb], -1)
    return np.asarray(Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC), np.float64) / 255


YY, XX = np.mgrid[0:H, 0:W]
Z = ((XX - W / 2) + 1j * (YY - H / 2)) / (H / 2)            # 以半高为 1 的复平面
R1, R2 = 0.42, 1.0                                          # 每一圈缩小 2.4 倍：一屏里能看到好几层
PERIOD = np.log(R2 / R1)
ALPHA = np.arctan(PERIOD / (2 * np.pi))
FAC = np.cos(ALPHA) * np.exp(1j * ALPHA)
LZ = np.log(Z + 1e-9) / FAC
DROSTE_HUD = ["t = {t:6.2f} s   REBUILD  08 DROSTE", f"w = exp( log(z) / f ),  f = cos(a) e^(ia),  a = {np.degrees(ALPHA):.1f} deg", f"scale per turn = {R2 / R1:.2f}x   falling inward"]
_fall = {"x": 0.0, "t": None}


def fx_droste(img, t, t0, t1):
    """Escher 式对数螺旋：画面套画面，一直往里掉。下落速度跟着这一段的进度和张力慢慢加快（不跟单个拍子跳 —— 👤：拍点推全局 = cheap）"""
    if _fall["t"] is None or t < _fall["t"]:
        _fall["x"] = 0.0
    dt = 1 / FPS if _fall["t"] is not None else 0
    _fall["t"] = t
    u = (t - t0) / (t1 - t0)
    _fall["x"] += dt * PERIOD * (0.25 + 0.9 * u * u + 0.3 * A.ch("tension", t))   # 越到后面掉得越快 = something is coming
    re = np.mod(LZ.real - _fall["x"] - np.log(R1), PERIOD) + np.log(R1)
    w = np.exp(re + 1j * LZ.imag)
    u_ = np.clip(w.real * (H / 2) + W / 2, 0, W - 1).astype(int)
    v_ = np.clip(w.imag * (H / 2) + H / 2, 0, H - 1).astype(int)
    return img[v_, u_]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fx", choices=["dct", "droste"])
    ap.add_argument("--from", dest="t0", type=float, default=E.parse_time("2:33.91"))
    ap.add_argument("--to", dest="t1", type=float, default=E.parse_time("2:53.11"))
    a = ap.parse_args()
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    out = S.MV / "样片" / f"段落试验_{a.fx}_{int(a.t0)}-{int(a.t1)}.mp4"
    p = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                          "-ss", f"{a.t0:.3f}", "-t", f"{a.t1 - a.t0:.3f}", "-i", str(S.MUSIC),
                          "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-c:a", "aac", "-shortest", str(out)], stdin=subprocess.PIPE)
    Mh, Mw = dct_mats()
    hung = np.array(A.SENSE["channels"]["hunger"])
    seg = hung[int(a.t0 * FPS):int(a.t1 * FPS)]
    lo, hi = seg.min(), seg.max()
    for t, img in frames_from_edl(a.t0, a.t1):
        u = (A.ch("hunger", t) - lo) / (hi - lo + 1e-9)          # 这一段里的相对饥饿 0~1
        frac = 10 ** (-1.3 - 2.7 * u)                            # 5% → 0.01%：越饿越只剩低频（一束光）
        frac = min(0.05, frac * (1 + 60 * A.ch("threat", t)))    # 起音时记忆短暂回来一下，再被压回去（对刺激的反应，不是一条死曲线）
        img = fx_dct(img, frac, Mh, Mw) if a.fx == "dct" else fx_droste(img, t, a.t0, a.t1)
        im = Image.fromarray((img * 255).astype(np.uint8))
        d = ImageDraw.Draw(im)
        n = int((W // 2) * (H // 2) * frac)
        for j, line in enumerate([f"t = {t:6.2f} s   BREAKDOWN  07 DCT", f"coefficients kept = {n:,} / {(W // 2) * (H // 2):,}  ({frac * 100:.3f} %)", f"hunger = {A.ch('hunger', t):.2f}"]):
            if a.fx == "dct":
                d.text((32, H - 95 + 21 * j), line, font=F, fill=(150, 160, 168))
        if a.fx == "droste":
            for j, line in enumerate(DROSTE_HUD):
                d.text((32, H - 95 + 21 * j), line.format(t=t), font=F, fill=(150, 160, 168))
        p.stdin.write(im.tobytes())
    p.stdin.close(); p.wait()
    print("->", out)


if __name__ == "__main__":
    main()
