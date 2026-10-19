#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🎬样片-粗剪 0:00-0:52-看镜头混在一起是什么感觉（👤2026-10-19 问）。1280×720 草稿画质，配原曲。
  01 0:00-0:10 延迟嵌入（直接用已渲好的 01_delay_embedding_10s.mp4）
  02 0:10-0:46 Kuramoto：锚点 = 「0:46 同步成一个」；规则 = Kuramoto；每个拍点给耦合 K 加一格，威胁通道踢一下相位
  03 0:46-0:52 吸气：这首歌前 46 秒的频谱当地形，点云山；语言卡「US」（还没有人形模型，先用语言顶）
读 Projekt_Visual/MV_Cant_Kill_Us/感官.json（Eidolon 感官层）。
用法：python tools/animatic_intro.py
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stills as S  # noqa: E402

W, H, FPS = 1280, 720, 24
OUT = S.MV / "样片"
SENSE = json.loads((S.MV / "感官.json").read_text("utf-8"))
F = ImageFont.truetype(S.FONT, 15)
FB = ImageFont.truetype("C:/Windows/Fonts/consolab.ttf", 120)


def ch(name, t):
    v = SENSE["channels"][name]
    return v[min(int(t * FPS), len(v) - 1)]


def hud(im, lines, title):
    d = ImageDraw.Draw(im)
    for j, line in enumerate(lines):
        d.text((32, H - 32 - 21 * (len(lines) - j)), line, font=F, fill=(150, 160, 168))
    d.text((W - 32 - 9 * len(title), 26), title, font=F, fill=(110, 118, 124))
    return im


def to_im(img):
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))


def shot_kuramoto(t0=10.0, t1=46.0, N=3000, seed=1):
    rng = np.random.default_rng(seed)
    w = np.clip(rng.standard_cauchy(N) * 0.5, -3, 3)
    th = rng.uniform(0, 2 * np.pi, N)
    beats = [b for b in SENSE["events"]["beats"] if t0 <= b < t1]
    K0, K1 = 0.3, 2.6                            # 锚点：0:46 时 K = 2.6，远超临界 K_c = 1
    step = (K1 - K0) / max(1, len(beats))
    acc = np.zeros((H, W)); accr = np.zeros((H, W))
    r = 150 + 32 * w
    for i in range(int((t1 - t0) * FPS)):
        t = t0 + i / FPS
        K = K0 + step * sum(1 for b in beats if b <= t)
        for _ in range(4):                      # 每帧 4 个子步
            z = np.exp(1j * th).mean()
            th = th + (1 / FPS / 4) * 2.2 * (w + 0.6 * ch("heartbeat", t) + K * abs(z) * np.sin(np.angle(z) - th))
        th = th + rng.normal(0, 0.35 * ch("threat", t), N)     # 威胁 = 相位被踢
        R = abs(np.exp(1j * th).mean())
        lock = np.abs(w) < K * R
        x, y = W / 2 + r * np.cos(th), H / 2 + r * np.sin(th)
        acc = acc * 0.82 + S.splat(x, y, W, H)
        accr = accr * 0.82 + S.splat(x[lock], y[lock], W, H)
        v = np.clip(np.log1p(acc) / 3.0, 0, 1) ** 0.6
        vr = np.clip(np.log1p(accr) / 3.0, 0, 1) ** 0.6 * (R ** 2)   # 同步得越好，红得越多
        img = v[..., None] * S.GREY * (1 - vr[..., None]) + vr[..., None] * S.RED
        img = S.glow(np.clip(img, 0, 1), 3, 0.5)
        yield hud(to_im(img), [f"t = {t:6.2f} s   N = {N}", f"K = {K:.2f}   (K_c = 1.00)   +{step:.3f} per beat", f"R = {R:.3f}   locked = {lock.mean()*100:4.1f} %"], "02  KURAMOTO")


def shot_inhale(t0=46.0, t1=52.0):
    x, sr = S.audio(sr=11025, start=0, dur=46.0)
    n, hop = 1024, 2048
    frames = (len(x) - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(frames)[:, None]
    spec = np.abs(np.fft.rfft(x[idx] * np.hanning(n), axis=1))
    fq = np.fft.rfftfreq(n, 1 / sr)
    rows = np.searchsorted(fq, np.geomspace(40, 5000, 90))
    Z = np.log1p(spec[:, rows] * 8)                # (时间, 频率)
    Z = Z / Z.max()
    T, Fn = np.meshgrid(np.linspace(-1, 1, Z.shape[0]), np.linspace(-1, 1, Z.shape[1]), indexing="ij")
    rng = np.random.default_rng(9)
    # 每个格点撒几个点，抖一点，像原图那种点云山
    P = np.repeat(np.stack([T.ravel(), Fn.ravel(), Z.ravel()], 1), 3, 0)
    P[:, :2] += rng.normal(0, 0.006, (len(P), 2))
    for i in range(int((t1 - t0) * FPS)):
        t = t0 + i / FPS; u = i / ((t1 - t0) * FPS)
        yaw = -0.5 + 0.25 * u
        cx, cy = np.cos(yaw), np.sin(yaw)
        X = P[:, 0] * cx - P[:, 1] * cy; Y = P[:, 0] * cy + P[:, 1] * cx
        Zh = P[:, 2] ** 1.6 * 0.9
        depth = Y + 2.6 - 0.3 * u                 # 镜头慢慢推近
        sx = W / 2 + X / depth * 900; sy = H * 0.72 - (Zh - 0.15) / depth * 900
        img = np.clip(np.log1p(S.splat(sx, sy, W, H, weight=1 / depth)) / 2.2, 0, 1) ** 0.7
        im = to_im(S.glow(img[..., None] * S.GREY, 2, 0.4))
        d = ImageDraw.Draw(im)
        al = min(1.0, u * 3)                      # 「US」半秒内淡入
        d.text((W / 2 - 120, H * 0.18), "US", font=FB, fill=(int(255 * al), int(42 * al), int(52 * al)))
        yield hud(im, [f"t = {t:6.2f} s   terrain = spectrum 0:00-0:46", "40 Hz - 5 kHz, log", "first sync. R > 0.9"], "03  INHALE")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    tmp = OUT / "_tmp_02_03.mp4"
    p = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(tmp)], stdin=subprocess.PIPE)
    k = 0
    for gen in (shot_kuramoto(), shot_inhale()):
        for im in gen:
            p.stdin.write(im.tobytes()); k += 1
            if k % 120 == 0:
                print("frame", k, flush=True)
    p.stdin.close(); p.wait()
    out = OUT / "02_粗剪_0-52.mp4"
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(OUT / "01_delay_embedding_10s.mp4"), "-i", str(tmp),
                    "-t", "52", "-i", str(S.MUSIC),
                    "-filter_complex", f"[0:v]scale={W}:{H},fps={FPS}[a];[1:v]fps={FPS}[b];[a][b]concat=n=2:v=1:a=0[v]",
                    "-map", "[v]", "-map", "2:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-c:a", "aac", "-b:a", "256k", "-shortest", str(out)], check=True)
    tmp.unlink()
    print("->", out)


if __name__ == "__main__":
    main()
