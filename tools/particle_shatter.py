#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🎬样片-粒子-碎裂 v4（学 The Crew 3:13–3:14 的逐帧机理，见 `_设计/方法_剪辑与情绪.md` §四⅞½）：
长静 → 褪色预告 → 单帧前震 → 2 帧横向拖影 → 1 帧无关噪点 → 1 帧白色横线 → 6–8 帧爆散 → 尘埃慢飘。
碎片双尺度：1–2 px 尘埃 + 整行宽的扫描线条块（她本来就是扫描线全息），没有中等碎片；单方向（向右），机位锁死，2D。
所有时刻锚在感官层：冲击 = 用户耳朵定的 drop 拍点；前震 = 冲击前最后一个起音。点数守恒，HUD 写真数。
用法：python tools/particle_shatter.py [--start 47.5] [--dur 7] [--hit 51.51]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stills as S  # noqa: E402

FPS = 24


def smooth(t, a, b):
    u = np.clip((t - a) / (b - a), 0, 1)
    return u * u * (3 - 2 * u)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=float, default=47.5)
    ap.add_argument("--dur", type=float, default=7.0)
    ap.add_argument("--hit", type=float, default=51.51)
    ap.add_argument("--img", default=str(S.MV / "生成" / "角色_她_正面_灰_pro.jpg"))
    ap.add_argument("--N", type=int, default=420000)
    ap.add_argument("--slab", type=float, default=0.35, help="属于扫描线条块的点的比例（其余是尘埃）")
    ap.add_argument("--out", default=str(S.MV / "样片" / "08_粒子_碎裂_v4.mp4"))
    a = ap.parse_args()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    W, H = S.W, S.H
    rng = np.random.default_rng(23)

    im = np.asarray(Image.open(a.img).convert("RGB").resize((W, H), Image.LANCZOS)) / 255.0
    L = im.mean(2); lum = np.clip((L - 0.05) / 0.95, 0, 1)
    p = (lum ** 1.2).ravel(); p /= p.sum()
    idx = rng.choice(W * H, size=a.N, p=p)
    sy, sx = np.divmod(idx, W)
    x0 = sx + rng.random(a.N); y0 = sy + rng.random(a.N)
    b = lum.ravel()[idx]; tint = im.reshape(-1, 3)[idx]           # 原色（极淡的青）→ 冲击前褪成纯白
    ys, xs = np.nonzero(lum > 0.25); top, bot = ys.min(), ys.max(); hgt = bot - top
    face = np.exp(-(((x0 - xs.mean()) / 80) ** 2 + ((y0 - (top + 0.11 * hgt)) / 100) ** 2))
    legs = smooth(y0 - (top + 0.70 * hgt), 0, 80)
    hitw = (1 - 0.95 * face) * (1 - 0.8 * legs)                    # 受力权重：脸和腿基本留下

    # ---- 双尺度：条块 = 同一行带（高 2–4 px）的点一起走；尘埃 = 每点各走各的
    band_h = rng.integers(2, 5, size=H)                            # 每行属于多高的条带
    band = np.cumsum(band_h)[np.minimum(np.searchsorted(np.cumsum(band_h), y0.astype(int)), H - 1)] // 1
    _, band = np.unique(band, return_inverse=True); nb = band.max() + 1
    is_slab = (rng.random(nb) < a.slab)[band] & (rng.random(a.N) < 0.95)
    vb = rng.lognormal(np.log(5200), 0.55, nb)                      # 条块要横穿画面（The Crew 的条块飞满全宽）                      # 每条带一个水平速度（向右）
    vx_i = np.where(is_slab, vb[band], rng.lognormal(np.log(1500), 0.7, a.N)) * hitw
    vy_i = np.where(is_slab, rng.normal(0, 12, nb)[band], rng.normal(0, 160, a.N)) * hitw
    gone = hitw > 0.08 + 0.5 * rng.random(a.N) ** 2                # 抖动阈值：脸的边缘也碎成颗粒，不是一块光滑的椭圆（👤：「为啥留一个边缘清晰的脸」）

    sense = json.loads((S.MV / "感官.json").read_text("utf-8"))
    pre = [t for t in sense["events"]["onsets"] if a.hit - 1.6 < t < a.hit - 0.15]
    pre_t = pre[-1] if pre else a.hit - 0.6                         # 前震 = 冲击前最后一个起音

    f = ImageFont.truetype(S.FONT, 18)
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.Popen([ff, "-y", "-loglevel", "error",
                             "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                             "-ss", str(a.start), "-t", str(a.dur), "-i", str(S.MUSIC),
                             "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16",
                             "-c:a", "aac", "-b:a", "256k", "-shortest", str(out)], stdin=subprocess.PIPE)
    n = int(round(a.dur * FPS)); fh = int(round((a.hit - a.start) * FPS))   # 冲击帧号
    x, y = x0.copy(), y0.copy(); vx = np.zeros(a.N); vy = np.zeros(a.N)
    t0w = time.time(); phase = "STILL"
    for i in range(n):
        tt = a.start + i / FPS; k = i - fh
        drain = smooth(tt, a.hit - 1.0, a.hit - 0.1)               # 褪色：原色 → 纯白剪影
        g = tint.mean(1, keepdims=True) * np.array([[1.0, 1.0, 1.02]])
        col = tint * (1 - drain) + g * drain * 1.15                # 褪色：去掉那点青，只略提亮（别过曝成白块）
        img = None
        if k < -2:
            phase = "STILL"
            px, py = x0, y0
            if abs(tt - pre_t) < 0.5 / FPS:                        # 单帧前震：一条带横向错位
                phase = "PRE-SHOCK"
                yb = top + rng.random() * hgt; sel = np.abs(y0 - yb) < 26
                px = x0 + sel * rng.normal(0, 90)
        elif k in (-2, -1):                                         # 2 帧横向拖影
            phase = "SMEAR"
            sm = (k + 3) * 70.0 * hitw
            px = np.concatenate([x0 + sm * u for u in np.linspace(0, 1, 6)]); py = np.tile(y0, 6)
            col = np.tile(col, (6, 1)) if col.ndim == 2 else col
        elif k == 0:                                                # 1 帧无关噪点（信号断了）
            phase = "NOISE"
            nz = rng.random((H // 4, W // 4)) ** 3 * (rng.random((H // 4, 1)) < 0.6)
            v = np.kron(nz, np.ones((4, 4)))
            img = np.clip(v[..., None] * np.array([0.8, 0.82, 0.85]), 0, 1)
        elif k == 1:                                                # 1 帧白色横线
            phase = "FLASH"
            rows = np.zeros(H); r = rng.choice(H, 40, replace=False); rows[r] = rng.random(40)
            core = np.exp(-((np.arange(H) - (top + 0.4 * hgt)) / 30.0) ** 2)
            v = np.clip(rows[:, None] * (0.6 + 0.4 * rng.random((1, W))) + core[:, None] * 1.4, 0, 1)
            img = v[..., None] * np.ones(3)
            vx = vx_i.copy(); vy = vy_i.copy()
        else:                                                       # 爆散（6–8 帧急刹）→ 尘埃慢飘
            phase = "SHATTER" if k < 9 else "DUST"
            damp = 0.80 if k < 9 else 0.93
            vx *= np.where(gone, damp, 0); vy *= np.where(gone, damp, 0)
            if k >= 9:                                              # 尘埃：很慢的漂移 + 一点布朗
                vx += gone * rng.normal(8, 14, a.N); vy += gone * rng.normal(-3, 10, a.N)
            x = x + vx / FPS; y = y + vy / FPS
            px, py = x, y
        if img is None:
            stretch = 1.0
            if phase == "SHATTER":                                  # 条块带运动模糊：沿 x 拉成线
                px = np.concatenate([px - vx / FPS * u for u in np.linspace(0, 1, 4)]); py = np.tile(py, 4)
                stretch = 4.0
            w = np.tile(b, len(px) // a.N) / stretch
            cc = col if col.ndim == 2 and len(col) == len(px) else np.tile(col if col.ndim == 2 else np.ones((a.N, 3)), (len(px) // a.N, 1))
            img = np.stack([S.splat(px, py, weight=w * cc[:, c]) for c in range(3)], -1)
            img = np.clip(img / 1.5, 0, 1) ** 0.8
            img = S.glow(img, 3, 0.25 if phase != "SHATTER" else 0.5)
        fr = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        dr = ImageDraw.Draw(fr)
        moved = int(((np.hypot(x - x0, y - y0) > 6) & gone).sum()) if k > 1 else 0
        hud = [f"t = {tt:6.2f} s   {phase:<9}   pre-shock @ {pre_t:.2f}   impact @ {a.hit:.2f}",
               f"N = {a.N:,}   body {a.N - moved:7,}   dust {int((moved and (~is_slab & gone).sum()) or 0):7,}   slabs {int((moved and (is_slab & gone).sum()) or 0):7,}   mass in = mass out"]
        if phase not in ("NOISE", "FLASH"):
            for jj, line in enumerate(hud):
                dr.text((48, H - 48 - 26 * (len(hud) - jj)), line, font=f, fill=(150, 160, 168))
        proc.stdin.write(fr.tobytes())
        if i % 48 == 0:
            print(f"frame {i}/{n}  {time.time() - t0w:.0f}s  {phase}", flush=True)
    proc.stdin.close(); proc.wait()
    print("->", out)


if __name__ == "__main__":
    main()
