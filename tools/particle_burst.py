#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🎬样片-粒子-吸气→炸碎（v3）：她全身就是点（亮度 = 设定图），没有「图片层」可挖。
45.11–51.51 吸气：只随心跳通道轻微呼吸，静。51.51 drop1：胸口受一记冲击 —— 冲量按距离衰减、带后仰（上身往后上方）、
有深度（z），阻尼减速；之后每个强起音再踢一次碎片；1 秒后碎片被这段 drop 低音的延迟嵌入轨道吸走。点数守恒。
👤对 v2 的批评：粒子在静止躯体旁边飘很怪；碎裂要短、猛，身体本身受力。
用法：python tools/particle_burst.py [--start 45.11] [--dur 9] [--hit 51.51]
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
    ap.add_argument("--start", type=float, default=45.11)
    ap.add_argument("--dur", type=float, default=9.0)
    ap.add_argument("--hit", type=float, default=51.51)
    ap.add_argument("--img", default=str(S.MV / "生成" / "角色_她_正面_灰_pro.jpg"))
    ap.add_argument("--N", type=int, default=420000)
    ap.add_argument("--out", default=str(S.MV / "样片" / "07_粒子_吸气炸碎_v3.mp4"))
    a = ap.parse_args()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    W, H = S.W, S.H
    rng = np.random.default_rng(11)

    # ---- 她 = N 个点：按亮度采样，点亮度 = 原图亮度（密到能看清脸）
    im = np.asarray(Image.open(a.img).convert("L").resize((W, H), Image.LANCZOS)) / 255.0
    lum = np.clip((im - 0.05) / 0.95, 0, 1)
    p = (lum ** 1.2).ravel(); p /= p.sum()
    idx = rng.choice(W * H, size=a.N, p=p)
    sy, sx = np.divmod(idx, W)
    x = (sx + rng.random(a.N)).astype(np.float64); y = (sy + rng.random(a.N)).astype(np.float64)
    z = (lum.ravel()[idx] - 0.5) * 40.0                       # 亮的地方略靠前：一点体积感
    b = lum.ravel()[idx]
    ys, xs = np.nonzero(lum > 0.25)
    top, bot = ys.min(), ys.max(); hgt = bot - top
    chest = np.array([xs.mean(), top + 0.40 * hgt])
    x0, y0 = x.copy(), y.copy()
    vx = np.zeros(a.N); vy = np.zeros(a.N); vz = np.zeros(a.N)

    # ---- 碎块：身体切成 K 块（最近种子点 = Voronoi），每块是一个刚体：同一个速度 + 自转 → 看起来是「碎开」不是「雾化」
    K = 420
    seeds = np.stack([x, y], 1)[rng.choice(a.N, K, replace=False)]
    piece = np.empty(a.N, np.int32)
    for s0 in range(0, a.N, 40000):
        P = np.stack([x[s0:s0 + 40000], y[s0:s0 + 40000]], 1)
        piece[s0:s0 + 40000] = np.argmin(((P[:, None, :] - seeds[None]) ** 2).sum(-1), 1)
    cxp = np.bincount(piece, x, K) / np.maximum(np.bincount(piece, minlength=K), 1)
    cyp = np.bincount(piece, y, K) / np.maximum(np.bincount(piece, minlength=K), 1)

    # ---- 冲击（按块）：胸口为中心，冲量 ∝ 1/(1+(d/r0)^2)；主要往**画面里面**推（+z = 远离镜头），上身后仰，脸和腿受得少
    dxp, dyp = cxp - chest[0], cyp - chest[1]
    dp = np.hypot(dxp, dyp) + 1e-6
    kp = 1.0 / (1.0 + (dp / (0.17 * hgt)) ** 2)
    upper = smooth(-(cyp - (top + 0.62 * hgt)), -40, 120)        # 腰以上 → 1
    face = np.exp(-(((cxp - xs.mean()) / 75) ** 2 + ((cyp - (top + 0.11 * hgt)) / 95) ** 2))
    Jp = 2300.0 * kp * (0.3 + 0.7 * upper) * (1 - 0.9 * face) * (0.5 + rng.random(K))
    Jp[kp * (0.3 + 0.7 * upper) * (1 - 0.9 * face) < 0.12] = 0   # 弱的块不动：腿和脸留下
    pvx = Jp * dxp / dp * 0.45 + Jp * 0.55 + rng.normal(0, 90, K) * (Jp > 0)   # 冲击从左前方来（和情绪大图 04 的光一致）→ 碎块往右后方飞，不对称
    pvy = Jp * dyp / dp * 0.45 - 220 * upper * (Jp > 0)           # 后仰：上身往上走一点
    pvz = Jp * 1.2 + 500 * upper * (Jp > 0)                        # 往后炸（远离镜头）
    pw = rng.normal(0, 2.5, K) * (Jp > 0)                          # 每块自转（rad/s）
    ivx, ivy, ivz = pvx[piece], pvy[piece], pvz[piece]

    # ---- 音乐：起音（含强度）+ 心跳 + drop 低音的延迟嵌入轨道
    sense = json.loads((S.MV / "感官.json").read_text("utf-8"))
    thr = np.array(sense["channels"]["threat"]); heart = np.array(sense["channels"]["heartbeat"])
    ons = [(t, thr[max(0, int(t * FPS) - 3):int(t * FPS) + 4].max()) for t in sense["events"]["onsets"]
           if a.hit + 0.05 < t < a.start + a.dur]
    big = [(t, s) for t, s in ons if s >= np.quantile([s for _, s in ons], 0.6)] if ons else []
    sr = 11025
    xa, _ = S.audio(sr=sr, start=a.start, dur=a.dur + 1.0)
    kk = int(sr / 120)
    xl = np.convolve(np.convolve(xa, np.ones(kk) / kk, mode="same"), np.ones(kk) / kk, mode="same")
    tau = int(sr * 0.012); win = int(sr * 0.5)
    xl = np.concatenate([np.zeros(win + 2 * tau), xl])
    slot = rng.random(a.N); orbit_c = np.array([W * 0.5, H * 0.46]); m_ema = 0.3
    F = 1400.0                                                    # 透视焦距（px）

    f = ImageFont.truetype(S.FONT, 18)
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.Popen([ff, "-y", "-loglevel", "error",
                             "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                             "-ss", str(a.start), "-t", str(a.dur), "-i", str(S.MUSIC),
                             "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16",
                             "-c:a", "aac", "-b:a", "256k", "-shortest", str(out)], stdin=subprocess.PIPE)
    n = int(round(a.dur * FPS)); dt = 1.0 / FPS
    hit_done = False; kicks = 0; t0w = time.time()
    for i in range(n):
        tt = a.start + i / FPS
        if not hit_done and tt >= a.hit:
            vx += ivx; vy += ivy; vz += ivz; hit_done = True
        for (to, s) in big:                                      # 之后的强起音：再踢一次已经离开身体的碎片
            if tt - dt < to <= tt:
                kick = 260 * s * rng.standard_normal((3, K)) * (Jp > 0)   # 按块踢：碎块保持完整
                vx += kick[0][piece]; vy += kick[1][piece]; vz += kick[2][piece]; kicks += 1
        if hit_done:
            age = tt - a.hit
            e = int((tt - a.start) * sr) + 2 * tau + win
            seg = xl[e - win - 2 * tau:e]
            m_ema = 0.85 * m_ema + 0.15 * max(np.abs(seg).max(), 1e-3); seg = seg / m_ema
            A, B, C = seg[2 * tau:], seg[tau:-tau], seg[:-2 * tau]
            th = 0.4 + 0.25 * age; ph = 0.3 + 0.1 * age
            j = (slot * (len(A) - 1)).astype(int)
            R = H * 0.34
            tx = orbit_c[0] + (A[j] * np.cos(th) - C[j] * np.sin(th)) * R
            ty = orbit_c[1] + ((A[j] * np.sin(th) + C[j] * np.cos(th)) * np.sin(ph) + B[j] * np.cos(ph)) * R
            tz = (A[j] * np.sin(th) + C[j] * np.cos(th)) * np.cos(ph) * R * 0.8 + 900   # 轨道在她身后的深处（远离镜头）
            moved = (Jp > 0)[piece]                               # 只有被打下来的碎块进轨道；脸和腿留在原地
            pull = 3.0 * smooth(age, 1.0, 3.0) * moved
            vx += (tx - x) * pull * dt; vy += (ty - y) * pull * dt; vz += (tz - z) * pull * dt
            damp = 0.965 + 0.015 * smooth(age, 0.0, 1.0) - 0.10 * smooth(age, 1.0, 2.0)   # 先保住动量飞出去；被轨道吸住时加阻尼（临界阻尼，不过冲成雾）
            vx *= damp; vy *= damp; vz *= damp
            x += vx * dt; y += vy * dt; z += vz * dt
            spin = pw * dt * (1 - smooth(age, 1.0, 2.8))                # 碎块自转（进轨道后停）
            if spin.any():
                n_ = np.maximum(np.bincount(piece, minlength=K), 1)
                gx = np.bincount(piece, x, K) / n_; gy = np.bincount(piece, y, K) / n_
                c_, s_ = np.cos(spin)[piece], np.sin(spin)[piece]
                ox_, oy_ = x - gx[piece], y - gy[piece]
                x = gx[piece] + ox_ * c_ - oy_ * s_; y = gy[piece] + ox_ * s_ + oy_ * c_
        else:                                                     # 吸气：只随心跳呼吸（±1.5px）
            hb = heart[min(int(tt * FPS), len(heart) - 1)]
            s_ = 1 + 0.004 * hb
            x = chest[0] + (x0 - chest[0]) * s_; y = chest[1] + (y0 - chest[1]) * s_
        # ---- 透视投影 + splat
        zz = np.clip(F + z, 200, None)
        px = W / 2 + (x - W / 2) * F / zz; py = H / 2 + (y - H / 2) * F / zz
        dens = S.splat(px, py, weight=b * (F / zz) ** 2)
        v = np.clip(dens / 1.6, 0, 1) ** 0.75
        img = v[..., None] * S.GREY * 1.08
        if hit_done and tt - a.hit < 2 / FPS:                     # 冲击那两帧：胸口一点红
            r = np.exp(-(((np.arange(W)[None, :] - chest[0]) ** 2 + (np.arange(H)[:, None] - chest[1]) ** 2) / (2 * 18.0 ** 2)))
            img = img * (1 - r[..., None]) + r[..., None] * S.RED
        img = S.glow(np.clip(img, 0, 1), 3, 0.3)
        fr = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        dr = ImageDraw.Draw(fr)
        loose = int(((Jp > 0)[piece]).sum()) if hit_done else 0
        hud = [f"t = {tt:6.2f} s   {'INHALE' if not hit_done else 'DROP 1'}   impulse @ {a.hit:.2f} s   kicks {kicks}",
               f"N = {a.N:,}   body {a.N - loose:7,}   shards {loose:7,}   mass in = mass out",
               f"|v| mean = {np.mean(np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)):7.1f} px/s"]
        for jj, line in enumerate(hud):
            dr.text((48, H - 48 - 26 * (len(hud) - jj)), line, font=f, fill=(150, 160, 168))
        dr.text((W - 48 - 260, 40), "THYMOS 03-04  INHALE / SHATTER", font=f, fill=(110, 118, 124))
        proc.stdin.write(fr.tobytes())
        if i % 48 == 0:
            print(f"frame {i}/{n}  {time.time() - t0w:.0f}s", flush=True)
    proc.stdin.close(); proc.wait()
    print("->", out)


if __name__ == "__main__":
    main()
