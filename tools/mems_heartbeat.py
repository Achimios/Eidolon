#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🎬样片-MEMS 机械心跳：drop1 的抽搐音（👤纠正后：1:03、1:15 两次；全片 4 次 + drop3 三次弱回声）（`动机_抽搐.json`）各切进 1 秒 MEMS 梳齿谐振器（工程图样式）。
位移 = 「other」分轨（抽搐音所在层）在这一秒的真实包络 × 可见频率的正弦（真实 f0=27.1 kHz 太快，按比例放慢画，HUD 写明）。
同一元素、每次推进（👤：先机械后有机）：#1 一阶平动 → #2 幅度更大、折叠梁发亮 → #3 加二阶摇摆 → #4 梳齿开始弯成曲线（有机的前兆，接 drop2/3 的克拉尼）。
其余时间全黑（插入镜头，配原曲看节奏）。
用法：python tools/mems_heartbeat.py [--start 55.5] [--end 76.6]
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

MV = Path("C:/Users/Administrator/Desktop/Projekt_Visual/MV_Cant_Kill_Us")
MUSIC = MV.parent / "Assets" / "The Glitch Mob - Can't Kill Us [2020 Edit]_190189282.mp4"
FPS, W, H, SS = 24, 1920, 1080, 2           # SS = 超采样（线条抗锯齿）
GREY, DIM, RED = (214, 220, 226), (90, 96, 102), (232, 44, 52)
F0, F_VIS = 27.1e3, 7.0                     # 真实一阶频率；画面上的可见频率


def drum_env(t0, t1):
    p = next(MV.glob("分轨/**/other.wav"))
    x, sr = sf.read(p); x = x.mean(1) if x.ndim > 1 else x
    seg = np.abs(x[int(t0 * sr):int(t1 * sr)])
    k = int(sr / 40)
    env = np.convolve(seg, np.ones(k) / k, mode="same")
    return env / (env.max() + 1e-9), sr


def draw_mems(d, u, v, stage, glow, s):
    """u = 平动位移（px，归一化后乘尺度），v = 二阶摇摆（rad），stage 1..4，glow = 折叠梁亮度"""
    cx, cy = W * s / 2, H * s / 2 + 20 * s
    L = lambda pts, c, w=1: d.line([(cx + x * s, cy + y * s) for x, y in pts], fill=c, width=max(1, int(w * s)))
    # 锚点（固定）
    for sx_ in (-1, 1):
        d.rectangle([cx + (sx_ * 560 - 40) * s, cy - 300 * s, cx + (sx_ * 560 + 40) * s, cy + 300 * s], outline=DIM, width=s)
    # 梭子（中间的质量块）：平动 u + 绕中心摇摆 v
    def P(x, y):
        c, sn = np.cos(v), np.sin(v)
        return (x * c - y * sn + u, x * sn + y * c)
    sh = [P(-180, -120), P(180, -120), P(180, 120), P(-180, 120), P(-180, -120)]
    L(sh, GREY, 1.4)
    # 折叠梁（上下各两组，像一对翅膀）：端部跟梭子，形状 = 悬臂梁挠曲线 3ξ²−2ξ³
    for sy_ in (-1, 1):
        for sx_ in (-1, 1):
            xs = np.linspace(0, 1, 40)
            for off in (0, 14, 28):
                y0 = sy_ * (140 + off)
                pts = []
                for xi in xs:
                    x = sx_ * (180 + xi * 340)
                    defl = (3 * (1 - xi) ** 2 - 2 * (1 - xi) ** 3)      # 靠梭子一端 = 1，锚点一端 = 0
                    ex, ey = P(sx_ * 180, y0)
                    pts.append((x + (ex - sx_ * 180) * defl, y0 + (ey - y0) * defl))
                c = tuple(int(DIM[k] + (GREY[k] - DIM[k]) * glow) for k in range(3))
                L(pts, c, 1.0)
    # 梳齿：动齿（跟梭子）和定齿交错；stage 4 动齿尖端开始弯（有机的前兆）
    for sy_ in (-1, 1):
        for j in range(9):
            yy = sy_ * (20 + j * 11)
            if abs(yy) > 115:
                continue
            for sx_ in (-1, 1):
                # 定齿（从锚点伸出）
                L([(sx_ * 520, yy + 5), (sx_ * 250, yy + 5)], DIM, 0.9)
                # 动齿（从梭子伸出）
                n = 18; pts = []
                for q in range(n + 1):
                    xi = q / n
                    x0, y0 = sx_ * (180 + xi * 260), yy
                    bend = (stage >= 4) * 26 * xi ** 2 * np.sin(3 * j + 2 * xi * np.pi + u * 0.02)
                    px, py = P(x0, y0 + bend)
                    pts.append((px, py))
                L(pts, GREY, 0.9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=float, default=55.5)
    ap.add_argument("--end", type=float, default=76.6)
    a = ap.parse_args()
    hits = [h["t"] for h in json.loads((MV / "动机_抽搐.json").read_text("utf-8"))["hits"] if a.start <= h["t"] < a.end]
    tpl = json.loads((MV / "动机_抽搐.json").read_text("utf-8"))["template"]; dur = tpl[1] - tpl[0]
    envs = [drum_env(t, t + dur) for t in hits]
    f = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 18)
    import imageio_ffmpeg
    out = MV / "样片" / "09_MEMS心跳_drop1.mp4"
    proc = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                             "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                             "-ss", str(a.start), "-t", str(a.end - a.start), "-i", str(MUSIC),
                             "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                             "-c:a", "aac", "-b:a", "192k", "-shortest", str(out)], stdin=subprocess.PIPE)
    n = int((a.end - a.start) * FPS)
    black = np.zeros((H, W, 3), np.uint8).tobytes()
    for i in range(n):
        t = a.start + i / FPS
        k = next((q for q, h in enumerate(hits) if h <= t < h + dur), None)
        if k is None:
            proc.stdin.write(black); continue
        stage = k + 1; tl = t - hits[k]
        env, sr = envs[k]; e = env[min(int(tl * sr), len(env) - 1)]
        amp = [55, 80, 75, 90][min(k, 3)]
        u = amp * e * np.sin(2 * np.pi * F_VIS * tl)
        v = (stage >= 3) * 0.06 * e * np.sin(2 * np.pi * F_VIS * 1.618 * tl + 0.7)
        im = Image.new("RGB", (W * SS, H * SS), (6, 7, 8)); d = ImageDraw.Draw(im)
        draw_mems(d, u, v, stage, glow=min(1, 0.2 + 0.3 * k + e * 0.5), s=SS)
        # 红：位移最大的那一点（梭子右上角）只在包络峰值附近亮
        if e > 0.8:
            cx, cy = W * SS / 2 + (180 + u) * SS, H * SS / 2 + (20 - 120) * SS
            d.ellipse([cx - 7 * SS, cy - 7 * SS, cx + 7 * SS, cy + 7 * SS], fill=RED)
        im = im.resize((W, H), Image.LANCZOS); dd = ImageDraw.Draw(im)
        hud = [f"HEARTBEAT {stage}/{len(hits)}    t = {t:6.2f} s    motif 'twitch' #{stage} @ {hits[k]:.2f} s",
               f"x(t) = {amp} px * env_other(t) * sin(2pi * {F_VIS:.0f} Hz * t)    drawn {F0 / F_VIS:,.0f}x slower than f0 = 27.1 kHz",
               ["MODE 1  TRANSLATION", "MODE 1  FLEXURES LOADED", "MODE 1 + 2  ROCKING", "MODE 1 + 2  FINGERS YIELDING"][min(k, 3)]]
        for jj, line in enumerate(hud):
            dd.text((48, H - 48 - 26 * (len(hud) - jj)), line, font=f, fill=(150, 160, 168))
        dd.text((W - 48 - 230, 40), "13  MEMS  COMB DRIVE", font=f, fill=(110, 118, 124))
        proc.stdin.write(im.tobytes())
    proc.stdin.close(); proc.wait()
    print("->", out)


if __name__ == "__main__":
    main()
