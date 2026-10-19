#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🎬样片-粗剪v3-整首歌：0:00-0:52 用 v2；之后把原版 MV 的画面当素材，按规则重剪（👤2026-10-19：原版太平太慢，要更多 emotion inception、uprise、stronger movement；需要 3D 的先标文字，找用户生成）。
规则（锚点 → 乐句格子 13.11 + 6.4k，整首歌对得上 drop / breakdown / outro）：
  · 每半句（3.2 s）换一段原片画面（v3 初稿是每小节，👤嫌插入太多后改），从「同一段落」的原片里挑（保住原片的叙事顺序），不重复上一段
  · 乐句边界硬切 + 2 帧破碎；每拍 bump（drop 里更大）+ 威胁越大越抖
  · 强起音 break 2 帧；小节第一拍 blink 第二章记忆 1 帧（隔一个小节一次）
  · uprise：每次 drop 前那一小节每拍都切、播放加速、往里推、频闪
  · breakdown 放慢到 0.6 倍；drop 1.25 倍
  · 需要 3D 的段落开头打 1.5 秒文字卡「[3D 待生成] …」
用法：python tools/animatic_v3.py [--from 52] [--to 290]
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stills as S  # noqa: E402
import animatic_intro as A  # noqa: E402
import animatic_v2 as V  # noqa: E402

W, H, FPS = A.W, A.H, A.FPS
PH0, PL = V.PH0, V.PHASE_LEN
BAR, BEAT = PL / 4, PL / 16
G = lambda k: PH0 + PL * k   # noqa: E731  第 k 个乐句边界
# 段落：名字, 开始, 结束, 播放速度, 概念
SECTIONS = [("DROP 1", G(6), G(15), 1.25, "03 LOGISTIC r + 04 REACTION DIFFUSION"),
            ("BRIDGE", G(15), G(18), 1.0, "05 DLA"),
            ("DROP 2", G(18), G(22), 1.25, "06 SANDPILE"),
            ("BREAKDOWN", G(22), G(25), 0.6, "07 DCT"),
            ("REBUILD", G(25), G(30), 0.9, "08 DROSTE"),
            ("DROP 3", G(30), G(42), 1.25, "09 OPTIMAL TRANSPORT"),
            ("OUTRO", G(42), 290.0, 0.8, "10 ATTRACTOR")]
DROPS = [G(6), G(18), G(30)]
CARDS = {  # 段落开始 + 2 小节处，1.5 秒；需要用户生成的 3D 内容
    "DROP 1": "[3D 待生成]  人形（数学组成的她）表面长出图灵斑纹，r 推过 3.57 后斑纹失控",
    "BRIDGE": "[3D 待生成]  一只手 / 一个点里长出枝晶（DLA），像闪电",
    "DROP 3": "[3D 待生成]  人形碎成颗粒 → 沿最优传输流成新的形 → 再碎再聚",
    "OUTRO": "[3D 待生成]  完整的她回到吸引子：被扰动后又回到同一个形",
}
MV_SRC = S.MV.parent / "Assets" / "The Glitch Mob - Can't Kill Us [2020 Edit]_190189282.mp4"
FCN = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 30)


def scene_cuts():
    """原片的镜头切换点（拆镜头时 ffmpeg 场景检测的结果）"""
    out = [0.0]
    for line in (S.MV / "拆镜头" / "cuts.txt").read_text("utf-8").splitlines():
        if "pts_time:" in line:
            out.append(float(line.split("pts_time:")[1].split()[0]))
    return np.array(sorted(set(out + [295.0])))


def read_clip(t, dur):
    import imageio_ffmpeg
    raw = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-ss", f"{max(0, t):.3f}", "-t", f"{dur:.3f}", "-i", str(MV_SRC),
                          "-vf", f"scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    n = len(raw) // (W * H * 3)
    return np.frombuffer(raw[:n * W * H * 3], np.uint8).reshape(n, H, W, 3)


def section_at(t):
    for s in SECTIONS:
        if s[1] <= t < s[2]:
            return s
    return SECTIONS[-1]


def plan(t0, t1, rng):
    """把 [t0, t1) 切成一串「段」：(开始, 时长, 原片起点, 速度)。规则全在这里"""
    cuts = scene_cuts()
    segs = []
    t = t0
    prev_src = -1
    while t < t1 - 1e-6:
        sec = section_at(t)
        upr = any(d - BAR <= t < d for d in DROPS)            # drop 前那一小节
        step = BEAT if upr else 2 * BAR          # 克制（👤看 v2：插入太多）：平时半句换一段，只有 drop 前那一小节每拍切
        nxt = min(t1, (np.floor((t - PH0) / step + 1e-6) + 1) * step + PH0)
        dur = nxt - t
        speed = 2.0 if upr else sec[3]
        # 原片候选：和当前段落同一时段（±4 s）里的镜头
        pool = [i for i in range(len(cuts) - 1) if cuts[i + 1] > sec[1] - 4 and cuts[i] < sec[2] + 4 and cuts[i + 1] - cuts[i] > 0.3]
        pool = [i for i in pool if i != prev_src] or pool
        i = int(rng.choice(pool)); prev_src = i
        need = dur * speed
        a, b = cuts[i], cuts[i + 1]
        src = a + rng.random() * max(0, (b - a) - need)
        segs.append((t, dur, src, speed, upr))
        t = nxt
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="t0", type=float, default=52.0)
    ap.add_argument("--to", dest="t1", type=float, default=290.0)
    a = ap.parse_args()
    rng = np.random.default_rng(11)
    mem = V.memory_frames()
    strength = V.onset_strength()
    on = V.ON
    strong = set(np.round(on[strength > np.percentile(strength, 92)] * FPS).astype(int).tolist())   # 只要最强的 8%
    beats = V.BEATS
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    OUT = A.OUT
    tmp = OUT / "_tmp_v3.mp4"
    p = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(tmp)], stdin=subprocess.PIPE)
    segs = plan(a.t0, a.t1, rng)
    print(f"{len(segs)} segments", flush=True)
    phrase_edges = set(int(round(G(k) * FPS)) for k in range(60))
    downbeats = [PH0 + BAR * k for k in range(-8, 200)]
    fcount = 0
    for si, (ts, dur, src, speed, upr) in enumerate(segs):
        clip = read_clip(src, dur * speed + 0.2)
        n = int(round((ts + dur) * FPS)) - int(round(ts * FPS))
        sec = section_at(ts)
        for j in range(n):
            f = int(round(ts * FPS)) + j
            t = f / FPS
            k = min(len(clip) - 1, int(j * speed)) if len(clip) else 0
            img = clip[k].astype(np.float64) / 255 if len(clip) else np.zeros((H, W, 3))
            # 推：uprise 里一路往里推；drop 里随拍子 bump
            drop = sec[0].startswith("DROP")
            # bump 只在 drop 的小节第一拍（👤：每拍都跳 = 媚俗，情感平铺）
            kick = max((np.exp(-(t - b) * 9) for b in downbeats if 0 <= t - b < 0.4), default=0) if drop else 0
            zoom = 1 + 0.015 * kick + (0.12 * (j / max(1, n)) if upr else 0)
            img = V.bump(img, zoom)
            thr = A.ch("threat", t)
            if upr:
                img = np.roll(img, (int(rng.normal(0, 6 * thr)), int(rng.normal(0, 6 * thr))), (0, 1))
            tag = sec[0]
            if drop and any(abs(t - (G(k) + BEAT)) < 0.5 / FPS for k in range(60)):   # 每乐句一次（第 2 拍）
                img = mem[int(t * 13) % len(mem)]; tag += "  BLINK"
            if f in strong or f in phrase_edges or f - 1 in phrase_edges:
                img = V.glitch(img, rng, 1.0 if drop else 0.5); tag += "  BREAK"
            if upr and t > ts + dur - 0.5 * BAR and any(abs(t - b) < 1.5 / FPS for b in beats):   # 频闪只给 drop 前最后两拍
                img = np.clip(img + S.RED * 0.4, 0, 1); tag += "  UPRISE"
            im = A.to_im(img)
            card = CARDS.get(sec[0])
            if card and sec[1] + 2 * BAR <= t < sec[1] + 2 * BAR + 1.5:
                d = ImageDraw.Draw(im, "RGBA")
                tw = d.textlength(card, font=FCN)
                d.rectangle(((W - tw) / 2 - 24, H / 2 - 34, (W + tw) / 2 + 24, H / 2 + 34), fill=(0, 0, 0, 200), outline=(255, 42, 52, 255))
                d.text(((W - tw) / 2, H / 2 - 21), card, font=FCN, fill=(235, 235, 235))
            A.hud(im, [f"t = {t:6.2f} s   {tag}", f"concept: {sec[4]}   speed x{speed:.2f}", f"src = original MV @ {src + j * speed / FPS:6.2f} s"], f"{sec[0]}")
            p.stdin.write(im.tobytes()); fcount += 1
        if si % 20 == 0:
            print(f"seg {si}/{len(segs)}  t={ts:.1f}", flush=True)
    p.stdin.close(); p.wait()
    out = OUT / "04_粗剪v3_全曲.mp4"
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(OUT / "03_粗剪v2_0-52.mp4"), "-i", str(tmp), "-t", str(a.t1), "-i", str(S.MUSIC),
                    "-filter_complex", f"[0:v]trim=0:{a.t0},setpts=PTS-STARTPTS,fps={FPS}[a];[1:v]fps={FPS}[b];[a][b]concat=n=2:v=1:a=0[v]",
                    "-map", "[v]", "-map", "2:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-c:a", "aac", "-b:a", "256k", "-shortest", str(out)], check=True)
    tmp.unlink()
    print("->", out, fcount, "frames")


if __name__ == "__main__":
    main()
