#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🎬样片-粗剪v2-0:00-0:52：Kuramoto 段改成「规则剪辑」（👤2026-10-19 看 v1：太闷、节奏没体现、切得太克制）。
锚点（用户给的，见 Projekt_Visual/MV_Cant_Kill_Us/锚点.json）→ 规则：
  · 乐句格子：13.11 起每 6.4 s（150 BPM 的 4 小节）硬切，换一个「视角」；切口 2 帧破碎转场
  · 半句（3.2 s）处临时切入：另一种情绪（记忆实拍 / 入侵）持续 2 拍，再切回
  · 每小节第一拍（1.6 s）：blink —— 第二章的实拍记忆插 1~2 帧
  · 强起音：break —— 当前画面横条错位 + RGB 分离 2 帧
  · 每拍：bump —— 画面放大一下再弹回
  · 38.71 s（≈用户 38.5 「something is coming」）起：切口密度翻倍（每小节切）、K 加速、红色频闪越来越密、入侵从边缘长进来
同一个 Kuramoto 模拟，四个视角：环 / 微距 / 栅格（相位 × 振子，像 MACHINE BEHAVIOR）/ 序参量。
用法：python tools/animatic_v2.py
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stills as S  # noqa: E402
import animatic_intro as A  # noqa: E402

W, H, FPS = A.W, A.H, A.FPS
SENSE = A.SENSE
ON = np.array(SENSE["events"]["onsets"]); BEATS = np.array(SENSE["events"]["beats"])
PH0, PHASE_LEN = 13.11, 6.4
BAR = PHASE_LEN / 4
RISE = PH0 + 4 * PHASE_LEN          # 38.71
MEM_DIR = S.MV / "cache" / "记忆帧"
CH2 = S.PROJ / "第二章_墙里的东西" / "素材"
F = A.F


def onset_strength():
    """每个起音的强度 = 威胁通道在起音后 ±3 帧的最大值（直接按帧取会落在冲击之前，v1 的坑）"""
    th = np.array(SENSE["channels"]["threat"])
    return np.array([th[max(0, int(t * FPS) - 1):int(t * FPS) + 4].max() for t in ON])


def memory_frames(n_per=3):
    """第二章 Seedance 镜头里抽记忆帧，冷色去饱和；缓存到作品 cache/"""
    MEM_DIR.mkdir(parents=True, exist_ok=True)
    if len(list(MEM_DIR.glob("*.jpg"))) < 6:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        for k, v in enumerate(sorted(CH2.glob("*shot*.mp4"))):
            for j, t in enumerate(np.linspace(1.5, 8.5, n_per)):
                subprocess.run([ff, "-loglevel", "error", "-y", "-ss", f"{t:.2f}", "-i", str(v), "-frames:v", "1",
                                "-vf", f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}", str(MEM_DIR / f"m{k}{j}.jpg")], check=True)
    out = []
    for p in sorted(MEM_DIR.glob("*.jpg")):
        a = np.asarray(Image.open(p).convert("RGB"), np.float64) / 255
        g = a.mean(2, keepdims=True)
        out.append(np.clip((g * 0.7 + a * 0.3) * np.array([0.85, 0.95, 1.05]) * 1.1, 0, 1))   # 冷、去饱和，像记得不清的
    return out


def glitch(img, rng, amt):
    """break：横条错位 + RGB 分离"""
    out = img.copy()
    y = 0
    while y < H:
        h = int(rng.integers(4, 60)); dx = int(rng.normal(0, 40 * amt)) if rng.random() < 0.6 else 0
        out[y:y + h] = np.roll(img[y:y + h], dx, axis=1); y += h
    s = int(6 + 14 * amt)
    out[..., 0] = np.roll(out[..., 0], s, 1); out[..., 2] = np.roll(out[..., 2], -s, 1)
    return out


def bump(img, k):
    """bump：中心放大 k 倍（>1）"""
    if k <= 1.002:
        return img
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    w2, h2 = int(W / k), int(H / k)
    im = im.crop(((W - w2) // 2, (H - h2) // 2, (W - w2) // 2 + w2, (H - h2) // 2 + h2)).resize((W, H), Image.BILINEAR)
    return np.asarray(im, np.float64) / 255


def invade_mask(t, rd):
    """入侵：反应扩散的纹理从画面边缘往里长（38.71 s 后）"""
    u = np.clip((t - RISE) / (46.0 - RISE), 0, 1)
    yy, xx = np.mgrid[0:H, 0:W]
    edge = np.minimum(np.minimum(xx, W - xx) / W, np.minimum(yy, H - yy) / H) * 2   # 0 在边缘，1 在中心
    return ((edge < u * 0.55) * rd).astype(np.float64) * u


def shot_kuramoto_v2(t0=10.0, t1=46.0, N=3000, seed=1):
    rng = np.random.default_rng(seed)
    w = np.clip(rng.standard_cauchy(N) * 0.5, -3, 3)
    order = np.argsort(w)
    th = rng.uniform(0, 2 * np.pi, N)
    strength = onset_strength()
    strong = set(np.round(ON[(strength > np.percentile(strength[(ON > t0) & (ON < t1)], 92)) & (ON > t0) & (ON < t1)] * FPS).astype(int).tolist())
    beats = BEATS[(BEATS >= t0) & (BEATS < t1)]
    mem = memory_frames()
    rd = np.asarray(Image.open(S.MV / "静帧" / "04_reaction_diffusion.jpg").convert("L").resize((W, H)), np.float64) / 255 > 0.5
    # K：拍点推，38.71 后加速；锚点 0:46 时 K ≈ 3
    K = 0.8
    acc = np.zeros((H, W)); accr = np.zeros((H, W))
    raster = np.zeros((360, W))          # 栅格视角的滚动缓冲：每帧往左挪 4 列
    r = 150 + 32 * w
    views = ["ring", "macro", "raster", "order"]
    view = 0
    cuts = sorted(set([PH0 + k * PHASE_LEN for k in range(5)] +
                      [RISE + k * BAR for k in range(1, 5)]))           # 乐句硬切 + rise 后每小节切
    cutins = [PH0 + k * PHASE_LEN + PHASE_LEN / 2 for k in range(0, 4, 2)]   # 半句临时切入；克制（👤v2 反馈）：隔一个乐句才切入
    bars = [PH0 + k * PHASE_LEN + PHASE_LEN / 16 for k in range(0, 6)]   # blink：每乐句一次（第 2 拍），不是每小节
    last_cut = -1
    for i in range(int((t1 - t0) * FPS)):
        t = t0 + i / FPS
        f = int(round(t * FPS))
        rising = t >= RISE
        nb = int(((beats > t - 1 / FPS) & (beats <= t)).sum())
        K += nb * (0.035 if not rising else 0.17)
        for _ in range(4):
            z = np.exp(1j * th).mean()
            th = th + (1 / FPS / 4) * 2.2 * (w + 0.6 * A.ch("heartbeat", t) + K * abs(z) * np.sin(np.angle(z) - th))
        th = th + rng.normal(0, 0.35 * A.ch("threat", t), N)
        z = np.exp(1j * th).mean(); R = abs(z); psi = np.angle(z)
        lock = np.abs(w) < K * R
        x, y = W / 2 + r * np.cos(th), H / 2 + r * np.sin(th)
        acc = acc * 0.82 + S.splat(x, y, W, H); accr = accr * 0.82 + S.splat(x[lock], y[lock], W, H)
        raster = np.roll(raster, -4, 1)
        col = (np.cos(th[order] - psi) * 0.5 + 0.5)
        raster[:, -4:] = np.interp(np.linspace(0, N - 1, 360), np.arange(N), col)[:, None] ** 3
        # ---- 剪辑规则 ----
        tag = ""
        hit_cut = [c for c in cuts if c - 0.5 / FPS <= t < c + 0.5 / FPS]
        if hit_cut:
            view = (view + 1 + int(rng.integers(0, 2))) % len(views); last_cut = f; tag = "CUT"
        v = views[view]
        in_cutin = any(c <= t < c + 2 * 60 / SENSE["bpm"] for c in cutins)
        is_blink = any(abs(t - b) < 0.5 / FPS or abs(t - b - 1 / FPS) < 0.5 / FPS for b in bars) and not hit_cut
        # ---- 画视角 ----
        if v == "ring":
            base = A.S.colorize(np.clip(np.log1p(acc) / 3, 0, 1) ** 0.6)
            red = np.clip(np.log1p(accr) / 3, 0, 1) ** 0.6 * R ** 2
            img = base * (1 - red[..., None]) + red[..., None] * S.RED
        elif v == "macro":   # 跟着同步团的方向，3 倍放大
            cx, cy = W / 2 + 160 * np.cos(psi), H / 2 + 160 * np.sin(psi)
            mx, my = (x - cx) * 3 + W / 2, (y - cy) * 3 + H / 2
            d = S.splat(mx, my, W, H); dr = S.splat(mx[lock], my[lock], W, H)
            base = S.colorize(np.clip(S.blur(d, 2) * 6, 0, 1) ** 0.5); red = np.clip(S.blur(dr, 2) * 6, 0, 1) * R   # 微距：点要大要亮（v2 试渲太暗太稀）
            img = base * (1 - red[..., None]) + red[..., None] * S.RED
        elif v == "raster":
            g = np.zeros((H, W)); g[180:540] = raster
            img = S.colorize(g) * 0.9
        else:                # 序参量：一根矢量 + 相位分布的环
            d = np.zeros((H, W))
            hist, _ = np.histogram(np.mod(th - psi, 2 * np.pi), 180, (0, 2 * np.pi))
            ang = np.linspace(0, 2 * np.pi, 180, endpoint=False) + psi
            for a_, hh in zip(ang, hist / hist.max()):
                rr = np.linspace(220, 220 + 120 * hh, 30)
                d += S.splat(W / 2 + rr * np.cos(a_), H / 2 + rr * np.sin(a_), W, H)
            rr = np.linspace(0, 200 * R, 200)
            vec = S.splat(W / 2 + rr * np.cos(psi), H / 2 + rr * np.sin(psi), W, H)
            img = S.colorize(np.clip(d, 0, 1)) + np.clip(S.blur(vec, 2) * 6, 0, 1)[..., None] * S.RED
        img = S.glow(np.clip(img, 0, 1), 3, 0.5)
        # ---- 情绪切入 / 入侵 / blink / break / bump ----
        if in_cutin:
            m = mem[int(t * 7) % len(mem)]
            img = m * 0.85 if not rising else np.clip(m * 0.6 + img * 0.6, 0, 1); tag = "MEMORY"
        if rising:
            inv = invade_mask(t, rd)
            img = img * (1 - inv[..., None] * 0.8) + inv[..., None] * S.RED * 0.6
            if nb and rng.random() < np.clip((t - RISE) / 6, 0.0, 0.5):   # 红色频闪越来越密
                img = np.clip(img + S.RED * 0.35, 0, 1); tag = tag or "STROBE"
        if is_blink:
            img = mem[int(t * 13) % len(mem)]; tag = "BLINK"
        if f in strong or f - 1 in strong or (f - last_cut) in (1, 2):
            img = glitch(img, rng, 0.6 if not rising else 1.0); tag = tag or "BREAK"
        # bump：只在上扬段的小节第一拍（👤：每拍都让圆圈跳 = 媚俗，强烈情感平铺把别的情感压下去）
        dbs = [PH0 + k * BAR for k in range(0, 24)]
        kick = max((np.exp(-(t - b) * 9) for b in dbs if 0 <= t - b < 0.4), default=0) if rising else 0
        # 👤：「鼓点不要影响全局放大。This is too cheap. With no form in it.」→ 不再用拍点推整幅画面
        im = A.to_im(img)
        yield A.hud(im, [f"t = {t:6.2f} s   view = {v}   {tag}",
                         f"K = {K:.2f}   R = {R:.3f}   locked = {lock.mean() * 100:4.1f} %",
                         f"phrase {int((t - PH0) // PHASE_LEN) + 1 if t >= PH0 else 0}  bar {int((t - PH0) // BAR) % 4 + 1 if t >= PH0 else 0}" + ("   SOMETHING IS COMING" if rising else "")],
                    "02  KURAMOTO")


def main():
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    OUT = A.OUT
    tmp = OUT / "_tmp_v2.mp4"
    p = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(tmp)], stdin=subprocess.PIPE)
    k = 0
    for gen in (shot_kuramoto_v2(), A.shot_inhale()):
        for im in gen:
            p.stdin.write(im.tobytes()); k += 1
            if k % 120 == 0:
                print("frame", k, flush=True)
    p.stdin.close(); p.wait()
    out = OUT / "03_粗剪v2_0-52.mp4"
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(OUT / "01_delay_embedding_10s.mp4"), "-i", str(tmp),
                    "-t", "52", "-i", str(S.MUSIC),
                    "-filter_complex", f"[0:v]scale={W}:{H},fps={FPS}[a];[1:v]fps={FPS}[b];[a][b]concat=n=2:v=1:a=0[v]",
                    "-map", "[v]", "-map", "2:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-c:a", "aac", "-b:a", "256k", "-shortest", str(out)], check=True)
    tmp.unlink()
    print("->", out)


if __name__ == "__main__":
    main()
