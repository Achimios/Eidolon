#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🎬HTE-渲染-「代码重绘」引擎（MV_Happier_Than_Ever，2026-10-24，Fable）。
两层：底层 = Seedream 静帧（生成/Pxx.jpg）；表层 = 代码整个重绘 —— 观众只看到代码层。
重绘 = 三件事，全部是歌曲时间 t 的纯函数（无状态，噪声按帧号播种）：
  ① 方块（像素溶解，学 _参考_Short #1：方块尺寸 = 溶解进度，不是透明度）：底图按 s×s 的格子取均值 → 调色板量化
     （琥珀 / 青两套单色阶，只有橙辫子保留原色 = 唯一饱和色）。s 由「压力」p(t) 决定：前半 s=4 且只有少数格子 1–2 帧闪成 16 px；
     后半 s 随 p 长大（8 → 64 → 1/4 画幅），长满即硬切。
  ② 排线 / 等高线：底图亮度的 Sobel 边缘 → 细线（奶白），呼吸（线宽随拍点相位微变）；暗部加 45° 排线，密度 = 暗度。
  ③ 歌词字形：按 歌词_v4.json 的词时间出现（不吸附拍），入场 ≤ 1/8 拍（81 BPM → 0.093 s），只用 score ≥ 0.5 且不在人声段外的词；
     前半小字（打字机，画面下 1/5，左对齐，逐词累积成一行）；后半大字（占画高 1/4–1/3）；"alone" 全片最大（画高 1/2）。
  相机：每镜在底图上的裁切窗（缩放 + 平移 + 视差：若有 layer PNG 则前后层位移不同），镜内永远在动。
  分镜 = Synapsis 时间轴（node cli.js timeline --json），镜头节点的 src=plate:Pxx|裁法。
用法：
  python tools/mv_hte/redraw.py still --t 12.0 --out x.jpg
  python tools/mv_hte/redraw.py clip --t0 4.195 --t1 50 --out 样片/01_压着_0-50_v1.mp4
  python tools/mv_hte/redraw.py clip --t0 220 --t1 250 --out 样片/02_爆发_220-250_v1.mp4
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
PROJ = ROOT.parents[1] / "Projekt_Visual"
MV = PROJ / "MV_Happier_Than_Ever"
GEN = MV / "生成"
MUSIC = MV / "音乐_HTE_2883-3178.wav"
CLI = "C:/Users/Administrator/Desktop/Obsidian_V_Synapsis/core/cli.js"

W0, H0 = 1920, 1080
FONT_SERIF = "C:/Windows/Fonts/georgia.ttf"
FONT_SERIF_B = "C:/Windows/Fonts/georgiab.ttf"
FONT_MONO = "C:/Windows/Fonts/cour.ttf"
FONT_MONO_B = "C:/Windows/Fonts/courbd.ttf"

# 调色板：单色阶（黑 → 中 → 亮），橙保留
PAL = {
    "A": [(14, 11, 9), (92, 62, 34), (190, 140, 82), (240, 222, 190)],      # 琥珀
    "G": [(12, 12, 12), (70, 70, 68), (150, 148, 142), (226, 224, 218)],    # 灰（过门）
    "B": [(6, 12, 14), (18, 62, 70), (62, 140, 150), (200, 230, 232)],      # 深青
}
ORANGE = (255, 120, 20)
BEAT = 60 / 81.2


def ease2(u, a=0.6, t1=3 / 24, t2=9 / 24):
    if u <= 0:
        return 0.0
    return a * (1 - math.exp(-u / t1)) + (1 - a) * (1 - math.exp(-u / t2))


# ---------------------------------------------------------------- 数据
class Song:
    def __init__(self):
        ly = json.loads((MV / "歌词_v4.json").read_text("utf-8"))
        self.words = [w for w in ly["words"] if w["score"] >= 0.5 and not w.get("outside_voice")]
        self.lines = ly["lines"]
        sense = json.loads((MV / "感官.json").read_text("utf-8"))
        self.fps = sense["fps"]
        self.beats = np.array(sense["events"]["beats"])[::2]
        ch = sense["channels"]
        self.heart = np.array(ch["heartbeat"]); self.threat = np.array(ch["threat"]); self.tension = np.array(ch["tension"])
        self.onsets = np.array(sense["events"]["onsets"])
        # 分镜
        tl = json.loads(subprocess.run(["node", CLI, "timeline", "--json", "--ws", str(MV)], capture_output=True, text=True, encoding="utf-8").stdout)
        self.shots = []
        for pg in tl:
            if pg["page"] == "剪辑":
                continue
            items = json.loads(subprocess.run(["node", CLI, "timeline", pg["page"], "--json", "--ws", str(MV)], capture_output=True, text=True, encoding="utf-8").stdout)["items"]
            for it in items:
                self.shots.append(it)
        self.shots.sort(key=lambda s: s["t0"])

    def ch(self, arr, t):
        i = int(t * self.fps)
        return float(arr[min(max(i, 0), len(arr) - 1)])

    def shot(self, t):
        for s in self.shots:
            if s["t0"] <= t < s["t1"]:
                return s
        return None

    def pressure(self, t):
        """压力 p(t) 0~1：方块尺寸的来源。前半 ≈ 0.05–0.15（只闪）；过门 0.3 → 0.6；后半 = 0.4 + 0.6·heartbeat 平滑；最后一句到 1。"""
        if t < 133.0:
            return 0.06 + 0.08 * self.ch(self.threat, t)
        if t < 147.4:
            return 0.3 + 0.4 * (t - 133.0) / 14.4
        h = np.mean([self.ch(self.heart, t - k * 0.1) for k in range(8)])
        base = 0.22 + 0.25 * h                      # 6 → 16 px：画面还读得出，方块只在起音时翻倍
        if 236.0 <= t < 242.24:
            base = max(base, 0.45 + (t - 236.0) / 6.24 * 0.55)   # 最后一句：冲到满幅
        if 242.49 <= t < 256.0:                      # E05：从满幅方块里吐出空房间
            base = max(0.22, 1.0 - (t - 242.49) / 13.5 * 0.78)
        return float(min(1.0, base))

    def palette(self, t):
        if t < 124.0:
            return "A"
        if t < 147.4:
            return "G"
        return "B"


# ---------------------------------------------------------------- 底图
class Plates:
    def __init__(self):
        self.cache = {}

    def get(self, pid):
        if pid not in self.cache:
            # 取最高版本：Pxx_v3.jpg > Pxx_v2.jpg > Pxx.jpg（重出不覆盖，按版本另起名）
            cands = sorted(GEN.glob(f"{pid}_v*.jpg"), key=lambda q: int(q.stem.rsplit("_v", 1)[1]))
            p = cands[-1] if cands else GEN / f"{pid}.jpg"
            if not p.exists():
                p = GEN / "_missing.jpg"
                if not p.exists():
                    Image.new("RGB", (2048, 1152), (40, 40, 40)).save(p)
            im = Image.open(p).convert("RGB")
            self.cache[pid] = np.asarray(im).astype(np.float32) / 255
        return self.cache[pid]


def parse_src(src):
    """'plate:P01|裁法' → ('P01', 裁法)；多张时取第一张（快切镜头由 t 选）"""
    if not src or not src.startswith("plate:"):
        return None, ""
    body = src[6:]
    pid, _, how = body.partition("|")
    return pid.split(" "), how


# ---------------------------------------------------------------- 重绘
def quantize(img, pal, levels=None):
    """单色阶量化：亮度 → 4 级调色板插值（连续，不是硬阶）；橙辫子保留"""
    lum = img[..., 0] * 0.3 + img[..., 1] * 0.59 + img[..., 2] * 0.11
    p = np.array(pal, np.float32) / 255
    # 5 级硬阶（海报化）+ 轻微抖动（按亮度的小数部分做有序抖动，像版画）
    x = np.clip(lum, 0, 1) * 3
    i = np.clip(np.floor(x).astype(int), 0, 2); f = (x - i)
    H_, W_ = lum.shape
    yy, xx = np.mgrid[0:H_, 0:W_]
    bayer = ((xx % 2) * 2 + (yy % 2)) / 4.0 + 0.125
    step = (f > bayer).astype(int)
    i2 = np.clip(i + step, 0, 3)
    out = p[i2]
    # 橙：色相在橙区且饱和
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    mx = img.max(-1); mn = img.min(-1); sat = (mx - mn) / (mx + 1e-6)
    orange = (r > g * 1.25) & (g > b * 1.3) & (sat > 0.45) & (mx > 0.35)
    out[orange] = img[orange] * np.array([1.0, 0.92, 0.6])
    return out, lum


def blocks(img, s):
    """s×s 均值方块（s 像素）"""
    if s <= 1:
        return img
    H, W = img.shape[:2]
    h2, w2 = H // s, W // s
    small = img[:h2 * s, :w2 * s].reshape(h2, s, w2, s, 3).mean((1, 3))
    big = np.repeat(np.repeat(small, s, 0), s, 1)
    return np.pad(big, ((0, H - big.shape[0]), (0, W - big.shape[1]), (0, 0)), mode="edge")


def edges(lum):
    gx = np.zeros_like(lum); gy = np.zeros_like(lum)
    gx[:, 1:-1] = lum[:, 2:] - lum[:, :-2]
    gy[1:-1, :] = lum[2:, :] - lum[:-2, :]
    return np.sqrt(gx * gx + gy * gy)


class Scene:
    def __init__(self, song: Song, w=W0, h=H0):
        self.s = song; self.W, self.H = w, h
        self.plates = Plates()
        self.fonts = {}

    def font(self, path, size):
        k = (path, size)
        if k not in self.fonts:
            self.fonts[k] = ImageFont.truetype(path, max(6, int(size)))
        return self.fonts[k]

    def camera_crop(self, img, t, shot):
        """镜内缓慢缩放 + 平移（裁切窗），返回 (H, W, 3) 的帧大小底图"""
        H, W = img.shape[:2]
        u = (t - shot["t0"]) / max(shot["t1"] - shot["t0"], 1e-6)
        seed = abs(hash(shot["id"])) % 1000
        z0 = 1.06 + 0.04 * ((seed % 3) / 2)
        z1 = z0 + (0.06 if seed % 2 == 0 else -0.05)
        z = z0 + (z1 - z0) * u
        cw, ch_ = W / z, H / z
        px = 0.5 + 0.08 * math.sin(2 * math.pi * (t / 37.0 + seed / 7)) + (0.06 if seed % 2 else -0.06) * u
        py = 0.5 + 0.05 * math.cos(2 * math.pi * (t / 23.0 + seed / 5))
        x0 = int(min(max(px * W - cw / 2, 0), W - cw)); y0 = int(min(max(py * H - ch_ / 2, 0), H - ch_))
        crop = img[y0:y0 + int(ch_), x0:x0 + int(cw)]
        im = Image.fromarray((crop * 255).astype(np.uint8)).resize((self.W, self.H), Image.BILINEAR)
        return np.asarray(im).astype(np.float32) / 255

    def draw(self, t):
        s = self.s
        W, H = self.W, self.H
        shot = s.shot(t)
        pal_key = s.palette(t); pal = PAL[pal_key]
        p = s.pressure(t)
        if shot is None or shot["label"].startswith("E04"):
            return Image.new("RGB", (W, H), tuple(pal[0]))
        pids, how = parse_src(shot.get("src") or "")
        if pids is None:
            pids = ["P01"]
        # 快切镜头：按拍选底图
        if len(pids) > 1:
            k = int((t - shot["t0"]) / BEAT) % len(pids)
            pid = pids[k]
        else:
            pid = pids[0]
        base = self.plates.get(pid)
        img = self.camera_crop(base, t, shot)
        # 入场：从黑长出（E01 / 从黑帧回来）
        fade = 1.0
        if shot["label"].startswith("E01"):
            fade = ease2(t - shot["t0"], t1=0.6, t2=1.8)
        elif shot["label"].startswith("E05"):
            fade = ease2(t - shot["t0"], t1=0.4, t2=1.2)
        q, lum = quantize(img, pal)

        # ① 方块：尺寸 = 压力
        if p < 0.2:
            sz = 6
        else:
            sz = int(4 * 2 ** ((p - 0.2) / 0.8 * 6))     # 0.2 → 4px, 1.0 → 256px
            sz = min(sz, 270)
        bl = blocks(q, sz)
        # 前半：少数格子 1–2 帧闪成 16 px（信号 = 起音）
        frame = int(t * 24)
        near_onset = np.any(np.abs(s.onsets - t) < 0.05)
        if p < 0.2 and near_onset:
            rng = np.random.default_rng(frame // 2)
            big = blocks(q, 24)
            n = 3 + int(12 * s.ch(s.threat, t))
            ys = rng.integers(0, H // 24, n); xs = rng.integers(0, W // 24, n)
            for y, x in zip(ys, xs):
                bl[y * 24:(y + 1) * 24, x * 24:(x + 1) * 24] = big[y * 24:(y + 1) * 24, x * 24:(x + 1) * 24]
        # 后半：方块按起音「长一级又缩回」
        if p >= 0.2 and near_onset:
            bl = blocks(q, min(270, sz * 2))
        out = bl.copy()

        # ② 边缘线 + 暗部排线（只在方块 ≤ 16 px 时画；方块大了就只剩方块）
        if sz <= 16:
            e = edges(lum)
            ph = (t % BEAT) / BEAT
            thr = 0.18 + 0.04 * math.sin(2 * math.pi * ph)
            line = np.clip((e - thr) / 0.10, 0, 1)[..., None]
            cream = np.array(pal[3], np.float32) / 255
            out = out * (1 - 0.95 * line) + cream * 0.95 * line
            # 排线：两组 —— 暗部密（间隔 5）、中间调疏（间隔 10），45°，随拍呼吸
            yy, xx = np.mgrid[0:H, 0:W]
            sh_ = int(ph * 5)
            h1 = (((xx + yy) + sh_) % 5 == 0).astype(np.float32)
            h2 = (((xx + yy) + sh_) % 10 == 0).astype(np.float32)
            dark = np.clip((0.28 - lum) / 0.28, 0, 1)
            mid = np.clip(1 - np.abs(lum - 0.42) / 0.18, 0, 1)
            hm = (h1 * dark * 0.6 + h2 * mid * 0.45)[..., None]
            out = out * (1 - hm) + (np.array(pal[1], np.float32) / 255) * hm

        out = np.clip(out * fade, 0, 1)
        im = Image.fromarray((out * 255).astype(np.uint8))
        d = ImageDraw.Draw(im, "RGBA")

        # ③ 歌词
        self.lyrics(d, t, pal_key, pal, p)

        # 颗粒
        rng = np.random.default_rng(frame)
        g = rng.normal(0, 3.0, (H, W, 1))
        im = Image.fromarray(np.clip(np.asarray(im).astype(np.float32) + g, 0, 255).astype(np.uint8))
        return im

    def lyrics(self, d, t, pal_key, pal, p):
        s = self.s
        W, H = self.W, self.H
        cream = tuple(pal[3])
        # 当前行
        line = next((L for L in s.lines if L["t0"] - 0.1 <= t <= L["t1"] + 1.2), None)
        if line is None:
            return
        ws = [w for w in s.words if line["t0"] - 0.05 <= w["t0"] <= line["t1"] + 0.05 and w["t0"] <= t]
        if not ws:
            return
        big = t >= 147.4
        if not big:
            # 前半：打字机小字，画面下 1/5，左对齐，逐词累积
            f = self.font(FONT_MONO, 30)
            x = int(W * 0.08); y = int(H * 0.84)
            for w in ws:
                u = (t - w["t0"]) / (BEAT / 8)
                a = int(255 * min(1.0, max(0.0, u)))
                d.text((x, y), w["w"], font=f, fill=(*cream, a))
                x += d.textlength(w["w"] + " ", font=f)
            # 光标：拍点闪
            if (t % BEAT) < BEAT / 2:
                d.rectangle([x, y + 4, x + 14, y + 30], fill=(*cream, 200))
        else:
            # 后半：当前词大字（最多两个词：上一个词退成一半大小），整行小字在底部；
            # "alone" 全片最大（橙）；"fucking" 只在方块最大时（p > 0.75）才是大字
            last = ws[-1]
            u = (t - last["t0"]) / (BEAT / 8)
            a = int(255 * min(1.0, max(0.0, u)))
            cur = last["w"].upper().strip(".,!?")
            size = int(H * 0.12 + H * 0.14 * p)
            if cur == "ALONE":
                size = int(H * 0.5)
            elif "FUCKING" in cur:
                size = int(H * 0.3) if p > 0.75 else int(H * 0.10)
            f = self.font(FONT_SERIF_B, size)
            tw = d.textlength(cur, font=f)
            # 位置：按词序左右交替（不总在正中），y 随压力微升
            k = len(ws)
            x = W * (0.5 + (0.12 if k % 2 else -0.12)) - tw / 2
            x = min(max(x, W * 0.04), W - tw - W * 0.04)
            y = H * 0.5 - size * 0.6 - H * 0.08 * p
            col = ORANGE if cur == "ALONE" else cream
            d.text((x + 3, y + 3), cur, font=f, fill=(0, 0, 0, 170))
            d.text((x, y), cur, font=f, fill=(*col, a))
            if len(ws) >= 2 and cur != "ALONE":
                prev = ws[-2]["w"].upper().strip(".,!?")
                f2 = self.font(FONT_SERIF_B, int(size * 0.45))
                d.text((x - d.textlength(prev, font=f2) - 24, y + size * 0.55), prev, font=f2, fill=(*cream, 120))
            # 整行小字
            fs = self.font(FONT_MONO, 26)
            full = " ".join(w["w"] for w in ws)
            d.text((W * 0.08, H * 0.88), full, font=fs, fill=(*cream, 150))


def contact_sheet(frames, out, cols=4):
    w, h = frames[0][0].size
    tw, th = w // 3, h // 3
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (tw * cols, th * rows), (10, 10, 10))
    f = ImageFont.truetype(FONT_MONO_B, 22)
    for i, (im, lab) in enumerate(frames):
        im = im.resize((tw, th), Image.LANCZOS)
        ImageDraw.Draw(im).text((8, 6), lab, fill=(255, 120, 20), font=f)
        sheet.paste(im, ((i % cols) * tw, (i // cols) * th))
    sheet.save(out, quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["still", "clip", "sheet"])
    ap.add_argument("--t", type=float, default=12.0)
    ap.add_argument("--t0", type=float, default=4.195)
    ap.add_argument("--t1", type=float, default=50.0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if not out.is_absolute():
        out = MV / out
    if out.exists():
        sys.exit(f"已存在，不覆盖：{out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    sc = Scene(Song())
    if a.mode == "still":
        sc.draw(a.t).save(out, quality=92); print(out); return
    if a.mode == "sheet":
        ts = np.linspace(a.t0, a.t1, 12, endpoint=False)
        contact_sheet([(sc.draw(float(t)), f"t={t:.2f}") for t in ts], out); print(out); return
    import imageio_ffmpeg, time
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    n = int(round((a.t1 - a.t0) * a.fps))
    cmd = [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W0}x{H0}", "-r", str(a.fps), "-i", "-",
           "-ss", f"{a.t0:.3f}", "-t", f"{a.t1 - a.t0:.3f}", "-i", str(MUSIC),
           "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-shortest", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    sheet_idx = set(np.linspace(0, n - 1, 12).round().astype(int).tolist()); sheet = []; t_start = time.time()
    for i in range(n):
        t = a.t0 + i / a.fps
        im = sc.draw(t)
        proc.stdin.write(im.tobytes())
        if i in sheet_idx:
            sheet.append((im, f"t={t:.2f}"))
        if i % 96 == 0:
            el = time.time() - t_start
            print(f"{i}/{n}  {el:.0f}s  eta {el / max(i, 1) * (n - i):.0f}s", flush=True)
    proc.stdin.close(); proc.wait()
    contact_sheet(sheet, out.with_name(out.stem + "_接触表.jpg"))
    print(out)


if __name__ == "__main__":
    main()
