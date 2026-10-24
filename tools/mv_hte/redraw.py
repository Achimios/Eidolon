#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🎬HTE-渲染-「代码重绘」引擎 v2（MV_Happier_Than_Ever，2026-10-24，Fable；ReadScreen 第 1 轮后重做）。
v1 的问题：代码层看不见（= AI 插画 + 滤镜，Ken Burns）；每个词都放大 = 模板；小字太淡。
v2（观众只看代码层，底图只透 ~30%）：
  ① 刻版排线（scratchboard）：黑底上用亮线画亮部 —— 四档线密度 = 亮度（交叉密 / 45° / 疏 / 零星）+ 边缘线；线相位随拍慢漂（呼吸）。
  ② 真压缩：她身上的区域做 8×8 DCT，只保留前 k 个系数（JPEG 的机理）。前半 k = 2（一张被压缩的脸）；每个歌词词落下的瞬间 k → 64
     （局部解压 0.5 s 再压回）；后半 k = 64 常驻；最后一句（236 s 起）换成方块越长越大吃掉画面。「礼貌是压缩，真话是解压」直接上画面。
  ③ 2.5D 视差：Seedream layer_decomposition 的图层（背景 / 她 / 前景）按 z 做不同位移，背景放大补缝；没有图层的底图退回裁切窗。
  ④ 字：全片同一种打字机字（Courier）。前半逐字打出（字符在词的真实时长里依次出现）、色带压痕、偶尔打错退格；
     后半同样的小字在底部打，只有每行**一个**承载意义的词（KEY_WORDS，按语义手选）放大 —— 同一封信、同一种字，只是尺寸解压了；"ALONE" 最大、橙色。
  一切都是 t 的纯函数；噪声按帧号播种。
用法同 v1：still / sheet / clip --t0 --t1 --out
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
PROJ = ROOT.parents[1] / "Projekt_Visual"
MV = PROJ / "MV_Happier_Than_Ever"
GEN = MV / "生成"
MUSIC = MV / "音乐_HTE_2883-3178.wav"
CLI = "C:/Users/Administrator/Desktop/Obsidian_V_Synapsis/core/cli.js"

W0, H0 = 1920, 1080
FONT_MONO = "C:/Windows/Fonts/cour.ttf"
FONT_MONO_B = "C:/Windows/Fonts/courbd.ttf"

PAL = {
    "A": [(12, 9, 7), (96, 64, 34), (196, 146, 86), (244, 226, 196)],      # 琥珀
    "G": [(10, 10, 10), (72, 72, 70), (156, 154, 148), (228, 226, 220)],    # 灰
    "B": [(5, 11, 13), (20, 66, 74), (70, 150, 160), (206, 234, 236)],      # 深青
}
ORANGE = (255, 122, 24)
BEAT = 60 / 81.2
# 每行一个承载意义的词（后半才放大；前半全是小字）
KEY_WORDS = {25: "again", 26: "influence", 27: "death", 28: "friends", 29: "relate", 30: "relate", 31: "shitty", 32: "hate",
             33: "internet", 34: "anything", 35: "everything", 36: "sad", 37: "waste", 38: "bad", 39: "time", 40: "empty",
             41: "mother", 42: "kid", 43: "ruined", 44: "misunderstood", 45: "moments", 46: "alone"}


def ease2(u, a=0.6, t1=3 / 24, t2=9 / 24):
    if u <= 0:
        return 0.0
    return a * (1 - math.exp(-u / t1)) + (1 - a) * (1 - math.exp(-u / t2))


# ---------------------------------------------------------------- DCT（8×8，JPEG 机理）
def dct_matrix(n=8):
    D = np.zeros((n, n), np.float32)
    for k in range(n):
        for i in range(n):
            D[k, i] = math.sqrt((1 if k == 0 else 2) / n) * math.cos(math.pi * (2 * i + 1) * k / (2 * n))
    return D


DCT = dct_matrix()
ZIG = np.array([0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
                35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63])


def jpeg_keep(img, k):
    """每个 8×8 块只保留 zigzag 前 k 个 DCT 系数（k=64 = 无损）。img (H,W)，H、W 为 8 的倍数。"""
    if k >= 64:
        return img
    H, W = img.shape
    b = img.reshape(H // 8, 8, W // 8, 8).transpose(0, 2, 1, 3)        # (bh, bw, 8, 8)
    coef = DCT @ b @ DCT.T
    mask = np.zeros(64, np.float32); mask[ZIG[:k]] = 1
    coef = coef * mask.reshape(8, 8)
    rec = DCT.T @ coef @ DCT
    return rec.transpose(0, 2, 1, 3).reshape(H, W)


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
        self.heart = np.array(ch["heartbeat"]); self.threat = np.array(ch["threat"])
        self.onsets = np.array(sense["events"]["onsets"])
        tl = json.loads(subprocess.run(["node", CLI, "timeline", "--json", "--ws", str(MV)], capture_output=True, text=True, encoding="utf-8").stdout)
        self.shots = []
        for pg in tl:
            if pg["page"] == "剪辑":
                continue
            items = json.loads(subprocess.run(["node", CLI, "timeline", pg["page"], "--json", "--ws", str(MV)], capture_output=True, text=True, encoding="utf-8").stdout)["items"]
            self.shots += items
        self.shots.sort(key=lambda s: s["t0"])

    def ch(self, arr, t):
        i = int(t * self.fps)
        return float(arr[min(max(i, 0), len(arr) - 1)])

    def shot(self, t):
        for s in self.shots:
            if s["t0"] <= t < s["t1"]:
                return s
        return None

    def line_index(self, t):
        for i, L in enumerate(self.lines):
            if L["t0"] - 0.1 <= t <= L["t1"] + 1.2:
                return i + 1, L
        return None, None

    def pressure(self, t):
        """最后一句 / 余韵的方块压力（0 = 不用方块）"""
        if 236.0 <= t < 242.24:
            return 0.45 + (t - 236.0) / 6.24 * 0.55
        if 242.49 <= t < 256.0:
            return max(0.0, 1.0 - (t - 242.49) / 13.5)
        return 0.0

    def keep_k(self, t):
        """DCT 保留系数数 k(t)：前半 2，词落下瞬间解压；过门 2 → 64；后半 64"""
        if t >= 147.4:
            return 64
        if t >= 133.0:
            return int(2 + 62 * (t - 133.0) / 14.4)
        k = 2
        for w in self.words:
            if 0 <= t - w["t0"] < 0.5:
                u = t - w["t0"]
                k = max(k, int(2 + 62 * (1 - u / 0.5) ** 2))
        return k

    def palette(self, t):
        if t < 124.0:
            return "A"
        if t < 147.4:
            return "G"
        return "B"


class Plates:
    def __init__(self):
        self.cache = {}

    def path(self, pid):
        cands = sorted(GEN.glob(f"{pid}_v*.jpg"), key=lambda q: int(q.stem.rsplit("_v", 1)[1]))
        return cands[-1] if cands else GEN / f"{pid}.jpg"

    def get(self, pid):
        """→ {'base': (H,W,3) float, 'layers': [(rgba, z, name)] 按 z 升序, 'person': 索引或 None}"""
        if pid in self.cache:
            return self.cache[pid]
        p = self.path(pid)
        if not p.exists():
            base = np.full((1152, 2048, 3), 0.15, np.float32)
            self.cache[pid] = {"base": base, "layers": [], "person": None}; return self.cache[pid]
        base = np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255
        layers = []; person = None
        side = p.with_name(p.stem + "_layers.png.json")
        if side.exists():
            meta = json.loads(side.read_text("utf-8"))
            Hb, Wb = base.shape[:2]
            for L in meta.get("layers", []):
                lp = p.with_name(L["file"])
                if not lp.exists():
                    continue
                # 图层 PNG 是按包围盒裁过、再放大的：缩回包围盒大小，贴到整幅透明画布上
                x0, y0, x1, y1 = L["bounding_box"]["absolute"]
                im = Image.open(lp).convert("RGBA").resize((max(1, x1 - x0), max(1, y1 - y0)), Image.BILINEAR)
                canvas = Image.new("RGBA", (Wb, Hb), (0, 0, 0, 0)); canvas.paste(im, (x0, y0))
                arr = np.asarray(canvas).astype(np.float32) / 255
                layers.append((arr, L.get("z_index", 0), (L.get("name") or "") + "  " + (L.get("description") or "")))   # 两个空格分隔名字和描述
            layers.sort(key=lambda x: x[1])
            # 最远层 = 整幅底图，但把所有前景层（她、道具）的区域挖掉、用模糊归一化填补（简易 inpaint），
            # 再把拆层给的背景层盖上去 —— 这样视差一动不会出现两个她 / 两把汤匙（ReadScreen 第 2 轮抓到的重影）
            if layers:
                from PIL import ImageFilter
                fg = np.zeros(base.shape[:2], np.float32)
                for arr, _, _ in layers[1:]:
                    fg = np.maximum(fg, arr[..., 3])
                fg = np.clip((fg - 0.2) / 0.3, 0, 1)
                # 稍微膨胀遮罩（边缘残留）
                fgi = Image.fromarray((fg * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(9))
                fg = np.asarray(fgi).astype(np.float32) / 255
                keep = (1 - fg)[..., None]
                num = Image.fromarray((np.clip(base * keep, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(60))
                den = Image.fromarray((keep[..., 0] * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(60))
                num = np.asarray(num).astype(np.float32) / 255; den = np.asarray(den).astype(np.float32)[..., None] / 255
                fill = num / np.maximum(den, 1e-3)
                far = base * keep + fill * (1 - keep)
                bg = layers[0][0]
                far = far * (1 - bg[..., 3:4]) + bg[..., :3] * bg[..., 3:4]
                full = np.concatenate([np.clip(far, 0, 1), np.ones(base.shape[:2] + (1,), np.float32)], -1)
                layers[0] = (full, layers[0][1], layers[0][2])
            for i, (_, _, nm) in enumerate(layers):
                if i == 0:
                    continue                                    # 背景层的描述里常有「behind the person」，别认错
                if any(w in nm.lower().split("  ")[0] for w in ("person", "woman", "girl", "character", "figure")):   # 只看名字
                    person = i; break
        self.cache[pid] = {"base": base, "layers": layers, "person": person}
        return self.cache[pid]


def parse_src(src):
    if not src or not src.startswith("plate:"):
        return None, ""
    body = src[6:]
    pid, _, how = body.partition("|")
    return pid.split(" "), how


# ---------------------------------------------------------------- 场景
class Scene:
    def __init__(self, song: Song, w=W0, h=H0):
        self.s = song; self.W, self.H = w, h
        self.plates = Plates(); self.fonts = {}
        yy, xx = np.mgrid[0:h, 0:w]
        self.xx, self.yy = xx, yy

    def font(self, path, size):
        k = (path, size)
        if k not in self.fonts:
            self.fonts[k] = ImageFont.truetype(path, max(6, int(size)))
        return self.fonts[k]

    # 相机参数（镜内永远在动）：缩放 z、平移 (px, py) ∈ 0..1
    def cam(self, t, shot):
        u = (t - shot["t0"]) / max(shot["t1"] - shot["t0"], 1e-6)
        seed = abs(hash(shot["id"])) % 1000
        z0 = 1.08 + 0.04 * ((seed % 3) / 2); z1 = z0 + (0.07 if seed % 2 == 0 else -0.06)
        z = z0 + (z1 - z0) * u
        px = 0.5 + 0.07 * math.sin(2 * math.pi * (t / 37.0 + seed / 7)) + (0.07 if seed % 2 else -0.07) * u
        py = 0.5 + 0.04 * math.cos(2 * math.pi * (t / 23.0 + seed / 5))
        return z, px, py

    def crop(self, img, z, px, py, extra=1.0):
        H, W = img.shape[:2]
        cw, ch_ = W / (z * extra), H / (z * extra)
        x0 = min(max(px * W - cw / 2, 0), W - cw); y0 = min(max(py * H - ch_ / 2, 0), H - ch_)
        mode = "RGBA" if img.shape[2] == 4 else "RGB"
        im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8), mode).crop((int(x0), int(y0), int(x0 + cw), int(y0 + ch_))).resize((self.W, self.H), Image.BILINEAR)
        return np.asarray(im).astype(np.float32) / 255

    def compose(self, pl, t, shot):
        """底图 + 视差 → 帧大小的 RGB 和「她」的遮罩（0~1）"""
        z, px, py = self.cam(t, shot)
        ell = np.exp(-(((self.xx - self.W * 0.55) / (self.W * 0.22)) ** 2 + ((self.yy - self.H * 0.5) / (self.H * 0.4)) ** 2)).astype(np.float32)
        if not pl["layers"]:
            self.has_person = False
            return self.crop(pl["base"], z, px, py), ell
        zs = [L[1] for L in pl["layers"]]
        zmin, zmax = min(zs), max(zs)
        out = None; mask = None
        for i, (arr, zi, _) in enumerate(pl["layers"]):
            depth = 0.0 if zmax == zmin else (zi - zmin) / (zmax - zmin)        # 0 = 最远
            shift = (depth - 0.35) * 0.05                                         # 视差：远层随镜头平移少、近层多
            dx = shift * (px - 0.5) * 2 + shift * 0.3
            rgba = self.crop(arr, z, px + dx, py + shift * 0.2, extra=(0.93 if i == 0 else 1.0))   # 背景多放大补缝
            rgb, a = rgba[..., :3], rgba[..., 3]
            if i == 0:
                out = rgb; continue
            out = out * (1 - a[..., None]) + rgb * a[..., None]
            if pl["person"] is not None and i == pl["person"]:
                mask = a
        self.has_person = mask is not None
        return out, (mask if mask is not None else ell)   # 没有「她」图层时用椭圆（压缩用）；橙色检测另见 draw

    # 刻版排线：黑底亮线，密度 = 亮度
    def scratch(self, lum, t):
        ph = (t % (BEAT * 4)) / (BEAT * 4)
        sh = int(ph * 12)                                   # 线相位慢漂（4 拍一周期）
        xx, yy = self.xx, self.yy
        d1 = (xx + yy + sh); d2 = (xx - yy + sh)
        ink = np.zeros_like(lum)
        for thr, sp, dd, gain in ((0.18, 11, d1, 0.55), (0.34, 7, d1, 0.7), (0.52, 4, d1, 0.85), (0.72, 4, d2, 0.9)):
            band = np.clip((lum - thr) / 0.12, 0, 1)
            ink = np.maximum(ink, ((dd % sp) == 0).astype(np.float32) * band * gain)
        gx = np.zeros_like(lum); gy = np.zeros_like(lum)
        gx[:, 1:-1] = lum[:, 2:] - lum[:, :-2]; gy[1:-1, :] = lum[2:, :] - lum[:-2, :]
        e = np.sqrt(gx * gx + gy * gy)
        return np.maximum(ink, np.clip((e - 0.10) / 0.10, 0, 1))

    def draw(self, t):
        s = self.s; W, H = self.W, self.H
        shot = s.shot(t)
        pal_key = s.palette(t); pal = PAL[pal_key]
        ground = np.array(pal[0], np.float32) / 255
        if shot is None or shot["label"].startswith("E04"):
            return Image.new("RGB", (W, H), tuple(pal[0]))
        pids, how = parse_src(shot.get("src") or "")
        pids = pids or ["P01"]
        pid = pids[int((t - shot["t0"]) / BEAT) % len(pids)] if len(pids) > 1 else pids[0]
        pl = self.plates.get(pid)
        img, mask = self.compose(pl, t, shot)
        fade = 1.0
        if shot["label"].startswith("E01"):
            fade = ease2(t - shot["t0"], t1=0.6, t2=1.8)
        elif shot["label"].startswith("E05"):
            fade = ease2(t - shot["t0"], t1=0.4, t2=1.2)

        lum = img[..., 0] * 0.3 + img[..., 1] * 0.59 + img[..., 2] * 0.11
        # ② 压缩：她身上的区域 DCT 只留 k 个系数
        k = s.keep_k(t)
        if k < 64:
            comp = jpeg_keep(lum, k)
            m = np.clip(mask, 0, 1)
            lum = lum * (1 - m) + comp * m
        # 最后一句：方块越长越大（整幅）
        p = s.pressure(t)
        if p > 0:
            sz = int(min(270, 8 * 2 ** (p * 5)))
            if np.any(np.abs(s.onsets - t) < 0.05):
                sz = min(270, sz * 2)
            # 网格随时间慢漂（§16 船：方块阶段也不能有完全静止的帧）
            ox = int((t * 41.0) % sz); oy = int((t * 23.0) % sz)
            padded = np.pad(lum, ((oy, sz), (ox, sz)), mode="edge")
            h2, w2 = padded.shape[0] // sz, padded.shape[1] // sz
            small = padded[:h2 * sz, :w2 * sz].reshape(h2, sz, w2, sz).mean((1, 3))
            big = np.repeat(np.repeat(small, sz, 0), sz, 1)
            lum = big[oy:oy + H, ox:ox + W] * (1.0 + 0.05 * math.sin(2 * math.pi * t * 1.3))   # 块亮度缓慢呼吸（不是闪）
            # 连续缓慢变焦（巨块阶段块边界几乎不动，靠这个保证没有静止帧）
            zz = 1.0 + 0.06 * (0.5 + 0.5 * math.sin(2 * math.pi * t / 7.0))
            cw, chh = int(W / zz), int(H / zz)
            x0 = (W - cw) // 2; y0 = (H - chh) // 2
            lum = np.asarray(Image.fromarray((np.clip(lum[y0:y0 + chh, x0:x0 + cw], 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)).astype(np.float32) / 255

        # ① 刻版：黑底 + 底图透 30%（海报化）+ 亮线；方块阶段只有块
        if p > 0.0:
            ink = np.zeros_like(lum); under = 0.9
        else:
            ink = self.scratch(lum, t); under = 0.30
        lv = np.clip(lum * 3, 0, 3); i = np.clip(np.floor(lv).astype(int), 0, 2); f = (lv - i)[..., None]
        P_ = np.array(pal, np.float32) / 255
        poster = P_[i] * (1 - f) + P_[i + 1] * f
        out = ground * (1 - under) + poster * under
        ink_col = np.array(pal[3], np.float32) / 255
        out = out * (1 - ink[..., None] * 0.92) + ink_col * ink[..., None] * 0.92
        # 橙辫子：唯一的饱和色
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        mx = img.max(-1); sat = (mx - img.min(-1)) / (mx + 1e-6)
        if self.has_person:   # 有「她」图层：只认 alpha 很实的像素里的橙（图层边缘会带进墙和灯光）
            orange = (((r > g * 1.5) & (g > b * 1.6) & (sat > 0.55) & (mx > 0.4)) & (mask > 0.9)).astype(np.float32)[..., None]   # 灯罩 r/g≈1.4 过不了，辫子 ≈2
        else:                 # 没拆层：严格阈值（灯、奖杯的暖色过不了）
            orange = ((r > g * 1.7) & (g > b * 2.0) & (sat > 0.68) & (mx > 0.5)).astype(np.float32)[..., None]
        out = out * (1 - orange) + img * np.array([1.0, 0.9, 0.55]) * orange
        out = np.clip(out * fade, 0, 1)
        im = Image.fromarray((out * 255).astype(np.uint8))
        d = ImageDraw.Draw(im, "RGBA")
        self.lyrics(d, t, pal)
        rng = np.random.default_rng(int(t * 24))
        g_ = rng.normal(0, 2.5, (H, W, 1))
        return Image.fromarray(np.clip(np.asarray(im).astype(np.float32) + g_, 0, 255).astype(np.uint8))

    # 打字机：字符在词的真实时长里依次出现；色带压痕；偶尔打错退格
    def typed(self, d, x, y, w, t, font, col, alpha, seed):
        txt = w["w"]; n = len(txt)
        dur = max(w["t1"] - w["t0"], 0.12)
        typo = (seed % 6 == 0) and n > 3
        shown = ""
        for i, ch_ in enumerate(txt):
            tc = w["t0"] + dur * i / n * 0.8
            if t < tc:
                break
            c = ch_
            if typo and i == 1 and t - tc < 0.18 and ch_.isalpha():
                c = chr(((ord(ch_.lower()) - 97 + 1) % 26) + 97)
            shown += c
        rng = np.random.default_rng(seed)
        cx = x
        for c in shown:
            a = int(alpha * (0.75 + 0.25 * rng.random()))
            dx, dy = rng.uniform(-0.6, 0.6), rng.uniform(-0.8, 0.8)
            d.text((cx + dx, y + dy), c, font=font, fill=(*col, a))
            d.text((cx + dx + 0.7, y + dy + 0.5), c, font=font, fill=(*col, int(a * 0.35)))
            cx += d.textlength(c, font=font)
        return x + d.textlength(txt, font=font)

    def lyrics(self, d, t, pal):
        s = self.s; W, H = self.W, self.H
        cream = tuple(pal[3])
        li, line = s.line_index(t)
        if line is None:
            return
        ws = [w for w in s.words if line["t0"] - 0.05 <= w["t0"] <= line["t1"] + 0.05 and w["t0"] <= t]
        if not ws:
            return
        f = self.font(FONT_MONO, 38)
        x = W * 0.08; y = H * 0.83
        for w in ws:
            x = self.typed(d, x, y, w, t, f, cream, 235, abs(hash(w["w"] + str(w["t0"]))) % 1000)
            x += d.textlength(" ", font=f)
        if (t % BEAT) < BEAT / 2:
            d.rectangle([x, y + 6, x + 16, y + 36], fill=(*cream, 190))
        if t >= 147.4 and li in KEY_WORDS:
            key = KEY_WORDS[li]
            kw = next((w for w in ws if w["w"].lower().strip(".,!?'") == key), None)
            if kw is not None:
                u = (t - kw["t0"]) / (BEAT / 8)
                a = int(255 * min(1.0, max(0.0, u)))
                big = key == "alone"
                size = int(H * 0.5) if big else int(H * 0.22)
                fb = self.font(FONT_MONO_B, size)
                txt = key.upper(); tw = d.textlength(txt, font=fb)
                xk = min(max(W * (0.5 + (0.1 if li % 2 else -0.1)) - tw / 2, W * 0.04), W - tw - W * 0.04)
                yk = H * 0.42 - size * 0.55
                col = ORANGE if big else cream
                d.text((xk + 4, yk + 4), txt, font=fb, fill=(0, 0, 0, 170))
                d.text((xk, yk), txt, font=fb, fill=(*col, a))
            fk = next((w for w in ws if "fucking" in w["w"].lower()), None)
            if fk is not None and s.pressure(t) > 0.75 and t - fk["t0"] < 0.6:
                fb = self.font(FONT_MONO_B, int(H * 0.3))
                d.text((W * 0.08, H * 0.1), "FUCKING", font=fb, fill=(*cream, 230))


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
