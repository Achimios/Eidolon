#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🖼情绪大图-全曲按段一格：图（Seedream 草图，带参考图）+ 时间 + 一句「她此刻」+ 这一段的感官曲线（真数）。
数据在作品文件夹 `情绪大图/草案.json`（用户 / LLM 改文字和 prompt），本工具只负责生成和排版 —— 改了哪段就删哪段的图重跑。
👤：人是视觉动物，给人看稠密的图，向人要稀疏的决定（哲学 §13）。
用法：python tools/emotion_board.py [--gen] [--mdl seedream_flash] [--only 04,05]
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import ark  # noqa: E402

MV = Path("C:/Users/Administrator/Desktop/Projekt_Visual/MV_Cant_Kill_Us")
BOARD = MV / "情绪大图"
FONT = "C:/Windows/Fonts/msyh.ttc"
MONO = "C:/Windows/Fonts/consola.ttf"
STYLE = "（画面语法：纯黑或近黑背景，冷灰白、低饱和，全画面只有一处红色强调；不要文字、不要水印、不要 UI；电影宽银幕构图，克制。）"
CW, CH = 640, 360          # 每格图
PAD, TXT = 18, 132         # 格间距、文字区高度
BG, FG, DIM = (8, 9, 10), (205, 210, 214), (120, 126, 132)
RED = (232, 44, 52)


def gen_one(item, alias, mdl):
    out = BOARD / f"{item['id']}.jpg"
    if out.exists():
        return out, "已有"
    refs = [str(MV / alias[r]) for r in item.get("refs", [])]
    ark.image(item["prompt"] + STYLE, out, mdl=mdl, size="1920x1080", refs=refs or None, sources=item.get("refs"))
    return out, "新生成"


def spark(d, sense, t0, t1, w, h):
    """这一段的感官曲线：威胁（灰）/ 饥饿（暗）/ 张力（红）—— 都是 0~1 的真值"""
    fps = sense["fps"]; i, j = int(t0 * fps), max(int(t1 * fps), int(t0 * fps) + 2)
    for name, col in (("hunger", (70, 76, 82)), ("threat", (170, 176, 182)), ("tension", RED)):
        v = np.array(sense["channels"][name][i:j])
        if len(v) < 2:
            continue
        k = max(1, len(v) // w)
        v = v[: len(v) // k * k].reshape(-1, k).mean(1)
        xs = np.linspace(0, w - 1, len(v)); ys = h - 1 - np.clip(v, 0, 1) * (h - 2)
        d.line(list(zip(xs.tolist(), ys.tolist())), fill=col, width=1)


def cell(item, sense, f, fm, fs):
    im = Image.new("RGB", (CW, CH + TXT), BG)
    p = BOARD / f"{item['id']}.jpg"
    if p.exists():
        im.paste(Image.open(p).convert("RGB").resize((CW, CH), Image.LANCZOS), (0, 0))
    d = ImageDraw.Draw(im)
    y = CH + 8
    if "t" in item:
        t0, t1 = item["t"]
        d.text((0, y), f"{item['id']}  {int(t0 // 60)}:{t0 % 60:05.2f}–{int(t1 // 60)}:{t1 % 60:05.2f}", font=fm, fill=DIM)
        sp = Image.new("RGB", (200, 26), BG); spark(ImageDraw.Draw(sp), sense, t0, t1, 200, 26); im.paste(sp, (CW - 200, y - 2))
    else:
        d.text((0, y), item["id"], font=fm, fill=DIM)
    d.text((0, y + 22), item["名"], font=f, fill=FG)
    line = item.get("她此刻", "")
    d.text((0, y + 54), line, font=fs, fill=FG)
    if item.get("记忆"):
        d.text((0, y + 82), "◇ " + item["记忆"], font=fs, fill=DIM)
    return im


def board(spec, sense, cols=3):
    f = ImageFont.truetype(FONT, 24); fs = ImageFont.truetype(FONT, 17); fm = ImageFont.truetype(MONO, 16)
    items = spec["段"] + [{**j, "她此刻": "John 的暗示方案 " + j["id"]} for j in spec.get("John_暗示试验", [])]
    rows = -(-len(items) // cols)
    W = cols * CW + (cols + 1) * PAD
    head = 230
    out = Image.new("RGB", (W, head + rows * (CH + TXT + PAD) + PAD), BG)
    d = ImageDraw.Draw(out)
    d.text((PAD, 18), "CAN'T KILL US — 情绪大图 草案", font=ImageFont.truetype(FONT, 34), fill=FG)
    d.text((PAD, 66), "Thymos：她被炸碎，坠入更高维，吸收残响而不失控，顿悟更高的目的，回到 John 身边。每格一句「她此刻」，等你亲耳听过再改。", font=fs, fill=DIM)
    # 全曲时间轴：感官曲线 + 段落边界
    tl_y, tl_h, dur = 110, 90, sense["duration"]
    tl = Image.new("RGB", (W - 2 * PAD, tl_h), (14, 15, 16)); spark(ImageDraw.Draw(tl), sense, 0, dur, W - 2 * PAD, tl_h); out.paste(tl, (PAD, tl_y))
    for s in spec["段"]:
        x = PAD + s["t"][0] / dur * (W - 2 * PAD)
        d.line([(x, tl_y), (x, tl_y + tl_h + 14)], fill=(60, 64, 68))
        d.text((x + 3, tl_y + tl_h + 2), s["id"], font=fm, fill=DIM)
    d.text((W - PAD - 420, tl_y - 22), "灰 = 威胁(起音)  暗 = 饥饿(安静)  红 = 张力(亮度)", font=fs, fill=DIM)
    for k, it in enumerate(items):
        r, c = divmod(k, cols)
        out.paste(cell(it, sense, f, fm, fs), (PAD + c * (CW + PAD), head + PAD + r * (CH + TXT + PAD)))
    p = BOARD / "_情绪大图.jpg"
    out.save(p, quality=90)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", action="store_true")
    ap.add_argument("--mdl", default="seedream_flash")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    spec = json.loads((BOARD / "草案.json").read_text("utf-8"))
    sense = json.loads((MV / "感官.json").read_text("utf-8"))
    if a.gen:
        todo = spec["段"] + spec.get("John_暗示试验", [])
        if a.only:
            keep = set(a.only.split(",")); todo = [t for t in todo if t["id"] in keep]
        with ThreadPoolExecutor(4) as ex:
            for fut in [ex.submit(gen_one, t, spec["refs_alias"], a.mdl) for t in todo]:
                try:
                    out, st = fut.result(); print(st, out.name, flush=True)
                except Exception as e:  # 一张失败不拖累整批（多半是审核或 400）
                    print("失败", e, flush=True)
    print("->", board(spec, sense))


if __name__ == "__main__":
    main()
