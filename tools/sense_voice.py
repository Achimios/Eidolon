#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<👂感官层-人声-CLI：作品文件夹的 分轨/vocals.wav → 往 感官.json 里**加** `voice` 节点（不动已有字段）+ 一张 人声_条带.jpg。
用法：python tools/sense_voice.py <作品文件夹> [--vocals 路径] [--method crepe|crepe_raw|yin] [--crepe-model full|tiny] [--dry]
- f0 默认 crepe-full + YIN 细化（对照见 tools/voice_compare.py）；yin = 旧算法（回退）
- 条带图不覆盖：已有 人声_条带.jpg 就出 _v2、_v3 ……
- 帧数 = 已有 channels 的帧数（同 fps、同长度 —— 必须相等）；时间 = 播放时间（vocals.wav 由 separate.py 出，已对齐）
- 第一次加 voice 前把原文件拷成 `感官.json.bak_2026-10-23_voice`；写完重读，除 voice 外逐字段比对旧文件，不等就不落盘
- 交叉校验：人声分轨在门以上的能量有多少落在乐句里（应 ≈ 100%）
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import sensorium as S  # noqa: E402
import voice as V  # noqa: E402

FONT = "C:/Windows/Fonts/consola.ttf"
BAK = "感官.json.bak_2026-10-23_voice"


def coverage(fr_data, phrases):
    """门以上的人声能量里，落在乐句 [start, end) 内的比例；以及乐句帧里有声帧的比例"""
    fr = fr_data["fr"]; db = fr_data["db"]; loud = fr_data["loud"]
    inside = np.zeros(len(db), bool)
    for p in phrases:
        inside[int(round(p["start"] * fr)):int(round(p["end"] * fr))] = True
    pw = 10 ** (db / 10) * (loud > 0)
    e_in = pw[inside].sum() / (pw.sum() + 1e-12)
    v_in = fr_data["voiced"][inside].mean() if inside.any() else 0.0
    return float(e_in), float(v_in)


def strip(x, sr, d, sense, path, title):
    """波形 + 响度 + f0（对数纵轴，标 C 音）+ 有声概率 / 颤音；乐句底色、换气红线、人声段顶条"""
    F = d["_frames"]; fr = F["fr"]
    dur = len(x) / sr
    pps = 20                                     # 每秒 20 像素
    L, Rm = 70, 20
    gw = int(dur * pps); Wd = L + gw + Rm
    lanes = [("wave", 90), ("loudness", 70), ("f0", 220), ("voiced/vib", 70)]
    top = 46; H = top + sum(h for _, h in lanes) + 10 * len(lanes) + 40
    im = Image.new("RGB", (Wd, H), (12, 13, 15)); g = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, 13); fb = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 16)   # 标题有中文路径

    def X(t):
        return L + t * pps
    # 乐句底色 + 人声段顶条
    for s0, s1 in d["sections"]:
        g.rectangle([X(s0), 30, X(s1), 38], fill=(200, 60, 70))
    for p in d["phrases"]:
        g.rectangle([X(p["start"]), top, X(p["end"]), H - 40], fill=(26, 30, 36))
    # 重拍（淡）
    for t in sense["events"].get("downbeats", []):
        g.line([(X(t), top), (X(t), H - 40)], fill=(34, 36, 40))
    y = top
    for name, h in lanes:
        g.text((6, y + h // 2 - 7), name, font=f, fill=(150, 156, 162))
        if name == "wave":
            env = np.abs(x)
            n = gw
            idx = (np.arange(n + 1) / pps * sr).astype(int)
            mx = np.array([env[idx[i]:max(idx[i] + 1, idx[i + 1])].max() for i in range(n)])
            mx = mx / (mx.max() + 1e-9)
            for i in range(n):
                g.line([(L + i, y + h / 2 - mx[i] * h / 2), (L + i, y + h / 2 + mx[i] * h / 2)], fill=(120, 130, 140))
        elif name == "loudness":
            v = F["loud"]; tt = np.arange(len(v)) / fr
            k = np.clip((np.arange(gw) / pps * fr).astype(int), 0, len(v) - 1)
            g.line(list(zip(L + np.arange(gw), y + h - v[k] * h)), fill=(230, 233, 236), width=1)
            del tt
        elif name == "f0":
            lo, hi = np.log2(80), np.log2(1100)

            def Y(hz):
                return y + h - (np.log2(hz) - lo) / (hi - lo) * h
            for midi, lab in ((48, "C3"), (60, "C4"), (72, "C5"), (84, "C6")):
                hz = 440 * 2 ** ((midi - 69) / 12)
                g.line([(L, Y(hz)), (L + gw, Y(hz))], fill=(40, 44, 50))
                g.text((L - 26, Y(hz) - 7), lab, font=f, fill=(110, 116, 122))
            f0 = F["f0"]; vib = F["vib"]
            ks = np.arange(len(f0)); tx = L + ks / fr * pps
            ok = np.isfinite(f0)
            for k in ks[ok]:
                c = (255, 42 + int(150 * (1 - vib[k])), 52 + int(150 * (1 - vib[k]))) if vib[k] > 0 else (235, 210, 120)
                g.point((tx[k], Y(f0[k])), fill=c)
        else:
            v = F["pv"]; k = np.clip((np.arange(gw) / pps * fr).astype(int), 0, len(v) - 1)
            g.line(list(zip(L + np.arange(gw), y + h - v[k] * h)), fill=(90, 140, 200), width=1)
            vv = F["vib"][k]
            g.line(list(zip(L + np.arange(gw), y + h - vv * h)), fill=(255, 42, 52), width=1)
        g.line([(L, y + h + 4), (L + gw, y + h + 4)], fill=(30, 32, 36))
        y += h + 10
    for b0, b1 in d["breaths"]:
        xm = X((b0 + b1) / 2)
        g.line([(xm, top), (xm, H - 40)], fill=(255, 42, 52))
    for t in range(0, int(dur) + 1, 5):
        g.line([(X(t), H - 40), (X(t), H - 34)], fill=(120, 120, 120))
        if t % 10 == 0:
            g.text((X(t) + 2, H - 32), f"{t // 60}:{t % 60:02d}", font=f, fill=(150, 150, 150))
    g.text((6, 6), title, font=fb, fill=(230, 233, 236))
    g.text((L + 520, 8), "red bar = voice section | dark band = phrase | red line = breath | f0: yellow = steady, red = vibrato | blue = voiced prob, red = vibrato",
           font=f, fill=(140, 146, 152))
    im.save(path, quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mv")
    ap.add_argument("--vocals", default=None)
    ap.add_argument("--dry", action="store_true", help="只算 + 出图，不写 感官.json")
    ap.add_argument("--method", default="crepe", choices=["crepe", "crepe_raw", "yin"])
    ap.add_argument("--crepe-model", default="full", choices=["full", "tiny"])
    a = ap.parse_args()
    mv = Path(a.mv)
    sp = mv / "感官.json"
    old_text = sp.read_text("utf-8")
    sense = json.loads(old_text)
    voc = Path(a.vocals) if a.vocals else mv / "分轨" / "vocals.wav"
    n_out = len(sense["channels"]["heartbeat"])
    x = S.decode(voc, sr=V.SR, fmt="f32le").astype(np.float64)
    d = V.analyze_voice(x, fps=sense["fps"], n_out=n_out, f0_method=a.method, crepe_model=a.crepe_model)
    F = d["_frames"]
    e_in, v_in = coverage(F, d["phrases"])
    try:
        rel = voc.resolve().relative_to(mv.resolve()).as_posix()
    except ValueError:
        rel = voc.name
    node = {k: v for k, v in d.items() if not k.startswith("_")}
    node["source"] = rel
    nvo = sum(z is not None for z in node["f0"])
    print(f"voice: {len(node['phrases'])} phrases, {len(node['breaths'])} breaths, sections {node['sections']}")
    print(f"  voiced frames {nvo}/{n_out} ({nvo / n_out * 100:.1f}%)  energy-in-phrases {e_in * 100:.1f}%  voiced-within-phrases {v_in * 100:.1f}%")
    ff = np.array([z for z in node["f0"] if z is not None])
    if len(ff):
        print(f"  f0 p5/p50/p95 = {np.percentile(ff, 5):.0f}/{np.median(ff):.0f}/{np.percentile(ff, 95):.0f} Hz")
    sp_img = mv / "人声_条带.jpg"
    k = 2
    while sp_img.exists():                       # 不覆盖用户可能看过的图
        sp_img = mv / f"人声_条带_v{k}.jpg"; k += 1
    strip(x, V.SR, d, sense, sp_img, f"{mv.name}  voice  ({rel})  f0: {d['method'].split('(')[0]}")
    print(f"  method {d['method']}  -> {sp_img.name}")
    if a.dry:
        return
    if "voice" not in sense and not (mv / BAK).exists():
        shutil.copy2(sp, mv / BAK)
    new = dict(sense); new["voice"] = node
    src = new.pop("source", None)
    tmp = mv / "感官.json.tmp"
    S.write(new, tmp, source=src)
    back = json.loads(tmp.read_text("utf-8"))
    vb = back.pop("voice")
    old = dict(sense); old.pop("voice", None)
    assert back == old, "除 voice 外有字段变了 —— 不落盘"
    assert list(back) == list(old), "字段顺序变了"
    for k in ("loudness", "voiced", "f0", "vibrato", "vibrato_cents", "vibrato_hz"):
        assert len(vb[k]) == n_out, k
    os.replace(tmp, sp)
    print(f"  -> {sp}（其余字段逐个比对一致；备份 {BAK}）")


if __name__ == "__main__":
    main()
