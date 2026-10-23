#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<👂感官层-CLI-整首歌 → 感官.json + 一张通道条带图（人看一眼对不对得上段落）。
用法：python tools/sense_song.py [音频/视频] [--out 目录]   默认：Can't Kill Us → Projekt_Visual/MV_Cant_Kill_Us/
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sensorium as S  # noqa: E402
import stills  # noqa: E402

COL = {"heartbeat": (230, 233, 236), "threat": (255, 42, 52), "hunger": (120, 140, 160), "tension": (190, 200, 210), "rhythm": (90, 96, 102)}


def strip(d, path, sections=()):
    W, lane = 2400, 90
    names = list(d["channels"])
    H = lane * len(names) + 60
    im = Image.new("RGB", (W, H)); g = ImageDraw.Draw(im); f = ImageFont.truetype(stills.FONT, 14)
    L = 140; gw = W - L - 20; dur = d["duration"]
    for i, k in enumerate(names):
        v = np.array(d["channels"][k]); y0 = i * lane + 10
        xs = L + np.arange(len(v)) / len(v) * gw
        pts = list(zip(xs[::2], (y0 + lane - 10 - v[::2] * (lane - 20))))
        g.line(pts, fill=COL[k], width=1)
        g.text((10, y0 + lane // 2 - 8), k, font=f, fill=COL[k])
    for t, name in sections:
        x = L + t / dur * gw
        g.line([(x, 0), (x, H - 40)], fill=(60, 64, 70))
        g.text((x + 3, H - 36), name, font=f, fill=(140, 146, 152))
    g.text((10, H - 36), f"bpm {d['bpm']}", font=f, fill=(200, 200, 200))
    im.save(path, quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?", default=str(stills.MUSIC))
    ap.add_argument("--out", default=str(stills.MV))
    a = ap.parse_args()
    x = S.load_audio(a.src)
    d = S.analyze(x)
    out = Path(a.out)
    S.write(d, out / "感官.json", source=Path(a.src).name)
    # 段落标签只属于《Can't Kill Us》；别的歌不画（2026-10-23：新歌会被盖上错的段名）
    sec = [(0, "INTRO"), (10, "BUILD"), (46, "INHALE"), (52, "DROP1"), (110, "BRIDGE"), (128, "DROP2"),
           (154, "BREAKDOWN"), (172, "REBUILD"), (204, "DROP3"), (282, "OUTRO")] if Path(a.src).resolve() == Path(stills.MUSIC).resolve() else []
    strip(d, out / "感官_条带.jpg", sec)
    print(f"bpm {d['bpm']}  onsets {len(d['events']['onsets'])}  beats {len(d['events']['beats'])}  -> {out / '感官.json'}")


if __name__ == "__main__":
    main()
