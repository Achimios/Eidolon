#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🎬HTE-自查-视差重影：对每个用了拆层底图的镜头，渲首 / 中 / 尾三帧拼成横条图，人眼看有没有物体出现两次。
同时给一个机器信号：把「她」图层区域从远层里挖掉之后，远层在该区域与底图原像素的相关性应该低（高 = 没挖干净）。
用法：python tools/mv_hte/ghost_check.py --t0 4 --t1 50 --out <目录>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import redraw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0", type=float, default=4.0); ap.add_argument("--t1", type=float, default=50.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sc = redraw.Scene(redraw.Song())
    rows = []
    for sh in sc.s.shots:
        if sh["t1"] <= a.t0 or sh["t0"] >= a.t1:
            continue
        pids, _ = redraw.parse_src(sh.get("src") or "")
        if not pids:
            continue
        pl = sc.plates.get(pids[0])
        if not pl["layers"]:
            continue
        ts = [sh["t0"] + 0.05, (sh["t0"] + sh["t1"]) / 2, sh["t1"] - 0.05]
        ims = [sc.draw(t).resize((640, 360)) for t in ts]
        strip = Image.new("RGB", (640 * 3, 360))
        for i, im in enumerate(ims):
            strip.paste(im, (i * 640, 0))
        name = sh["label"].split(" ")[0]
        strip.save(out / f"ghost_{name}.jpg", quality=85)
        # 机器信号：远层挖掉区域 vs 原底图的相关性
        far = pl["layers"][0][0][..., :3]; base = pl["base"]
        fg = np.zeros(base.shape[:2], np.float32)
        for arr, _, _ in pl["layers"][1:]:
            fg = np.maximum(fg, arr[..., 3])
        m = fg > 0.5
        if m.sum() > 100:
            lf = far[m].mean(-1); lb = base[m].mean(-1)
            corr = float(np.corrcoef(lf - lf.mean(), lb - lb.mean())[0, 1]) if lf.std() > 1e-6 else 0.0
        else:
            corr = 0.0
        rows.append((name, pids[0], round(corr, 3)))
        print(name, pids[0], "far/base corr in fg region:", round(corr, 3), flush=True)
    print("done", len(rows))


if __name__ == "__main__":
    main()
