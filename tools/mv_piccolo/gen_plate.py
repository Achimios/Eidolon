#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🎬Piccolo-生成-Seedream 底图（环境板）：旧剧场后台 / 一盏钨丝灯 / 没有人。代码只画洞、光和他。
每次花额度记一行 MV_Piccolo/生成/ledger.jsonl（时间、张数、prompt 摘要），这条线封顶 20 张。
用法：.venv/Scripts/python tools/mv_piccolo/gen_plate.py --n 3 [--layers]   # --layers = 对选中的那张做图层拆分
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
PROJ = ROOT.parents[1] / "Projekt_Visual"
GEN = PROJ / "MV_Piccolo" / "生成"
LEDGER = GEN / "ledger.jsonl"
CAP = 20

PROMPT = ("16:9 cinematic still, backstage of an empty 1920s vaudeville theatre at night. A bare brick wall with peeling "
          "paint and old faded posters, wooden floorboards, hanging ropes and a sandbag, thick dust in the air. "
          "A single bare tungsten bulb on a wire casts warm amber light from the lower left, strong falloff into deep darkness, "
          "long soft shadows on the wall. Nobody in the frame. Film photography, Kodak Vision3 500T pushed, heavy grain, "
          "shallow depth of field, slight haze. Muted palette: amber, umber, near-black. No text, no signage, no people, no modern objects.")


def used():
    if not LEDGER.exists():
        return 0
    return sum(json.loads(l)["n"] for l in LEDGER.read_text("utf-8").splitlines() if l.strip())


def log(n, note, files):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"t": datetime.datetime.now().isoformat(timespec="seconds"), "line": "piccolo", "n": n,
                            "note": note, "files": [str(x) for x in files]}, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--layers", default=None, help="对这张本地图做图层拆分（图生图 + layer_decomposition）")
    a = ap.parse_args()
    import ark
    u = used()
    if u + (a.n if not a.layers else 1) > CAP:
        sys.exit(f"额度：已用 {u}/{CAP}，这次要 {a.n}，停。")
    GEN.mkdir(parents=True, exist_ok=True)
    if a.layers:
        src = Path(a.layers)
        out = GEN / f"{src.stem}_layers.png"
        p = ark.image("Decompose this image into depth layers: background wall, mid-ground objects, foreground floor and dust. Keep the image unchanged.",
                      out, mdl="seedream_flash", size="1.5K", refs=[src], layer_decomposition=True, output_format="png")
        log(1, "layer_decomposition", [p])
        print(p); return
    files = []
    for i in range(a.n):
        k = len([x for x in GEN.glob("plate_*.jpg")]) + 1
        out = GEN / f"plate_{k:02d}.jpg"
        p = ark.image(PROMPT, out, mdl="seedream_flash", size="1.5K")
        files.append(p); print(p, flush=True)
    log(len(files), "plate candidates: backstage, bulb, no people", files)


if __name__ == "__main__":
    main()
