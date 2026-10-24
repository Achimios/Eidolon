#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🎬Dune-底图-Seedream 关键帧（MV_Dune，2026-10-24，Fable）。预算 30 张；ledger MV_Dune/生成/ledger.jsonl。
原创角色（测绘员，不是 HTE 的女孩）：裹风衣、脸几乎看不见、一只手和一个测距仪。不用 Dune 的任何设计。
用法：python tools/mv_dune/plates.py list | gen --ids P01,P02 [--refs 图] | layers --ids P01
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
MV = ROOT.parents[1] / "Projekt_Visual" / "MV_Dune"
GEN = MV / "生成"
LEDGER = GEN / "ledger.jsonl"
CAP = 30

CHARACTER = ("CHARACTER: a surveyor seen mostly from behind or by her hands, wrapped in a long weathered dust-colored canvas coat with a high "
             "collar and a hood pulled low so the face is almost never visible, heavy laced boots, a worn leather strap across the back, "
             "holding a small matte-black handheld rangefinder with a blank dark display (no text, no digits). Stylized CG, not photoreal.")
STYLE = ("STYLE: stylized CG illustration, matte shading, after-sunset light with a single warm horizon glow, very dark exposure, "
         "muted palette (ochre sand, slate blue night), film grain, 16:9 cinematic still, no text, no logos, no vehicles, no creatures, "
         "no symbols, nothing resembling any existing film. Enormous scale contrast: person tiny, dunes vast.")

PLATES = {
 "P01": ("SCENE: ultra-wide shot of endless rippled sand dunes just after sunset, a single tiny hooded figure standing on a crest in the far middle distance, long shadow, no footprints yet.", "A-end B"),
 "P02": ("SCENE: close shot from behind her shoulder: a gloved hand holding a small matte-black rangefinder with a blank display, pointed at a distant dune crest, horizon glow.", "B 读数"),
 "P03": ("SCENE: macro close-up of a single boot print in fine sand, sharp edges, wind beginning to blow grains into it, low raking light.", "B/C 靴印"),
 "P04": ("SCENE: a large hand-drawn paper map of dunes (contour lines only, no text) lifted by the wind off the sand, half folded, dusk light.", "B 地图"),
 "P05": ("SCENE: top-down shot of gloved hands tearing a paper map in half, torn edge mid-air, sand below, cold blue night light.", "D 撕"),
 "P06": ("SCENE: night, deep slate blue, vast dunes under starless sky, a tiny figure with a small warm handheld light far away, dune crests like frozen waves.", "C 夜"),
 "P07": ("SCENE: extreme close-up of wind ripples on a sand surface at dusk, ripples looking exactly like water ripples, grains lifting off the crests.", "C 风纹"),
 "P08": ("SCENE: wide low-angle shot from behind: the hooded figure small in the foreground, and behind her the dunes rising into one enormous curling wave-like crest, night blue, no water, only sand.", "D 浪"),
 "P09": ("SCENE: side medium shot of the hooded figure bent over a map spread on the sand, holding it down with both hands against the wind, dusk.", "B/C 看图"),
 "P10": ("SCENE: a huge dune face seen from below at dusk, wind pouring sand off the crest like a waterfall, no person.", "A 中段"),
}
V1_IDS = ["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09", "P10"]


def used():
    if not LEDGER.exists():
        return 0
    return sum(json.loads(l)["n"] for l in LEDGER.read_text("utf-8").splitlines() if l.strip())


def log(n, note, files):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"t": datetime.datetime.now().isoformat(timespec="seconds"), "line": "dune", "n": n, "note": note,
                            "files": [str(x) for x in files]}, ensure_ascii=False) + "\n")


def prompt_for(pid):
    scene, _ = PLATES[pid]
    return f"{scene} {CHARACTER if 'no person' not in scene else ''} {STYLE}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["list", "gen", "layers"])
    ap.add_argument("--ids", default=None); ap.add_argument("--refs", default=None); ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    if a.mode == "list":
        print(f"底图 {len(PLATES)}（预算 {CAP}，已用 {used()}）")
        for k, (sc, serves) in PLATES.items():
            print(k, serves, sc[:80]); return
    import ark
    ids = a.ids.split(",") if a.ids else V1_IDS
    if used() + len(ids) > CAP:
        sys.exit(f"额度：已用 {used()}/{CAP}，这次要 {len(ids)}，停。")
    GEN.mkdir(parents=True, exist_ok=True)
    files = []
    if a.mode == "layers":
        for pid in ids:
            src = GEN / f"{pid}.jpg"; out = GEN / f"{pid}_layers.png"
            if out.exists():
                continue
            p_ = ark.image("Decompose this image into depth layers: background (sky and far dunes, fully inpainted), the person as one layer, foreground sand. Keep the image unchanged.",
                           out, mdl="seedream_flash", size="1.5K", refs=[src], layer_decomposition=True, output_format="png")
            files.append(p_); print(p_, flush=True)
        log(len(files), f"layers {','.join(ids)}", files); return
    refs = a.refs.split(",") if a.refs else None
    for pid in ids:
        out = GEN / f"{pid}{a.suffix}.jpg"
        if out.exists():
            print("已存在，跳过：", out); continue
        pr = prompt_for(pid)
        if refs:
            pr = "Keep the same character and coat as in the reference image (图1). " + pr
        p_ = ark.image(pr, out, mdl="seedream_flash", size="1.5K", refs=refs)
        files.append(p_); print(p_, flush=True)
    log(len(files), f"plates {','.join(ids)}{' refs' if refs else ''}", files)


if __name__ == "__main__":
    main()
