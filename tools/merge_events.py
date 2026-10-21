#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<👂感官层-事件合并：把 节拍.json（beat_this）、锚点.json 的乐句格子、动机_*.json 并进 感官.json，规则和现场 rig 只读一个文件。
- events.beats      ← beat_this 的拍（替换掉 sensorium 自己估的，那个只是谱通量自相关）
- events.downbeats  ← beat_this 的重拍
- events.phrases    ← 锚点.json rules.phrase_grid（用户两个切点推出的 6.4 s 格子），每格吸附到 ±40 ms 内最近的重拍；吸不上的保留格子值
- motifs            ← 每个 动机_<名>.json：{名: {template, track, hits}}（用户耳朵指出的声音 → 全曲的每次出现）
- bpm               ← beat_this 的 bpm
用法：python tools/merge_events.py <作品文件夹>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import sensorium  # noqa: E402

SNAP = 0.040


def main():
    mv = Path(sys.argv[1] if len(sys.argv) > 1 else "C:/Users/Administrator/Desktop/Projekt_Visual/MV_Cant_Kill_Us")
    sense = json.loads((mv / "感官.json").read_text("utf-8"))
    bt = json.loads((mv / "节拍.json").read_text("utf-8"))
    grid = next(r for r in json.loads((mv / "锚点.json").read_text("utf-8"))["rules"] if r["name"] == "phrase_grid")
    dur = sense["duration"]
    down = bt["downbeats"]
    g = []
    t = grid["start"]
    while t > grid["period"]:
        t -= grid["period"]                       # 格子也往前铺（13.11 之前的 6.71 / 0.31）
    while t < dur:
        g.append(t); t += grid["period"]
    phrases, snapped = [], 0
    for x in g:
        near = min(down, key=lambda d: abs(d - x))
        if abs(near - x) <= SNAP:
            phrases.append(round(near, 3)); snapped += 1
        else:
            phrases.append(round(x, 3))
    motifs = {}
    for p in sorted(mv.glob("动机_*.json")):
        m = json.loads(p.read_text("utf-8"))
        motifs[m["name"]] = {"template": m["template"], "track": m.get("main_track"), "hits": [h["t"] for h in m["hits"]]}
    sense["bpm"] = bt["bpm"]
    sense["beat_source"] = bt.get("model", "beat_this")
    sense["events"]["beats"] = [round(b, 3) for b in bt["beats"]]
    sense["events"]["downbeats"] = [round(b, 3) for b in down]
    sense["events"]["phrases"] = phrases
    if motifs:
        sense["motifs"] = motifs
    src = sense.pop("source", None)
    sensorium.write(sense, mv / "感官.json", source=src)
    print(f"beats {len(bt['beats'])}  downbeats {len(down)}  phrases {len(phrases)} (snapped {snapped})  motifs {list(motifs)}")


if __name__ == "__main__":
    main()
