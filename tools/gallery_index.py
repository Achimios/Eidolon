#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<📇样片总览-所有样片 / 生成物一张表（👤2026-10-23：「文件分散在各处可能还有漏了的」）。
  python tools/gallery_index.py        → Projekt_Visual/_样片总览.md（每次全量重写；只读文件，不动任何样片）
按文件夹分组，同一系列（去掉 _vN / 配色后缀后同名）只把最新版本加粗，旧版本折叠成一行；有接触表就给链接。
"""
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path("C:/Users/Administrator/Desktop/Projekt_Visual")
OUT = ROOT / "_样片总览.md"
SCAN = ["MV_Cant_Kill_Us/样片", "MV_Cant_Kill_Us/生成", "MV_Cant_Kill_Us/素材"]
VER = re.compile(r"_v(\d+)([a-z]?)")
TAIL = re.compile(r"(_v\d+[a-z]?)|(_(bwr|acid|黑白红|酸性|酸绿|冷灰))", re.I)


def series_key(stem):
    return TAIL.sub("", stem)


def version_of(stem):
    m = VER.findall(stem)
    return (int(m[-1][0]), m[-1][1]) if m else (0, "")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    L = [f"# 样片总览（自动生成 {time.strftime('%Y-%m-%d %H:%M')}，`python 4_Eidolon/tools/gallery_index.py` 重出）", "",
         "> 每组里**加粗** = 最新版本；接触表在括号里。只列视频，按文件夹 → 系列。", ""]
    total = 0
    for rel in SCAN:
        base = ROOT / rel
        if not base.exists():
            continue
        vids = sorted(p for p in base.rglob("*.mp4") if p.is_file())
        if not vids:
            continue
        L.append(f"## {rel}（{len(vids)} 个）")
        by_dir = defaultdict(list)
        for p in vids:
            by_dir[p.parent].append(p)
        for d in sorted(by_dir):
            sub = d.relative_to(base).as_posix()
            if sub != ".":
                L.append(f"### {sub}")
            groups = defaultdict(list)
            for p in by_dir[d]:
                groups[series_key(p.stem)].append(p)
            for key in sorted(groups, key=lambda k: -max(q.stat().st_mtime for q in groups[k])):
                ps = sorted(groups[key], key=lambda q: (version_of(q.stem), q.stat().st_mtime))
                top = max(version_of(q.stem) for q in ps)
                newest = [q for q in ps if version_of(q.stem) == top]
                for q in newest:
                    cs = q.with_name(q.stem + "_接触表.jpg")
                    link = f"[{q.name}](<{q.relative_to(ROOT).as_posix()}>)"
                    extra = f"（[接触表](<{cs.relative_to(ROOT).as_posix()}>)）" if cs.exists() else ""
                    L.append(f"- **{link}**{extra} · {time.strftime('%m-%d %H:%M', time.localtime(q.stat().st_mtime))}")
                old = [q for q in ps if q not in newest]
                if old:
                    L.append(f"  - 旧版 {len(old)} 个：" + "、".join(q.name for q in old))
                total += len(ps)
        L.append("")
    OUT.write_text("\n".join(L) + "\n", "utf-8", newline="\n")
    print("->", OUT, total, "个视频")


if __name__ == "__main__":
    main()
