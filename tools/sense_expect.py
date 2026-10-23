#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<👂感官层-期待-CLI：往 感官.json 里**加** `expect` 节点（意外度 + 情绪主导的显著度），出一张 感官_期待审计.jpg（已有就 _v2、_v3 …… 不覆盖）；给了剪辑表时分歧点按时间编号（A1… / B1…），同名 .md 出清单（编号 | 时间 | 意外度 | 当前画面 | 一句建议）。
⚠️这是积木 / 审计，不是导演（👤2026-10-23：「核心还是跟着 Thymos 和 emotion flow，数学是 building block 和审计视野」）。
它不改剪辑表、不改画面、不改任何已有字段；分歧点只画出来给人看。

用法：python tools/sense_expect.py <作品文件夹> [--edl 剪辑表] [--video 按该剪辑表出的样片] [--audit-end 秒] [--dry]
- 情绪段落 = <作品>/情绪大图/草案.json 的「段」；她在等什么 = 同目录的 期待.json（没有 → 只出意外度，显著度为 null）
- 没有情绪大图 → 滑动窗口（expect.TAU_SLIDE）当上下文，显著度为 null
- 事件流：onset（起音，大小 = 起音那一帧的 threat）、motif（动机 hits）、voice_on / voice_off（人声乐句起 / 止）、absence（该来的起音没来）
- 写入：除 expect 外逐字段比对旧文件，不等就不落盘；第一次加 expect 前备份 感官.json.bak_2026-10-23_expect
- ⚠️顺序：sense_voice.py 之后再跑（expect 读 voice）；sense_song.py 会整个重写 感官.json（voice / expect 都会丢）
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import expect as E  # noqa: E402
import sensorium as S  # noqa: E402

FONT = "C:/Windows/Fonts/consola.ttf"
FCN = "C:/Windows/Fonts/msyh.ttc"
BAK = "感官.json.bak_2026-10-23_expect"
STREAM_COL = {"onset": (150, 156, 162), "motif": (235, 200, 90), "voice_on": (90, 150, 230), "voice_off": (150, 190, 240),
              "absence": (180, 110, 230)}
EXP_COL = {"冲击": (150, 40, 46), "John": (150, 118, 40), "自己的声音": (40, 80, 140), "自己": (60, 70, 90),
           "记忆": (70, 90, 100), "无": (44, 46, 50)}


# ───────────────────────── 计算 ─────────────────────────
def load_context(mv):
    """情绪段落 + 期待；没有就 (None, None)"""
    dp = mv / "情绪大图" / "草案.json"
    if not dp.exists():
        return None, None, None
    segs = json.loads(dp.read_text("utf-8"))["段"]
    ep = mv / "情绪大图" / "期待.json"
    exp = None
    if ep.exists():
        ex = {e["id"]: e for e in json.loads(ep.read_text("utf-8"))["段"]}
        missing = [s["id"] for s in segs if s["id"] not in ex]
        assert not missing, f"期待.json 少了这些段：{missing}"
        exp = [ex[s["id"]] for s in segs]
    return segs, exp, ep if exp else None


def streams(sense):
    fps = sense["fps"]; thr = np.array(sense["channels"]["threat"])
    out = {}
    on = np.array(sense["events"]["onsets"], float)
    k = np.clip(np.round(on * fps).astype(int), 0, len(thr) - 1)
    mag = np.array([thr[max(0, j - 1):j + 2].max() for j in k])
    out["onset"] = (on, mag, dict(use_rate=True, use_mag=True))
    hits = sorted(h for m in sense.get("motifs", {}).values() for h in m["hits"])
    if hits:
        out["motif"] = (np.array(hits, float), np.ones(len(hits)), dict(use_rate=True, use_mag=False))
    ph = sense.get("voice", {}).get("phrases", [])
    if ph:
        out["voice_on"] = (np.array([p["start"] for p in ph]), np.array([p["peak"] for p in ph]), dict(use_rate=True, use_mag=True))
        # 乐句的结束由开始决定，不算「来没来」，只算「这么长的一句」有多罕见
        out["voice_off"] = (np.array([p["end"] for p in ph]), np.array([p["end"] - p["start"] for p in ph]), dict(use_rate=False, use_mag=True))
    return out


def compute(sense, segs, exp):
    fps = sense["fps"]; n = len(sense["channels"]["heartbeat"]); beat = 60 / sense["bpm"]
    seglist = [tuple(s["t"]) for s in segs] if segs else None
    evs = []
    st = streams(sense)
    for name, (ts, mags, kw) in st.items():
        r = E.surprisal(ts, mags, seglist, beat=beat, **kw)
        for i in range(len(ts)):
            evs.append({"t": round(float(ts[i]), 3), "s": name, "m": round(float(mags[i]), 3),
                        "bits": round(float(r["bits"][i]), 2), "rate": round(float(r["rate"][i]), 2), "mag": round(float(r["mag"][i]), 2)})
    ft = np.arange(n) / fps
    on = st["onset"][0]
    ab = E.absence(on, ft, seglist)
    for t, b, d in E.absence_events(ab, fps):
        evs.append({"t": round(t, 3), "s": "absence", "m": round(d, 2), "bits": round(b, 2), "rate": round(b, 2), "mag": 0.0})
    evs.sort(key=lambda e: (e["t"], e["s"]))
    seg = E.seg_index([e["t"] for e in evs], seglist)
    for e, g in zip(evs, seg):
        e["seg"] = segs[g]["id"] if segs else None
        e["sal"] = round(E.salience(e["bits"], e["s"], g, exp), 2) if exp else None
    pt = [e for e in evs if e["s"] != "absence"]
    surprise = E.envelope([e["t"] for e in pt], [e["bits"] for e in pt], n, fps)
    sal = E.envelope([e["t"] for e in evs], [e["sal"] for e in evs], n, fps) if exp else None
    return evs, surprise, ab, sal


def node(sense, segs, exp, ep, mv, evs, surprise, ab, sal):
    rel = lambda p: p.resolve().relative_to(mv.resolve()).as_posix()  # noqa: E731
    return {
        "version": "2026-10-23",
        "role": "积木 / 审计，不是导演：标出「哪里罕见」和「她此刻会不会被它抓住」，不改剪辑、不改画面",
        "method": f"意外度 = −log2 p，p 只用这一刻之前听到的；同段全记、前段 {E.TAU_OUT:g} s 内淡掉（无段落时滑动窗口 {E.TAU_SLIDE:g} s）；"
                  f"bits = 这一拍里来一个的罕见度 + 这么大或更大的罕见度；absence = 该来的起音没来（> {E.ABS_ON:g} 比特记一次）；上限 {E.CAP:g} 比特。见 src/expect.py",
        "context": (f"情绪大图/草案.json 的段（{len(segs)} 段）" if segs else f"滑动窗口 {E.TAU_SLIDE:g} s（这首没有情绪大图）"),
        "expect_source": (rel(ep) + "（⚠️草稿，待👤过）") if ep else None,
        "salience_note": None if exp else ("没有 期待.json：显著度留空" if segs else "没有情绪大图：显著度留空（要先有段落和「她此刻」）"),
        "params": {"tau_out": E.TAU_OUT, "tau_slide": None if segs else E.TAU_SLIDE, "alpha_n": E.ALPHA_N, "alpha_m": E.ALPHA_M,
                   "cap_bits": E.CAP, "absence_on": E.ABS_ON, "gate_bits": E.GATE if exp else None,
                   "unattended": E.UNATTENDED if exp else None, "beat": round(60 / sense["bpm"], 4)},
        "segments": [[s["t"][0], s["t"][1], s["id"], s["名"]] for s in segs] if segs else None,
        "events": evs,
        "surprise": np.round(surprise, 2).tolist(),
        "absence": np.round(ab, 2).tolist(),
        "salience": np.round(sal, 2).tolist() if sal is not None else None,
    }


# ───────────────────────── 审计：剪辑表 + 样片 ─────────────────────────
LINE = re.compile(r"^(\d+:\d+(?:\.\d+)?)\s+(\S+)(?:\s+([\d.]+))?\s*(?:#\s*(.*))?$")


def parse_edl(path):
    rows = []
    for ln in Path(path).read_text("utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        m = LINE.match(ln)
        if not m:
            continue
        mm, ss = m.group(1).split(":")
        rows.append({"t": int(mm) * 60 + float(ss), "src": m.group(2), "speed": float(m.group(3) or 1), "note": m.group(4) or ""})
    for a, b in zip(rows, rows[1:]):
        a["dur"] = b["t"] - a["t"]
    rows = [r for r in rows if "dur" in r and r["src"] != "黑"]
    durs = np.array([r["dur"] for r in rows])
    for i, r in enumerate(rows):
        nb = np.r_[durs[max(0, i - 4):i], durs[i + 1:i + 5]]
        med = float(np.median(nb)) if len(nb) else r["dur"]
        r["med"] = med
        why = []
        if r["dur"] >= max(2.4, 1.5 * med):
            why.append(f"停留 {r['dur']:.1f}s（邻近中位 {med:.1f}s）")
        if r["src"].startswith("图:") and r["speed"] >= 1 and r["dur"] >= 1.2:
            why.append("静帧慢推")
        if "⭐" in r["note"]:
            why.append("星标")                  # 图里写字：msyh 没有 ⭐ 字形
        r["long"] = why
        r["short"] = r["dur"] < 0.3
    for i in range(2, len(rows)):          # 闪帧之后接回同一个素材 = 原镜头的续，不是新开的长码（码是那一下闪帧）
        a, b, c = rows[i - 2], rows[i - 1], rows[i]
        if b["short"] and a["src"].split("@")[0] == c["src"].split("@")[0]:
            c["long"] = []; c["cont"] = True
    return rows


def motion(video, n):
    """样片逐帧运动量：96×54 灰度的相邻帧绝对差，切成 6×6 块，取最大的那块（0~1）。
    取块而不是全图平均：拍点该作用在有形的东西上（哲学 §8），一根手指卷曲在全图平均里看不见"""
    import imageio_ffmpeg
    w, h = 96, 54
    raw = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-i", str(video), "-vf", f"scale={w}:{h}",
                          "-pix_fmt", "gray", "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, h, w).astype(np.float32) / 255
    df = np.abs(np.diff(fr, axis=0)).reshape(-1, 6, h // 6, 6, w // 6).mean((2, 4))
    d = np.r_[0, df.reshape(len(df), -1).max(1)]
    out = np.zeros(n); out[:min(n, len(d))] = d[:n]
    return out, len(fr)


def audit(evs, rows, mot, fps, end):
    """分歧点。A：显著度（没有就意外度）高，但附近没有切点、样片也没有运动尖峰 —— 画面没回应。
              B：剪辑给了长码（停留 / 慢推 / ⭐），但切点前 0.1 s 到后 0.6 s 内意外度 < 3 比特、且低于本段六成起音。只标，不改。"""
    key = "sal" if evs and evs[0]["sal"] is not None else "bits"
    inr = [e for e in evs if e["t"] < end]
    v = np.array([e[key] for e in inr])
    thr = max(2.0, float(np.percentile(v, 92))) if len(v) else 99
    cuts = np.array([r["t"] for r in rows])
    A = []
    for e in inr:
        if e[key] < thr:
            continue
        k = int(round(e["t"] * fps))
        near_cut = len(cuts) and np.abs(cuts - e["t"]).min() <= 0.2
        # 长码镜头在事件前 ≤ 0.6 s 开 = 镜头是为它开的（如 ⭐抽搐的 MEMS 插入：切在 0.42 s 前，镜头里一直在动，抽搐只是速度鼓包，帧差抓不到）
        opened = any(r["long"] and 0 <= e["t"] - r["t"] <= 0.6 for r in rows)
        spike = False
        if mot is not None and k < len(mot):
            lo, hi = max(0, k - 48), min(len(mot), k + 49)
            base = np.median(mot[lo:hi]) + 1e-4
            spike = mot[max(0, k - 3):k + 6].max() >= 2.5 * base      # 画面可以晚到 5 帧（≈ 0.2 s），不能早于 3 帧
        if not near_cut and not opened and not spike:
            A.append(e)
    A.sort(key=lambda e: -e[key])
    B = []
    for r in rows:
        if not r["long"] or r["t"] < 1.0 or r["t"] >= end:
            continue
        near = [e for e in evs if -0.1 <= e["t"] - r["t"] <= 0.6]       # 切点这一刻，到镜头开出来后 0.6 s（镜头可以为马上要来的事件开）
        mb = max([e["bits"] for e in near], default=0.0)
        seg = [e["bits"] for e in evs if e["seg"] == (near[0]["seg"] if near else None) and e["s"] == "onset"]
        pct = float(np.mean(np.array(seg) < mb)) if seg else 0.0
        # 「平常的一刻」：比这一段里六成的起音都不意外，且 < 3 比特
        if mb < 3.0 and pct < 0.6:
            B.append((r, mb, pct))
    return A, B, key, thr


def _mmss(t):
    return f"{int(t // 60)}:{t % 60:05.2f}"


def number(A, B, rows):
    """分歧点按时间编号（A1, A2… / B1, B2…），让人能用一句话指认：「A3 加闪帧」「B2 留」。
    每个点附上当前画面（落在哪一段剪辑）和一句**草稿**建议（规则给的，人来定）。"""
    def shot(t):
        for r in rows:
            if r["t"] <= t < r["t"] + r["dur"]:
                return r
        return None
    A = sorted(A, key=lambda e: e["t"])
    for i, e in enumerate(A, 1):
        e["id"] = f"A{i}"
        r = shot(e["t"])
        e["shot"] = r["src"] if r else "—"
        into = e["t"] - r["t"] if r else 0.0
        e["advice"] = {"onset": f"加闪帧（2–3 帧）或把切点挪到 {_mmss(e['t'])}（现在镜头开了 {into:.1f} s）",
                       "voice_on": "人声进来：给她一个回应 —— 切点、亮度起或推近，对齐到起音",
                       "voice_off": "人声收：画面跟着落（减速 / 暗下）",
                       "motif": "动机：插独立镜头（同前几次出现的处理）",
                       "absence": "该来的没来：可以留空白 / 定格，或不动"}.get(e["s"], "看一眼再定")
    B = sorted(B, key=lambda b: b[0]["t"])
    out = []
    for i, (r, mb, pct) in enumerate(B, 1):
        out.append((r, mb, pct, f"B{i}"))
    return A, out


def write_md(path, img, mv, edl, video, A, B, key, sense):
    """和审计图同名的 .md 清单：编号 | 时间 | 意外度 | 当前画面 | 一句建议"""
    L = [f"# {mv.name} 期待 / 意外 审计清单（{Path(img).name}）", "",
         f"- 剪辑表：`{Path(edl).name}`；样片运动量：`{Path(video).name if video else '（没给，只看切点）'}`",
         "- ⚠️积木 / 审计，不是导演。建议是规则起的**草稿**；你说「A3 加闪帧」「B2 留」这样就行，我按编号改剪辑表。",
         f"- A = {('显著度' if key == 'sal' else '意外度')}高但画面没回应（没切点、没为它开的长码、样片没运动尖峰）；B = 剪辑给了长码，但那一刻很平常。", "",
         "| 编号 | 时间 | 意外度（比特） | 事件 | 当前画面 | 一句建议 | 你的决定 |", "|---|---|---|---|---|---|---|"]
    for e in A:
        sal = f"，显著 {e['sal']}" if e.get("sal") is not None else ""
        L.append(f"| **{e['id']}** | {_mmss(e['t'])} | {e['bits']:.1f}{sal} | {e['s']} | `{e['shot']}` | {e['advice']} |  |")
    for r, mb, pct, bid in B:
        alt = [x for x in sense.get("_evs", []) if abs(x["t"] - r["t"]) <= 1.5 and x["bits"] >= 3.0 and x["s"] != "voice_off"]
        if alt:
            x = max(alt, key=lambda x: x["bits"])
            adv = f"把这个长码挪到 {_mmss(x['t'])}（{x['s']}，{x['bits']:.1f} 比特），或缩到 ~{r['med']:.1f} s"
        else:
            adv = f"缩到 ~{r['med']:.1f} s（邻近中位），或留（如果这份停留本身是情绪）"
        L.append(f"| **{bid}** | {_mmss(r['t'])} | {mb:.1f}（本段第 {pct * 100:.0f} 百分位） | 长码：{'；'.join(r['long'])} | `{r['src']}` | {adv} |  |")
    Path(path).write_text("\n".join(L) + "\n", "utf-8", newline="\n")


# ───────────────────────── 图 ─────────────────────────
def draw(path, title, sense, segs, exp, evs, surprise, ab, sal, rows=None, mot=None, A=(), B=(), key="bits", audit_end=None):
    dur = sense["duration"]; fps = sense["fps"]
    f = ImageFont.truetype(FONT, 13); fc = ImageFont.truetype(FCN, 14); fb = ImageFont.truetype(FCN, 18)
    L, R = 90, 20

    def panel(t0, t1, pps, with_edit):
        gw = int((t1 - t0) * pps)
        lanes = [("段落", 54), ("意外度", 150), ("absence", 50)] + ([("显著度", 150)] if sal is not None else [])
        if with_edit:
            lanes += [("剪辑", 70), ("样片运动", 60)]
        TOP = 58 if (A or B) else 0                 # 分歧点编号的标签带（3 层错开，避免互相压字）
        H = sum(h + 8 for _, h in lanes) + 30 + TOP
        im = Image.new("RGB", (L + gw + R, H), (12, 13, 15)); g = ImageDraw.Draw(im)

        def X(t):
            return L + (t - t0) * pps
        y = 4 + TOP
        # 乐句（淡竖线）
        for t in sense["events"].get("phrases", []):
            if t0 <= t <= t1:
                g.line([(X(t), 4), (X(t), H - 28)], fill=(30, 32, 36))
        for name, h in lanes:
            g.text((6, y + h // 2 - 9), name, font=fc, fill=(150, 156, 162))
            if name == "段落":
                if segs:
                    for i, s in enumerate(segs):
                        a, b = max(t0, s["t"][0]), min(t1, s["t"][1])
                        if a >= b:
                            continue
                        e = exp[i] if exp else {}
                        c = EXP_COL.get(e.get("expecting", "无"), (50, 52, 56))
                        g.rectangle([X(a), y, X(b) - 1, y + h], fill=c)
                        lab = f"{s['id']} {s['名']}"
                        if e:
                            lab2 = f"等:{e['expecting']} 忍:{e['tolerance']}"
                        else:
                            lab2 = ""
                        g.text((X(a) + 3, y + 3), lab, font=fc, fill=(235, 236, 238))
                        if lab2 and (b - a) * pps > 90:
                            g.text((X(a) + 3, y + 26), lab2, font=fc, fill=(200, 202, 206))
                else:
                    g.rectangle([L, y, L + gw, y + h], fill=(30, 32, 36))
                    g.text((L + 6, y + 16), f"没有情绪大图：上下文 = 滑动窗口 {E.TAU_SLIDE:g} s，显著度留空（底下红条 = 人声段）", font=fc, fill=(200, 200, 200))
                    for s0, s1 in sense.get("voice", {}).get("sections", []):
                        g.rectangle([X(max(t0, s0)), y + h - 6, X(min(t1, s1)), y + h], fill=(200, 60, 70))
            elif name in ("意外度", "显著度"):
                fld = "bits" if name == "意外度" else "sal"
                top = E.CAP if name == "意外度" else max(4.0, max((e["sal"] or 0) for e in evs))
                for b in (2, 4, 8):
                    if b < top:
                        yy = y + h - b / top * h
                        g.line([(L, yy), (L + gw, yy)], fill=(34, 36, 40))
                        g.text((L - 22, yy - 7), str(b), font=f, fill=(100, 104, 110))
                for e in evs:
                    if not (t0 <= e["t"] <= t1) or e[fld] is None:
                        continue
                    hh = min(1.0, e[fld] / top) * h
                    g.line([(X(e["t"]), y + h), (X(e["t"]), y + h - hh)], fill=STREAM_COL[e["s"]], width=2 if e["s"] != "onset" else 1)
            elif name == "absence":
                k0, k1 = int(t0 * fps), min(len(ab), int(t1 * fps))
                pts = [(X(k / fps), y + h - min(1, ab[k] / E.CAP) * h) for k in range(k0, k1, 2)]
                if len(pts) > 1:
                    g.line(pts, fill=STREAM_COL["absence"], width=1)
                yy = y + h - E.ABS_ON / E.CAP * h
                g.line([(L, yy), (L + gw, yy)], fill=(50, 40, 60))
            elif name == "剪辑":
                for r in rows:
                    if not (t0 <= r["t"] < t1):
                        continue
                    x0, x1 = X(r["t"]), X(min(t1, r["t"] + r["dur"]))
                    if r["long"]:
                        g.rectangle([x0, y + 18, x1 - 1, y + h], fill=(70, 74, 80))
                    elif r["short"]:
                        g.rectangle([x0, y + 50, max(x0 + 2, x1), y + h], fill=(200, 200, 200))
                    else:
                        g.rectangle([x0, y + 40, x1 - 1, y + h], fill=(36, 38, 42))
                    g.line([(x0, y + 10), (x0, y + h)], fill=(230, 233, 236))
                g.text((L + 4, y), "亮竖线 = 切点   高灰块 = 长码（停留 / 慢推 / 星标）   白矮块 = 闪帧（短码）", font=fc, fill=(120, 124, 130))
            elif name == "样片运动" and mot is not None:
                top = np.percentile(mot[mot > 0], 99) if (mot > 0).any() else 1
                k0, k1 = int(t0 * fps), min(len(mot), int(t1 * fps))
                for k in range(k0, k1):
                    hh = min(1, mot[k] / top) * h
                    g.line([(X(k / fps), y + h), (X(k / fps), y + h - hh)], fill=(110, 116, 122))
            g.line([(L, y + h + 4), (L + gw, y + h + 4)], fill=(30, 32, 36))
            y += h + 8
        # 分歧点
        marks = [(e["t"], e.get("id", ""), (255, 42, 52), "A") for e in A] + [(b[0]["t"], b[3] if len(b) > 3 else "", (255, 160, 40), "B") for b in B]
        marks.sort()
        lvl_end = [-1e9, -1e9, -1e9]                 # 每层上一个标签的右端（像素）
        for t, mid, col, kind in marks:
            if not (t0 <= t <= t1):
                continue
            x = X(t)
            lab = f"{mid} {_mmss(t)}" if with_edit else mid
            wl = g.textlength(lab, font=fc) + 8
            lv = next((i for i in range(3) if lvl_end[i] < x - 2), int(np.argmin(lvl_end)))
            lvl_end[lv] = x + wl
            ly = 4 + lv * 18
            g.text((x + 2, ly), lab, font=fc, fill=col)
            if kind == "A":
                g.polygon([(x - 6, TOP + 4), (x + 6, TOP + 4), (x, TOP + 16)], fill=col)
                g.line([(x, ly + 2), (x, TOP + 4)], fill=col)
                g.line([(x, TOP + 16), (x, H - 28)], fill=(120, 30, 36))
            else:
                g.polygon([(x - 6, H - 28), (x + 6, H - 28), (x, H - 40)], fill=col)
                g.line([(x, ly + 2), (x, H - 40)], fill=(110, 70, 20))
        if audit_end and t0 < audit_end < t1 and with_edit is False:
            g.line([(X(audit_end), 4), (X(audit_end), H - 28)], fill=(255, 160, 40))
        step = 5 if pps >= 16 else 10
        for t in range(int(t0), int(t1) + 1):
            if t % step == 0:
                g.line([(X(t), H - 28), (X(t), H - 22)], fill=(120, 120, 120))
                g.text((X(t) + 2, H - 20), f"{t // 60}:{t % 60:02d}", font=f, fill=(150, 150, 150))
        return im

    pans = [panel(0, dur, 8 if dur > 200 else 16, False)]
    if rows is not None:
        pans.append(panel(0, audit_end, 24, True))
    # 图例 + 分歧点清单
    leg = [f"{title}", "意外度 = 在她此刻所在段落里有多罕见（比特，−log2 p，只用之前听到的）；显著度 = 期待.json 的乘子 × 门 × 意外度。这是积木 / 审计，不是导演。",
           "颜色：灰 = 起音  黄 = 动机  蓝 = 人声起  浅蓝 = 人声止  紫 = 该来的没来（absence）。段落底色 = 她在等什么（红 = 冲击，金 = John，蓝 = 自己的声音）。"]
    if rows is not None:
        leg.append(f"▼红 = {('显著度' if key == 'sal' else '意外度')}高（≥ 前 8% 且 ≥ 2）但 ±0.2 s 没有切点、前 0.6 s 内也没开长码镜头、样片任何一块（6×6）也没有运动尖峰（−3~+5 帧，≥ 附近中位 2.5 倍）；▲橙 = 剪辑给了长码，但切点前 0.1 s 到后 0.6 s 内最大意外度 < 3 比特、且低于本段六成起音。只标出来，不改任何东西。")
        for e in A:
            leg.append(f"  ▼ {e.get('id', ''):<4} {_mmss(e['t'])}  {e['s']:<9} 意外 {e['bits']:.1f} 比特  显著 {e['sal'] if e['sal'] is not None else '—'}  段 {e['seg']}  画面 {e.get('shot', '')[:40]}")
        for r, mb, pct, *bid in B:
            leg.append(f"  ▲ {(bid[0] if bid else ''):<4} {_mmss(r['t'])}  {r['src'][:46]}  长码：{'；'.join(r['long'])}  这一刻意外 {mb:.1f} 比特（本段第 {pct * 100:.0f} 百分位）")
        leg.append("  编号可直接用来说：「A3 加闪帧」「B2 留」—— 同名 .md 是清单。")
    lh = 22
    W = max(p.width for p in pans)
    Ht = len(leg) * lh + 16 + sum(p.height + 10 for p in pans)
    im = Image.new("RGB", (W, Ht), (12, 13, 15)); g = ImageDraw.Draw(im)
    y = 8
    for i, s in enumerate(leg):
        g.text((10, y), s, font=fb if i == 0 else fc, fill=(230, 233, 236) if i == 0 else (170, 174, 180)); y += lh
    y += 8
    for p in pans:
        im.paste(p, (0, y)); y += p.height + 10
    im.save(path, quality=88)


# ───────────────────────── 主程序 ─────────────────────────
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows 控制台默认 GBK，⚠️ 等字符会让 print 崩
    except AttributeError:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("mv")
    ap.add_argument("--edl", default=None, help="审计用的剪辑表（不写进 感官.json）")
    ap.add_argument("--video", default=None, help="按该剪辑表出的样片（逐帧运动量；没有就只看切点）")
    ap.add_argument("--audit-end", type=float, default=None, help="剪辑表只到这一秒有真剪辑（之后不审）")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    mv = Path(a.mv); sp = mv / "感官.json"
    sense = json.loads(sp.read_text("utf-8"))
    segs, exp, ep = load_context(mv)
    evs, surprise, ab, sal = compute(sense, segs, exp)
    nd = node(sense, segs, exp, ep, mv, evs, surprise, ab, sal)
    by = {}
    for e in evs:
        by.setdefault(e["s"], []).append(e["bits"])
    print(f"context: {nd['context']}  expect: {nd['expect_source']}")
    for s, v in by.items():
        v = np.array(v)
        print(f"  {s:<9} n={len(v):4d}  bits p50 {np.median(v):.2f}  p90 {np.percentile(v, 90):.2f}  max {v.max():.2f}")
    rows = mot = None; A = B = (); key = "bits"; end = a.audit_end
    if a.edl:
        rows = parse_edl(a.edl)
        end = end or (rows[-1]["t"] + rows[-1]["dur"])
        if a.video:
            mot, nfr = motion(a.video, len(surprise))
            print(f"  样片 {nfr} 帧")
        A, B, key, thr = audit(evs, rows, mot, sense["fps"], end)
        print(f"audit 0–{end:.1f}s：{len(rows)} 段，长码 {sum(bool(r['long']) for r in rows)} 段；A（{key} ≥ {thr:.2f} 无回应）{len(A)} 个，B（长码低意外）{len(B)} 个")
        A, B = number(A, B, rows)
        for e in A:
            print(f"  {e['id']:<4} {e['t']:7.2f} {e['s']:<9} bits {e['bits']:.2f} sal {e['sal']} seg {e['seg']}")
        for r, mb, pct, bid in B:
            print(f"  {bid:<4} {r['t']:7.2f} {r['src']}  {r['long']}  bits {mb:.2f}  pct {pct:.2f}")
    img = mv / "感官_期待审计.jpg"
    k = 2
    while img.exists():                           # 不覆盖用户可能看过的图（2026-10-23：voice 重算后重跑会撞名）
        img = mv / f"感官_期待审计_v{k}.jpg"; k += 1
    draw(img, f"{mv.name}  期待 / 意外 审计", sense, segs, exp, evs, surprise, ab, sal, rows, mot, A, B, key, end)
    print(f"  -> {img}")
    if rows is not None:
        md = img.with_suffix(".md")
        write_md(md, img, mv, a.edl, a.video, A, B, key, {**sense, "_evs": evs})
        print(f"  -> {md}")
    if a.dry:
        return
    if "expect" not in sense and not (mv / BAK).exists():
        shutil.copy2(sp, mv / BAK)
    new = dict(sense); new["expect"] = nd
    src = new.pop("source", None)
    tmp = mv / "感官.json.tmp"
    S.write(new, tmp, source=src)
    back = json.loads(tmp.read_text("utf-8"))
    back.pop("expect")
    old = dict(sense); old.pop("expect", None)
    assert back == old, "除 expect 外有字段变了 —— 不落盘"
    assert list(back) == list(old), "字段顺序变了"
    os.replace(tmp, sp)
    print(f"  -> {sp}（其余字段逐个比对一致；备份 {BAK}）")


if __name__ == "__main__":
    main()
