#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🎬HTE-分镜-按歌词行生成整首的 Synapsis 分镜（MV_Happier_Than_Ever，2026-10-24，Fable）。
规则（ReadScreen / Leo 的硬要求）：切点先锚歌词行（歌词_v4.json 的 lines），再吸附到最近的拍（感官.json 的 beats，162.43 = 双倍速，
真拍 = 每隔一个）；段落边界换空间；切的密度 = 情绪曲线：前半每行一镜（4–10 s）、过门转折、147 s 后每行拆两镜（≈2 s）、
184 s 后每镜 ≤ 2 s（按词间隙拆）、最后一句每拍一切 → 黑帧休止 → 长镜头余韵。
输出：Synapsis batch JSON（stdout 或 --out），再 node cli.js batch 写进 __synapsis。
"""
import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MV = ROOT.parents[1] / "Projekt_Visual" / "MV_Happier_Than_Ever"

# 每行歌词 → (场景 / 画面, 镜头运动, 切入方式)。行号 = 歌词_v4.json 的顺序（1 起）。
SHOTS = {
 1: ("浴缸·全景：她坐在没有水的浴缸里，打字机放膝上，一盏台灯；琥珀。字 'When I'm away from you' 像打字机一样一个词一个词落在画面下 1/5（小字）", "极慢横移", "从黑长出"),
 2: ("浴缸·俯拍打字机纸：纸上只有这一行；方块在纸角闪 1 帧", "缓推", "硬切"),
 3: ("浴缸·侧面近景她的脸：面无表情，辫子（信号橙）垂在肩上；方块在脸上闪 1–2 帧就消失", "静中微漂", "硬切"),
 4: ("浴缸·高角度：浴缸 + 地上摊开的信纸；她不动；字小、靠右", "缓慢下降", "硬切"),
 5: ("厨房·全景：凌晨四点，她用一座奖杯吃麦片，手机扣在桌上", "缓推", "硬切（换空间）"),
 6: ("厨房·近景：勺子停在半空，麦片牛奶滴下来；字排版在桌面上", "静中微漂", "硬切"),
 7: ("厨房·打字机特写：'To write myself a letter' 真的被打出来（词时间 = 打字时间）", "缓推", "硬切"),
 8: ("厨房·俯拍手：把信对折、再对折；方块第一次长到 16 px 又缩回（鼓没进，只是她的手）", "极慢横移", "硬切"),
 9: ("冰箱前·全景：她剪杂志，剪下的脸贴在冰箱门上（脸是 CG、不像任何人）", "缓推", "硬切（换空间）"),
 10: ("特写：剪刀剪掉一张脸；'interviews' 字被剪刀剪开成两半", "静中微漂", "硬切"),
 11: ("走廊·全景：她背对镜头站在一扇门前，门缝有光；'avenue' 沿门缝竖排", "极慢推", "硬切（换空间）"),
 12: ("走廊·近景门缝的光：光里有人影走过去（他，只是影子）", "静中微漂", "硬切"),
 13: ("空车位·全景：夜里的停车场，一盏路灯，一个空车位，她对着空车位说话", "缓慢环绕", "硬切（换空间）"),
 14: ("空车位·近景她的嘴：面无表情地说 'Be cool'；字小", "静中微漂", "硬切"),
 15: ("空车位·俯拍：她站在车位的白线框里，像站在一个格子里；方块闪", "缓慢上升", "硬切"),
 16: ("厨房·全景：她坐在打开的冰箱里面，冰箱灯是唯一光源；'more afraid' 小字", "缓推", "硬切（换空间）"),
 17: ("冰箱里·近景：她的脸被冰箱灯打亮，呼出白气", "静中微漂", "硬切"),
 18: ("冰箱里·特写：手里揉皱的信，纸上的字在方块化", "缓推", "硬切"),
 19: ("厨房·全景：厨房灯灭，只剩冰箱灯；'miserable' 是前半最大的字（占画高 1/6），停 2 s 后方块吃掉它", "极慢横移", "硬切"),
 20: ("厨房·近景：厨房灯灭后只剩冰箱灯，她在灯里抬头（'So if you really wanna know' 小字）", "静中微漂", "硬切"),
 21: ("浴缸·全景（= 第 1 行的构图，重复一次）：琥珀更暗；字稍大", "极慢横移（反方向）", "硬切（换空间）"),
 22: ("浴缸·俯拍打字机纸（= 第 2 行）：纸上已经有两行同样的话", "缓推", "硬切"),
 23: ("浴缸·侧面近景（= 第 3 行）：方块闪得更频（每 0.8 s 一次）", "静中微漂", "硬切"),
 24: ("浴缸·高角度（= 第 4 行）：'I wish it wasn't true' 拉长 8 s，字一个个方块化；色温开始从琥珀褪向灰", "缓慢下降", "硬切"),
 25: ("空车位·全景：一辆车的车灯亮起来（车来了，她不在车里）；深青；字中等偏大", "缓推向车灯", "硬切（换色温 = 转折后第一镜）"),
 26: ("车·近景：从乘客座看空的驾驶位，雨在车窗上；'under the influence' 字被雨水拖", "静中微漂", "硬切"),
 27: ("路边·全景：她站在路边，车灯从背后打过来，影子拉到镜头前", "缓慢横移", "硬切"),
 28: ("特写：手机屏幕上一串没有名字的联系人；'fucking friends' 第一次大字（占画高 1/4）", "缓推", "硬切"),
 29: ("她的脸·正面近景：'I don't relate to you' 大字横过脸；方块开始长（32 px）", "静中微漂", "硬切"),
 30: ("她的脸·反打（另一侧）：同一句，字更大；方块 48 px", "静中微漂", "硬切"),
 31: ("天台边缘·全景：她背对镜头，辫子被风吹直（唯一的橙）；城市在下面", "缓慢上升", "硬切（换空间）"),
 32: ("城市天际线·大字 'hate this city' 占画高 1/3；方块把天际线一块块吃掉", "缓推", "硬切"),
 33: ("天台·她转身面对镜头：'internet' 字从她身后升起", "快推", "硬切"),
 34: ("嘴·特写：'Never told anyone' 方块在嘴上长到 64 px", "静中微漂", "硬切"),
 35: ("冰箱门·全景：贴满剪下的脸；'everything' 大字盖在上面", "缓推", "硬切（换空间）"),
 36: ("全屏大字：'fucking sad'（全片最大的字之一，占画高 1/3）；底图是浴缸被方块吃到一半", "静中微漂", "硬切"),
 37: ("厨房·全景：她用吸尘器吸一地的字；字被吸进去", "缓慢横移", "硬切（换空间）"),
 38: ("近景：她的手推向镜头（'don't try'）；方块从手心长出", "快推再停", "硬切"),
 39: ("空椅子·全景：桌边一把空椅，'showed up on time' 字停在椅子上", "缓推", "硬切"),
 40: ("打字机特写：一行空的（'empty line'），光标闪 = 拍点", "静中微漂", "硬切"),
 41: ("三把空椅·全景：'mother or friends'；每把椅子上一个被方块化的名字", "缓慢横移", "硬切（换空间）"),
 42: ("她在画面里很小·超广：'I was a kid'，她缩在画面角落，方块从四周长进来", "缓慢拉远", "硬切"),
 43: ("浴缸·全景被方块吃掉：'ruined everything good'，浴缸最后一次出现，长到 1/4 画幅就硬切", "静中微漂", "硬切（换空间）"),
 44: ("杂志脸·特写：被剪掉的脸贴满画面；'misunderstood' 大字", "缓推", "硬切"),
 45: ("快切：浴缸 / 厨房 / 走廊 / 车位 / 天台五个场景，每个 ≤ 1 s，都在被方块吃", "各自原运动", "硬切 ×5"),
 46: ("'leave me alone'：每拍一切，五个场景的方块版；最后一个词 'alone' 单独一镜，橙辫子是画面里唯一的颜色 → 黑帧 6 帧", "静", "硬切 / 黑帧休止"),
}
# 没有歌词的段落（时间，秒）
EXTRA = [
 ("00 吸气", 0.0, 4.20, [("E01 黑到琥珀", 0.0, 4.20, "黑 → 琥珀慢慢亮起来；一行小字先在角落排版（'When I'm away from you' 的词以极小字预排，还没唱）", "静中微漂", "从黑长出")]),
 ("04 信被撕了", 133.09, 147.59, [
     ("E02 撕信", 133.09, 140.0, "俯拍：她把信撕成方块，按大小排好；色温从琥珀褪到灰；没有字", "缓慢上升", "硬切（换空间）"),
     ("E03 方块长满", 140.0, 147.59, "排好的方块自己长大，长到 1/4 画幅把画面吃掉 → 147.59 硬切到深青的车位", "静中微漂", "溶解是切的前奏 → 硬切"),
 ]),
 ("07 休止与喊", 242.24, 280.0, [
     ("E04 黑帧休止", 242.24, 242.49, "黑 6 帧", "静", "黑帧"),
     ("E05 吃掉再吐出", 242.49, 256.0, "一个长镜头：五个场景的方块版叠在一起被整个吃掉，再从方块里吐出一个空房间（没有她）；没有字", "缓慢拉远", "从黑长出"),
     ("E06 喊", 256.0, 270.0, "空房间·全景：喊声（没有词）每一下让方块长一级又缩回；橙辫子不在画面里", "静中微漂", "硬切"),
     ("E07 余韵", 270.0, 280.0, "空房间·近景地上一根橙辫子；溶解只到一半就停（不收干净）", "缓推", "硬切"),
 ]),
 ("08 空", 280.0, 295.0, [("E08 青褪到黑", 280.0, 295.0, "青褪到黑；最后 3 s 只剩辫子的一点橙，然后没了", "静中微漂", "溶解出")]),
]
SEGMENTS = [  # (名字, t0, t1, 行号范围)
 ("01 礼貌版", 4.20, 49.70, (1, 8)),
 ("02 数他做过的事", 52.72, 103.0, (9, 19)),
 ("03 副歌重复", 104.0, 133.09, (20, 24)),
 ("05 真话版", 147.59, 184.0, (25, 32)),
 ("06 一行比一行狠", 184.0, 242.24, (33, 46)),
]


def snap(t, beats, max_shift=0.26):
    """吸附到最近的真拍（≤ max_shift 才吸）"""
    i = int(np.argmin(np.abs(beats - t)))
    return float(beats[i]) if abs(beats[i] - t) <= max_shift else float(t)


def split_line(line, words, n):
    """把一行拆成 n 段：目标 = 等分点，切点取离目标最近的词起点（切在词边界上，长度尽量均匀）"""
    ws = [w for w in words if line["t0"] - 0.05 <= w["t0"] <= line["t1"] + 0.05]
    if n <= 1 or len(ws) < 2:
        return []
    dur = line["t1"] - line["t0"]
    cuts = []
    for k in range(1, n):
        target = line["t0"] + dur * k / n
        cand = [w["t0"] for w in ws[1:] if w["t0"] not in cuts]
        if cand:
            cuts.append(min(cand, key=lambda x: abs(x - target)))
    return sorted(set(cuts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ly = json.loads((MV / "歌词_v4.json").read_text("utf-8"))
    sense = json.loads((MV / "感官.json").read_text("utf-8"))
    beats = np.array(sense["events"]["beats"])[::2]      # 真拍（81 BPM）
    lines = ly["lines"]; words = ly["words"]
    batch = [["add", ".", "剪辑", {"ifMissing": True}],
             ["set", "剪辑", "timeline=true", "beats=感官.json", "fps=24"],
             ["body", "剪辑", "Happier Than Ever 的剪辑真源（Fable，2026-10-24）。切点锚歌词行（歌词_v4.json），吸附到最近真拍（81 BPM）；段落边界换空间；密度 = 情绪曲线：前半每行一镜（4–10 s）→ 过门转折 → 147 s 后每行两镜 → 184 s 后每镜 ≤ 2 s → 最后一句每拍一切 → 黑帧 6 帧 → 长镜头余韵。歌词字按词时间出现（不吸附拍），只用 score ≥ 0.5 且在人声段内的词。"]]
    segs = []
    first = snap(lines[0]["t0"], beats)
    for name, t0, t1, extras in EXTRA:
        segs.append((t0, name, [(n_, s0, (first if name == "00 吸气" else s1), desc, cam, cut) for n_, s0, s1, desc, cam, cut in extras]))
    for name, t0, t1, (l0, l1) in SEGMENTS:
        shots = []
        for li in range(l0, l1 + 1):
            L = lines[li - 1]
            desc, cam, cut = SHOTS[li]
            start = snap(L["t0"], beats)
            nxt = lines[li]["t0"] if li < len(lines) else t1
            end = snap(nxt, beats) if li < l1 else t1
            # 密度：147 s 后每行两镜；184 s 后每镜 ≤ 2 s；最后一句每拍
            if li == 46:
                bts = [b for b in beats if start < b < end]
                pts = [start] + [float(b) for b in bts] + [end]
                for k in range(len(pts) - 1):
                    shots.append((re.sub(r"[,]|->", "", f"L{li:02d}-{k + 1} {L['text'][:18]}").strip(), pts[k], pts[k + 1], desc, cam, cut))
                continue
            if li >= 33:
                n = max(1, math.ceil((end - start) / 2.0))
            elif li >= 25:
                n = 2
            else:
                n = 1
            cuts = [snap(c, beats, 0.19) for c in split_line(L, words, n)]
            pts = [start] + cuts + [end]
            for k in range(len(pts) - 1):
                tag = f"L{li:02d}" + (f"-{k + 1}" if len(pts) > 2 else "")
                shots.append((re.sub(r"[,]|->", "", f"{tag} {L['text'][:22]}").strip(), pts[k], pts[k + 1], desc + (f"（第 {k + 1}/{len(pts) - 1} 镜）" if len(pts) > 2 else ""), cam, cut if k == 0 else "硬切（半行）"))
        segs.append((t0, name, shots))
    segs.sort()
    # 段落边界对齐：上一段最后一镜的结束 = 下一段第一镜的开始（都已吸附到拍）
    for i in range(len(segs) - 1):
        a_, b_ = segs[i][2], segs[i + 1][2]
        if a_ and b_:
            n_, s0, s1, desc, cam, cut = a_[-1]
            a_[-1] = (n_, s0, b_[0][1], desc, cam, cut)
    for _, name, shots in segs:
        sid = f"剪辑/{name}"
        batch += [["add", "剪辑", name, {"ifMissing": True}], ["set", sid, "timeline=true", "beats=感官.json", "fps=24"]]
        for n_, s0, s1, desc, cam, cut in shots:
            nid = f"{sid}/{n_}"
            batch += [["add", sid, n_, {"ifMissing": True}],
                      ["set", nid, f"t0={s0:.3f}", f"t1={s1:.3f}", "track=0"],
                      ["body", nid, f"画面：{desc}\n镜头：{cam}\n切入：{cut}\n切出：{s1:.3f}"]]
    js = json.dumps(batch, ensure_ascii=False, indent=0)
    if a.out:
        Path(a.out).write_text(js, "utf-8"); print(a.out, "shots:", sum(len(s[2]) for s in segs))
    else:
        print(js)


if __name__ == "__main__":
    main()
