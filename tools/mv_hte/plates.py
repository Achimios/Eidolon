#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🎬HTE-底图-Seedream 关键帧表 + 镜头 → 底图映射 + 生成（MV_Happier_Than_Ever，2026-10-24，Fable）。
预算 50 张（Leo 定），每张服务几镜；留 10 张给重出。ledger：MV_Happier_Than_Ever/生成/ledger.jsonl。
角色规范（角色规范.md 的 CHARACTER_SPEC / STYLE_SPEC）原样进每条 prompt。
用法：
  python tools/mv_hte/plates.py list                       # 打印底图表与每镜映射、数张数
  python tools/mv_hte/plates.py gen --ids P01,P02 [--refs 已有图.jpg]   # 生成（flash，1.5K）
  python tools/mv_hte/plates.py assign                     # 把 src=plate:Pxx|裁法 写进 Synapsis 镜头节点
"""
import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
MV = ROOT.parents[1] / "Projekt_Visual" / "MV_Happier_Than_Ever"
GEN = MV / "生成"
LEDGER = GEN / "ledger.jsonl"
CAP = 50
CLI = "C:/Users/Administrator/Desktop/Obsidian_V_Synapsis/core/cli.js"

CHARACTER = ("CHARACTER: a stylized 3D-illustrated young woman, clearly CG (not photoreal), early twenties, pale matte skin with visible "
             "render-like smoothness, deadpan expression, heavy-lidded eyes, bleached white eyebrows, short jet-black bob with blunt straight bangs, "
             "and ONE long thin braid on her left side dyed SIGNAL ORANGE (the only saturated color in the frame), large round wire glasses "
             "with one lens cracked, mismatched earrings (a tiny spoon, a single die), wearing a child-sized pale grey school cardigan that is "
             "clearly TOO SMALL (sleeves end mid-forearm, buttoned one hole off) over a ribbed black turtleneck, bare feet. "
             "Slightly asymmetrical posture, doing something absurd with complete seriousness.")
STYLE = ("STYLE: stylized CG illustration, matte shading, soft single-source lamp light, muted palette, film grain, 16:9 cinematic still, "
         "no text, no logos, no photoreal human faces. Not an anime style, not Pixar; closer to a painted maquette.")
PAL_A = "PALETTE: amber and warm grey, 4 a.m. lamp light, everything desaturated except the orange braid."
PAL_B = "PALETTE: deep teal night, car headlights, cold, everything desaturated except the orange braid."

# id: (半, 场景 prompt, 服务的镜头)
PLATES = {
 "P01": ("A", "SCENE: wide shot, a small tiled bathroom, she sits inside an empty bathtub (no water) with a vintage typewriter on her knees, one desk lamp on the floor, loose sheets of paper on the tiles.", "L01 L21 L43"),
 "P02": ("A", "SCENE: top-down close shot of a vintage typewriter on her knees inside an empty bathtub, a blank sheet in the typewriter, her hands resting on the keys, lamp light from the side.", "L02 L22"),
 "P03": ("A", "SCENE: side profile close-up of her face and shoulder inside the empty bathtub, the orange braid falling over her shoulder, lamp light raking across the face, deadpan.", "L03 L23"),
 "P04": ("A", "SCENE: high-angle shot looking down at the empty bathtub with her small inside it, sheets of a letter spread on the bathroom floor around the tub.", "L04 L24"),
 "P05": ("A", "SCENE: wide shot, a cramped kitchen at 4 a.m., she sits at a small table eating cereal out of a golden sports trophy with a spoon, a phone face down on the table, one hanging lamp.", "L05"),
 "P06": ("A", "SCENE: close-up of a spoon held still in mid-air over a golden trophy full of cereal, milk dripping, her chin and mouth at the top of the frame, kitchen lamp light.", "L06"),
 "P07": ("A", "SCENE: extreme close-up of a vintage typewriter on a kitchen table with a half-typed letter in it, her fingers on two keys, warm lamp light, shallow depth of field.", "L07 L40"),
 "P08": ("A", "SCENE: top-down shot of her hands folding a letter in half on a kitchen table, very precise, a trophy and a bowl at the edge of frame.", "L08"),
 "P09": ("A", "SCENE: wide shot of her standing in front of a refrigerator cutting faces out of a magazine with scissors; several cut-out CG faces already taped to the fridge door.", "L09"),
 "P10": ("A", "SCENE: macro close-up of scissors cutting a stylized illustrated face out of a magazine page, her fingers holding the page.", "L10"),
 "P11": ("A", "SCENE: wide shot, a narrow dim hallway, she stands with her back to the camera facing a closed door, a thin line of light under the door, the orange braid down her back.", "L11"),
 "P12": ("A", "SCENE: close shot of the gap under a closed door with light coming through, the shadow of someone walking past on the other side, dusty floor.", "L12"),
 "P13": ("A", "SCENE: wide night shot of an empty parking lot, one street lamp, one painted empty parking space, she stands beside it talking to the empty space.", "L13 L15"),
 "P14": ("A", "SCENE: extreme close-up of her mouth and chin mid-sentence, deadpan, street lamp light from above, dark background.", "L14 L34"),
 "P16": ("A", "SCENE: wide shot of a kitchen at night, she sits inside an open refrigerator with the door open, the fridge light is the only light source.", "L16 L19 L20"),
 "P17": ("A", "SCENE: close-up of her face lit only by the cold refrigerator light from below, breath visible, shelves of the fridge around her.", "L17"),
 "P18": ("A", "SCENE: close-up of her hands holding a crumpled letter, fridge light, paper creases sharp.", "L18"),
 "P20": ("A", "SCENE: top-down shot of a table where a letter has been torn into small squares arranged in rows by size, a knocked-over golden trophy with cereal spilled, lamp light turning grey.", "E02 E03"),
 "P21": ("B", "SCENE: wide night shot of a parking lot, deep teal, a dark sedan with its headlights just turned on facing the camera, she stands small at the edge of the frame, nobody inside the car visible.", "L25"),
 "P22": ("B", "SCENE: shot from the passenger seat of a car at night looking at the empty driver's seat, rain running down the windshield, dashboard glow, teal.", "L26"),
 "P23": ("B", "SCENE: wide shot, she stands on a wet roadside at night, car headlights from behind her throw her long shadow toward the camera, teal.", "L27"),
 "P24": ("B", "SCENE: close-up of a phone screen in her hand at night showing a long contact list with no names, only grey circles, teal reflections on her glasses.", "L28"),
 "P25": ("B", "SCENE: frontal close-up of her face at night lit by passing headlights, teal, deadpan, the cracked lens catching light.", "L29 L30 L33"),
 "P26": ("B", "SCENE: wide shot, a city rooftop edge at night, she stands with her back to the camera, the orange braid blown straight sideways by wind, city lights below, teal.", "L31 L32"),
 "P27": ("B", "SCENE: wide city skyline at night from a rooftop, no people, teal, scattered windows lit, haze.", "L32"),
 "P30": ("B", "SCENE: a refrigerator door completely covered with cut-out stylized faces, lit by a cold light, no person, teal.", "L35 L44"),
 "P31": ("B", "SCENE: wide shot, a kitchen floor covered with printed words on paper scraps, she pushes a vacuum cleaner across them, cold light, teal.", "L37"),
 "P32": ("B", "SCENE: close shot of her open hand pushed toward the camera, palm filling half the frame, her face out of focus behind, teal.", "L38"),
 "P33": ("B", "SCENE: an empty chair beside a kitchen table at night, cold light, no person, teal.", "L39"),
 "P35": ("B", "SCENE: three empty chairs in a row against a bare wall, cold light, no person, teal.", "L41"),
 "P36": ("B", "SCENE: ultra-wide shot of a huge empty dark room, she is tiny in the far corner sitting on the floor, one cold light, teal.", "L42"),
 "P37": ("B", "SCENE: a wall covered edge to edge with cut-out stylized magazine faces, cold light, teal, no person.", "L44 L45"),
 "P38": ("B", "SCENE: an empty room at night, bare floor, one window with cold light, no person, teal.", "E05 E06"),
 "P39": ("B", "SCENE: close-up of a single orange braid lying on a bare floor, cold teal light, nothing else.", "E07 E08"),
}
# 镜头 → (底图, 裁法 / 视差)。字典键 = Synapsis 节点名前缀（L 行号 / E 编号）
SHOT_PLATE = {
 "E01": ("P01", "黑 → 整张慢慢亮起，只取左 2/3"), "L01": ("P01", "全幅，极慢横移"), "L02": ("P02", "全幅缓推"), "L03": ("P03", "全幅"),
 "L04": ("P04", "全幅缓降"), "L05": ("P05", "全幅缓推"), "L06": ("P06", "全幅"), "L07": ("P07", "全幅缓推"), "L08": ("P08", "全幅"),
 "L09": ("P09", "全幅缓推"), "L10": ("P10", "全幅"), "L11": ("P11", "全幅极慢推"), "L12": ("P12", "全幅"), "L13": ("P13", "全幅缓慢环绕（视差）"),
 "L14": ("P14", "全幅"), "L15": ("P13", "裁上半 1/2 当俯拍，缓升"), "L16": ("P16", "全幅缓推"), "L17": ("P17", "全幅"), "L18": ("P18", "全幅缓推"),
 "L19": ("P16", "压暗 60%，只留冰箱灯"), "L20": ("P17", "裁眼睛以上 + 抬头"), "L21": ("P01", "= L01，更暗，横移反向"), "L22": ("P02", "= L02"),
 "L23": ("P03", "= L03"), "L24": ("P04", "= L04，色温褪灰"), "E02": ("P20", "全幅缓升"), "E03": ("P20", "方块长满"),
 "L25": ("P21", "两镜：全幅 → 裁车灯"), "L26": ("P22", "两镜：全幅 → 裁雨窗"), "L27": ("P23", "两镜：全幅 → 裁影子"), "L28": ("P24", "两镜：全幅 → 裁屏幕"),
 "L29": ("P25", "两镜：正面 → 镜像反打"), "L30": ("P25", "两镜：更近 → 更近"), "L31": ("P26", "两镜：全幅 → 裁辫子"), "L32": ("P27", "两镜：天际线 → 回到 P26"),
 "L33": ("P25", "裁眼睛"), "L34": ("P14", "青色重调 + 裁嘴"), "L35": ("P30", "全幅缓推"), "L36": ("P01", "浴缸被方块吃到一半"),
 "L37": ("P31", "全幅横移"), "L38": ("P32", "快推再停"), "L39": ("P33", "全幅缓推"), "L40": ("P07", "青色重调，裁空行"), "L41": ("P35", "横移"),
 "L42": ("P36", "缓慢拉远"), "L43": ("P01", "被方块吃掉"), "L44": ("P37", "缓推"), "L45": ("P01 P05 P11 P13 P26", "快切五场景"),
 "L46": ("P01 P05 P11 P13 P26", "每拍一切，最后 alone 单独一镜 = P26 裁辫子"), "E04": ("—", "黑帧"), "E05": ("P38", "五场景方块叠 → 吐出空房间"),
 "E06": ("P38", "全幅"), "E07": ("P39", "缓推"), "E08": ("P39", "青褪黑"),
}
V1_IDS = ["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P33", "P35", "P36", "P37", "P38", "P26"]   # 第一版两段对照样片要的


def used():
    if not LEDGER.exists():
        return 0
    return sum(json.loads(l)["n"] for l in LEDGER.read_text("utf-8").splitlines() if l.strip())


def log(n, note, files):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"t": datetime.datetime.now().isoformat(timespec="seconds"), "line": "hte", "n": n, "note": note,
                            "files": [str(x) for x in files]}, ensure_ascii=False) + "\n")


def prompt_for(pid):
    half, scene, _ = PLATES[pid]
    return f"{scene} {CHARACTER if 'no person' not in scene else ''} {STYLE} {PAL_A if half == 'A' else PAL_B}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["list", "gen", "assign"])
    ap.add_argument("--ids", default=None)
    ap.add_argument("--refs", default=None, help="参考图（逗号分隔），用于角色一致性")
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    if a.mode == "list":
        print(f"底图 {len(PLATES)} 张（预算 {CAP}，已用 {used()}）；第一版样片要 {len(V1_IDS)} 张：{' '.join(V1_IDS)}")
        for k, (h, sc, serves) in PLATES.items():
            print(f"{k} [{h}] 服务 {serves:16s} {sc[:70]}…")
        return
    if a.mode == "assign":
        batch = []
        import re
        for page in subprocess.run(["node", CLI, "timeline", "--json", "--ws", str(MV)], capture_output=True, text=True, encoding="utf-8").stdout.splitlines():
            pass
        tl = json.loads(subprocess.run(["node", CLI, "timeline", "--json", "--ws", str(MV)], capture_output=True, text=True, encoding="utf-8").stdout)
        for pg in tl:
            if pg["page"] == "剪辑":
                continue
            items = json.loads(subprocess.run(["node", CLI, "timeline", pg["page"], "--json", "--ws", str(MV)], capture_output=True, text=True, encoding="utf-8").stdout)["items"]
            for it in items:
                key = it["label"].split(" ")[0].split("-")[0]
                if key in SHOT_PLATE:
                    pid, how = SHOT_PLATE[key]
                    batch.append(["set", it["id"], f"src=plate:{pid}|{how}"])
        tmp = GEN / "_assign_batch.json"; GEN.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(batch, ensure_ascii=False), "utf-8")
        r = subprocess.run(["node", CLI, "batch", str(tmp), "--ws", str(MV)], capture_output=True, text=True, encoding="utf-8")
        print(r.stdout[-300:], r.stderr[-300:]); print("assigned", len(batch)); return
    import ark
    ids = a.ids.split(",") if a.ids else V1_IDS
    if used() + len(ids) > CAP:
        sys.exit(f"额度：已用 {used()}/{CAP}，这次要 {len(ids)}，停。")
    GEN.mkdir(parents=True, exist_ok=True)
    refs = a.refs.split(",") if a.refs else None
    files = []
    for pid in ids:
        out = GEN / f"{pid}{a.suffix}.jpg"
        if out.exists():
            print("已存在，跳过（换 --suffix）：", out); continue
        pr = prompt_for(pid)
        if refs:
            pr = "Keep the same character as in the reference image (图1). " + pr
        p = ark.image(pr, out, mdl="seedream_flash", size="1.5K", refs=refs)
        files.append(p); print(p, flush=True)
    log(len(files), f"plates {','.join(ids)}{' refs' if refs else ''}", files)


if __name__ == "__main__":
    main()
