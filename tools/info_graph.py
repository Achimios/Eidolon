#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🕸信息图-入口-节点 / 边 / 坐标 / 标签的信息可视化（参考 `_参考库/信息可视化_节点坐标.gif`、`AI Strategy.mp4`；
拆解在 `Projekt_Visual/_参考库/分析_信息可视化/分析.md`）。⭐符号说真话：每个节点、每条边、每个数字都来自真实数据。

两个版本：
  memory  A：《Can't Kill Us》Thymos 的「记忆图谱」—— 06「一个名字」（115.51–128.31 s）她在记忆里找 John。
          节点 = 剪辑表_v4_0-97 的每一个镜头（到 06 为止她「活过」的东西）；
          节点特征 = 该镜头时间段上 感官.json 五通道的平均（心跳 β / 威胁 θ / 饥饿 η / 张力 τ）+ 起音密度 ρ（个/秒），z 标准化；
          位置 = 特征的 PCA 前两维（坐标轴写出真实载荷），再做最多 N 步的防重叠推开（推开量写进 .json）；
          边 = ①故事线（剪辑顺序 i→i+1，实线）②情绪近邻（特征空间 kNN，虚弧）③同一素材再次出现（点线，= 回想）；
          搜索 = 从「此刻」（06 段自己的特征向量，同样 z 标准化）出发的 Dijkstra（边权 = 特征欧氏距离），
          每个威胁 ≥ thr 的真实起音弹出一个节点（拍点作用在节点 / 边上，不推全局）；到 John（02 走廊背影）后，
          路径红色回溯，后面的强起音沿路径各跑一次脉冲。CONFIDENCE = (1 + cos(此刻, John)) / 2（真算）。
  track   B：独立版 —— 对一段真实视频做亮斑追踪（阈值 → 连通域 → 逐帧最近邻关联），
          节点 = 连通域（框 = 外接框，框里 = 该连通域的真遮罩 / 源图放大），标签 = 真实像素质心 + 面积，
          边 = 当前帧质心的欧氏最小生成树（Prim，箭头 = 从最大连通域出发的父 → 子方向），轨迹 = 追踪 ID 的历史。

用法（仓库 .venv）：
  python tools/info_graph.py memory [--palette bwr|acid] [--name 22_信息图_记忆图谱_一个名字_v1] [--still 6.0]
  python tools/info_graph.py track --src <视频> --start <歌曲时间> [--palette ...] [--name ...] [--v 3]
  例（B 当前版）：track --v 3 --src MV/素材/全息女性视频运镜描述_裁肩_v2.mp4 --start 45.12 --levels 0.45,0.65,0.85
                  --min-area 40,10,3 --name 22_信息图_成分树_她的脸_v3
  --v（版式版本，旧版可复现）：memory 1 = 首版 / 2 = John 大框常驻 + 右上 z 向量对照 / 3（默认）= 注意力分层标签 + 最小字号 + 开场已在长 /
                               4 = 三段：搜 + 心魔被压下 → 切进局部躁动 → 四声锁定 → 冲进 John 接粒子海 v4（`memory --v 4`，到 2:08.36）；
                               track 1 = 遮罩框盖原位 / 2 = 标本栏 + 直方图 / 3 = 标本栏滞回 + 放大镜优先新生高光
输出：视频（配乐）+ 同名 .json（节点 / 边 / 搜索顺序 / 每帧相机）+ _接触表.jpg。不覆盖已有文件（换 --name）。
人读版：-给人类的解释/信息可视化怎么做的.md
"""
import argparse
import heapq
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]
PROJ = ROOT.parents[1] / "Projekt_Visual"
MV = PROJ / "MV_Cant_Kill_Us"
MUSIC = PROJ / "Assets" / "The Glitch Mob - Can't Kill Us [2020 Edit]_190189282.mp4"
SENSE = MV / "感官.json"
OUT_A = MV / "样片" / "信息可视化"
OUT_B = PROJ / "MotionDesign" / "信息可视化"
CACHE = MV / "cache" / "info_graph"
FPS = 24
W, H = 1920, 1080
SS = 2                                   # 超采样（PIL 线条没有抗锯齿 → 2× 画再缩）
FMONO = "C:/Windows/Fonts/consola.ttf"
FCN = "C:/Windows/Fonts/msyh.ttc"
CH = ["heartbeat", "threat", "hunger", "tension"]
SYM = {"heartbeat": "β", "threat": "θ", "hunger": "η", "tension": "τ", "rho": "ρ"}


def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


# ───────────────────────── 配色 ─────────────────────────
PALETTES = {
    # 黑白红：白 / 灰的层级，红只给「John + 找到的路径」（物件.md：一粒红点）
    "bwr": dict(bg=(4, 4, 5), ink=(236, 236, 232), mid=(150, 150, 148), dim=(78, 78, 80), faint=(38, 38, 40),
                accent=(236, 236, 232), hot=(232, 36, 30), thumb="gray", box=(58, 58, 60)),
    # 酸性点缀：暖灰底 + 酸黄绿给「正在搜索」的前沿，红仍只给 John / 路径
    "acid": dict(bg=(7, 7, 6), ink=(232, 228, 214), mid=(146, 142, 128), dim=(80, 78, 70), faint=(40, 39, 35),
                 accent=(206, 255, 46), hot=(255, 48, 72), thumb="desat", box=(52, 52, 46)),
}


def mix(c, a, bg=(0, 0, 0)):
    """颜色 c 以不透明度 a 盖在 bg 上（PIL 直接画不混合 → 预乘）"""
    return tuple(int(round(bg[i] + (c[i] - bg[i]) * a)) for i in range(3))


_FONTS = {}


def font(size, cn=False):
    k = (size, cn)
    if k not in _FONTS:
        _FONTS[k] = ImageFont.truetype(FCN if cn else FMONO, int(round(size * SS)))
    return _FONTS[k]


# ───────────────────────── 几何：虚线二次贝塞尔 + 箭头 ─────────────────────────
def bezier(p0, p1, curv, n=48):
    """二次贝塞尔：控制点 = 中点 + 法向 × curv × 弦长（curv 正负 = 弯向哪边）。返回 (n,2) 点和累计弧长。"""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = float(np.hypot(*d)) + 1e-9
    nrm = np.array([-d[1], d[0]]) / L
    c = (p0 + p1) / 2 + nrm * curv * L
    u = np.linspace(0, 1, n)[:, None]
    pts = (1 - u) ** 2 * p0 + 2 * (1 - u) * u * c + u ** 2 * p1
    seg = np.hypot(*np.diff(pts, axis=0).T)
    return pts, np.concatenate([[0], np.cumsum(seg)])


def draw_path(dr, pts, s, col, width=1, dash=None, upto=1.0, phase=0.0):
    """沿折线画（可虚线）；upto ∈ [0,1] = 只画到弧长的这一比例（边「长出来」）。dash = (on, off) 像素（1× 单位）。"""
    if upto <= 0:
        return
    Ltot = s[-1] * upto
    w = max(1, int(round(width * SS)))
    if dash is None:
        k = int(np.searchsorted(s, Ltot))
        P = [tuple(p) for p in pts[:k]]
        if k < len(pts):
            P.append(tuple(_interp(pts, s, Ltot)))
        if len(P) >= 2:
            dr.line(P, fill=col, width=w)
        return
    on, off = dash[0] * SS, dash[1] * SS
    per = on + off
    a = -((phase * SS) % per)
    while a < Ltot:
        b0, b1 = max(a, 0), min(a + on, Ltot)
        if b1 > b0:
            dr.line([tuple(_interp(pts, s, b0)), tuple(_interp(pts, s, b1))], fill=col, width=w)
        a += per


def _interp(pts, s, l):
    i = int(np.clip(np.searchsorted(s, l) - 1, 0, len(s) - 2))
    f = (l - s[i]) / (s[i + 1] - s[i] + 1e-9)
    return pts[i] + (pts[i + 1] - pts[i]) * f


def draw_arrow(dr, pts, s, at, col, size=5.0):
    """在弧长比例 at 处画一个沿切向的小 V 形箭头"""
    l = s[-1] * at
    p = _interp(pts, s, l)
    q = _interp(pts, s, max(0, l - 2 * SS))
    d = p - q
    n = np.hypot(*d) + 1e-9
    d = d / n
    o = np.array([-d[1], d[0]])
    a = size * SS
    dr.line([tuple(p - d * a + o * a * 0.6), tuple(p), tuple(p - d * a - o * a * 0.6)], fill=col, width=max(1, SS))


_MINF = {"on": False}
READ_MIN, READ_MIN_DIM = 11, 12     # 1080p 实测（scratchpad fonts.png）：Consolas 8–9 px 读不清；10 px 只有亮色勉强；暗色要 ≥ 12


def text(dr, xy, s, col, size=11, cn=False, anchor="la", knock=None):
    """knock = 底色：先在字的外框里铺一块底色（字压在边上时把边挖掉 → 标签永远可读）。
    _MINF on（memory v3 起）：字号至少 READ_MIN；调用方负责暗色字用 ≥ READ_MIN_DIM。"""
    if _MINF["on"]:
        size = max(size, READ_MIN)
        if sum(col) / 3 < 100:                            # 暗色字：再大一号，否则读不清
            size = max(size, READ_MIN_DIM)
    if knock is not None:
        x0, y0, x1, y1 = dr.textbbox((xy[0], xy[1]), s, font=font(size, cn), anchor=anchor)
        pad = 2 * SS
        dr.rectangle([x0 - pad, y0 - pad, x1 + pad, y1 + pad], fill=knock)
    dr.text((xy[0], xy[1]), s, fill=col, font=font(size, cn), anchor=anchor)


# ───────────────────────── 相机 ─────────────────────────
def spring_follow(targets, x0, v0, omega, dt=1 / FPS):
    """临界阻尼弹簧跟随（不会过冲、不会停在静止 —— 目标在动它就在动）。targets: (T, D)。"""
    x, v = np.array(x0, float), np.array(v0, float)
    out = []
    for tg in targets:
        a = omega * omega * (tg - x) - 2 * omega * v
        v = v + a * dt
        x = x + v * dt
        out.append(x.copy())
    return np.array(out)


# ───────────────────────── 感官 ─────────────────────────
def load_sense():
    d = json.loads(SENSE.read_text("utf-8"))
    fps = d["fps"]
    ch = {k: np.asarray(d["channels"][k], float) for k in CH}
    ev = {k: np.asarray(v, float) for k, v in d["events"].items()}
    return d, fps, ch, ev


def onset_strength(ch, fps, t):
    """起音处的威胁峰：±3 帧取最大（按帧直接取会落在冲击之前 —— 方法_剪辑与情绪.md §一）"""
    i = int(round(t * fps))
    th = ch["threat"]
    return float(th[max(0, i - 3):i + 4].max())


def span_features(ch, fps, onsets, t0, t1):
    a, b = int(round(t0 * fps)), max(int(round(t0 * fps)) + 1, int(round(t1 * fps)))
    f = {k: float(ch[k][a:b].mean()) for k in CH}
    f["rho"] = float(((onsets >= t0) & (onsets < t1)).sum() / max(t1 - t0, 1e-6))
    return f


# ───────────────────────── A：记忆图谱 ─────────────────────────
def short_name(src):
    if src.startswith("原片@"):
        return "原片 " + src.split("@")[1][:4]
    if src.startswith("图:"):
        st = Path(src[2:]).stem
        return ("记忆 " + st) if "记忆帧" in src else ("静帧 " + "_".join(st.split("_")[:2]))
    path = src.split(":", 1)[1].rsplit("@", 1)[0]
    st = Path(path).stem.split("_")
    return " ".join(st[:2])


def thumb(src, play_t, start, speed, size=320):
    """镜头在 play_t 那一刻的真实画面（缓存）"""
    import edl
    CACHE.mkdir(parents=True, exist_ok=True)
    key = f"{abs(hash((src, round(play_t, 2)))) % 10**10}.jpg"
    p = CACHE / key
    if p.exists():
        return Image.open(p).convert("RGB")
    vid, at, im = edl.resolve(src)
    if im is None:
        t = at + (play_t - start) * speed
        r = subprocess.run([ffmpeg(), "-loglevel", "error", "-ss", f"{t:.3f}", "-i", str(vid), "-frames:v", "1",
                            "-vf", f"scale={size}:-2", "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True)
        from io import BytesIO
        im = Image.open(BytesIO(r.stdout)).convert("RGB")
    im = im.copy()
    im.thumbnail((size, size))
    im.save(p, quality=90)
    return im


def pca2(X):
    mu = X.mean(0)
    U, S_, Vt = np.linalg.svd(X - mu, full_matrices=False)
    Vt = Vt[:2] * np.sign(Vt[:2, :1] + 1e-12)        # 符号定死（第一个分量为正）→ 可复现
    return (X - mu) @ Vt.T, Vt, S_ ** 2 / (S_ ** 2).sum()


def relax(P, rad, iters=200):
    """防重叠：两点距离 < rad 就沿连线各推开一半，最多 iters 轮。返回新坐标和每点推开量（像素）。"""
    Q = P.copy()
    for _ in range(iters):
        d = Q[:, None, :] - Q[None, :, :]
        r = np.hypot(d[..., 0], d[..., 1]) + np.eye(len(Q)) * 1e9
        m = r < rad
        if not m.any():
            break
        push = ((rad - r) * m / r)[..., None] * d * 0.5
        Q += push.sum(1)
    return Q, np.hypot(*(Q - P).T)


def dijkstra_order(n, adj, src):
    """返回弹出顺序 [(节点, 距离, 父)]（标准 Dijkstra，边权非负）"""
    dist = {src: 0.0}
    par = {src: -1}
    pq = [(0.0, src)]
    done, order = set(), []
    while pq:
        d, u = heapq.heappop(pq)
        if u in done:
            continue
        done.add(u)
        order.append((u, d, par[u]))
        for v, w in adj[u]:
            nd = d + w
            if v not in dist or nd < dist[v] - 1e-12:
                dist[v], par[v] = nd, u
                heapq.heappush(pq, (nd, v))
    return order


def read_notes(path):
    """剪辑表每行 # 后面的备注（edl.read_edl 丢掉了）→ {开始秒: 短备注}"""
    import edl
    out = {}
    for raw in Path(path).read_text("utf-8").splitlines():
        s_ = raw.strip()
        if not s_ or s_.startswith("#") or "#" not in s_:
            continue
        t = edl.parse_time(s_.split()[0])
        note = s_.split("#", 1)[1].strip().lstrip("⭐").strip()
        for sep in ("：", "（", "(", "，", "；"):
            note = note.split(sep)[0]
        out[round(t, 2)] = note.strip()[:14]
    return out


@dataclass
class Memory:
    nodes: list            # dict: i, t0, t1, src, name, feat, z, pos, push, thumb
    edges: list            # (a, b, kind, w)
    order: list            # Dijkstra 弹出顺序
    path: list             # src → john 的节点序列
    john: int
    now: int
    pcs: np.ndarray
    evr: np.ndarray
    conf: float
    keys: list
    scale: float = 1.0          # 世界像素 / PCA 单位
    origin: tuple = (0.0, 0.0)  # PCA 原点在世界坐标里的位置


def build_memory(edl_path=None, seg=(115.51, 128.31), k=3, span=(2400, 1500)):
    import edl
    ep = Path(edl_path or MV / "剪辑表_v4_0-97.txt")
    rows = edl.read_edl(ep)
    notes = read_notes(ep)
    _, fps, ch, ev = load_sense()
    on = ev["onsets"]
    nodes = []
    for (t0, src, sp), (t1, _, _) in zip(rows[:-1], rows[1:]):
        if src == "黑":
            continue
        f = span_features(ch, fps, on, t0, t1)
        nodes.append(dict(t0=t0, t1=t1, src=src, speed=sp, name=short_name(src), note=notes.get(round(t0, 2), ""), feat=f))
    nodes.append(dict(t0=seg[0], t1=seg[1], src="此刻", speed=1, name="此刻 · 06 一个名字",
                      feat=span_features(ch, fps, on, *seg)))
    keys = CH + ["rho"]
    X = np.array([[n["feat"][k_] for k_ in keys] for n in nodes])
    mu, sd = X[:-1].mean(0), X[:-1].std(0) + 1e-9     # 只用记忆（不含「此刻」）定标准化 → 此刻是被投进记忆空间的
    Z = (X - mu) / sd
    P, Vt, evr = pca2(Z[:-1])
    Pn = (Z[-1:] - Z[:-1].mean(0)) @ Vt.T
    P = np.vstack([P, Pn])
    sc = np.array(span) / (P.max(0) - P.min(0) + 1e-9)
    s = float(sc.min())
    Pw = (P - P.mean(0)) * s
    Pw2, push = relax(Pw, 120.0)
    n = len(nodes)
    D = np.linalg.norm(Z[:, None] - Z[None], axis=2)
    E = {}
    for i in range(n):
        for j in np.argsort(D[i])[1:k + 1]:
            a, b = min(i, j), max(i, j)
            E.setdefault((a, b), "knn")
    for i in range(n - 2):                       # 故事线（「此刻」不在剪辑表里）
        E[(i, i + 1)] = "story"
    last = {}                                   # 回想：同一素材再出现 → 连到它上一次出现（链，不是两两全连）
    for i in range(n - 1):
        key = nodes[i]["src"].rsplit("@", 1)[0]
        if key in last and (last[key], i) not in E:
            E[(last[key], i)] = "same"
        last[key] = i
    edges = [(a, b, kd, float(D[a, b])) for (a, b), kd in sorted(E.items())]
    adj = [[] for _ in range(n)]
    for a, b, _, w in edges:
        adj[a].append((b, w))
        adj[b].append((a, w))
    now = n - 1
    john = next(i for i, nd in enumerate(nodes) if "走廊背影" in nd["src"])
    order = dijkstra_order(n, adj, now)
    par = {u: p for u, _, p in order}
    path = [john]
    while path[-1] != now:
        path.append(par[path[-1]])
    path = path[::-1]
    conf = float((1 + Z[now] @ Z[john] / (np.linalg.norm(Z[now]) * np.linalg.norm(Z[john]) + 1e-12)) / 2)
    for i, nd in enumerate(nodes):
        nd.update(i=i, z=Z[i].tolist(), pca=P[i].tolist(), pos=Pw2[i].tolist(), push=float(push[i]))
    org = tuple((-P.mean(0) * s).tolist())
    return Memory(nodes, edges, order, path, john, now, Vt, evr, conf, keys, s, org)


def schedule(mem, seg, thr=0.3, t_first=0.0, pulse_thr=0.5):
    """把 Dijkstra 的弹出顺序排到真实起音上：每个威胁峰 ≥ thr 的起音弹一个节点（第一个 = 此刻）。
    找到 John 之后：路径按节点逐个变红（每个后续强起音一格），之后的强起音沿路径跑脉冲。"""
    _, fps, ch, ev = load_sense()
    ons = [(float(t), onset_strength(ch, fps, t)) for t in ev["onsets"] if seg[0] <= t < seg[1]]
    hits = [(t - seg[0], s) for t, s in ons if s >= thr and t - seg[0] >= t_first]
    pop_t = {}
    k = 0
    jk = next(i for i, (u, _, _) in enumerate(mem.order) if u == mem.john)
    for t, s in hits:
        if k <= jk:
            pop_t[mem.order[k][0]] = (t, s)
            k += 1
        else:
            break
    rest = hits[k:]
    path_t = []                                 # 路径变红：从 John 往回（她「顺着线索回想到自己」）→ 用一个起音整条回溯
    pulses = []
    if rest:
        path_t = rest[0]
        last = path_t[0]
        for tt, ss in rest[1:]:                 # 脉冲只给强起音（威胁峰 ≥ pulse_thr），且间隔 ≥ 0.3 s → 不每拍都跑
            if ss >= pulse_thr and tt - last >= 0.3:
                pulses.append((tt, ss))
                last = tt
    return dict(pop=pop_t, path=path_t, pulses=pulses, hits=hits, all_onsets=ons, jk=jk)


def ease_pop(dt, tau=0.12):
    """事件之后的强度：满幅 2 帧，然后 exp 衰减（冲击 → 衰减）"""
    if dt < 0:
        return 0.0
    if dt < 2 / FPS:
        return 1.0
    return math.exp(-(dt - 2 / FPS) / tau)


def render_memory(mem, sch, seg, pal, n_frames=None, only=None, zoom_in=1.65, omega=3.4, v=3):
    john_big = v >= 2
    _MINF["on"] = v >= 3
    nbr = {i: set() for i in range(len(mem.nodes))}
    for a_, b_, _, _ in mem.edges:
        nbr[a_].add(b_)
        nbr[b_].add(a_)
    P = PALETTES[pal]
    dur = seg[1] - seg[0]
    T = n_frames or int(round(dur * FPS))
    pos = np.array([nd["pos"] for nd in mem.nodes])
    n = len(pos)
    thumbs = {}
    for nd in mem.nodes:
        if nd["src"] == "此刻":
            continue
        mid = (nd["t0"] + nd["t1"]) / 2
        im = thumb(nd["src"], mid, nd["t0"], nd["speed"])
        if P["thumb"] == "gray":
            im = im.convert("L").convert("RGB")
        else:
            g = np.asarray(im.convert("L").convert("RGB"), float)
            im = Image.fromarray(np.clip(g * 0.55 + np.asarray(im, float) * 0.45, 0, 255).astype(np.uint8))
        thumbs[nd["i"]] = im

    # ── 相机：开场已经在高速平移（从故事起点那一侧扫向「此刻」），先快后慢；之后弹簧跟住最新弹出的节点；
    #    找到 John 后拉远到能装下整条路径。整段不停：目标一直在变。
    pop = sch["pop"]
    t_path = sch["path"][0] if sch["path"] else dur
    targets, zooms = [], []
    newest = mem.now
    pop_sorted = sorted(pop.items(), key=lambda kv: kv[1][0])
    path_pos = pos[mem.path]
    pc = (path_pos.min(0) + path_pos.max(0)) / 2
    pz = min((W * 0.62) / (np.ptp(path_pos[:, 0]) + 300), (H * 0.62) / (np.ptp(path_pos[:, 1]) + 260))
    for f in range(T):
        t = f / FPS
        for u, (tu, _) in pop_sorted:
            if tu <= t:
                newest = u
        vis = [u for u, (tu, _) in pop_sorted if tu <= t]
        c_vis = pos[vis].mean(0) if vis else pos[mem.now]
        tg = 0.8 * pos[newest] + 0.2 * c_vis
        z = zoom_in
        if t >= t_path:
            a = min(1.0, (t - t_path) / 2.5)
            a = a * a * (3 - 2 * a)
            tg = (1 - a) * tg + a * pc
            z = (1 - a) * z + a * min(pz, 1.0)
        targets.append(tg)
        zooms.append(z)
    targets = np.array(targets)
    start = pos[0] if v < 2 else pos[mem.now] + (pos[0] - pos[mem.now]) * (0.55 if v < 3 else 0.32)   # 开场：从故事起点（0:00）那一侧高速扫向「此刻」
    v0 = (pos[mem.now] - start) * (3.0 if v < 2 else 4.2)
    cam = spring_follow(targets, start, v0, omega=omega)        # 每弹出一个节点目标跳一次 → 镜头「进」一下再「缓」
    zs = spring_follow(np.array(zooms)[:, None], [zoom_in * 0.8], [0.25], omega=1.8)[:, 0]

    _, fps_s, ch, ev = load_sense()
    tl0, tl1 = 0.0, seg[1]
    strip = dict(x0=150, x1=W - 150, y0=H - 118, y1=H - 46)

    frames_meta = []
    for f in range(T):
        if only is not None and f not in only:
            continue
        t = f / FPS
        img = Image.new("RGB", (W * SS, H * SS), P["bg"])
        dr = ImageDraw.Draw(img)
        z = float(zs[f])
        cx, cy = cam[f]

        def S2(p):
            return np.array([(W / 2 + (p[0] - cx) * z) * SS, (H / 2 + (p[1] - cy) * z) * SS])

        # 坐标系：PCA 轴（过 PCA 原点）+ 每 1 个单位一个刻度（单位 = 标准化后的 1σ）+ 载荷
        ox, oy = mem.origin
        s_world = mem.scale
        ax_col = P["faint"]
        x_axis = [S2((ox - 4.6 * s_world, oy)), S2((ox + 4.6 * s_world, oy))]
        y_axis = [S2((ox, oy - 3.2 * s_world)), S2((ox, oy + 3.2 * s_world))]
        dr.line([tuple(x_axis[0]), tuple(x_axis[1])], fill=ax_col, width=SS)
        dr.line([tuple(y_axis[0]), tuple(y_axis[1])], fill=ax_col, width=SS)
        for kk in range(-4, 5):
            px = S2((ox + kk * s_world, oy))
            py = S2((ox, oy + kk * s_world))
            dr.line([(px[0], px[1] - 4 * SS), (px[0], px[1] + 4 * SS)], fill=P["dim"], width=SS)
            if abs(kk) <= 3:
                dr.line([(py[0] - 4 * SS, py[1]), (py[0] + 4 * SS, py[1])], fill=P["dim"], width=SS)
            if kk:
                text(dr, (px[0] + 3 * SS, px[1] + 6 * SS), f"{kk:+d}", P["dim"], 9)
                if abs(kk) <= 3:
                    text(dr, (py[0] + 6 * SS, py[1] - 5 * SS), f"{kk:+d}", P["dim"], 9)
        ld = mem.pcs
        lx = "PC1 " + " ".join(f"{ld[0, i]:+.2f}{SYM[k_]}" for i, k_ in enumerate(mem.keys)) + f"  ({mem.evr[0] * 100:.0f}% var)"
        ly = "PC2 " + " ".join(f"{ld[1, i]:+.2f}{SYM[k_]}" for i, k_ in enumerate(mem.keys)) + f"  ({mem.evr[1] * 100:.0f}% var)"
        text(dr, (x_axis[1][0] - 8 * SS, x_axis[1][1] - 16 * SS), lx, P["dim"], 10, anchor="ra")
        text(dr, (y_axis[0][0] + 8 * SS, y_axis[0][1] + 14 * SS), ly, P["dim"], 10)

        popped = {u: tu for u, (tu, _) in pop.items() if tu <= t}
        red_on = sch["path"] and t >= sch["path"][0]
        path_edges = set()
        for a, b in zip(mem.path[:-1], mem.path[1:]):
            path_edges.add((min(a, b), max(a, b)))
        par = {u: p for u, _, p in mem.order}

        # 边
        for a, b, kind, w in mem.edges:
            pa, pb = S2(pos[a]), S2(pos[b])
            curv = 0.0 if kind == "story" else (0.22 if (a + b) % 2 else -0.22)
            if kind == "same":
                curv = 0.35
            pts, s = bezier(pa, pb, curv)
            seen = a in popped or b in popped
            if kind == "story":
                if v >= 3:                                  # v3 开场：故事线从 0:00 那一侧一路「长」过来，第一帧已经长到三分之一
                    ta = -0.35 + 1.0 * a / max(1, len(mem.nodes) - 2)
                    gs = min(1.0, max(0.0, (t - ta) / 0.25))
                    draw_path(dr, pts, s, P["dim"] if gs < 1 else P["faint"], 1, upto=gs)
                else:
                    draw_path(dr, pts, s, P["faint"], 1)
            elif kind == "same":
                draw_path(dr, pts, s, P["mid"] if seen else P["dim"], 1, dash=(1.2, 3.0))
            else:
                draw_path(dr, pts, s, P["dim"] if seen else P["faint"], 1, dash=(4, 5), phase=t * 9)
                if v >= 3 and mem.now in (a, b) and t < 1.2:      # v3 开场：「此刻」的情绪近邻边第一帧就已在往外长（搜索在预热）
                    g0 = min(1.0, (t + 0.3) / 0.6)
                    if mem.now == b:
                        pts, s = pts[::-1], np.concatenate([[0], np.cumsum(np.hypot(*np.diff(pts[::-1], axis=0).T))])
                    draw_path(dr, pts, s, mix(P["accent"], 0.8 * (1 - max(0.0, t - 0.6) / 0.6), P["bg"]), 1.4, dash=(6, 4),
                              upto=g0, phase=-t * 14)
        # 搜索树的边：父 → 子，在弹出那一刻「长出来」（0.25 s），箭头跑在生长前沿
        for u, tu in popped.items():
            p = par.get(u, -1)
            if p < 0:
                continue
            g = min(1.0, (t - tu) / 0.25)
            g = 1 - (1 - g) ** 3
            pa, pb = S2(pos[p]), S2(pos[u])
            pts, s = bezier(pa, pb, 0.18)
            on_p = (min(p, u), max(p, u)) in path_edges and red_on
            fade = math.exp(-max(0.0, t - tu - 0.4) / 1.2)          # 新边亮，旧边退成暗的历史（层级 = 时间）
            col = mix(P["accent"], 0.35 + 0.65 * fade, P["bg"])
            draw_path(dr, pts, s, col, 1.0 + 0.6 * fade, dash=(6, 4), upto=g, phase=-t * 14)
            if fade > 0.05 or g < 1:
                draw_arrow(dr, pts, s, g if g < 1 else 0.55, col, 5)
            _ = on_p
        # 找到 John：路径从「此刻」沿搜索树长到 John（0.5 s，按弧长），红色实线
        if red_on:
            curves = [bezier(S2(pos[a_]), S2(pos[b_]), 0.18) for a_, b_ in zip(mem.path[:-1], mem.path[1:])]
            Ls = np.array([c[1][-1] for c in curves])
            tot = Ls.sum()
            grow = min(1.0, (t - sch["path"][0]) / 0.5) * tot
            acc = 0.0
            for (pts, s_), L in zip(curves, Ls):
                if grow > acc:
                    draw_path(dr, pts, s_, P["hot"], 2.2, upto=min(1.0, (grow - acc) / L))
                acc += L
            # 路径脉冲：之后每个强起音，一个红点沿同一条曲线从此刻跑到 John（0.6 s）
            for tp, sp in sch["pulses"]:
                u_ = (t - tp) / 0.6
                if 0 <= u_ <= 1:
                    l = u_ * tot
                    k_ = int(np.searchsorted(np.cumsum(Ls), l))
                    k_ = min(k_, len(curves) - 1)
                    q = _interp(curves[k_][0], curves[k_][1], l - (np.cumsum(Ls)[k_ - 1] if k_ else 0.0))
                    r = (3 + 4 * sp) * SS
                    dr.ellipse([q[0] - r, q[1] - r, q[0] + r, q[1] + r], fill=P["hot"])

        # 节点
        for nd in mem.nodes:
            i = nd["i"]
            p = S2(pos[i])
            vis = i in popped
            e = ease_pop(t - popped[i]) if vis else 0.0
            is_j, is_now = i == mem.john, i == mem.now
            col = P["hot"] if (is_j and (vis or red_on)) else (P["ink"] if vis else P["mid"])
            r = 3.5 * SS
            half = r
            if vis and i in thumbs:
                im = thumbs[i]
                big = (64 + 56 * min(1.0, (nd["t1"] - nd["t0"]) / 6.4))       # 框大小 ∝ 镜头时长（满 = 一个乐句）
                age = t - popped[i]
                g = min(1.0, age / (3 / FPS))                                    # 3 帧弹开
                k_ = math.exp(-max(0.0, age - 0.6) / 0.9)                        # 0.6 s 后缩回常驻小框（只留最近几个大的）
                if is_j and john_big:                                            # v2：找到的那个不缩回 —— 全段唯一一个大框
                    k_ = 1.0
                    big = 210
                bw = (0.3 + 0.7 * g) * (22 + (big - 22) * k_) * z * SS
                ar = im.height / im.width
                bwi, bhi = int(bw), int(bw * ar)
                half = bwi / 2
                if bwi > 4 and bhi > 4:
                    tim = im.resize((bwi, bhi), Image.BILINEAR)
                    if e > 0:
                        tim = Image.fromarray(np.clip(np.asarray(tim, float) * (1 + 1.2 * e), 0, 255).astype(np.uint8))
                    x0, y0 = int(p[0] - bwi / 2), int(p[1] - bhi / 2)
                    img.paste(tim, (x0, y0))
                    dr.rectangle([x0, y0, x0 + bwi, y0 + bhi], outline=col, width=SS)
            elif is_now:
                rr = (6 + 10 * e) * SS
                dr.rectangle([p[0] - rr, p[1] - rr, p[0] + rr, p[1] + rr], outline=P["accent"], width=SS * 2)
                dr.line([(p[0] - rr * 1.8, p[1]), (p[0] + rr * 1.8, p[1])], fill=P["accent"], width=SS)
                dr.line([(p[0], p[1] - rr * 1.8), (p[0], p[1] + rr * 1.8)], fill=P["accent"], width=SS)
            else:
                dr.rectangle([p[0] - r, p[1] - r, p[0] + r, p[1] + r], outline=col, width=SS)
            if is_j:
                rr = 4.5 * SS
                dr.ellipse([p[0] - rr, p[1] - rr, p[0] + rr, p[1] + rr], fill=P["hot"])
            lx, ly = p[0] + half + 7 * SS, p[1] - 12 * SS
            if v >= 3:
                _label_v3(dr, P, mem, nd, i, lx, ly, p, half, popped, t, seg, red_on, nbr, is_j, is_now)
                continue
            # 标签：三级 —— 时间（亮）/ 名字（中）/ 特征（暗，只给已访问的）
            tcol = P["ink"] if vis or is_now else P["mid"]
            if is_j and (vis or red_on):
                tcol = P["hot"]
            tt = f"{_fmt(nd['t0'])}" if not is_now else f"{_fmt(seg[0])}–{_fmt(seg[1])}"
            text(dr, (lx, ly), tt, tcol, 11, knock=P["bg"])
            nm = nd["name"] + ("  · JOHN" if is_j else "")
            if nd.get("note") and nd["note"] not in nm:
                nm += "  " + nd["note"]
            text(dr, (lx, ly + 13 * SS), nm, mix(tcol, 0.75, P["bg"]), 10, cn=True, knock=P["bg"])
            fresh = vis and (t - popped[i] < 2.2)
            on_path = red_on and i in mem.path
            if fresh or is_now or on_path:
                fz = nd["feat"]
                fl = " ".join(f"{SYM[k_]}{fz[k_]:.2f}" for k_ in CH) + f" ρ{fz['rho']:.1f}/s"
                text(dr, (lx, ly + 29 * SS), fl, P["mid"], 9, knock=P["bg"])
                dd = next(d for u, d, _ in mem.order if u == i)
                text(dr, (lx, ly + 41 * SS), f"[{nd['pca'][0]:+.2f}, {nd['pca'][1]:+.2f}]  d={dd:.2f}", P["mid"], 9, knock=P["bg"])

        # 注释引线：最新弹出的节点 → 底部时间轴上它的时间段
        last = max(popped.items(), key=lambda kv: kv[1])[0] if popped else None
        _draw_strip(dr, P, ch, fps_s, ev, strip, tl0, tl1, mem, popped, t, seg, red_on)
        if last is not None:
            nd = mem.nodes[last]
            p = S2(pos[last])
            xm = strip["x0"] + (nd["t0"] + nd["t1"]) / 2 / (tl1 - tl0) * (strip["x1"] - strip["x0"])
            q = np.array([xm * SS, (strip["y0"] - 6) * SS])
            pts, s = bezier(p, q, 0.0)
            draw_path(dr, pts, s, P["dim"], 1, dash=(2, 3))

        if v >= 2:
            _draw_compare(dr, P, mem, popped, red_on, t, sch)
        # HUD（左上）：真参数
        nvis = len(popped)
        hud = [f"06  A NAME    {_fmt(seg[0])}–{_fmt(seg[1])}   t={seg[0] + t:7.2f}s",
               f"MEMORY GRAPH  |V| {n}  |E| {len(mem.edges)}  (story {sum(e[2] == 'story' for e in mem.edges)} · "
               f"kNN k=3 {sum(e[2] == 'knn' for e in mem.edges)} · recall {sum(e[2] == 'same' for e in mem.edges)})",
               f"DIJKSTRA FROM NOW   visited {nvis}/{n}   w = ‖z_i − z_j‖₂",
               ]
        for k_, line in enumerate(hud):
            text(dr, (56 * SS, (40 + 17 * k_) * SS), line, P["ink"] if k_ == 0 else P["mid"], 12 if k_ == 0 else 11)
        if red_on and v >= 3:
            dj = next(d for u, d, _ in mem.order if u == mem.john)
            text(dr, (56 * SS, (40 + 17 * 3 + 8) * SS), f"MATCH: JOHN-117   CONFIDENCE {mem.conf:.2f}", P["hot"], 15)
            text(dr, (56 * SS, (40 + 17 * 3 + 30) * SS), f"FOUND AT POP {sch['jk'] + 1}/{n}  ·  path {len(mem.path) - 1} hops  ·  d={dj:.2f}",
                 P["hot"], 12)
            text(dr, (56 * SS, (40 + 17 * 3 + 48) * SS), "CONFIDENCE = (1 + cos(z_now, z_john)) / 2", P["mid"], 11)
        elif red_on:
            dj = next(d for u, d, _ in mem.order if u == mem.john)
            text(dr, (56 * SS, (40 + 17 * 3 + 8) * SS), f"MATCH: JOHN-117   CONFIDENCE {mem.conf:.2f}", P["hot"], 15)
            text(dr, (56 * SS, (40 + 17 * 3 + 30) * SS),
                 f"= (1+cos(z_now, z_john))/2    path {len(mem.path) - 1} hops  d={dj:.2f}", P["dim"], 10)
        out = img.resize((W, H), Image.LANCZOS)
        frames_meta.append(dict(f=f, t=round(seg[0] + t, 3), cam=[round(float(cx), 1), round(float(cy), 1)],
                                zoom=round(z, 3), visited=nvis))
        yield f, out, frames_meta[-1]


# ───────────────────────── A v4：三段（搜 → 心魔被压下 / 切进局部躁动 / 四声锁定 → 冲进 John）─────────────────────────
V4 = dict(unstable_t=118.552, cut_t=121.96, hops=(126.554, 126.943, 127.342, 127.741), end_t=128.36,
          n_phase1b=6, sea={"bwr": "20_粒子海_吸收_128.36-134.78_黑白红_v4.mp4", "acid": "20_粒子海_吸收_128.36-134.78_酸绿_v4.mp4"})


def schedule_v4(mem, seg, unstable_node=None, thr=0.3):
    """只改节奏，不改顺序 / 数值（👤2026-10-23）：
    · 第一段 115.51–121.96：弹出 0..k_u−1 落在 118.552 之前的真实事件上，第 k_u 个 =「心魔」节点正好在 118.552（起音）弹出；
      之后 n_phase1b 个落在 118.552–121.96；
    · 第二段 121.96–126.554：剩下的（到 John 前一个）全挤进来 → 弹出加快；
    · 结尾：路径 4 跳逐跳点亮在 126.554 / 126.943 / 127.342 / 127.741 四声上，第四声 = 弹出 John（第 jk+1 个）= 锁定。
      搜索在找到目标时停（Dijkstra 提前终止），之后的节点不再弹出。
    事件池 = 起音（威胁峰 ≥ thr）∪ 拍点（感官.json 的 beats），相距 < 0.1 s 的合并；不够就在最大间隔的中点补（标 sub）。"""
    _, fps, ch, ev = load_sense()
    s0 = seg[0]
    order = [u for u, _, _ in mem.order]
    jk = order.index(mem.john)
    if unstable_node is None:                                   # 心魔 = 第一段里和 John 最不像的那段记忆（cos 最低）
        zj = np.array(mem.nodes[mem.john]["z"])
        cand = order[1:12]
        unstable_node = min(cand, key=lambda u: np.array(mem.nodes[u]["z"]) @ zj / np.linalg.norm(mem.nodes[u]["z"]) / np.linalg.norm(zj))
    ku = order.index(unstable_node)
    pool = []
    for t in ev["onsets"]:
        s = onset_strength(ch, fps, t)
        if s >= thr:
            pool.append((float(t), s, "onset"))
    for b in ev["beats"]:
        if not any(abs(b - p[0]) < 0.1 for p in pool):
            pool.append((float(b), onset_strength(ch, fps, b), "beat"))
    pool.sort()

    def pick(a, b, k, first=None):
        ev_ = [p for p in pool if a < p[0] < b]
        if first is not None:
            ev_ = [p for p in ev_ if p[0] != first[0]]
        chosen = sorted(sorted(ev_, key=lambda p: -p[1])[:max(0, k - (first is not None))])
        if first is not None:
            chosen = [first] + chosen
        while len(chosen) < k:                                   # 补：最大间隔中点
            ts = [a] + [c[0] for c in chosen] + [b]
            g = int(np.argmax(np.diff(ts)))
            chosen.append(((ts[g] + ts[g + 1]) / 2, 0.0, "sub"))
            chosen.sort()
        return chosen

    ut, cut, hops = V4["unstable_t"], V4["cut_t"], V4["hops"]
    first = min(pool, key=lambda p: abs(p[0] - s0 - 0.04))
    e1 = pick(s0 - 0.01, ut - 0.05, ku, first=first)
    e_u = [min(pool, key=lambda p: abs(p[0] - ut))]
    n1b = V4["n_phase1b"]
    e2 = pick(ut + 0.05, cut, n1b)
    first3 = min([p for p in pool if cut < p[0] < hops[0]], key=lambda p: p[0])      # 切过去后第一个事件必弹（第二段一进来就在动）
    e3 = pick(cut, hops[0] - 0.05, jk - ku - 1 - n1b, first=first3)
    events = e1 + e_u + e2 + e3 + [(hops[3], onset_strength(ch, fps, hops[3]), "onset")]
    assert len(events) == jk + 1, (len(events), jk + 1)
    pop = {order[k]: (events[k][0] - s0, events[k][1], events[k][2]) for k in range(jk + 1)}
    return dict(pop={u: (t, s) for u, (t, s, _) in pop.items()}, kinds={u: k for u, (_, _, k) in pop.items()},
                jk=jk, unstable=unstable_node, ku=ku, hops=[h - s0 for h in hops], cut=cut - s0, ut=ut - s0,
                path=(hops[0] - s0, 1.0), pulses=[], end=V4["end_t"] - s0)


def takens_points(t0, t1, f_lo=30.0, f_hi=250.0):
    """那段记忆自己的声音（原曲 t0–t1）→ 带通 → 延迟嵌入 (x(t), x(t−τ), x(t−2τ))，τ = 主频周期的 1/4。全是真数。"""
    sys.path.insert(0, str(ROOT / "src"))
    import sensorium
    sr = 11025
    x = sensorium.load_audio(MUSIC, t0, t1 - t0, sr=sr)
    X = np.fft.rfft(x)
    fr = np.fft.rfftfreq(len(x), 1 / sr)
    X[(fr < f_lo) | (fr > f_hi)] = 0
    xb = np.fft.irfft(X, len(x))
    f0 = float(fr[np.argmax(np.abs(X))])
    tau = max(1, int(round(sr / (4 * f0))))
    P3 = np.stack([xb[2 * tau:], xb[tau:-tau], xb[:-2 * tau]], 1)
    P3 = P3 / (np.abs(P3).max() + 1e-12)
    return P3, dict(f0=f0, tau_ms=1000 * tau / sr, sr=sr, n=len(P3))


def _ch_at(ch, fps, t, key):
    return float(ch[key][min(len(ch[key]) - 1, max(0, int(round(t * fps))))])


def render_memory_v4(mem, sch, seg, pal, only=None):
    _MINF["on"] = True
    P = PALETTES[pal]
    s0 = seg[0]
    T = int(round((sch["end"]) * FPS))
    base = np.array([nd["pos"] for nd in mem.nodes])
    n = len(base)
    nbr = {i: set() for i in range(n)}
    for a_, b_, _, _ in mem.edges:
        nbr[a_].add(b_)
        nbr[b_].add(a_)
    thumbs = {}
    for nd in mem.nodes:
        if nd["src"] == "此刻":
            continue
        im = thumb(nd["src"], (nd["t0"] + nd["t1"]) / 2, nd["t0"], nd["speed"])
        if P["thumb"] == "gray":
            im = im.convert("L").convert("RGB")
        else:
            g = np.asarray(im.convert("L").convert("RGB"), float)
            im = Image.fromarray(np.clip(g * 0.55 + np.asarray(im, float) * 0.45, 0, 255).astype(np.uint8))
        thumbs[nd["i"]] = im
    sea = read_video(MV / "样片" / "粒子海" / V4["sea"][pal], 0, 0.05)[0]          # 下一镜（粒子海 v4）的首帧
    sea_im = Image.fromarray(sea)
    jt = thumbs[mem.john]                                                         # John 框统一成 16:9（冲进去时铺满画面）
    jw, jh = jt.size
    if jw / jh > 16 / 9:
        nw = int(jh * 16 / 9); jt = jt.crop(((jw - nw) // 2, 0, (jw - nw) // 2 + nw, jh))
    else:
        nh = int(jw * 9 / 16); jt = jt.crop((0, (jh - nh) // 2, jw, (jh - nh) // 2 + nh))
    thumbs[mem.john] = jt.resize((640, 360), Image.LANCZOS)
    un = sch["unstable"]
    und = mem.nodes[un]
    tk, tk_meta = takens_points(und["t0"], und["t1"])
    zj = np.array(mem.nodes[mem.john]["z"])
    zu = np.array(und["z"])
    cos_u = float(zu @ zj / np.linalg.norm(zu) / np.linalg.norm(zj))
    _, fps_s, ch, ev = load_sense()
    ons = [(float(t) - s0, onset_strength(ch, fps_s, t)) for t in ev["onsets"] if s0 - 0.5 <= t <= seg[1] + 0.5]
    pop = sch["pop"]
    order = [u for u, _, _ in mem.order]
    par = {u: p for u, _, p in mem.order}
    hops = sch["hops"]
    t_lock, t_cut, t_ut = hops[3], sch["cut"], sch["ut"]
    t_dive = t_lock + 0.12
    t_last = (T - 1) / FPS
    path = mem.path
    z_final = W / 210.0

    # ── 相机 ──
    pop_sorted = sorted(pop.items(), key=lambda kv: kv[1][0])

    def newest_at(t):
        nw = mem.now
        for u, (tu, _) in pop_sorted:
            if tu <= t:
                nw = u
        return nw
    tg1, tg2 = [], []
    for f in range(T):
        t = f / FPS
        nw = newest_at(t)
        vis = [u for u, (tu, _) in pop_sorted if tu <= t]
        cv = base[vis].mean(0) if vis else base[mem.now]
        tg1.append(0.8 * base[nw] + 0.2 * cv)
        if t >= hops[0]:                                          # 四声：镜头一跳一跳追着路径点亮的那一跳
            k = sum(t >= h for h in hops) - 1
            tg2.append(0.75 * (base[path].min(0) + base[path].max(0)) / 2 + 0.25 * base[path[k + 1]])
        else:
            tg2.append(0.7 * base[nw] + 0.3 * cv)
    tg1, tg2 = np.array(tg1), np.array(tg2)
    f_cut = int(math.ceil(t_cut * FPS))
    st = base[mem.now] + (base[0] - base[mem.now]) * 0.32
    cam = np.zeros((T, 2))
    cam[:f_cut] = spring_follow(tg1[:f_cut], st, (base[mem.now] - st) * 4.2, omega=3.4)
    c0 = tg2[f_cut] + np.array([140.0, -60.0])                    # 硬切：第二个画面一进来就在横移
    cam[f_cut:] = spring_follow(tg2[f_cut:], c0, np.array([-260.0, 90.0]), omega=5.0)
    zs = np.zeros(T)
    zs[:f_cut] = spring_follow(np.full((f_cut, 1), 1.65), [1.32], [0.25], omega=1.8)[:, 0]
    pp_ = base[path]
    z_fit = min(W * 0.6 / (np.ptp(pp_[:, 0]) + 300), H * 0.55 / (np.ptp(pp_[:, 1]) + 260))
    for f in range(f_cut, T):
        t = f / FPS
        zs[f] = 2.55 + 0.45 * min(1.0, (t - t_cut) / (hops[0] - t_cut))    # 第二段：慢慢逼近（张力）
        if t >= hops[0]:                                                   # 四声：0.25 s 拉到整条路径都在画面里
            a_ = min(1.0, (t - hops[0]) / 0.25)
            a_ = 1 - (1 - a_) ** 3
            zs[f] = (1 - a_) * 3.0 + a_ * z_fit
    f_dive = int(math.ceil(t_dive * FPS))
    c_d, z_d = cam[f_dive - 1].copy(), zs[f_dive - 1]
    t_full = t_last - 2 / FPS                                      # 倒数第 3 帧起 John 框正好铺满画面（之后 = 下一镜首帧）
    for f in range(f_dive, T):                                    # 冲进 John：变焦按指数（每帧同比放大），中心收向 John
        u = min(1.0, (f / FPS - t_dive) / (t_full - t_dive))
        zs[f] = z_d * (z_final / z_d) ** (u ** 1.5)
        a = 1 - (1 - u) ** 3
        cam[f] = (1 - a) * c_d + a * base[mem.john]

    # 抖动：第二段的线和节点自己抖，幅度接真实通道（屏幕像素）A = ramp·(2 + 9·张力 + 16·威胁)；
    # 相位沿图传播（离最新节点的跳数越远越晚）→ 是沿结构走的波，不是各点独立加噪。锁定那一刻全部停住。
    rng = np.random.default_rng(7)
    dirs = rng.normal(size=(n, 2)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    def hopdist(src):
        d = {src: 0}
        q = [src]
        while q:
            u = q.pop(0)
            for w_ in nbr[u]:
                if w_ not in d:
                    d[w_] = d[u] + 1
                    q.append(w_)
        return np.array([d.get(i, 6) for i in range(n)])

    def jitter_amp(t):
        if t < t_cut or t >= t_lock:
            return 0.0
        tp = s0 + t
        ramp = min(1.0, (t - t_cut) / 0.8) * (1 + 0.8 * (t - t_cut) / (hops[0] - t_cut))
        return ramp * (2 + 9 * _ch_at(ch, fps_s, tp, "tension") + 16 * _ch_at(ch, fps_s, tp, "threat"))

    amps = [jitter_amp(f / FPS) for f in range(T)]
    strip = dict(x0=150, x1=W - 150, y0=H - 118, y1=H - 46)
    meta = []
    for f in range(T):
        if only is not None and f not in only:
            continue
        t = f / FPS
        z = float(zs[f])
        cx, cy = cam[f]
        A = amps[f]
        nw = newest_at(t)
        hd = hopdist(nw)
        pos = base.copy()
        if A > 0:
            ph = 2 * math.pi * 7.0 * t - 1.4 * hd
            pos = base + (A / z) * np.sin(ph)[:, None] * dirs
        img = Image.new("RGB", (W * SS, H * SS), P["bg"])
        dr = ImageDraw.Draw(img)

        def S2(p):
            return np.array([(W / 2 + (p[0] - cx) * z) * SS, (H / 2 + (p[1] - cy) * z) * SS])

        diving = t >= t_dive
        fade_ui = 0.0 if not diving else min(1.0, (t - t_dive) / 0.2)
        popped = {u: tu for u, (tu, _) in pop.items() if tu <= t}
        n_lit = sum(t >= h for h in hops)
        locked = t >= t_lock
        flick = t >= t_cut and not locked and any(0 <= t - to < 2 / FPS for to, _ in ons)
        # 坐标轴（第二段近景里只剩刻度线）
        ox, oy = mem.origin
        sw = mem.scale
        xa = [S2((ox - 4.6 * sw, oy)), S2((ox + 4.6 * sw, oy))]
        ya = [S2((ox, oy - 3.2 * sw)), S2((ox, oy + 3.2 * sw))]
        dr.line([tuple(xa[0]), tuple(xa[1])], fill=P["faint"], width=SS)
        dr.line([tuple(ya[0]), tuple(ya[1])], fill=P["faint"], width=SS)
        if t < t_cut:
            ld = mem.pcs
            text(dr, (xa[1][0] - 8 * SS, xa[1][1] - 18 * SS), "PC1 " + " ".join(f"{ld[0, i]:+.2f}{SYM[k_]}" for i, k_ in enumerate(mem.keys))
                 + f"  ({mem.evr[0] * 100:.0f}% var)", P["dim"], 12, anchor="ra")
        # 边
        for a, b, kind, w in mem.edges:
            curv = 0.0 if kind == "story" else (0.22 if (a + b) % 2 else -0.22)
            if kind == "same":
                curv = 0.35
            if A > 0 and kind != "story":
                curv += 0.05 * (A / 10) * math.sin(2 * math.pi * 5.0 * t + a)
            pts, s = bezier(S2(pos[a]), S2(pos[b]), curv)
            seen = a in popped or b in popped
            if kind == "story":
                ta = -0.35 + 1.0 * a / max(1, n - 2)
                gs = min(1.0, max(0.0, (t - ta) / 0.25))
                draw_path(dr, pts, s, P["dim"] if gs < 1 else P["faint"], 1, upto=gs)
            elif kind == "same":
                draw_path(dr, pts, s, P["mid"] if seen else P["dim"], 1, dash=(1.2, 3.0))
            else:
                draw_path(dr, pts, s, P["dim"] if seen else P["faint"], 1, dash=(4, 5), phase=t * (9 + 3 * A))
                if mem.now in (a, b) and t < 1.2:
                    g0 = min(1.0, (t + 0.3) / 0.6)
                    if mem.now == b:
                        pts = pts[::-1]
                        s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
                    draw_path(dr, pts, s, mix(P["accent"], 0.8 * (1 - max(0.0, t - 0.6) / 0.6), P["bg"]), 1.4, dash=(6, 4),
                              upto=g0, phase=-t * 14)
        # 搜索树的边
        for u, tu in popped.items():
            p = par.get(u, -1)
            if p < 0:
                continue
            g = 1 - (1 - min(1.0, (t - tu) / (0.25 if tu < t_cut else 0.14))) ** 3
            pts, s = bezier(S2(pos[p]), S2(pos[u]), 0.18)
            fade = math.exp(-max(0.0, t - tu - 0.4) / 1.2)
            col = mix(P["accent"], 0.35 + 0.65 * fade, P["bg"])
            draw_path(dr, pts, s, col, 1.0 + 0.6 * fade, dash=(6, 4), upto=g, phase=-t * 14)
            if fade > 0.05 or g < 1:
                draw_arrow(dr, pts, s, g if g < 1 else 0.55, col, 5)
        # 四声：路径逐跳点亮（每一声一跳，0.12 s 长满），第四声到 John = 锁定
        for k in range(n_lit):
            a_, b_ = path[k], path[k + 1]
            pts, s = bezier(S2(pos[a_]), S2(pos[b_]), 0.18)
            gk = min(1.0, (t - hops[k]) / 0.12)
            e = ease_pop(t - hops[k], 0.2)
            draw_path(dr, pts, s, P["hot"], 2.2 + 2.5 * e, upto=gk)
            q = S2(pos[b_]) if gk >= 1 else _interp(pts, s, s[-1] * gk)
            r = (4 + 6 * e) * SS
            dr.ellipse([q[0] - r, q[1] - r, q[0] + r, q[1] + r], fill=P["hot"])
        # 节点
        for nd in mem.nodes:
            i = nd["i"]
            p = S2(pos[i])
            vis = i in popped
            is_j, is_now = i == mem.john, i == mem.now
            e = ease_pop(t - popped[i]) if vis else 0.0
            col = P["hot"] if (is_j and vis) else (P["ink"] if vis else P["mid"])
            r = 3.5 * SS
            half = r
            un_live = i == un and t_ut <= t < t_ut + 2.0
            if un_live:
                half = _draw_unstable(img, dr, P, p, t - t_ut, tk, tk_meta, ch, fps_s, s0 + t, ons, t, und, cos_u, z)
            elif vis and i in thumbs and not (i == un):
                im = thumbs[i]
                age = t - popped[i]
                g = min(1.0, age / (3 / FPS))
                if is_j:
                    bw = 210 * z * SS
                else:
                    big = 64 + 56 * min(1.0, (nd["t1"] - nd["t0"]) / 6.4)
                    k_ = math.exp(-max(0.0, age - (0.6 if popped[i] < t_cut else 0.35)) / 0.9)
                    bw = (0.3 + 0.7 * g) * (22 + (big - 22) * k_) * min(z, 1.9) * SS
                bwi, bhi = int(bw), int(bw * im.height / im.width)
                half = bwi / 2
                if 4 < bwi < 6 * W * SS:
                    if is_j and diving:                           # 冲进去：框里换成下一镜（粒子海 v4 首帧）
                        ud = (t - t_dive) / (t_last - 2 / FPS - t_dive)
                        wsea = min(1.0, max(0.0, (ud - 0.2) / 0.6))
                        im = Image.blend(im, sea_im.resize(im.size, Image.LANCZOS), wsea)
                    x0, y0 = int(p[0] - bwi / 2), int(p[1] - bhi / 2)
                    # 只缩放落在画面里的那一块（冲进去时框比画面大很多）
                    vx0, vy0, vx1, vy1 = max(0, x0), max(0, y0), min(W * SS, x0 + bwi), min(H * SS, y0 + bhi)
                    if vx1 > vx0 and vy1 > vy0:
                        sx_, sy_ = im.width / bwi, im.height / bhi
                        crop = im.crop(((vx0 - x0) * sx_, (vy0 - y0) * sy_, (vx1 - x0) * sx_, (vy1 - y0) * sy_))
                        tim = crop.resize((vx1 - vx0, vy1 - vy0), Image.BILINEAR)
                        if e > 0:
                            tim = Image.fromarray(np.clip(np.asarray(tim, float) * (1 + 1.2 * e), 0, 255).astype(np.uint8))
                        img.paste(tim, (vx0, vy0))
                    if bwi < 0.97 * W * SS:
                        dr.rectangle([x0, y0, x0 + bwi, y0 + bhi], outline=col, width=SS * (2 if is_j else 1))
            elif is_now:
                rr = (6 + 10 * e) * SS
                dr.rectangle([p[0] - rr, p[1] - rr, p[0] + rr, p[1] + rr], outline=P["accent"], width=SS * 2)
                dr.line([(p[0] - rr * 1.8, p[1]), (p[0] + rr * 1.8, p[1])], fill=P["accent"], width=SS)
                dr.line([(p[0], p[1] - rr * 1.8), (p[0], p[1] + rr * 1.8)], fill=P["accent"], width=SS)
            else:
                dr.rectangle([p[0] - r, p[1] - r, p[0] + r, p[1] + r], outline=col, width=SS)
            if is_j and not diving:
                rr = 4.5 * SS
                dr.ellipse([p[0] - rr, p[1] - rr, p[0] + rr, p[1] + rr], fill=P["hot"])
            if diving:
                continue
            lx, ly = p[0] + half + 7 * SS, p[1] - 12 * SS
            if un_live:
                continue
            _label_v3(dr, P, mem, nd, i, lx, ly, p, half, popped, t, seg, locked, nbr, is_j and locked, is_now, flick=flick)
            if i == un and t >= t_ut + 2.0:
                text(dr, (lx, ly + 18 * SS), "SUPPRESSED", P["dim"], 12, knock=P["bg"])
        # 第一段：底部时间轴；第二段（另一个画面）：换成右下的抖动示波（A(t) 的真实曲线）
        if t < t_cut:
            _draw_strip(dr, P, ch, fps_s, ev, strip, 0.0, seg[1], mem, popped, t, seg, False)
            last = max(popped.items(), key=lambda kv: kv[1])[0] if popped else None
            if last is not None:
                nd = mem.nodes[last]
                pp = S2(pos[last])
                xm = strip["x0"] + (nd["t0"] + nd["t1"]) / 2 / seg[1] * (strip["x1"] - strip["x0"])
                pts, s = bezier(pp, np.array([xm * SS, (strip["y0"] - 6) * SS]), 0.0)
                draw_path(dr, pts, s, P["dim"], 1, dash=(2, 3))
        elif not diving:
            _draw_scope(dr, P, amps, f, t_cut, hops, ch, fps_s, s0, t)
        if not diving:
            _draw_compare(dr, P, mem, popped, locked, t, sch)
        # HUD
        if fade_ui < 1:
            if t < t_cut:
                ph_ = "PHASE 1 · SEARCH"
            elif t < hops[0]:
                ph_ = f"PHASE 2 · DIG IN   jitter A = {A:4.1f}px = ramp·(2 + 9τ + 16θ)"
            elif not locked:
                ph_ = f"TRACE {n_lit}/4"
            else:
                ph_ = "LOCK 4/4 · ENTER"
            hud = [f"06  A NAME    {_fmt(seg[0])}–{_fmt(V4['end_t'])}   t={s0 + t:7.2f}s",
                   f"MEMORY GRAPH  |V| {n}  |E| {len(mem.edges)}   DIJKSTRA FROM NOW   visited {len(popped)}/{n}",
                   ph_]
            for k_, line in enumerate(hud):
                text(dr, (56 * SS, (40 + 18 * k_) * SS), line, P["ink"] if k_ != 1 else P["mid"], 12)
            if locked:
                dj = next(d for u, d, _ in mem.order if u == mem.john)
                text(dr, (56 * SS, (40 + 18 * 3 + 8) * SS), f"MATCH: JOHN-117   CONFIDENCE {mem.conf:.2f}", P["hot"], 15)
                text(dr, (56 * SS, (40 + 18 * 3 + 30) * SS), f"FOUND AT POP {sch['jk'] + 1}/{n}  ·  path {len(path) - 1} hops  ·  d={dj:.2f}", P["hot"], 12)
        out = img.resize((W, H), Image.LANCZOS)
        if f >= T - 4:                                            # 最后 4 帧并到下一镜首帧（最后一帧 = 粒子海 v4 第 0 帧）
            out = Image.blend(out, sea_im, (f - (T - 4) + 1) / 4)
        meta.append(dict(f=f, t=round(s0 + t, 3), cam=[round(float(cx), 1), round(float(cy), 1)], zoom=round(z, 3),
                         visited=len(popped), jitter_px=round(A, 2), lit=n_lit))
        yield f, out, meta[-1]
    _MINF["on"] = False


def _draw_unstable(img, dr, P, p, s, tk, meta, ch, fps, tp, ons, t, und, cos_u, z):
    """心魔：框里是那段记忆自己的声音的延迟嵌入（在转、在走），抖 / 闪 / 收缩 / 变暗 / 退回成点。
    s = 弹出后的秒数：0–0.15 弹开；0.15–1.15 活着（抖幅 = 4 + 18·当前威胁）；1.15–2.0 被压下（三次缓入收缩到点）。"""
    th = _ch_at(ch, fps, tp, "threat")
    if s < 0.15:
        sc = s / 0.15
    elif s < 1.15:
        sc = 1.0
    else:
        u = (s - 1.15) / 0.85
        sc = max(0.03, 1 - u ** 3 * 0.97 - 0.35 * u)
    sc = max(0.03, sc)
    flash = any(0 <= t - to < 2 / FPS and so > 0.4 for to, so in ons)
    bright = (0.35 if flash else 1.0) * (1.0 if s < 1.15 else max(0.2, 1 - (s - 1.15) / 0.85))
    shake = (4 + 18 * th) * (1.0 if s < 1.15 else max(0.0, 1 - (s - 1.15) / 0.6))
    rng = np.random.default_rng(int(t * FPS) + 11)
    j = rng.normal(size=2) * shake * SS
    size = 200 * sc * SS
    cx, cy = p[0] + j[0], p[1] + j[1]
    col = mix(P["accent"] if P["accent"] != P["ink"] else P["ink"], bright, P["bg"])
    if size > 6 * SS:
        x0, y0 = cx - size / 2, cy - size / 2
        dr.rectangle([x0, y0, x0 + size, y0 + size], fill=P["bg"], outline=col, width=SS)
        th_r = 0.9 * s + 0.4
        N = len(tk)
        win = int(0.25 * meta["sr"])
        c = int((s * 0.35 * meta["sr"])) % max(1, N - win)
        seg_ = tk[c:c + win:6]
        allp = tk[::max(1, N // 900)]
        for pts_, cc in ((allp, mix(col, 0.25, P["bg"])), (seg_, col)):
            X = pts_[:, 0] * math.cos(th_r) + pts_[:, 2] * math.sin(th_r)
            Y = pts_[:, 1]
            q = np.stack([cx + X * size * 0.44, cy + Y * size * 0.44], 1)
            if len(q) > 1:
                dr.line([tuple(v) for v in q], fill=cc, width=SS)
        lab = "UNSTABLE" if s < 1.15 else "SUPPRESSING"
        text(dr, (x0, y0 - 34 * SS), f"#{und['i']}  {lab}   {_fmt(und['t0'])}  {und['name']}", col, 12, cn=True, knock=P["bg"])
        text(dr, (x0, y0 - 17 * SS), f"TAKENS (x(t), x(t−τ), x(t−2τ))  τ={meta['tau_ms']:.1f}ms  f₀={meta['f0']:.0f}Hz  "
             f"cos(z, z_john)={cos_u:+.2f}  θ={th:.2f}", mix(col, 0.8, P["bg"]), 11, knock=P["bg"])
        return size / 2
    r = 3 * SS
    dr.rectangle([cx - r, cy - r, cx + r, cy + r], outline=col, width=SS)
    return r


def _draw_scope(dr, P, amps, f, t_cut, hops, ch, fps, s0, t):
    """第二段右下：抖动幅度 A(t) 的示波 + 当前张力 / 威胁（真数）"""
    x0, y0, w_, h_ = W - 420, H - 150, 360, 80
    dr.rectangle([(x0 - 10) * SS, (y0 - 26) * SS, (x0 + w_ + 10) * SS, (y0 + h_ + 30) * SS], fill=P["bg"], outline=P["faint"], width=SS)
    f0, f1 = int(t_cut * FPS), int(hops[0] * FPS)
    mx = max(1e-6, max(amps[f0:f1 + 1]))
    xs, ys = [], []
    for k in range(f0, min(f, f1) + 1):
        xs.append((x0 + (k - f0) / max(1, f1 - f0) * w_) * SS)
        ys.append((y0 + h_ - amps[k] / mx * h_) * SS)
    if len(xs) > 1:
        dr.line(list(zip(xs, ys)), fill=P["accent"], width=SS)
    tp = s0 + t
    text(dr, (x0 * SS, (y0 - 20) * SS), f"JITTER A(t)  max {mx:.1f}px   τ={_ch_at(ch, fps, tp, 'tension'):.2f}  θ={_ch_at(ch, fps, tp, 'threat'):.2f}",
         P["mid"], 11)
    text(dr, (x0 * SS, (y0 + h_ + 6) * SS), f"{_fmt(s0 + t_cut)}", P["dim"], 12)
    text(dr, ((x0 + w_) * SS, (y0 + h_ + 6) * SS), f"{_fmt(s0 + hops[0])}", P["dim"], 12, anchor="ra")


def _fmt(t):
    return f"{int(t // 60)}:{t % 60:05.2f}"


def _label_v3(dr, P, mem, nd, i, lx, ly, p, half, popped, t, seg, red_on, nbr, is_j, is_now, flick=False):
    """v3「注意力」分层（gif 只标约 1/3 的节点）：
    焦点 = 最新弹出的节点（前沿，0.5 s 内弹出的都算）+ 它的图邻居 + 此刻（+ 找到后：整条路径）→ 时间 13 / 名字 12，前沿再加特征和坐标；
    已访问的 → 一行：时间 + PCA 坐标（11，中灰）；未访问的 → 只剩 #编号（12，暗）。"""
    vis = i in popped
    front = {u for u, tu in popped.items() if t - tu < 0.5}
    if popped:
        front.add(max(popped.items(), key=lambda kv: kv[1])[0])
    near = set().union(*[nbr[u] for u in front]) if front else set()
    on_path = red_on and i in mem.path
    focus = i in front or i in near or is_now or on_path or (is_j and red_on)
    hot = is_j and (vis or red_on)
    if focus:
        col = P["hot"] if hot else (P["accent"] if i in front and not red_on else P["ink"])
        if flick and not hot:                                # v4 第二段：起音那 2 帧焦点标签反白（闪）
            text(dr, (lx, ly - 2 * SS), f"#{i}  {_fmt(nd['t0'])}", P["bg"], 13, knock=col)
            text(dr, (lx, ly + 15 * SS), nd["name"], P["bg"], 12, cn=True, knock=col)
            return
        tt = f"{_fmt(nd['t0'])}" if not is_now else f"{_fmt(seg[0])}–{_fmt(seg[1])}"
        text(dr, (lx, ly - 2 * SS), f"#{i}  {tt}", col, 13, knock=P["bg"])
        nm = nd["name"] + ("  · JOHN" if is_j else "")
        if nd.get("note") and nd["note"] not in nm:
            nm += "  " + nd["note"]
        text(dr, (lx, ly + 15 * SS), nm, mix(col, 0.8, P["bg"]), 12, cn=True, knock=P["bg"])
        if i in front or is_now or on_path:
            fz = nd["feat"]
            fl = " ".join(f"{SYM[k_]}{fz[k_]:.2f}" for k_ in CH) + f" ρ{fz['rho']:.1f}/s"
            dd = next(d for u, d, _ in mem.order if u == i)
            text(dr, (lx, ly + 33 * SS), fl, P["mid"], 11, knock=P["bg"])
            text(dr, (lx, ly + 48 * SS), f"[{nd['pca'][0]:+.2f}, {nd['pca'][1]:+.2f}]  d={dd:.2f}", P["mid"], 11, knock=P["bg"])
    elif vis:
        text(dr, (lx, ly + 2 * SS), f"{_fmt(nd['t0'])}  [{nd['pca'][0]:+.2f}, {nd['pca'][1]:+.2f}]",
             P["hot"] if hot else P["mid"], 11, knock=P["bg"])
    else:
        text(dr, (p[0] + 6 * SS, p[1] - 6 * SS), f"#{i}", P["dim"], READ_MIN_DIM)


def _draw_compare(dr, P, mem, popped, red_on, t, sch):
    """右上：此刻 vs 最新弹出节点（找到后 = John）的 z 向量，逐维条形 + cos。全是真数（z = 标准化特征）。"""
    x0, y0 = W - 360, 40
    last = max(popped.items(), key=lambda kv: kv[1])[0] if popped else mem.now
    other = mem.john if red_on else last
    zn, zo = np.array(mem.nodes[mem.now]["z"]), np.array(mem.nodes[other]["z"])
    cos = float(zn @ zo / (np.linalg.norm(zn) * np.linalg.norm(zo) + 1e-12))
    oc = P["hot"] if red_on else P["accent"]
    hgt = 22 + len(mem.keys) * 15 + 26 + (len(mem.path) * 13 + 14 if red_on else 0)
    dr.rectangle([(x0 - 14) * SS, (y0 - 10) * SS, (W - 16) * SS, (y0 + hgt) * SS], fill=P["bg"], outline=P["faint"], width=SS)
    text(dr, (x0 * SS, y0 * SS), "z-SCORE   NOW  vs  " + ("JOHN" if red_on else f"#{other} {_fmt(mem.nodes[other]['t0'])}"),
         P["ink"], 11)
    cx = x0 + 150
    for k, key in enumerate(mem.keys):
        yy = y0 + 22 + k * 15
        text(dr, (x0 * SS, (yy - 1) * SS), f"{SYM[key]}", P["mid"], 10)
        dr.line([(cx * SS, (yy - 2) * SS), (cx * SS, (yy + 9) * SS)], fill=P["faint"], width=SS)
        for val, col, dy in ((zn[k], P["ink"], 0), (zo[k], oc, 5)):
            L = np.clip(val, -3, 3) * 40
            dr.line([(cx * SS, (yy + dy) * SS), ((cx + L) * SS, (yy + dy) * SS)], fill=col, width=SS * 2)
        text(dr, ((x0 + 20) * SS, (yy - 1) * SS), f"{zn[k]:+.2f}", P["ink"], 9)
        text(dr, ((x0 + 64) * SS, (yy - 1) * SS), f"{zo[k]:+.2f}", oc, 9)
    yy = y0 + 22 + len(mem.keys) * 15 + 4
    text(dr, (x0 * SS, yy * SS), f"cos = {cos:+.3f}    ‖Δz‖ = {np.linalg.norm(zn - zo):.2f}", oc, 10)
    if red_on:
        dmap = {u: d for u, d, _ in mem.order}
        for k, u in enumerate(mem.path):
            nd = mem.nodes[u]
            text(dr, (x0 * SS, (yy + 20 + k * 13) * SS),
                 f"{k}  {_fmt(nd['t0']) if u != mem.now else 'NOW    '}  d={dmap[u]:.2f}  {nd['name'][:12]}", P["hot"] if u == mem.john else P["mid"], 9, cn=True)


_SEG = []
_ONS = []


def segments():
    if not _SEG:
        d = json.loads((MV / "情绪大图" / "草案.json").read_text("utf-8"))
        _SEG.extend(d["段"])
    return _SEG


def onset_list():
    if not _ONS:
        _, fps, ch, ev = load_sense()
        _ONS.extend((float(t), onset_strength(ch, fps, t)) for t in ev["onsets"])
    return _ONS


def _draw_strip(dr, P, ch, fps, ev, st, tl0, tl1, mem, popped, t, seg, red_on):
    """底部时间轴（0 → 06 结束）：四条感官通道 + 起音刻度 + 乐句格 + 每个记忆镜头的区间；播放头 = 歌曲时间。"""
    x0, x1, y0, y1 = st["x0"], st["x1"], st["y0"], st["y1"]
    X = lambda tt: (x0 + (tt - tl0) / (tl1 - tl0) * (x1 - x0)) * SS   # noqa: E731
    dr.rectangle([0, (y0 - 44) * SS, W * SS, H * SS], fill=P["bg"])          # 底板：时间轴是独立的一层，图不压上来
    dr.line([(x0 * SS, (y0 - 44) * SS), (x1 * SS, (y0 - 44) * SS)], fill=P["faint"], width=SS)
    for sg in segments():
        a_, b_ = sg["t"]
        if a_ >= tl1:
            continue
        xa = X(a_)
        dr.line([(xa, (y0 - 36) * SS), (xa, (y0 - 26) * SS)], fill=P["mid"], width=SS)
        cur = a_ <= seg[0] + t < b_
        room = (X(min(b_, tl1)) - xa) / SS                  # 段太短放不下名字 → 只写编号（不叠字）
        lab = f"{sg['id']} {sg['名']}" if room > 90 else sg["id"]
        text(dr, (xa + 3 * SS, (y0 - 38) * SS), lab, P["ink"] if cur else P["dim"], 9, cn=True)
    dr.line([(x0 * SS, y1 * SS), (x1 * SS, y1 * SS)], fill=P["dim"], width=SS)
    for ot, os_ in onset_list():                                              # 起音刻度：高 ∝ 威胁峰
        if tl0 <= ot <= tl1:
            hh_ = (2 + 9 * os_) * SS
            dr.line([(X(ot), (y1 + 27) * SS), (X(ot), (y1 + 27) * SS + hh_)],
                    fill=P["accent"] if ot <= seg[0] + t and ot >= seg[0] else P["dim"], width=SS)
    for ph in ev["phrases"]:
        if tl0 <= ph <= tl1:
            dr.line([(X(ph), (y0 - 4) * SS), (X(ph), (y1 + 5) * SS)], fill=P["faint"], width=SS)
            text(dr, (X(ph) + 2 * SS, (y1 + 6) * SS), f"{ph:.1f}", P["dim"], 8)
    hh = (y1 - y0) / 4
    names = {"heartbeat": "β heartbeat", "threat": "θ threat", "hunger": "η hunger", "tension": "τ tension"}
    for k_, c in enumerate(CH):
        v = ch[c]
        n = int((tl1 - tl0) * fps)
        idx = np.linspace(tl0 * fps, tl1 * fps - 1, 700).astype(int)
        vv = v[idx]
        if c == "threat":                           # 威胁是脉冲 → 每格取最大，避免抽样漏峰
            edges = np.linspace(tl0 * fps, tl1 * fps, 701).astype(int)
            vv = np.array([v[a:max(a + 1, b)].max() for a, b in zip(edges[:-1], edges[1:])])
        ys = (y0 + hh * k_ + hh * (1 - vv) * 0.9) * SS
        xs = np.linspace(x0, x1, len(vv)) * SS
        dr.line(list(zip(xs, ys)), fill=P["mid"] if c != "threat" else P["dim"], width=SS)
        text(dr, ((x0 - 8) * SS, (y0 + hh * k_ + 1) * SS), names[c], P["dim"], 8, anchor="ra")
        _ = n
    for nd in mem.nodes:
        a, b = X(nd["t0"]), X(nd["t1"])
        yy = (y1 + 18) * SS
        col = P["hot"] if (nd["i"] == mem.john and red_on) else (P["ink"] if nd["i"] in popped else P["dim"])
        dr.line([(a + SS, yy), (b - SS, yy)], fill=col, width=SS * (2 if nd["i"] in popped else 1))
    ph = X(seg[0] + t)
    dr.line([(ph, (y0 - 8) * SS), (ph, (y1 + 22) * SS)], fill=P["accent"], width=SS)
    text(dr, (x0 * SS, (y0 - 20) * SS), "SENSORIUM 0:00 → 2:08.31   感官.json · 播放时间 · 24 fps", P["dim"], 9, cn=True)


# ───────────────────────── 输出 ─────────────────────────
def open_writer(path, start, dur):
    cmd = [ffmpeg(), "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(MUSIC), "-map", "0:v", "-map", "1:a",
           "-c:v", "libx264", "-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
           "-shortest", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def contact(frames, path, cols=4, tw=640):
    th = int(tw * H / W)
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * (th + 22)), (12, 12, 12))
    d = ImageDraw.Draw(sheet)
    fnt = ImageFont.truetype(FMONO, 15)
    for k, (lab, im) in enumerate(frames):
        x, y = (k % cols) * tw, (k // cols) * (th + 22)
        sheet.paste(im.resize((tw, th), Image.LANCZOS), (x, y + 22))
        d.text((x + 6, y + 3), lab, fill=(200, 200, 200), font=fnt)
    sheet.save(path, quality=88)


def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def safe_out(folder, name, ext):
    p = folder / f"{name}{ext}"
    if p.exists():
        sys.exit(f"已存在，不覆盖：{p}（换 --name，按版本另起 _v2）")
    return p


# ───────────────────────── B：亮斑追踪 + 成分树 ─────────────────────────
def read_video(path, start=0.0, dur=None, w=None, h=None):
    """解码成 (T, h, w, 3) uint8（w/h 给了就缩放）"""
    cmd = [ffmpeg(), "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(path)]
    if dur:
        cmd += ["-t", f"{dur:.3f}"]
    vf = [f"fps={FPS}"]
    if w:
        vf.append(f"scale={w}:{h}")
    cmd += ["-vf", ",".join(vf), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    if not w:
        r = subprocess.run([ffmpeg(), "-i", str(path)], capture_output=True).stderr.decode("utf8", "replace")
        import re
        w, h = map(int, re.search(r"Video:.*?(\d{3,5})x(\d{3,5})", r).groups())
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def label_cc(mask):
    """4 邻接连通域标记（无 scipy）：最小标号传播 + 指针跳跃，直到不变。返回 (labels, n)，背景 = −1，标号 0..n−1。"""
    H_, W_ = mask.shape
    BIG = H_ * W_
    lab = np.where(mask, np.arange(BIG).reshape(H_, W_), BIG)
    while True:
        old = lab
        m = lab.copy()
        m[1:] = np.minimum(m[1:], lab[:-1])
        m[:-1] = np.minimum(m[:-1], lab[1:])
        m[:, 1:] = np.minimum(m[:, 1:], lab[:, :-1])
        m[:, :-1] = np.minimum(m[:, :-1], lab[:, 1:])
        m = np.where(mask, m, BIG)
        flat = np.append(m.ravel(), BIG)
        for _ in range(3):                                   # 指针跳跃：label ← label[label]
            flat[:-1] = np.where(flat[:-1] < BIG, flat[np.minimum(flat[:-1], BIG)], BIG)
        m = flat[:-1].reshape(H_, W_)
        if np.array_equal(m, old):
            break
        lab = m
    u, inv = np.unique(lab, return_inverse=True)
    inv = inv.reshape(H_, W_)
    n = len(u) - (1 if u[-1] == BIG else 0)
    return np.where(mask, inv, -1), n


def components(lum, thr, min_area):
    """一层阈值的连通域：质心 / 外接框 / 面积 / 平均亮度（真像素统计）"""
    lab, n = label_cc(lum >= thr)
    out = []
    if n == 0:
        return out, lab
    ys, xs = np.nonzero(lab >= 0)
    ls = lab[ys, xs]
    area = np.bincount(ls, minlength=n)
    cx = np.bincount(ls, xs, n) / np.maximum(area, 1)
    cy = np.bincount(ls, ys, n) / np.maximum(area, 1)
    mx = np.bincount(ls, lum[ys, xs], n) / np.maximum(area, 1)
    x0 = np.full(n, 1 << 30); x1 = np.full(n, -1); y0 = np.full(n, 1 << 30); y1 = np.full(n, -1)
    np.minimum.at(x0, ls, xs); np.maximum.at(x1, ls, xs); np.minimum.at(y0, ls, ys); np.maximum.at(y1, ls, ys)
    for k in range(n):
        if area[k] >= min_area:
            out.append(dict(k=k, a=int(area[k]), c=(float(cx[k]), float(cy[k])), box=(int(x0[k]), int(y0[k]), int(x1[k]), int(y1[k])),
                            lum=float(mx[k])))
    return out, lab


def mst_prim(P):
    """欧氏最小生成树（Prim，O(n²)）。返回 [(父, 子)]，根 = 0。"""
    n = len(P)
    if n < 2:
        return []
    D = np.hypot(*(P[:, None] - P[None]).transpose(2, 0, 1))
    inT = np.zeros(n, bool); inT[0] = True
    best = D[0].copy(); par = np.zeros(n, int)
    E = []
    for _ in range(n - 1):
        b = np.where(inT, np.inf, best)
        j = int(np.argmin(b))
        E.append((int(par[j]), j))
        inT[j] = True
        nb = D[j] < best
        par[nb & ~inT] = j
        best = np.minimum(best, D[j])
    return E


class Tracker:
    """逐帧最近邻关联（贪心，按距离从小到大配对，超过 gate 像素就当新目标）。ID 单调递增，历史质心留着画轨迹。"""
    def __init__(self, gate):
        self.gate, self.next, self.prev, self.hist, self.born = gate, 0, [], {}, {}

    def step(self, comps, f):
        P = np.array([c["c"] for c in comps]) if comps else np.zeros((0, 2))
        Q = np.array([c["c"] for c in self.prev]) if self.prev else np.zeros((0, 2))
        ids = [-1] * len(comps)
        if len(P) and len(Q):
            D = np.hypot(*(P[:, None] - Q[None]).transpose(2, 0, 1))
            usedq = set()
            for idx in np.argsort(D, axis=None):
                i, j = divmod(int(idx), D.shape[1])
                if D[i, j] > self.gate:
                    break
                if ids[i] < 0 and j not in usedq:
                    ids[i] = self.prev[j]["id"]
                    usedq.add(j)
        for i, c in enumerate(comps):
            if ids[i] < 0:
                ids[i] = self.next
                self.born[self.next] = f
                self.next += 1
            c["id"] = ids[i]
            self.hist.setdefault(ids[i], []).append((f, c["c"]))
        self.prev = comps
        return comps


def analyse_track(frames, levels, min_area, gate, crop_bottom=0):
    """每帧：3 层阈值（高 → 低）的连通域 + 成分树（高层的每个连通域恰好落在低一层的一个连通域里 → 父子）+ 追踪"""
    T = len(frames)
    trk = [Tracker(gate) for _ in levels]
    out = []
    for f in range(T):
        im = frames[f].astype(np.float32)
        lum = (0.2126 * im[..., 0] + 0.7152 * im[..., 1] + 0.0722 * im[..., 2]) / 255
        if crop_bottom:
            lum[-crop_bottom:] = 0
        lv = []
        labs = []
        for li, thr in enumerate(levels):
            comps, lab = components(lum, thr, min_area[li])
            trk[li].step(comps, f)
            lv.append(comps)
            labs.append(lab)
        for li in range(1, len(levels)):                       # 父 = 低一层标号图在子质心像素（及其所在连通域任一像素）的值
            lab_lo = labs[li - 1]
            klo = {c["k"]: c for c in lv[li - 1]}
            for c in lv[li]:
                x0, y0, x1, y1 = c["box"]
                sub = labs[li][y0:y1 + 1, x0:x1 + 1] == c["k"]
                vals = lab_lo[y0:y1 + 1, x0:x1 + 1][sub]
                pk = int(vals[0]) if len(vals) else -1
                c["parent"] = klo[pk]["id"] if pk in klo else None
        for li, comps in enumerate(lv):
            for c in comps:
                x0, y0, x1, y1 = c["box"]
                c["mask"] = (labs[li][y0:y1 + 1, x0:x1 + 1] == c["k"])
        out.append(lv)
    return out, trk


def render_track(frames, data, trk, levels, pal, t0_song, src_scale, bg_gain=0.32, v=2):
    """v=1：遮罩框盖在原位（像 AI Strategy）；v=2：脸上只留四角括号，遮罩挪到左边的「标本栏」+ 引线，左下加亮度直方图与每层节点数曲线"""
    P = PALETTES[pal]
    counts = np.array([[len(x) for x in d] for d in data])
    shown = []
    T, h, w, _ = frames.shape
    _, fps_s, ch, ev = load_sense()
    ons = [(float(t) - t0_song, onset_strength(ch, fps_s, t)) for t in ev["onsets"] if t0_song <= t < t0_song + T / FPS]
    strong = [(t, s) for t, s in ons if s >= 0.55]
    # 相机：开场就在推（z 1.32 → 1.06，指数先快后慢）+ 横移（一直在动，速度衰减但不归零）
    tt = np.arange(T) / FPS
    zc = 1.06 + 0.26 * np.exp(-tt / 1.1)
    px = 0.5 * W * 0.0 + (-60 * np.exp(-tt / 1.1) + 6 * tt)
    py = 25 * np.exp(-tt / 1.1)
    sx = W / w
    for f in range(T):
        t = f / FPS
        z = float(zc[f])

        def S2(p):
            return np.array([(W / 2 + (p[0] * sx - W / 2 + px[f]) * z) * SS, (H / 2 + (p[1] * sx - H / 2 + py[f]) * z) * SS])

        src = Image.fromarray(frames[f]).convert("L" if P["thumb"] == "gray" else "RGB")
        base = src.convert("RGB").resize((int(w * sx * z * SS), int(h * sx * z * SS)), Image.BILINEAR)
        arr = np.asarray(base, np.float32) * bg_gain + np.array(P["bg"], np.float32) * (1 - bg_gain)
        img = Image.new("RGB", (W * SS, H * SS), P["bg"])
        o = S2((0, 0))
        img.paste(Image.fromarray(arr.clip(0, 255).astype(np.uint8)), (int(o[0]), int(o[1])))
        dr = ImageDraw.Draw(img)
        lv = data[f]
        top = lv[0]                                            # 最低阈值 = 大块（成分树的根层）
        mid = lv[1] if len(lv) > 1 else []
        hi = lv[2] if len(lv) > 2 else []
        # ① 中层质心的 MST：虚弧 + 箭头（父 → 子；根 = 面积最大的中层连通域）
        if len(mid) >= 2:
            order = sorted(range(len(mid)), key=lambda i: -mid[i]["a"])
            Pm = np.array([mid[i]["c"] for i in order])
            for a_, b_ in mst_prim(Pm):
                pa, pb = S2(Pm[a_]), S2(Pm[b_])
                cv = 0.22 if (order[a_] + order[b_]) % 2 else -0.22
                pts, s_ = bezier(pa, pb, cv)
                draw_path(dr, pts, s_, P["mid"], 1, dash=(4, 5), phase=-t * 10)
                draw_arrow(dr, pts, s_, 0.5, P["mid"], 4)
        # ② 成分树：高层 → 中层 → 低层 的父子实线（短、暗）
        idx_mid = {c["id"]: c for c in mid}
        idx_top = {c["id"]: c for c in top}
        lim = 170 * SS                                         # 显示过滤：父的质心离得太远（弧形大块的质心在块外）就不画线；.json 里父子关系全留着
        for c in hi:
            pc = idx_mid.get(c.get("parent"))
            if pc:
                a_, b_ = S2(c["c"]), S2(pc["c"])
                if np.hypot(*(a_ - b_)) < lim:
                    dr.line([tuple(a_), tuple(b_)], fill=P["dim"], width=SS)
        for c in mid:
            pc = idx_top.get(c.get("parent"))
            if pc:
                a_, b_ = S2(c["c"]), S2(pc["c"])
                if np.hypot(*(a_ - b_)) < lim:
                    dr.line([tuple(a_), tuple(b_)], fill=P["faint"], width=SS)
        # ③ 轨迹：中层追踪 ID 的最近 1 s 质心
        for c in mid:
            hs = [p for ff, p in trk[1].hist[c["id"]] if f - 24 <= ff <= f]
            if len(hs) >= 3:
                dr.line([tuple(S2(p)) for p in hs], fill=P["dim"], width=SS)
        if v >= 2:
            shown = _specimens(dr, img, P, top, S2, src_scale, w, shown if v >= 3 else None)
            _hist_panel(dr, P, frames[f], levels, counts, f)
        # ④ 低层框：外接框 + 遮罩（灰底白形，真遮罩）
        for c in (sorted(top, key=lambda c: -c["a"])[:8] if v < 2 else []):
            x0, y0, x1, y1 = c["box"]
            a0, a1 = S2((x0, y0)), S2((x1 + 1, y1 + 1))
            bw, bh = int(a1[0] - a0[0]), int(a1[1] - a0[1])
            if bw < 6 or bh < 6:
                continue
            if max(x1 - x0, y1 - y0) > 0.18 * w:               # 大块：只画四角括号 + 标签，不盖住画面
                L_ = 14 * SS
                for (qx, qy, sx_, sy_) in ((a0[0], a0[1], 1, 1), (a1[0], a0[1], -1, 1), (a0[0], a1[1], 1, -1), (a1[0], a1[1], -1, -1)):
                    dr.line([(qx, qy + sy_ * L_), (qx, qy), (qx + sx_ * L_, qy)], fill=P["mid"], width=SS)
                text(dr, (a0[0], a0[1] - 14 * SS), f"L0#{c['id']}  A {c['a']}px  box {x1 - x0 + 1}×{y1 - y0 + 1}", P["mid"], 9, knock=P["bg"])
                continue
            m = Image.fromarray((c["mask"] * 255).astype(np.uint8)).resize((bw, bh), Image.NEAREST)
            fill = Image.new("RGB", (bw, bh), P["box"])
            fill.paste(Image.new("RGB", (bw, bh), mix(P["ink"], 0.85, P["box"])), (0, 0), m)
            img.paste(fill, (int(a0[0]), int(a0[1])))
            dr.rectangle([a0[0], a0[1], a1[0], a1[1]], outline=P["mid"], width=SS)
            text(dr, (a0[0], a0[1] - 14 * SS), f"L0#{c['id']}  A {c['a']}px", P["mid"], 9, knock=P["bg"])
        # ⑤ 高层节点：小方框；新生的（< 0.5 s）带标签
        born = trk[2].born if len(trk) > 2 else {}
        for c in hi:
            p = S2(c["c"])
            age = (f - born.get(c["id"], f)) / FPS
            r = (3 + 2 * ease_pop(age)) * SS
            dr.rectangle([p[0] - r, p[1] - r, p[0] + r, p[1] + r], outline=P["ink"], width=SS)
        lab_c = sorted(mid, key=lambda c: -c["a"])[:14]
        for c in lab_c:
            p = S2(c["c"])
            r = 4 * SS
            dr.rectangle([p[0] - r, p[1] - r, p[0] + r, p[1] + r], outline=P["accent"], width=SS)
            X_, Y_ = c["c"][0] * src_scale, c["c"][1] * src_scale
            text(dr, (p[0] + 8 * SS, p[1] - 16 * SS), f"x: {X_:.0f}  y: {Y_:.0f}", P["accent"], 10, knock=None)
        # ⑥ 强起音：最亮的高层连通域开一个放大窗（源像素 ×4，0.5 s 衰减），引线连回去 —— 拍点作用在被追踪的那个东西上
        for ts, sv in strong:
            age = t - ts
            if not (0 <= age < 0.9) or not hi:
                continue
            k = int(round(ts * FPS))
            k = min(max(k, 0), T - 1)
            cand = data[k][2] if len(data[k]) > 2 else []
            if v >= 3:                                         # v3：优先「起音前后 0.5 s 内新出现的」高光（= 这一下打出来的东西），没有才取最亮最大
                bn = trk[2].born
                new = [c for c in cand if k - bn.get(c["id"], -99) <= 12]
                cand = new or cand
            tgt = max(cand, key=lambda c: c["lum"] * c["a"], default=None)
            if tgt is None:
                continue
            e = ease_pop(age, 0.25)
            cx, cy = tgt["c"]
            half = 18
            crop = frames[k][max(0, int(cy) - half):int(cy) + half, max(0, int(cx) - half):int(cx) + half]   # 起音那一帧的像素（定格的回想）
            if crop.size == 0:
                continue
            mag = int((150 + 60 * sv) * SS)
            ci = Image.fromarray(crop).resize((mag, mag * crop.shape[0] // max(crop.shape[1], 1)), Image.NEAREST)
            if P["thumb"] == "gray":
                ci = ci.convert("L").convert("RGB")
            ci = Image.fromarray((np.asarray(ci, np.float32) * (0.35 + 0.65 * e)).astype(np.uint8))
            p = S2((cx, cy))
            bx = p[0] + (90 * SS if p[0] < W * SS * 0.6 else -90 * SS - mag)
            by = p[1] - mag - 40 * SS
            if v >= 2:                                         # 放大镜进左侧栏（标本栏右边），引线横穿过去
                bx = 330 * SS
                by = p[1] - mag / 2
            by = min(max(by, 30 * SS), H * SS - mag - 30 * SS)
            img.paste(ci, (int(bx), int(by)))
            col = mix(P["hot"], 0.4 + 0.6 * e, P["bg"])
            dr.rectangle([bx, by, bx + ci.width, by + ci.height], outline=col, width=SS)
            corners = [(bx, by), (bx + ci.width, by), (bx, by + ci.height), (bx + ci.width, by + ci.height)]
            cn_ = min(corners, key=lambda q: (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2)
            dr.line([tuple(p), cn_], fill=col, width=SS)
            text(dr, (bx, by + ci.height + 4 * SS), f"#{tgt['id']}  x: {cx * src_scale:.0f} y: {cy * src_scale:.0f}  A {tgt['a']}px  "
                 f"L̄ {tgt['lum']:.2f}  θ {sv:.2f} @ {t0_song + ts:.2f}s", col, 9, knock=P["bg"])
        # HUD
        nn = [len(x) for x in lv]
        hud = [f"COMPONENT TREE · {len(levels)} LEVELS  L ≥ " + " / ".join(f"{v:.2f}" for v in levels),
               f"NODES  " + "  ".join(f"L{i} {n_}" for i, n_ in enumerate(nn)) + f"   MST |E| {max(0, nn[1] - 1) if len(nn) > 1 else 0}",
               f"TRACKS  {trk[1].next} ids   t = {t0_song + t:7.2f}s   frame {f:03d}"]
        for k_, line in enumerate(hud):
            text(dr, (56 * SS, (40 + 17 * k_) * SS), line, P["ink"] if k_ == 0 else P["mid"], 12 if k_ == 0 else 11)
        yield f, img.resize((W, H), Image.LANCZOS), dict(f=f, t=round(t0_song + t, 3), n=nn, zoom=round(z, 3))


def _specimens(dr, img, P, top, S2, src_scale, w, keep=None):
    """标本栏：面积最大的 6 个低层连通域，按追踪 ID 排（顺序稳定），左侧竖排：灰底 + 真遮罩 + 标签；引线连到它在脸上的外接框（脸上只画四角）"""
    x0, y0, cw, chh, gap = 64, 120, 230, 96, 22
    ranked = sorted(top, key=lambda c: -c["a"])
    if keep is not None:                                  # v3 滞回：上一帧展示过、这一帧还在前 10 的保留，空位再按面积补 → 标本栏不每帧换
        top10 = {c["id"]: c for c in ranked[:10]}
        sel = [top10[i] for i in keep if i in top10][:6]
        for c in ranked:
            if len(sel) >= 6:
                break
            if c not in sel:
                sel.append(c)
        sel = sorted(sel, key=lambda c: c["id"])          # 位置按 ID 定死（滞回下不跳）
    else:
        sel = sorted(ranked[:6], key=lambda c: c["c"][1])   # 按脸上的高度排 → 引线少交叉
    for k, c in enumerate(sel):
        bx0, by0, bx1, by1 = c["box"]
        a0, a1 = S2((bx0, by0)), S2((bx1 + 1, by1 + 1))
        L_ = 10 * SS
        for (qx, qy, sx_, sy_) in ((a0[0], a0[1], 1, 1), (a1[0], a0[1], -1, 1), (a0[0], a1[1], 1, -1), (a1[0], a1[1], -1, -1)):
            dr.line([(qx, qy + sy_ * L_), (qx, qy), (qx + sx_ * L_, qy)], fill=P["mid"], width=SS)
        m = c["mask"]
        sc = min(cw / m.shape[1], chh / m.shape[0])
        mw, mh = max(2, int(m.shape[1] * sc * SS)), max(2, int(m.shape[0] * sc * SS))
        sy = y0 + k * (chh + gap + 14)
        fill = Image.new("RGB", (mw, mh), P["box"])
        mi = Image.fromarray((m * 255).astype(np.uint8)).resize((mw, mh), Image.NEAREST)
        fill.paste(Image.new("RGB", (mw, mh), mix(P["ink"], 0.85, P["box"])), (0, 0), mi)
        img.paste(fill, (x0 * SS, sy * SS))
        dr.rectangle([x0 * SS, sy * SS, x0 * SS + mw, sy * SS + mh], outline=P["mid"], width=SS)
        text(dr, (x0 * SS, (sy - 13) * SS), f"L0#{c['id']}  A≈{c['a'] * src_scale ** 2:.0f}px  {(bx1 - bx0 + 1) * src_scale:.0f}×{(by1 - by0 + 1) * src_scale:.0f}  "
             f"c({c['c'][0] * src_scale:.0f},{c['c'][1] * src_scale:.0f})", P["mid"], 9, knock=P["bg"])
        q0 = np.array([x0 * SS + mw, sy * SS + mh / 2])
        q1 = np.array([a0[0], (a0[1] + a1[1]) / 2])
        mid_x = q0[0] + 30 * SS
        dr.line([tuple(q0), (mid_x, q0[1]), tuple(q1)], fill=P["dim"], width=SS)
        _ = w
    return [c["id"] for c in sel]


def _hist_panel(dr, P, frame, levels, counts, f):
    """左下：这一帧的亮度直方图（对数计数，64 格）+ 三条阈值线；下面三条曲线 = 每层节点数随时间（到当前帧）"""
    x0, y0, pw, ph = 64, H - 210, 360, 70
    im = frame.astype(np.float32)
    lum = (0.2126 * im[..., 0] + 0.7152 * im[..., 1] + 0.0722 * im[..., 2]) / 255
    hcount, _ = np.histogram(lum, bins=64, range=(0, 1))
    hv = np.log1p(hcount) / np.log1p(hcount.max() + 1)
    dr.rectangle([(x0 - 8) * SS, (y0 - 22) * SS, (x0 + pw + 8) * SS, (y0 + ph + 118) * SS], fill=P["bg"])
    text(dr, (x0 * SS, (y0 - 18) * SS), "LUMINANCE HISTOGRAM  log count · 64 bins", P["mid"], 9)
    bw = pw / 64
    for i, v_ in enumerate(hv):
        xx = x0 + i * bw
        dr.rectangle([xx * SS, (y0 + ph * (1 - v_)) * SS, (xx + bw - 1) * SS, (y0 + ph) * SS], fill=P["dim"])
    for li, thr in enumerate(levels):
        xx = (x0 + thr * pw) * SS
        n_above = int((lum >= thr).sum())
        col = P["accent"] if li == 1 else P["ink"]
        dr.line([(xx, (y0 - 4) * SS), (xx, (y0 + ph + 4) * SS)], fill=col, width=SS)
        text(dr, (xx + 3 * SS, (y0 + 2 + li * 11) * SS), f"L{li} ≥{thr:.2f}  {n_above}px", col, 8)
    gy = y0 + ph + 26
    text(dr, (x0 * SS, (gy - 14) * SS), f"NODES / FRAME   L0 {counts[f, 0]}  L1 {counts[f, 1]}  L2 {counts[f, 2]}", P["mid"], 9)
    T = len(counts)
    mx = max(1, counts.max())
    for li in range(counts.shape[1]):
        ys = gy + 70 - counts[:f + 1, li] / mx * 66
        xs = x0 + np.arange(f + 1) / max(1, T - 1) * pw
        if f >= 1:
            dr.line(list(zip(xs * SS, ys * SS)), fill=[P["ink"], P["accent"], P["mid"]][li], width=SS)
    dr.line([(x0 * SS, (gy + 70) * SS), ((x0 + pw) * SS, (gy + 70) * SS)], fill=P["faint"], width=SS)


def cmd_track(a):
    src = Path(a.src)
    fr = read_video(src, a.src_start, a.dur, a.aw, a.ah)
    full = read_video(src, a.src_start, 0.05)                   # 只为拿源分辨率
    src_scale = full.shape[2] / a.aw
    levels = [float(x) for x in a.levels.split(",")]
    mins = [int(x) for x in a.min_area.split(",")]
    print(f"源 {src.name}  {len(fr)} 帧  分析 {a.aw}×{a.ah}（×{src_scale:.2f} = 源像素）")
    data, trk = analyse_track(fr, levels, mins, a.gate, a.crop_bottom)
    print("每层平均节点数:", [round(np.mean([len(d[i]) for d in data]), 1) for i in range(len(levels))])
    OUT_B.mkdir(parents=True, exist_ok=True)
    if a.still is not None:
        f0 = int(round(a.still * FPS))
        for f, im, _ in render_track(fr, data, trk, levels, a.palette, a.start, src_scale, v=a.v):
            if f == f0:
                im.save(a.out)
                print("->", a.out)
                return
    vid = safe_out(OUT_B, a.name + f"_{a.palette}", ".mp4")
    dur = len(fr) / FPS
    w = open_writer(vid, a.start, dur)
    picks = set(np.linspace(0, len(fr) - 1, 16).round().astype(int).tolist())
    sheet, meta = [], []
    for f, im, m in render_track(fr, data, trk, levels, a.palette, a.start, src_scale, v=a.v):
        w.stdin.write(np.asarray(im).tobytes())
        meta.append(m)
        if f in picks:
            sheet.append((f"{m['t']:.2f}s  f{f}", im))
    w.stdin.close()
    w.wait()
    contact(sheet, vid.with_name(vid.stem + "_接触表.jpg"))
    side = dict(tool="tools/info_graph.py track", src=str(src), src_start=a.src_start, song_start=a.start, levels=levels,
                min_area=mins, gate=a.gate, analysis_size=[a.aw, a.ah], src_scale=src_scale, frames=meta,
                nodes=[[[dict(id=c["id"], c=[round(v * src_scale, 1) for v in c["c"]], a=c["a"], parent=c.get("parent"))
                         for c in lvl] for lvl in d] for d in data])
    vid.with_suffix(".json").write_text(json.dumps(side, ensure_ascii=False, default=_jsonable), "utf-8")
    print("->", vid)


def cmd_memory(a):
    if a.v >= 4:
        return cmd_memory_v4(a)
    seg = (a.seg0, a.seg1)
    mem = build_memory(seg=seg, k=a.k)
    sch = schedule(mem, seg, thr=a.thr)
    print(f"节点 {len(mem.nodes)}  边 {len(mem.edges)}  John = #{mem.john}  在第 {sch['jk'] + 1} 次弹出被找到；"
          f"路径 {[mem.nodes[i]['name'] for i in mem.path]}  conf {mem.conf:.3f}")
    print("弹出时刻:", {mem.nodes[u]["name"]: round(tt + seg[0], 2) for u, (tt, _) in sch["pop"].items()})
    print("路径变红:", sch["path"], " 脉冲:", [round(x + seg[0], 2) for x, _ in sch["pulses"]])
    OUT = OUT_A
    OUT.mkdir(parents=True, exist_ok=True)
    if a.still is not None:
        f = int(round(float(a.still) * FPS))
        for _, im, _ in render_memory(mem, sch, seg, a.palette, only={f}, v=a.v):
            p = Path(a.out or (Path(a.scratch) / f"memory_{a.palette}_{float(a.still):.2f}.png"))
            im.save(p)
            print("->", p)
        return
    vid = safe_out(OUT, a.name + f"_{a.palette}", ".mp4")
    dur = seg[1] - seg[0]
    w = open_writer(vid, seg[0], dur)
    T = int(round(dur * FPS))
    picks = set(np.linspace(0, T - 1, 16).round().astype(int).tolist())
    sheet, meta = [], []
    for f, im, m in render_memory(mem, sch, seg, a.palette, v=a.v):
        w.stdin.write(np.asarray(im).tobytes())
        meta.append(m)
        if f in picks:
            sheet.append((f"{m['t']:.2f}s  f{f}", im))
    w.stdin.close()
    w.wait()
    contact(sheet, vid.with_name(vid.stem + "_接触表.jpg"))
    side = dict(
        tool="tools/info_graph.py memory", seg=seg, palette=a.palette, k=a.k, thr=a.thr,
        features=mem.keys, pca_loadings=mem.pcs.round(4).tolist(), pca_var=mem.evr.round(4).tolist(),
        nodes=[{k_: v for k_, v in nd.items() if k_ != "thumb"} for nd in mem.nodes],
        edges=mem.edges, dijkstra=[dict(node=u, d=round(d, 4), parent=p) for u, d, p in mem.order],
        path=mem.path, john=mem.john, confidence=round(mem.conf, 4),
        pop_times={str(u): round(tt + seg[0], 3) for u, (tt, _) in sch["pop"].items()},
        path_time=(round(sch["path"][0] + seg[0], 3) if sch["path"] else None),
        pulses=[round(x + seg[0], 3) for x, _ in sch["pulses"]], frames=meta)
    vid.with_suffix(".json").write_text(json.dumps(side, ensure_ascii=False, indent=1, default=_jsonable), "utf-8")
    print("->", vid)


def cmd_memory_v4(a):
    seg = (a.seg0, V4["end_t"])
    mem = build_memory(seg=seg, k=a.k)
    sch = schedule_v4(mem, seg)
    order = [u for u, _, _ in mem.order]
    print(f"心魔 = #{sch['unstable']} {mem.nodes[sch['unstable']]['name']}（第 {sch['ku'] + 1} 个弹出）  John = 第 {sch['jk'] + 1} 个")
    print("弹出:", [(order.index(u) + 1, round(tt + seg[0], 3), sch["kinds"][u]) for u, (tt, _) in sorted(sch["pop"].items(), key=lambda kv: kv[1][0])])
    OUT_A.mkdir(parents=True, exist_ok=True)
    if a.still is not None:
        fs = {int(round(float(x) * FPS)) for x in str(a.still).split(",")}
        ims = [(f"{seg[0] + f / FPS:.2f}s f{f}", im) for f, im, _ in render_memory_v4(mem, sch, seg, a.palette, only=fs)]
        outp = Path(a.out or (Path(a.scratch) / f"v4_{a.palette}.jpg"))
        contact(ims, outp, cols=2, tw=960)
        print("->", outp)
        return
    vid = safe_out(OUT_A, a.name + f"_{a.palette}", ".mp4")
    dur = V4["end_t"] - seg[0]
    w = open_writer(vid, seg[0], dur)
    T = int(round(dur * FPS))
    picks = set(np.linspace(0, T - 1, 20).round().astype(int).tolist()) | {int(round((t - seg[0]) * FPS)) for t in
                                                                          (V4["unstable_t"] + 0.6, V4["unstable_t"] + 1.6, V4["cut_t"] + 0.05, *V4["hops"])}
    sheet, meta = [], []
    for f, im, m in render_memory_v4(mem, sch, seg, a.palette):
        w.stdin.write(np.asarray(im).tobytes())
        meta.append(m)
        if f in picks:
            sheet.append((f"{m['t']:.2f}s  f{f}", im))
    w.stdin.close()
    w.wait()
    contact(sheet, vid.with_name(vid.stem + "_接触表.jpg"))
    side = dict(tool="tools/info_graph.py memory --v 4", seg=[seg[0], V4["end_t"]], palette=a.palette, v4=V4,
                unstable=sch["unstable"], unstable_pop=sch["ku"] + 1, john_pop=sch["jk"] + 1, confidence=round(mem.conf, 4),
                path=mem.path, pop_times={str(u): [round(tt + seg[0], 3), sch["kinds"][u]] for u, (tt, _) in sch["pop"].items()},
                hops=list(V4["hops"]), dijkstra=[dict(node=u, d=round(d, 4), parent=p) for u, d, p in mem.order],
                nodes=[{k_: v for k_, v in nd.items()} for nd in mem.nodes], edges=mem.edges, frames=meta)
    vid.with_suffix(".json").write_text(json.dumps(side, ensure_ascii=False, indent=1, default=_jsonable), "utf-8")
    print("->", vid)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("memory")
    m.add_argument("--palette", default="bwr", choices=list(PALETTES))
    m.add_argument("--name", default="22_信息图_记忆图谱_一个名字_v1")
    m.add_argument("--seg0", type=float, default=115.51)
    m.add_argument("--seg1", type=float, default=128.31)
    m.add_argument("--k", type=int, default=3)
    m.add_argument("--v", type=int, default=3, help="版式版本：1 = 首版；2 = John 大框常驻 + 右上 z 向量对照 + 开场更近更快；"
                                                    "3 = 注意力分层标签 + 1080p 最小字号 + 开场故事线已在长 + FOUND AT POP")
    m.add_argument("--thr", type=float, default=0.3, help="起音威胁峰 ≥ thr 才弹出一个节点")
    m.add_argument("--still", default=None, help="只渲这一秒（相对段首）的一帧；v4 可逗号分隔多个 → 拼一张")
    m.add_argument("--out", default=None)
    m.add_argument("--scratch", default=".")
    t = sub.add_parser("track")
    t.add_argument("--src", required=True, help="源视频")
    t.add_argument("--src-start", type=float, default=0.0, help="从源视频第几秒开始")
    t.add_argument("--start", type=float, required=True, help="这一段对应歌曲的播放时间（配乐 + 起音）")
    t.add_argument("--dur", type=float, default=None)
    t.add_argument("--aw", type=int, default=640, help="分析分辨率（宽）")
    t.add_argument("--ah", type=int, default=360)
    t.add_argument("--levels", default="0.30,0.55,0.80", help="成分树的阈值（亮度 0–1，低 → 高）")
    t.add_argument("--min-area", default="60,12,3", help="每层最小面积（分析像素）")
    t.add_argument("--gate", type=float, default=24.0, help="追踪关联的最大位移（分析像素 / 帧）")
    t.add_argument("--crop-bottom", type=int, default=0, help="底部这么多行不参与（源视频自带 HUD）")
    t.add_argument("--palette", default="bwr", choices=list(PALETTES))
    t.add_argument("--v", type=int, default=3, help="版式：1 = 遮罩框盖在原位；2 = 标本栏 + 直方图；3 = 标本栏滞回 + 放大镜优先新生高光")
    t.add_argument("--name", default="22_信息图_亮斑追踪_v1")
    t.add_argument("--still", type=float, default=None)
    t.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.cmd == "memory":
        cmd_memory(a)
    else:
        cmd_track(a)


if __name__ == "__main__":
    main()
