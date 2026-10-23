#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🕸感官空间-入口-「感官 / 记忆的 3D 空间」：四条 Demucs 分轨各是一张随时间滚动的频谱点云平面，按 Z 排开；
感官通道 + 期待审计（意外度尖刺）是最前面的一层；记忆图谱的节点浮在最前面，搜索时镜头在层之间穿行（👤2026-10-23）。
⭐符号说真话：每个点 = 某条分轨在某个时刻、某个频带的真实能量；尖刺 = 感官.json expect 的真实意外度（比特）；节点 = 剪辑表的真实镜头。

空间（世界坐标）：
  X = 时间（鱼眼）：播放头在 x = 0；过去在左，x = −X1·ln(1 + Δt/τ)/ln(1 + t_now/τ)（τ = 1.5 s）→ 刚听到的展开、远的压扁；未来不画（她还没听到）
  Y = 频率（对数，30 Hz → 10 kHz，96 带）
  Z = 分轨深度：drums 0（离她最近的冲击）/ other 420（动机、抽搐）/ bass 840（心跳）/ vocals 1260（她自己的声音，最深 —— 这一段还没回来）
  感官层 z = −320（四条通道曲线 + 意外度尖刺，红 = 显著度 ≥ max(2, 本曲 92 百分位)）；记忆层 z = −680（节点浮在频谱上方）
拨动：每个起音（威胁峰 ≥ 0.3）落在那一刻频谱通量最大的那条分轨上 → 那一层从播放头处起一圈膜波（只这一层动），波前红色。
镜头：同 info_graph v6 的前馈排程（schedule_v4 + 分组 + 速度上限 + 提前 2 帧 + 1 − (1 − u)^3.5 快起慢收）；
  第一段（到 2:01.96）在层外冷静地看；2:01.96 冲进层间；John 2:04.51 浮出、2:05.05 NOTICED、2:06.554 起四声确认 / 锁定。

用法：python tools/sense_space.py [--still 秒,秒,...] [--name 24_感官空间_一个名字_v1]
输出：MV/样片/信息可视化/<name>.mp4（配乐）+ _接触表.jpg + _镜头速度.png + .json。不覆盖已有文件。
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import info_graph as G  # noqa: E402

ROOT = G.ROOT
sys.path.insert(0, str(ROOT / "src"))
import sensorium  # noqa: E402

W, H, FPS, SS = G.W, G.H, G.FPS, G.SS
STEMS = [("drums", 0.0), ("other", 420.0), ("bass", 840.0), ("vocals", 1260.0)]
Z_SENSE, Z_MEM = -320.0, -680.0
X1, TAU, HH = 1700.0, 1.5, 700.0
NB, NC, NR = 96, 300, 32
RIDGE = 55.0                     # 山脊隆起：满能量 = 2.5 个行距
F_LO, F_HI = 30.0, 10000.0
FOCAL = (W / 2) / math.tan(math.radians(25))
SR = 22050
HOP = 0.02
P = G.PALETTES["bwr"]


# ───────────────────────── 数据 ─────────────────────────
def stem_bands(name, t_end):
    """分轨 0 → t_end 的对数频带功率（dB，帧 = 20 ms）。缓存。"""
    G.CACHE.mkdir(parents=True, exist_ok=True)
    cp = G.CACHE / f"space_{name}_{t_end:.2f}.npy"
    if cp.exists():
        return np.load(cp)
    x = sensorium.load_audio(G.MV / "分轨" / f"{name}.wav", 0.0, t_end, sr=SR)
    n, hop = 2048, int(HOP * SR)
    frames = (len(x) - n) // hop + 1
    win = np.hanning(n)
    fr = np.fft.rfftfreq(n, 1 / SR)
    edges = np.geomspace(F_LO, F_HI, NB + 1)
    bi = np.clip(np.searchsorted(edges, fr) - 1, -1, NB)
    out = np.zeros((frames + n // (2 * hop), NB))
    for a in range(0, frames, 512):
        idx = np.arange(a, min(frames, a + 512))
        seg = x[idx[:, None] * hop + np.arange(n)[None, :]] * win
        Pw = np.abs(np.fft.rfft(seg, axis=1)) ** 2
        ok = (bi >= 0) & (bi < NB)
        for b in range(NB):
            m = ok & (bi == b)
            if m.any():
                out[idx + n // (2 * hop), b] = Pw[:, m].sum(1)          # 帧时间 = 窗口中心
    db = 10 * np.log10(out + 1e-12)
    np.save(cp, db)
    return db


def fisheye_x(dt, t_now):
    return -X1 * np.log1p(np.maximum(dt, 0) / TAU) / math.log1p(max(t_now, 1e-3) / TAU)


def fisheye_inv(x, t_now):
    u = -x / X1
    return TAU * ((1 + t_now / TAU) ** u - 1)


# ───────────────────────── 相机 ─────────────────────────
def cam_basis(st):
    la = st[:3]; yaw, pitch, dist = st[3], st[4], st[5]
    d = np.array([math.cos(pitch) * math.sin(yaw), math.sin(pitch), -math.cos(pitch) * math.cos(yaw)])
    pos = la + dist * d
    f = (la - pos) / np.linalg.norm(la - pos)
    r = np.cross([0, 1, 0], f); r /= np.linalg.norm(r)
    u = np.cross(f, r)
    return pos, f, r, u


def project(Pw, st):
    pos, f, r, u = cam_basis(st)
    q = Pw - pos
    z = q @ f
    zc = np.maximum(z, 1.0)
    return np.stack([W / 2 + FOCAL * (q @ r) / zc, H / 2 - FOCAL * (q @ u) / zc], -1), z


def camera_schedule(mem, sch, seg, T, node_pos, base):
    """前馈：每个事件提前 2 帧起步，状态 (lookat xyz, yaw, pitch, dist) 按 1 − (1 − u)^3.5 走；
    lookat 的移动按「屏幕角速度」限速（第一段 220、第二段 700 px/s）；限速后追不上余速的那一拍并进当前移动。"""
    V6 = G.V6
    lead = V6["lead_frames"] / FPS
    pop = sch["pop"]
    hops, t_cut = sch["hops"], sch["cut"]
    moves = []
    last_mv, group = -9.0, []
    for u, (tu, _) in sorted(pop.items(), key=lambda kv: kv[1][0]):
        if u == mem.john:
            continue
        group.append(u)
        if tu < t_cut:
            if tu - last_mv >= V6["group_s"]:
                la = 0.65 * base[:3] + 0.35 * node_pos[group].mean(0)
                moves.append((tu, dict(la=la), 1.1, "pop-group"))
                last_mv, group = tu, []
        else:
            la = 0.4 * np.array([-700.0, 380.0, 0.0]) + 0.6 * node_pos[u]
            moves.append((tu, dict(la=la), 0.5, "pop"))
            group = []
    jp = node_pos[mem.john]
    fr = [u for u, (tu, _) in pop.items() if tu >= t_cut][:5]
    rush = node_pos[fr].mean(0) * 0.6 + np.array([-700.0, 380.0, 100.0]) * 0.4
    moves.append((t_cut, dict(la=rush, yaw=0.62, pitch=0.12, dist=1700.0), 1.1, "rush-in"))     # 冲进层间（相机从层外推到记忆层和频谱层之间）
    moves.append((V6["faint_t"] - seg[0], dict(la=0.8 * rush + 0.2 * jp), 0.6, "faint"))
    moves.append((V6["notice_t"] - seg[0], dict(la=0.5 * rush + 0.5 * jp, dist=1600.0), 0.6, "noticed"))
    moves.append((hops[0], dict(la=0.2 * rush + 0.8 * jp, dist=1450.0, yaw=0.66), 0.6, "recognize"))
    for k in (1, 2, 3):
        moves.append((hops[k], dict(la=0.1 * rush + 0.9 * jp, dist=1450.0 - 110 * k), 0.45, f"confirm{k}"))
    moves.sort(key=lambda m: m[0])
    st0 = base.copy()
    cur = dict(t_s=-lead, x0=st0 + np.array([180.0, -40.0, -120.0, 0.08, 0.02, 350.0]), tgt=st0, D=1.4)
    out = np.zeros((T, 6))
    acc = []
    mi = 0
    pE = G.V5["ease_p"]
    for f in range(T):
        t = f / FPS
        while mi < len(moves) and moves[mi][0] - lead <= t:
            te, tg, D, kind = moves[mi]
            ts = te - lead
            u_ = (ts - cur["t_s"]) / cur["D"]
            now = cur["x0"] + (cur["tgt"] - cur["x0"]) * G.ease_ff(u_)
            tgt = now.copy()
            if "la" in tg:
                tgt[:3] = tg["la"]
            for k_, i_ in (("yaw", 3), ("pitch", 4), ("dist", 5)):
                if k_ in tg:
                    tgt[i_] = tg[k_]
            dl = tgt[:3] - now[:3]
            cap = V6["cap1"] if te < t_cut else V6["cap2"]
            avg = np.linalg.norm(dl) / max(now[5], 1) * FOCAL / D          # 屏幕像素 / 秒（近似：角速度 × 焦距）
            if avg > cap:
                tgt[:3] = now[:3] + dl * cap / avg
            v_res = np.linalg.norm(cur["tgt"][:3] - cur["x0"][:3]) / max(now[5], 1) * FOCAL * pE * max(0, 1 - u_) ** (pE - 1) / cur["D"]
            v_new = np.linalg.norm(tgt[:3] - now[:3]) / max(now[5], 1) * FOCAL * pE / D
            if v_new < 1.2 * v_res and set(tg) == {"la"}:
                mi += 1
                continue
            cur = dict(t_s=ts, x0=now, tgt=tgt, D=D)
            acc.append((te, kind))
            mi += 1
        out[f] = cur["x0"] + (cur["tgt"] - cur["x0"]) * G.ease_ff((t - cur["t_s"]) / cur["D"])
        out[f, 0] += 8.0 * t                                              # 很慢的漂移（不停在静止）
    return out, acc


def screen_speed(cams, refs):
    sp = np.zeros(len(cams))
    prev = None
    for f, st in enumerate(cams):
        q, z = project(refs, st)
        ok = z > 50
        if prev is not None:
            m = ok & prev[1]
            sp[f] = np.median(np.hypot(*(q[m] - prev[0][m]).T)) if m.any() else 0
        prev = (q, ok)
    return sp


def alignment(sp, events, T, lead=2):
    fes = sorted({int(round(t * FPS)) for t, _ in events})
    out = []
    for t, k in events:
        fe = int(round(t * FPS))
        nxt = [x for x in fes if x > fe]
        a, b = max(1, fe - 1), min(T - 1, fe + 6, (nxt[0] - lead) if nxt else T - 1)
        if b < a:
            continue
        out.append((t, k, a + int(np.argmax(sp[a:b + 1])) - fe))
    return out


# ───────────────────────── 渲染 ─────────────────────────
def onset_layer(stems_db, ons):
    """每个起音落在哪一条分轨：起音前后 ±2 帧里，各分轨正的频谱通量（dB 差，限 0–40 dB 区间）之和最大的那条"""
    out = []
    for t, s in ons:
        i = int(round(t / HOP))
        best, bk = -1, 0
        for k, db in enumerate(stems_db):
            a = db[max(0, i - 3):i + 3]
            if len(a) < 2:
                continue
            fl = np.clip(np.diff(np.clip(a, -80, -10), axis=0), 0, None).sum()
            if fl > best:
                best, bk = fl, k
        out.append((t, s, bk))
    return out


def render(seg, only=None):
    G._MINF["on"] = True
    s0, s1 = seg
    mem = G.build_memory(seg=seg)
    sch = G.schedule_v4(mem, seg, john_hop=0)
    T = int(round((s1 - s0) * FPS))
    _, fps_s, ch, ev = G.load_sense()
    sense = json.loads(G.SENSE.read_text("utf-8"))
    exv = [e for e in sense["expect"]["events"] if e["t"] <= s1]
    sal_all = np.array([e.get("sal") or 0 for e in sense["expect"]["events"]])
    sal_thr = max(2.0, float(np.percentile(sal_all, 92)))
    dbs = [stem_bands(n, s1 + 0.5) for n, _ in STEMS]
    pool = np.concatenate([d[d > -100].ravel()[::7] for d in dbs])      # 四条分轨共用一把尺 → 安静的分轨就是暗的（真）
    lo = [float(np.percentile(pool, 50))] * len(dbs)
    hi = [float(np.percentile(pool, 99.7))] * len(dbs)
    ons = [(float(t), G.onset_strength(ch, fps_s, t)) for t in ev["onsets"] if t <= s1]
    plucks = [p for p in onset_layer(dbs, [o for o in ons if o[1] >= 0.3]) if p[0] >= s0 - 1.0]
    # 记忆节点在空间里的位置：x = 鱼眼时间（以段首为「此刻」定位，之后随播放头滑动），y = 频谱上方 + PC2，z = 记忆层
    pc = np.array([nd["pca"] for nd in mem.nodes])

    def node_pos_at(t_now):
        P_ = np.zeros((len(mem.nodes), 3))
        for i, nd in enumerate(mem.nodes):
            tm = (nd["t0"] + nd["t1"]) / 2 if nd["src"] != "此刻" else t_now
            P_[i] = [fisheye_x(t_now - tm, t_now), HH * 0.55 + 110 * pc[i, 1], Z_MEM]   # 浮在自己那段时间的频谱前面
        return P_
    np0 = node_pos_at(s0 + 6.0)
    base = np.array([-760.0, 330.0, 300.0, 0.82, 0.17, 3000.0])
    cams, cam_ev = camera_schedule(mem, sch, seg, T, np0, base)
    refs = np.array([[x, y, z] for x in np.linspace(-X1, 0, 7) for y in (0, HH / 2, HH) for z in (Z_MEM, Z_SENSE, 0, STEMS[-1][1])])
    thumbs = {}
    for nd in mem.nodes:
        if nd["src"] != "此刻":
            im = G.thumb(nd["src"], (nd["t0"] + nd["t1"]) / 2, nd["t0"], nd["speed"]).convert("L").convert("RGB")
            thumbs[nd["i"]] = im
    par = {u: p for u, _, p in mem.order}
    path = mem.path
    hops, t_cut = sch["hops"], sch["cut"]
    lit_at = [hops[1], hops[1], hops[2], hops[3]]
    un = sch["unstable"]
    und = mem.nodes[un]
    tk, tk_meta = G.takens_points(und["t0"], und["t1"])
    zj = np.array(mem.nodes[mem.john]["z"])
    cos_u = float(np.array(und["z"]) @ zj / np.linalg.norm(und["z"]) / np.linalg.norm(zj))
    ons_rel = [(t - s0, s) for t, s in ons]
    xs_cols = np.linspace(-X1, 0, NC)
    ys_r = (np.arange(NR) + 0.5) / NR * HH
    meta = []
    for f in range(T):
        if only is not None and f not in only:
            continue
        t = f / FPS
        tn = s0 + t
        st = cams[f]
        # ── 点云：四张频谱平面（只画到「此刻」，鱼眼时间）
        acc = np.zeros((H, W))
        acc_r = np.zeros((H, W))
        dts = fisheye_inv(xs_cols, tn)
        tt = tn - dts
        for k, (name, zk) in enumerate(STEMS):
            db = dbs[k]
            # 每列 = 鱼眼一格覆盖的时间段里的平均功率（相邻列的中点做边界）
            edges_t = np.concatenate([[tt[0] + (tt[0] - tt[1]) / 2], (tt[1:] + tt[:-1]) / 2, [tn]])
            ia = np.clip((np.minimum(edges_t[:-1], edges_t[1:]) / HOP).astype(int), 0, len(db) - 1)
            ib = np.clip((np.maximum(edges_t[:-1], edges_t[1:]) / HOP).astype(int) + 1, 1, len(db))
            cs = np.vstack([np.zeros(NB), np.cumsum(10 ** (db / 10), axis=0)])
            pw = (cs[ib] - cs[ia]) / np.maximum(ib - ia, 1)[:, None]
            pw = pw.reshape(NC, NR, NB // NR).sum(2)                   # 96 带 → 32 条山脊（每条 = 3 个相邻频带的功率和）
            v = np.clip((10 * np.log10(pw + 1e-12) - lo[k] - 4.8) / (hi[k] - lo[k]), 0, 1)
            # 山脊线：每个频带一条沿时间的线，往上隆起 ∝ 能量（点沿线密采样 → 连成线）
            xf = np.linspace(-X1, 0, NC * 3)
            vf = np.stack([np.interp(xf, xs_cols, v[:, b]) for b in range(NR)], 1)
            X = np.repeat(xf[:, None], NR, 1)
            Y = ys_r[None, :] + vf * RIDGE
            v = vf
            Z = np.full_like(X, zk)
            red = np.zeros_like(X)
            for tp, sp_, kk in plucks:                          # 膜波：只这一层，从播放头 (x=0, 该层通量重心的频率) 起
                if kk != k or tp > tn or tn - tp > 1.0:
                    continue
                dtp = tn - tp
                r = np.hypot(X - 0.0, Y - HH * 0.35)
                front = 900 * dtp
                Z = Z - (70 + 90 * sp_) * np.exp(-dtp / 0.35) * np.sin(2 * np.pi * (r - front) / 260) * np.exp(-((r - front) / 220) ** 2)
                red = np.maximum(red, np.exp(-((r - front) / 40) ** 2) * np.exp(-dtp / 0.35))
            Pw = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
            q, zc = project(Pw, st)
            ok = zc > 60
            wv = (0.12 + 2.2 * v.ravel() ** 1.4) * np.clip(1400 / np.maximum(zc, 1), 0.3, 2.0)
            acc += G_splat(q[ok, 0], q[ok, 1], wv[ok])
            if red.max() > 0.02:
                acc_r += G_splat(q[ok, 0], q[ok, 1], (red.ravel() * 0.9)[ok])
        img_f = 1 - np.exp(-acc * 0.9)
        base_rgb = np.stack([img_f, img_f, img_f], -1) * np.array(P["ink"]) / 255.0
        rr = 1 - np.exp(-acc_r * 1.5)
        base_rgb = base_rgb * (1 - rr[..., None]) + rr[..., None] * np.array(P["hot"]) / 255.0
        bg = np.array(P["bg"]) / 255.0
        rgb = bg + (1 - bg) * base_rgb
        im = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).resize((W * SS, H * SS), Image.BILINEAR)
        dr = ImageDraw.Draw(im)

        def P2(p):
            q, z = project(np.atleast_2d(np.asarray(p, float)), st)
            return q[0] * SS, z[0]

        def line3(a, b, col, w=1, n=2, dash=None):
            pts = np.linspace(a, b, n)
            q, z = project(pts, st)
            if (z < 60).any():
                return
            q = q * SS
            s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(q, axis=0).T))])
            G.draw_path(dr, q, s, col, w, dash=dash)

        # 平面框 + 标签（层名、真实电平）
        i_now = int(tn / HOP)
        for k, (name, zk) in enumerate(STEMS):
            for a, b in (((-X1, 0, zk), (0, 0, zk)), ((-X1, HH, zk), (0, HH, zk)), ((0, 0, zk), (0, HH, zk))):
                line3(a, b, P["faint"], 1, n=12)
            lvl = 10 * np.log10(np.sum(10 ** (dbs[k][max(0, i_now - 10):i_now + 1] / 10), axis=1).mean() + 1e-12)
            q, z = P2((0, HH, zk))
            if z > 60:
                hit = any(kk == k and 0 <= tn - tp < 0.4 for tp, _, kk in plucks)
                G.text(dr, (q[0] + 8 * SS, q[1] - 16 * SS), f"{name.upper()}  z={zk:.0f}  {lvl:5.1f} dB", P["hot"] if hit else P["mid"], 12)
        for fz in (100, 1000, 5000):                          # 频率刻度（播放头那条边，最前一层）
            y = HH * math.log(fz / F_LO) / math.log(F_HI / F_LO)
            q, z = P2((0, y, 0))
            if z > 60:
                G.text(dr, (q[0] + 6 * SS, q[1] - 6 * SS), f"{fz if fz < 1000 else str(fz // 1000) + 'k'}Hz", P["dim"], 12)
        for back in (1, 5, 15, 40, 100):                      # 时间刻度（鱼眼）
            if back < tn:
                x = fisheye_x(back, tn)
                line3((x, -10, 0), (x, -30, 0), P["dim"], 1)
                q, z = P2((x, -46, 0))
                if z > 60:
                    G.text(dr, (q[0], q[1]), f"−{back}s", P["dim"], 12, anchor="ma")
        # 播放头平面（x = 0，跨所有层）
        line3((0, -330, Z_MEM), (0, -330, STEMS[-1][1]), P["mid"], 1, n=8)
        line3((0, HH, Z_MEM), (0, HH, STEMS[-1][1]), P["mid"], 1, n=8)
        # ── 感官层：四条通道 + 意外度尖刺
        tx = tn - fisheye_inv(np.linspace(-X1, 0, 500), tn)
        for k_, cname in enumerate(G.CH):
            vals = np.array([G._ch_at(ch, fps_s, x, cname) for x in tx])
            y0 = -320 + k_ * 65
            pts = np.stack([np.linspace(-X1, 0, 500), y0 + 55 * vals, np.full(500, Z_SENSE)], 1)
            q, z = project(pts, st)
            if (z > 60).all():
                qq = q * SS
                dr.line([tuple(p) for p in qq], fill=P["mid"] if cname != "threat" else P["ink"], width=SS)
                q0, _ = P2((0, y0 + 20, Z_SENSE))
                G.text(dr, (q0[0] + 8 * SS, q0[1] - 8 * SS), f"{G.SYM[cname]} {cname}  {G._ch_at(ch, fps_s, tn, cname):.2f}", P["mid"], 12)
        for e in exv:
            dt_ = tn - e["t"]
            if dt_ < 0 or e["t"] < 0.5:
                continue
            x = fisheye_x(dt_, tn)
            hi_ = (e.get("sal") or 0) >= sal_thr
            line3((x, -330, Z_SENSE), (x, -330 - 18 * min(e["bits"], 12), Z_SENSE), P["hot"] if hi_ else P["dim"], 1)
        q, z = P2((-X1, -330 - 18 * 12, Z_SENSE))
        if z > 60:
            G.text(dr, (q[0], q[1] - 16 * SS), f"EXPECT · surprise bits (−log₂p)   red = salience ≥ {sal_thr:.2f}", P["mid"], 12)
        # ── 记忆层：节点 + 搜索树 + 路径
        npos = node_pos_at(tn)
        popped = {u: tu for u, (tu, _) in sch["pop"].items() if tu <= t}
        for u, tu in popped.items():
            p_ = par.get(u, -1)
            if p_ >= 0:
                fade = math.exp(-max(0.0, t - tu - 0.4) / 1.2)
                line3(npos[p_], npos[u], G.mix(P["ink"], 0.3 + 0.6 * fade, P["bg"]), 1, n=24, dash=(6, 4))
        for u in popped:                                             # 引线：节点 → 它那段时间在 DRUMS 层上的位置（记忆 ↔ 当时听到的）
            if u != mem.now:
                line3(npos[u], (npos[u][0], npos[u][1], 0.0), P["faint"], 1, n=8, dash=(2, 4))
        n_lit = sum(t >= x for x in lit_at)
        for k in range(n_lit):
            line3(npos[path[k]], npos[path[k + 1]], P["hot"], 2, n=24)
        newest = max(popped.items(), key=lambda kv: kv[1])[0] if popped else mem.now
        for nd in mem.nodes:
            i = nd["i"]
            q, z = P2(npos[i])
            if z < 60:
                continue
            vis = i in popped
            is_j = i == mem.john
            sc = max(0.4, min(2.0, 1400 / z))
            if i == un and sch["ut"] <= t < sch["ut"] + 2.0:
                G._draw_unstable_v5(dr, P, q, t - sch["ut"], tk, tk_meta, ch, fps_s, tn, ons_rel, t, und, cos_u)
                continue
            show_j = is_j and t >= G.V6["faint_t"] - s0
            if (vis and i in thumbs and (t - popped[i] < 0.9 or is_j)) or show_j:
                if is_j:
                    a_ = 1.0 if vis else (0.55 if t >= G.V6["notice_t"] - s0 else 0.12)
                    bw = (150 + 30 * sum(t >= h for h in hops[1:])) if vis else (110 if a_ > 0.5 else 60)
                else:
                    a_, bw = 1.0, 120
                bw = int(bw * sc * SS)
                tim = thumbs[i].resize((bw, max(2, int(bw * thumbs[i].height / thumbs[i].width))), Image.BILINEAR)
                tim = Image.fromarray((np.asarray(tim, float) * a_ + np.array(P["bg"]) * (1 - a_)).astype(np.uint8))
                x0, y0 = int(q[0] - bw / 2), int(q[1] - tim.height / 2)
                im.paste(tim, (x0, y0))
                dr.rectangle([x0, y0, x0 + bw, y0 + tim.height], outline=P["hot"] if is_j and a_ > 0.5 else P["mid"], width=SS)
            r = 3.5 * SS
            dr.rectangle([q[0] - r, q[1] - r, q[0] + r, q[1] + r], outline=P["hot"] if (is_j and (vis or show_j)) else (P["ink"] if vis else P["dim"]), width=SS)
            if i == newest or (is_j and (vis or t >= G.V6["notice_t"] - s0)) or i == mem.now:
                lab = f"#{i} {G._fmt(nd['t0']) if i != mem.now else 'NOW'}  {nd['name']}"
                if is_j and not vis:
                    lab = f"#{i} NOTICED  {G._fmt(nd['t0'])}  {nd['name']}"
                G.text(dr, (q[0] + 10 * SS, q[1] - 22 * SS), lab, P["hot"] if is_j else P["ink"], 12, cn=True, knock=P["bg"])
        # HUD
        G.text(dr, (56 * SS, 40 * SS), f"06  A NAME    SENSORY SPACE    t={tn:7.2f}s", P["ink"], 12)
        G.text(dr, (56 * SS, 58 * SS), f"X = time (fisheye τ={TAU}s)   Y = log f {F_LO:.0f} Hz – {F_HI / 1000:.0f} kHz   Z = Demucs stem   "
               f"visited {len(popped)}/{len(mem.nodes)}", P["mid"], 12)
        if t >= hops[0]:
            G.text(dr, (56 * SS, 82 * SS), f"{'MATCH' if t >= hops[3] else 'CANDIDATE'}: JOHN-117   CONFIDENCE {mem.conf:.2f}   FOUND AT POP {sch['jk'] + 1}/36", P["hot"], 15)
        elif t >= G.V6["notice_t"] - s0:
            G.text(dr, (56 * SS, 82 * SS), "NOTICED: #8 JOHN-117   (search not there yet)", P["hot"], 15)
        out = im.resize((W, H), Image.LANCZOS)
        meta.append(dict(f=f, t=round(tn, 3), cam=[round(float(v), 3) for v in st]))
        yield f, out, meta[-1]
    G._MINF["on"] = False
    render.cams, render.events, render.refs = cams, cam_ev, refs


def G_splat(x, y, w):
    x0 = np.floor(x); y0 = np.floor(y); fx = x - x0; fy = y - y0
    out = np.zeros(H * W)
    for dx, dy, ww in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi = (x0 + dx).astype(np.int64); yi = (y0 + dy).astype(np.int64)
        ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        out += np.bincount(yi[ok] * W + xi[ok], (ww * w)[ok], minlength=H * W)
    return out.reshape(H, W)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seg0", type=float, default=115.51)
    ap.add_argument("--seg1", type=float, default=128.36)
    ap.add_argument("--name", default="24_感官空间_一个名字_v1")
    ap.add_argument("--still", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seg = (a.seg0, a.seg1)
    T = int(round((seg[1] - seg[0]) * FPS))
    if a.still:
        fs = {int(round(float(x) * FPS)) for x in a.still.split(",")}
        ims = [(f"{seg[0] + f / FPS:.2f}s f{f}", im) for f, im, _ in render(seg, only=fs)]
        G.contact(ims, a.out, cols=2, tw=960)
        print("->", a.out)
        return
    vid = G.safe_out(G.OUT_A, a.name, ".mp4")
    w = G.open_writer(vid, seg[0], T / FPS)
    picks = set(np.linspace(0, T - 1, 20).round().astype(int).tolist())
    sheet, meta = [], []
    for f, im, m in render(seg):
        w.stdin.write(np.asarray(im).tobytes())
        meta.append(m)
        if f in picks:
            sheet.append((f"{m['t']:.2f}s  f{f}", im))
    w.stdin.close()
    w.wait()
    G.contact(sheet, vid.with_name(vid.stem + "_接触表.jpg"))
    cams, evs = render.cams, render.events
    sp = screen_speed(cams, render.refs)
    al = alignment(sp, evs, T)
    print("镜头速度峰值 − 事件帧：", {d: sum(1 for *_, x in al if x == d) for d in sorted(set(x for *_, x in al))})
    # 速度图（复用 info_graph 的画法：把屏幕速度喂成「位移」）
    cam2 = np.cumsum(np.stack([sp, np.zeros_like(sp)], 1), axis=0)
    G.speed_plot(cam2, np.ones(T), evs, T, vid.with_name(vid.stem + "_镜头速度.png"), seg[0], vid.stem, lead=2)
    vid.with_suffix(".json").write_text(json.dumps(dict(tool="tools/sense_space.py", seg=seg, stems=STEMS, z_sense=Z_SENSE, z_mem=Z_MEM,
                                                        fisheye_tau=TAU, bands=[NB, F_LO, F_HI], camera_events=[[round(t + seg[0], 3), k] for t, k in evs],
                                                        camera_alignment=[[round(t + seg[0], 3), k, d] for t, k, d in al], frames=meta),
                                                   ensure_ascii=False, indent=1, default=G._jsonable), "utf-8")
    print("->", vid)


if __name__ == "__main__":
    main()
