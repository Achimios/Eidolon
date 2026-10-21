#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🎬样片-碎裂-3D：从 Tripo 模型 `3D/她.glb` 本身碎裂（和 03 抬头同一套全息等高线点溅射风格，见 headlift_render.py）。
机理照抄 v4（The Crew，§四⅞½）：褪色 → 单帧前震 → 2 帧拖影 → 1 帧噪点 → 1 帧白线 → 6–8 帧爆散 → 尘埃；
双尺度碎片 = 1 px 尘埃 + 等高线条块（她身上那一圈圈亮线整段剥下来横飞，没有中等碎片）；单方向（向后 = −x，屏幕上向右）；
胸口中弹 → 冲击波从胸口往外传（离胸口越远越晚放开）→ 上身后仰；脸最后走，边缘按抖动阈值碎成颗粒。
节奏（👤2026-10-21 看 v4.3，§四⅞¾）：蓄势段不许干等 —— 推镜 + 绕到 3/4 侧 + 呼吸 + 光在变 + 扫描线 + 越来越密的单帧前震；
碎裂本身只 1.6 s；之后的余波是同一个物理（一个真源、闭式解）换机位 / 景别拍的碎片飞行镜头，卡 0.4 s 的拍。
物理是闭式的：任何时刻的点位置都能直接算（不用逐帧积分）→ 余波镜头可以任意挑时刻和机位，运动模糊也是精确的。
用法：.venv/Scripts/python tools/shatter_3d.py --shot 蓄势|冲击|余波1..5|连续 [--ver 1] [--pts 8000000] [--no-hud]
      --stats：只打印碎片云在各时刻的分布（给余波镜头摆机位用）
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import headlift_render as HL  # noqa: E402

MV = HL.MV
DEV = HL.DEV
FPS = 24
W, H = 1280, 720
MUSIC = MV.parents[0] / "Assets" / "The Glitch Mob - Can't Kill Us [2020 Edit]_190189282.mp4"
FONT = "C:/Windows/Fonts/consola.ttf"

HIT = 51.52                      # 冲击 = drop 的强拍（感官.json downbeat）
CHEST = (0.075, 0.19, 0.0)       # 中弹点：胸口正前（她.glb：脸朝 +x，y 朝上）
FACE = (0.045, 0.415, 0.0)       # 脸心
WAIST = (0.0, -0.10)             # 后仰的枢轴（x, y）
CONTOUR_F = 230.0                # 等高线频率，和 headlift_render 一致
TINT_HOLO = np.array([0.86, 0.92, 1.0])
TINT_DRAIN = np.array([0.97, 0.97, 0.97])

# 余波 / 段落：song-time 窗口（拍点相隔 0.4 s，强拍 51.52 / 53.12 / 54.72 / 56.32）
SHOTS = {
    "蓄势": (47.50, 51.52),
    "冲击": (51.52, 53.12),
    "连续": (47.50, 53.12),
    "余波1": (53.12, 53.92),     # 流里：在尘埃中间顺流往前冲、斜着看，直线条块从身边擦过
    "余波2": (53.92, 54.32),     # 回放：从她背后低机位看冲击那 0.4 s，碎片迎面炸过来
    "余波3": (54.32, 55.12),     # 逆流（斜 30°）：在下游回头，条块迎面飞来、从左边掠过
    "余波4": (55.12, 56.32),     # 回放（0.8×）：正侧面，看她从胸口碎开、脸最后走，碎片横穿画面
    "余波5": (56.32, 57.92),     # 正俯拍：流向 = 屏幕横向，条块成一道道水平扫描线横穿（最像 The Crew 的一镜）
}


# 回放：余波镜头里哪些不是「接着往下」，而是换机位重看冲击（sim s0 起，速率）。其余余波 = 真实时间往下走
REPLAY = {"余波2": (0.045, 1.0), "余波4": (0.045, 0.8)}
EXPOSURE = {"余波1": 2.2, "余波3": 2.4, "余波5": 4.0}     # 稀疏的尘埃远看太暗：只是曝光，不改物理


def smooth(e0, e1, x):
    if isinstance(x, torch.Tensor):
        u = torch.clamp((x - e0) / (e1 - e0), 0, 1)
    else:
        u = np.clip((x - e0) / (e1 - e0), 0, 1)
    return u * u * (3 - 2 * u)


def bump(t, anchors, rise=2.5 / FPS, decay=0.18):
    """「进」2–3 帧加速 →「缓」指数衰减（§四⅞¾）。anchors = [(时刻, 强度)]"""
    v = 0.0
    for ta, amp in anchors:
        d = t - ta
        if d < 0:
            continue
        v += amp * (min(d / rise, 1.0) if d < rise else np.exp(-(d - rise) / decay))
    return v


def progress(t, t0, t1, anchors, base=1.0):
    """镜头进度 0→1：速度 = 基础 + 每个锚点一次进缓；首尾速度都不为零"""
    g = np.linspace(t0, t1, 600)
    sp = np.array([base + bump(x, anchors) for x in g])
    cum = np.concatenate([[0], np.cumsum((sp[1:] + sp[:-1]) / 2 * np.diff(g))])
    return float(np.interp(t, g, cum / cum[-1]))


# ---------------------------------------------------------------- 模型 → 点 + 属性
def load_scene(n_pts, seed=7):
    import trimesh
    m = trimesh.load(str(MV / "3D" / "她.glb"), force="mesh")
    V = np.asarray(m.vertices); Fc = np.asarray(m.faces)
    Fc = Fc[V[Fc].max(1)[:, 1] > -0.32]                   # 腰以上（下面隐入黑暗）
    sub = trimesh.Trimesh(V, Fc, process=False)
    pts, fi = trimesh.sample.sample_surface(sub, n_pts, seed=seed)
    area = float(sub.area)
    P = torch.tensor(pts, dtype=torch.float32, device=DEV)
    N = torch.tensor(sub.face_normals[fi], dtype=torch.float32, device=DEV)
    g = torch.Generator(device=DEV).manual_seed(seed)
    rnd = lambda *s: torch.rand(*s, device=DEV, generator=g)  # noqa: E731
    rng = np.random.default_rng(seed)
    S = {"P": P, "N": N, "ry": P[:, 1].clone(), "area": area, "n": n_pts}
    ry = S["ry"]
    S["keep"] = rnd(n_pts) < (0.05 + 0.95 * smooth(-0.30, -0.05, ry))   # 腰部点云消散（03 是肩部）
    S["fade"] = 0.12 + 0.88 * smooth(-0.30, 0.0, ry)
    cf = torch.frac(ry * CONTOUR_F)
    S["contour"] = torch.exp(-((cf - 0.5) / 0.09) ** 2)
    S["hw"] = HL.bend_weight(P)                                         # 头（复用 03 的脖子权重）
    S["bw"] = smooth(-0.10, 0.16, ry)                                   # 后仰权重：腰 0 → 胸 1
    C = torch.tensor(CHEST, device=DEV); Fp = torch.tensor(FACE, device=DEV)
    d = (P - C).norm(dim=1)
    face = torch.exp(-(((P - Fp) * torch.tensor([1.0, 0.8, 1.0], device=DEV)).norm(dim=1) / 0.055) ** 2)
    face = face * (P[:, 0] > -0.02)                                     # 只算脸前半（后脑照常走）
    S["face"] = face
    # 胸口呼吸权重
    S["chest"] = torch.exp(-((ry - 0.18) / 0.07) ** 2) * (ry > -0.05)
    # ---- 放开时刻：冲击波从胸口以 1.5 /s 传出；脸按抖动阈值最后走（边缘碎成颗粒）
    t_rel = d / 1.5 + 0.035 * rnd(n_pts) + face * (0.25 + 1.05 * rnd(n_pts) ** 0.7) * (rnd(n_pts) < 0.97)
    hitw = 0.45 + 0.9 * torch.exp(-(d / 0.16) ** 2)                     # 离胸口越近飞得越快
    # ---- 双尺度：条块 = 同一条等高线上一段弧（亮线整段剥下）；尘埃 = 其余每点各走各的
    band = torch.floor(ry * CONTOUR_F).long(); band -= band.min()
    nb = int(band.max()) + 1
    phi = torch.atan2(P[:, 2], P[:, 0] + 0.01)
    sw = torch.tensor(rng.uniform(0.12, 0.40, nb), dtype=torch.float32, device=DEV)   # 每条线的弧段宽（弧度）
    so = torch.tensor(rng.uniform(0, 6.3, nb), dtype=torch.float32, device=DEV)
    sec = torch.floor((phi + 3.2 + so[band]) / sw[band]).long()
    slab_id = band * 64 + sec
    uid, slab_id = torch.unique(slab_id, return_inverse=True); ns = len(uid)
    chosen = torch.tensor(rng.random(ns) < 0.42, device=DEV)
    is_slab = chosen[slab_id] & (S["contour"] > 0.25) & (face < 0.15) & (rnd(n_pts) < 0.96)
    # 条块整体走：放开时刻 / 速度取条块均值
    cnt = torch.zeros(ns, device=DEV).index_add_(0, slab_id, torch.ones(n_pts, device=DEV)).clamp(min=1)
    tr_s = torch.zeros(ns, device=DEV).index_add_(0, slab_id, t_rel) / cnt
    hw_s = torch.zeros(ns, device=DEV).index_add_(0, slab_id, hitw) / cnt
    vs = torch.tensor(rng.lognormal(np.log(3.6), 0.5, ns), dtype=torch.float32, device=DEV)
    vy_s = torch.tensor(rng.normal(0, 0.04, ns), dtype=torch.float32, device=DEV)
    vz_s = torch.tensor(rng.normal(0, 0.10, ns), dtype=torch.float32, device=DEV)
    S["t_rel"] = torch.where(is_slab, tr_s[slab_id], t_rel)
    S["is_slab"] = is_slab
    sp_d = torch.exp(torch.log(torch.tensor(1.5)) + 0.85 * torch.randn(n_pts, device=DEV, generator=g)) * hitw
    sp_d = sp_d * (1 - 0.75 * face)                                       # 脸：轻轻散开，不是被炸飞
    dyn = torch.randn(n_pts, device=DEV, generator=g) * 0.28; dzn = torch.randn(n_pts, device=DEV, generator=g) * 0.38
    v0 = torch.stack([-sp_d, sp_d * dyn, sp_d * dzn], 1)
    sp_s = vs[slab_id] * hw_s[slab_id] * (0.86 + 0.28 * rnd(n_pts))     # 条内各点速度不同 → 剪切流把弧拉直、拉长成扫描线
    S["zs"] = (torch.zeros(ns, device=DEV).index_add_(0, slab_id, P[:, 2]) / cnt)[slab_id]
    v0s = torch.stack([-sp_s, vy_s[slab_id], vz_s[slab_id] * vs[slab_id]], 1)
    S["v0"] = torch.where(is_slab[:, None], v0s, v0)
    S["tau"] = torch.where(is_slab, torch.full_like(ry, 0.19), 0.13 + 0.10 * rnd(n_pts))
    # 持续漂移（不归零）：条块留 18% 速度；尘埃一个慢的同向流 + 一点上浮
    vslow_d = torch.stack([-(0.18 + 0.35 * rnd(n_pts)), 0.015 + 0.03 * (rnd(n_pts) - 0.5), 0.05 * (rnd(n_pts) - 0.5)], 1)
    S["vslow"] = torch.where(is_slab[:, None], 0.18 * v0s, vslow_d)
    S["phs"] = rnd(n_pts, 2) * 6.283
    S["base_b"] = (0.30 + 0.70 * S["contour"]) * S["fade"]
    S["frag_b"] = S["base_b"] * torch.where(is_slab, torch.full_like(ry, 1.0), torch.full_like(ry, 0.22))   # 尘埃暗、条块亮
    S["n_slab"] = int(is_slab.sum()); S["n_slabs_groups"] = int(chosen.sum())
    return S


# ---------------------------------------------------------------- 身体姿态（蓄势：呼吸 + 抬头；冲击：后仰）
def head_angle(t):
    """度：接 03 抬头的结尾（仰着）慢慢收回到平视，冲击时被甩回去"""
    a = 16.0 - 7.0 * smooth(47.5, 51.4, t) + 1.2 * np.sin((t - 47.5) * 2.1)
    s = t - HIT
    if s > -2 / FPS:
        a += 20.0 * (1 - np.exp(-(s + 2 / FPS) / 0.07)) + 6.0 * max(s, 0)
    return a


def bend_angle(t):
    s = t - HIT + 2 / FPS
    return 0.0 if s < 0 else 16.0 * (1 - np.exp(-s / 0.06)) + 4.0 * s


def breath(t):
    """呼吸：慢的起伏 + 冲击前最后 1 s 深吸一口（吸气 → 炸碎）"""
    return 0.5 + 0.5 * np.sin((t - 47.5) * 2 * np.pi * 0.28) * 0.6 + 1.3 * smooth(HIT - 1.1, HIT - 0.1, t)


def rot_z(P, ang, pivot):
    """绕 z 轴、过 pivot(x, y) 转 ang（逐点弧度张量）"""
    c, s = torch.cos(ang), torch.sin(ang)
    x = P[:, 0] - pivot[0]; y = P[:, 1] - pivot[1]
    return torch.stack([x * c - y * s + pivot[0], x * s + y * c + pivot[1], P[:, 2]], 1)


def pose(S, t, idx):
    """t 时刻身体（未放开时）的姿态，只算 idx 这些点，返回 (P, N)"""
    P = S["P"][idx]; N = S["N"][idx]; sl = idx
    hw = S["hw"][sl]; bw = S["bw"][sl]; ch = S["chest"][sl]
    k = 0.012 * breath(t) * ch
    Q = torch.stack([P[:, 0] * (1 + k), P[:, 1] + 0.004 * breath(t) * ch, P[:, 2] * (1 + 0.6 * k)], 1)
    ah = torch.tensor(np.radians(head_angle(t)), device=DEV) * hw
    ab = torch.tensor(np.radians(bend_angle(t)), device=DEV) * bw
    Q = rot_z(rot_z(Q, ah, HL.PIVOT), ab, WAIST)
    Nn = rot_z(rot_z(N, ah, (0.0, 0.0)), ab, (0.0, 0.0))
    return Q, Nn


def released_pos(S, t, idx):
    """闭式：放开后的位置 = 放开时的姿态位置 + v0·τ(1−e^(−s/τ)) + 慢漂移·s + 小涡流"""
    tr = S["t_rel"][idx]
    s = (t - HIT - 1 / FPS) - tr                     # 白线帧放开
    s = s.clamp(min=0)
    # 放开时的姿态：用放开时刻的后仰角（逐点）
    th = (tr + HIT + 1 / FPS).cpu().numpy()
    ha = np.radians(np.array([head_angle(x) for x in np.linspace(HIT - 0.1, HIT + 2.0, 256)]))
    ba = np.radians(np.array([bend_angle(x) for x in np.linspace(HIT - 0.1, HIT + 2.0, 256)]))
    tt = np.clip((th - (HIT - 0.1)) / 2.1 * 255, 0, 255)
    ahp = torch.tensor(np.interp(tt, np.arange(256), ha), dtype=torch.float32, device=DEV)
    abp = torch.tensor(np.interp(tt, np.arange(256), ba), dtype=torch.float32, device=DEV)
    P0 = S["P"][idx]
    k = (0.012 * breath(HIT) * S["chest"][idx])
    Q = torch.stack([P0[:, 0] * (1 + k), P0[:, 1] + 0.004 * breath(HIT) * S["chest"][idx], P0[:, 2] * (1 + 0.6 * k)], 1)
    Q = rot_z(rot_z(Q, ahp * S["hw"][idx], HL.PIVOT), abp * S["bw"][idx], WAIST)
    sl = S["is_slab"][idx]
    zc = S["zs"][idx] + (Q[:, 2] - S["zs"][idx]) * torch.exp(-s / 0.12)    # 条块的弧收成一条直线（只剩 y 一层、沿 x 拉开）
    Q = torch.stack([Q[:, 0], Q[:, 1], torch.where(sl, zc, Q[:, 2])], 1)
    tau = S["tau"][idx][:, None]
    s1 = s[:, None]
    disp = S["v0"][idx] * tau * (1 - torch.exp(-s1 / tau)) + S["vslow"][idx] * s1
    ph = S["phs"][idx]
    amp = 0.035 * (1 - torch.exp(-s / 0.9)) * (~S["is_slab"][idx])
    tw = torch.stack([0 * s, torch.sin(9.0 * Q[:, 0] + 1.7 * s + ph[:, 0]), torch.sin(7.0 * Q[:, 1] + 1.3 * s + ph[:, 1])], 1)
    return Q + disp + amp[:, None] * tw, s


# ---------------------------------------------------------------- 相机与渲染
def look(pos, target, fov):
    pos = torch.tensor(pos, dtype=torch.float32, device=DEV); target = torch.tensor(target, dtype=torch.float32, device=DEV)
    fwd = F.normalize(target - pos, dim=0)
    right = F.normalize(torch.linalg.cross(fwd, torch.tensor([0.0, 1.0, 0.0], device=DEV)), dim=0)
    up = torch.linalg.cross(right, fwd)
    return {"pos": pos, "fwd": fwd, "right": right, "up": up, "f": (H / 2) / np.tan(np.radians(fov / 2))}


def project(cam, P):
    d = P - cam["pos"]
    zc = d @ cam["fwd"]; xc = d @ cam["right"]; yc = d @ cam["up"]
    zs = zc.clamp(min=1e-4)
    px = (xc / zs * cam["f"] + W / 2); py = (-yc / zs * cam["f"] + H / 2)
    ok = (zc > 0.02) & (px >= 0) & (px < W - 1) & (py >= 0) & (py < H - 1)
    return px, py, zc, ok


def render_body(S, cam, P, N, mask, look_p):
    """未放开的身体：z-buffer 取最近 + 补洞（和 03 一样）。返回 (img, zbuf)"""
    px, py, zc, ok = project(cam, P)
    ok = ok & mask
    idx = py[ok].long() * W + px[ok].long(); z = zc[ok]
    zbuf = torch.full((W * H,), 1e9, device=DEV).scatter_reduce(0, idx, z, "amin")
    vis = z <= zbuf[idx] + 0.0025
    idx = idx[vis]
    sel = torch.nonzero(ok).squeeze(1)[vis]
    n = N[sel]; view = -F.normalize(P[sel] - cam["pos"], dim=1)
    L = F.normalize(torch.tensor(look_p["L"], dtype=torch.float32, device=DEV), dim=0)
    lam = (n @ L).clamp(min=0)
    rim = (1 - (n * view).sum(1).abs()).clamp(0, 1) ** 3
    ry = S["ry"][sel]
    v = (0.05 + 0.75 * lam ** 1.4 + look_p["rim"] * rim) * (0.55 + 0.75 * S["contour"][sel]) * S["fade"][sel]
    if look_p.get("scan") is not None:                                # 扫描光带（蓄势：从头顶扫到胸口）
        for ys, a in look_p["scan"]:
            v = v + a * torch.exp(-((ry - ys) / 0.006) ** 2) * S["fade"][sel]
    v = v * look_p["gain"]
    img = torch.zeros(W * H, device=DEV).scatter_reduce(0, idx, v, "amax")
    cov = torch.zeros(W * H, device=DEV).index_fill(0, idx, 1.0)
    img = img.view(1, 1, H, W); cov = cov.view(1, 1, H, W)
    k = torch.ones(1, 1, 3, 3, device=DEV)
    for _ in range(2):
        s = F.conv2d(img, k, padding=1); c = F.conv2d(cov, k, padding=1)
        hole = (cov == 0) & (c >= 3)
        img = torch.where(hole, s / c.clamp(min=1), img); cov = torch.where(hole, torch.ones_like(cov), cov)
    return img[0, 0], zbuf.view(H, W)


def splat_add(cam, P, b, zbuf, area_per_pt, acc):
    """碎片：加法溅射（全息，半透明）；被身体挡住的不画。亮度按「每像素多少点」归一，近处的点至少是 1 px 亮点"""
    px, py, zc, ok = project(cam, P)
    pxl, pyl = px[ok].long(), py[ok].long(); z = zc[ok]
    idx = pyl * W + pxl
    vis = z <= zbuf.view(-1)[idx] + 0.004
    pix_area = (z / cam["f"]) ** 2
    w = (area_per_pt / pix_area).clamp(0.015, 1.6)
    acc.view(-1).index_add_(0, idx[vis], (b[ok] * w)[vis])


def finish(img, look_p, tint):
    rows = torch.arange(H, device=DEV, dtype=torch.float32)
    img = img * (0.72 + 0.28 * torch.cos(rows * np.pi / 1.5)[:, None] ** 2)        # 细扫描线（同 03）
    blur = F.avg_pool2d(F.avg_pool2d(img[None, None], 9, 1, 4), 9, 1, 4)[0, 0]
    img = img + look_p.get("glow", 0.35) * blur
    xs = (torch.arange(W, device=DEV) - W * look_p.get("cone_x", 0.5)) / W
    cone = torch.exp(-(xs[None, :] / (0.05 + 0.25 * rows[:, None] / H)) ** 2) * torch.exp(-rows[:, None] / (0.35 * H))
    img = img + look_p.get("cone", 0.10) * cone
    rgb = (img.clamp(0, 1)[..., None] * torch.tensor(tint, dtype=torch.float32, device=DEV)).clamp(0, 1)
    return (rgb * 255).byte().cpu().numpy()


# ---------------------------------------------------------------- 各镜头的相机
def onsets(t0, t1):
    sense = json.loads((MV / "感官.json").read_text("utf-8"))
    return [t for t in sense["events"]["onsets"] if t0 - 0.3 < t < t1]


def orbit(target, az_deg, el_deg, dist, screen_off=0.0):
    """绕 y 轴：az=0 正面（+x），负 = 往 −z 侧绕（−x 在屏幕上朝右）。screen_off>0 = 主体偏左"""
    az, el = np.radians(az_deg), np.radians(el_deg)
    dvec = np.array([np.cos(az) * np.cos(el), np.sin(el), np.sin(az) * np.cos(el)])
    right = np.array([np.sin(az), 0.0, -np.cos(az)])
    tg = np.array(target) + screen_off * right
    return tg + dist * dvec, tg


def cam_main(t):
    """蓄势 + 冲击是一条连续的机位：推 + 绕到 3/4，进度按起音进缓；冲击后只剩很慢的余速（不缩放不震）"""
    an = [(x, 1.6 + 2.2 * smooth(48.0, 51.4, x)) for x in onsets(47.5, HIT - 0.05)]
    if t <= HIT:
        u = progress(t, 47.5, HIT, an, base=1.0)
    else:
        u = 1.0 + (t - HIT) * 0.035          # 余速：约等于蓄势末段基础速度的 1/4
    az = -14 - 38 * u
    el = -3 - 4 * u
    dist = 0.80 - 0.20 * u
    tgt = (0.02, 0.37 - 0.07 * min(u, 1.0), 0.0)
    pos, tg = orbit(tgt, az, el, dist, screen_off=0.02 + 0.10 * min(u, 1.0))
    return pos, tg, 32.0


def lerp3(a, b, u):
    return tuple(float(x + (y - x) * u) for x, y in zip(a, b))


def cam_after(name, t):
    t0, t1 = SHOTS[name]
    beats = [(b, 1.8) for b in np.arange(51.52, 58.5, 0.4) if t0 - 0.3 < b < t1]
    u = progress(t, t0, t1, beats, base=1.3)
    if name == "余波1":      # 真实时间 s≈1.6–2.4：在流里顺 −x 冲，斜 45° 看（正对流向会成「曲速」放射线，太俗）
        pos = lerp3((-0.55, 0.30, -0.38), (-1.20, 0.27, -0.33), u)
        tg = tuple(np.array(pos) + np.array([-0.62, -0.10, 0.78]))
        return pos, tg, 46.0
    if name == "余波2":      # 回放：她背后、肩高、近，碎片朝镜头来
        pos = lerp3((-0.62, 0.22, -0.26), (-0.55, 0.23, -0.22), u)
        tg = lerp3((0.05, 0.27, 0.02), (0.02, 0.27, 0.02), u)
        return pos, tg, 42.0
    if name == "余波3":      # 真实时间 s≈2.8–3.6：在下游回头、斜 30°，条块迎面来、从画面左边掠过
        pos = lerp3((-3.00, 0.26, -0.70), (-3.35, 0.23, -0.72), u)
        tg = lerp3((-1.25, 0.18, 0.30), (-1.55, 0.17, 0.30), u)
        return pos, tg, 40.0
    if name == "余波4":      # 回放 0.8×：正侧面（−z 侧，−x = 屏幕右），慢慢跟着往 −x 横移
        pos = lerp3((-0.20, 0.27, -0.95), (-0.45, 0.25, -0.92), u)
        tg = lerp3((-0.20, 0.25, 0.0), (-0.45, 0.24, 0.0), u)
        return pos, tg, 40.0
    if name == "余波5":      # 真实时间 s≈4.8–6.4：正俯拍、略向 +z 倾（−x = 屏幕右），沿 −x 跟拍比流快一点
        pos = lerp3((-1.70, 0.80, -0.05), (-2.90, 0.72, -0.05), u)
        tg = (pos[0], 0.15, pos[2] + 0.30)
        return pos, tg, 44.0
    raise KeyError(name)


# ---------------------------------------------------------------- 主循环
def phase_of(t):
    k = int(round((t - HIT) * FPS))
    if k < -2:
        return "BUILD", k
    return {-2: "SMEAR", -1: "SMEAR", 0: "NOISE", 1: "FLASH"}.get(k, "SHATTER" if k < 10 else "DUST"), k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", default="冲击", choices=list(SHOTS))
    ap.add_argument("--ver", type=int, default=1)
    ap.add_argument("--pts", type=int, default=8_000_000)
    ap.add_argument("--no-hud", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--frames", default=None, help="只渲这几帧存 png（调试），如 0,10,40")
    ap.add_argument("--dbg-dir", default=".", help="--frames 的 png 存哪")
    a = ap.parse_args()
    S = load_scene(a.pts)
    N_all = S["n"]; area_per_pt = S["area"] / N_all
    allidx = torch.arange(N_all, device=DEV)
    if a.stats:
        for s in (0.1, 0.3, 0.6, 1.0, 1.6, 2.4, 3.2, 4.8, 6.4):
            Q, _ = released_pos(S, HIT + 1 / FPS + s, allidx)
            for nm, m in (("slab", S["is_slab"]), ("dust", ~S["is_slab"])):
                q = torch.quantile(Q[m][::50], torch.tensor([0.05, 0.5, 0.95], device=DEV), dim=0).cpu().numpy().round(2)
                print(f"s={s:4.1f} {nm}: x {q[:, 0]}  y {q[:, 1]}  z {q[:, 2]}")
        return
    t0, t1 = SHOTS[a.shot]
    name = f"16_碎裂_3D_{a.shot}" if not a.shot.startswith("余波") else f"16_碎裂_3D_余波_{a.shot[2:]}"
    out = MV / "样片" / f"{name}_v{a.ver}.mp4"
    assert not out.exists(), f"不覆盖已有文件：{out}"
    n = int(round((t1 - t0) * FPS))
    pre_shocks = [x for x in onsets(48.6, HIT - 0.12) if x > 48.6]
    f_hud = ImageFont.truetype(FONT, 13)
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    want = None if a.frames is None else {int(x) for x in a.frames.split(",")}
    proc = None
    if want is None:
        proc = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                                 "-ss", f"{t0:.3f}", "-t", f"{n / FPS:.3f}", "-i", str(MUSIC),
                                 "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16",
                                 "-c:a", "aac", "-b:a", "256k", "-shortest", str(out)], stdin=subprocess.PIPE)
    rng = np.random.default_rng(11)
    tw = time.time()
    for i in range(n):
        t_song = t0 + i / FPS
        if want is not None and i not in want:
            continue
        rp = REPLAY.get(a.shot)
        t = t_song if rp is None else HIT + 1 / FPS + rp[0] + (t_song - t0) * rp[1]     # 物理时刻
        ph, k = phase_of(t)
        main_cam = not a.shot.startswith("余波")
        pos, tg, fov = cam_main(t) if main_cam else cam_after(a.shot, t_song)
        cam = look(pos, tg, fov)
        # ---- 光
        an = [(x, 1.0) for x in onsets(47.5, HIT)]
        drain = float(smooth(HIT - 1.0, HIT - 0.1, t)) if t < HIT else 1.0
        tint = TINT_HOLO * (1 - drain) + TINT_DRAIN * drain
        ub = float(smooth(47.5, HIT, t))
        lp = {"L": (0.35 - 0.75 * ub, 1.0 - 0.35 * ub, -0.55 * ub),      # 顶光慢慢绕到背后侧上方
              "rim": 0.45 + 0.55 * ub, "gain": 1.0 + 0.30 * bump(t, an, decay=0.10) * (0.4 + 0.6 * ub),
              "cone": 0.10 * (1 - 0.6 * ub) if t < HIT else 0.0, "cone_x": 0.42, "scan": []}
        for x in onsets(47.5, HIT - 0.05):                               # 每个起音一道扫描光，从头顶落到胸口
            d = t - x
            if 0 <= d < 0.32:
                lp["scan"].append((0.49 - (0.49 - CHEST[1]) * (d / 0.32) ** 0.7, (0.5 + 1.2 * ub) * (1 - d / 0.32)))
        img = None
        moved = 0; n_d = n_s = 0
        if ph in ("NOISE", "FLASH"):
            if ph == "NOISE":                                             # 1 帧无关噪点（信号断了）
                nz = rng.random((H // 4, W // 4)) ** 3 * (rng.random((H // 4, 1)) < 0.6)
                v = np.kron(nz, np.ones((4, 4)))
                frame = (np.clip(v[..., None] * np.array([0.8, 0.82, 0.85]), 0, 1) * 255).astype(np.uint8)
            else:                                                         # 1 帧白色横线，核心在胸口的屏幕高度
                cy = float(project(cam, torch.tensor([CHEST], device=DEV))[1][0])
                rows = np.zeros(H); r = rng.choice(H, 34, replace=False); rows[r] = rng.random(34)
                core = np.exp(-((np.arange(H) - cy) / 22.0) ** 2)
                v = np.clip(rows[:, None] * (0.6 + 0.4 * rng.random((1, W))) + core[:, None] * 1.4, 0, 1)
                frame = (v[..., None] * np.ones(3) * 255).astype(np.uint8)
        else:
            rel_t = (t - HIT - 1 / FPS)
            rel = (S["t_rel"] <= rel_t) if k >= 2 else torch.zeros(N_all, dtype=torch.bool, device=DEV)
            body = ~rel
            bidx = allidx[body]
            Pb, Nb = pose(S, t, bidx)
            if ph == "BUILD" and any(abs(t - x) < 0.5 / FPS for x in pre_shocks):   # 单帧前震：一条带沿碎裂方向错开
                ph = "PRE-SHOCK"
                amp = 0.012 + 0.05 * float(smooth(48.6, HIT, t))
                yb = rng.uniform(0.05, 0.45); sel = (S["ry"][bidx] - yb).abs() < rng.uniform(0.012, 0.03)
                Pb = Pb.clone(); Pb[:, 0] -= sel * amp
                lp["gain"] *= 1.25
            mask = S["keep"][bidx]
            if ph == "SMEAR":                                             # 2 帧拖影：沿 −x 拉出 6 份
                sm = (k + 3) * 0.035
                hitw = (0.45 + 0.9 * torch.exp(-((S["P"][bidx] - torch.tensor(CHEST, device=DEV)).norm(dim=1) / 0.16) ** 2))
                acc = torch.zeros(H, W, device=DEV)
                zb = torch.full((H, W), 1e9, device=DEV)
                for u in np.linspace(0, 1, 6):
                    Pu = Pb.clone(); Pu[:, 0] -= sm * u * hitw
                    splat_add(cam, Pu[mask], S["base_b"][bidx][mask] * (0.26 - 0.16 * u), zb, area_per_pt, acc)
                img = 1 - torch.exp(-1.2 * acc)
            else:
                img, zbuf = render_body(S, cam, Pb, Nb, mask, lp)
                if rel.any():
                    ridx = allidx[rel]
                    acc = torch.zeros(H, W, device=DEV)
                    nsub = 4 if ph == "SHATTER" else 2
                    for j in range(nsub):                                 # 精确运动模糊：闭式位置在快门内取样
                        Q, s = released_pos(S, t - j / nsub * 0.6 / FPS * (1 if rp is None else rp[1]), ridx)
                        heat = (1 + 2.2 * torch.exp(-s / 0.10)) * (0.55 + 0.45 * torch.exp(-s / 0.9))
                        m = S["keep"][ridx]
                        splat_add(cam, Q[m], (S["frag_b"][ridx] * heat)[m] / nsub, zbuf, area_per_pt, acc)
                    img = img + (1 - torch.exp(-0.9 * EXPOSURE.get(a.shot, 1.0) * acc))
                    n_s = int((rel & S["is_slab"]).sum()); n_d = int(rel.sum()) - n_s
            frame = finish(img, {**lp, "glow": 0.5 if ph == "SHATTER" else 0.35}, tint)
            moved = n_s + n_d
        fr = Image.fromarray(frame)
        if not a.no_hud and ph not in ("NOISE", "FLASH"):
            dr = ImageDraw.Draw(fr)
            rpl = "" if rp is None else f"   REPLAY x{rp[1]:.1f}  sim t+{t - HIT:5.2f}"
            hud = [f"t = {t_song:6.2f} s   {ph:<9}   impact @ {HIT:.2f}{rpl}   src her.glb (Tripo), surface-sampled",
                   f"N = {N_all:,}   body {N_all - moved:9,}   dust {n_d:9,}   slabs {n_s:9,}   mass in = mass out"]
            for jj, line in enumerate(hud):
                dr.text((32, H - 30 - 18 * (len(hud) - jj)), line, font=f_hud, fill=(140, 150, 158))
        if want is not None:
            fr.save(Path(a.dbg_dir) / f"dbg_{a.shot}_{i:03d}.png")
            continue
        proc.stdin.write(fr.tobytes())
        if i % 24 == 0:
            print(f"{a.shot} frame {i}/{n}  {time.time() - tw:.0f}s  {ph}", flush=True)
    if proc:
        proc.stdin.close(); proc.wait()
        contact(out, n)
        print("->", out)


def contact(out, n):
    """接触表：冲击段逐帧（前 16 帧）+ 全段均匀取样"""
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    step = max(1, n // 16)
    cs = out.with_name(out.stem + "_接触表.jpg")
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(out), "-vf",
                    f"select='not(mod(n\\,{step}))',scale=320:-1,drawtext=fontfile='C\\:/Windows/Fonts/consola.ttf':text='%{{n}}':x=6:y=6:fontsize=16:fontcolor=gray,tile=4x4",
                    "-frames:v", "1", str(cs)], check=True)


if __name__ == "__main__":
    main()
