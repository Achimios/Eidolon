#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🧊四维切片-入口-4D 物体 → 3D 截面渲染器（「全息 AI 坠入高维」：高维的存在穿过我们的空间）。
4D 物体 = 有符号距离函数 f(x,y,z,w)（torch / GPU）；先在 4D 里转 R(θ_xw, θ_yw, θ_zw)，再用超平面 w' = c 去切：
    g(x,y,z) = f( [x, y, z, c] · R )            （4D 距离 ≥ 3D 里的距离 ⇒ g 仍是合法的距离下界，可以直接球追踪）
建模算子推广到 4D：extrude（沿 w 拉伸）/ spin（绕一个平面旋转，duocylinder、tiger、ditorus…）/ loft（沿 w 插值两个 3D SDF）/
link（S¹ 与 S² 在 4D 里互锁）/ sweep（3-球沿 4D 曲线扫）/ 正多胞体（超立方体、24-cell、120-cell：面超平面 max）。
红 = 超平面与 4D 表面相切的地方（4D 法向的 xyz 分量 → 0）：Morse 理论里截面拓扑只会在这种点变（裂开 / 长洞 / 生灭）。
HUD 的 b0 b1 b2 是在粗体素网格上真数的 Betti 数（连通块 / 洞 / 空腔，χ = b0 − b1 + b2）。
用法：
  python tools/four_d.py --obj tiger --op slice  --frames 48 --out out.mp4     # 超平面扫过（c 从 −c_max 到 +c_max）
  python tools/four_d.py --obj tiger --op rotate --frames 168 --out out.mp4    # 在 xw 平面里转，同时 c 缓慢移动
  python tools/four_d.py --obj tesseract --op project --frames 96 --out out.mp4   # 多胞体棱的 4D→3D→2D 透视投影
  python tools/four_d.py --sheet [--out x.jpg]   # 总览接触表（6 行物体 × 6 列切片 / 角度）
  python tools/four_d.py --list
--out 以 .mp4 结尾 → 视频（1920x1080, 24 fps）；否则当目录，写 PNG 帧。
"""
import argparse
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stills import FONT, GREY, MV, RED, H, W  # noqa: E402  同一套视觉语法

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PHI = (1 + 5 ** 0.5) / 2
GREY_T = torch.tensor(GREY, dtype=torch.float32, device=DEV)
RED_T = torch.tensor(RED, dtype=torch.float32, device=DEV)


# ---------- 4D 几何 ----------
def L(*a):
    """多个分量的欧氏长度"""
    return torch.sqrt(sum(x * x for x in a) + 1e-12)


def rot4(xw=0.0, yw=0.0, zw=0.0, xy=0.0):
    """>v<🧊四维切片-旋转-平面旋转的乘积 R = R_xw R_yw R_zw R_xy（角度：弧度）"""
    R = torch.eye(4, dtype=torch.float32)
    for (i, j), a in (((0, 3), xw), ((1, 3), yw), ((2, 3), zw), ((0, 1), xy)):
        G = torch.eye(4, dtype=torch.float32)
        c, s = math.cos(a), math.sin(a)
        G[i, i], G[i, j], G[j, i], G[j, j] = c, -s, s, c
        R = R @ G
    return R.to(DEV)


class Obj:
    def __init__(self, name, op, eq, f, cmax, facets=None, step=0.9, verts=None, bound=2.2, cam=5.2, R0=None):
        self.name, self.op, self.eq, self.f, self.cmax = name, op, eq, f, cmax
        self.cam, self.R0 = cam, R0   # 相机距离；R0 = 固定的基准朝向（比如让切片法向沿体对角线）
        self.facets, self.step, self.verts, self.bound = facets, step, verts, bound


def _cols(p):
    return p[:, 0], p[:, 1], p[:, 2], p[:, 3]


# --- spin：绕一个平面旋转（ρ = √(a² + b²) 代替一个坐标）---
def f_duocylinder(p, r=0.85):
    x, y, z, w = _cols(p)
    a, b = L(x, y) - r, L(z, w) - r
    return L(a.clamp(min=0), b.clamp(min=0)) + torch.maximum(a, b).clamp(max=0) - 0.04  # 倒一点点角


def f_tiger(p, R1=0.95, R2=0.95, r=0.38):
    x, y, z, w = _cols(p)
    return L(L(x, y) - R1, L(z, w) - R2) - r


def f_ditorus(p, R=1.05, R2=0.5, r=0.2):
    x, y, z, w = _cols(p)
    return L(L(L(x, y) - R, z) - R2, w) - r


def f_spheritorus(p, R=1.0, r=0.38):
    x, y, z, w = _cols(p)
    return L(L(x, y, z) - R, w) - r


def f_torisphere(p, R=1.0, r=0.42):
    x, y, z, w = _cols(p)
    return L(L(x, y) - R, z, w) - r


# --- extrude：3D 形状沿 w 拉成棱柱（精确的拉伸距离）---
def _torus3(x, y, z, R=0.85, r=0.3):
    return L(L(x, z) - R, y) - r


def f_extrude_torus(p, h=0.7):
    x, y, z, w = _cols(p)
    d, e = _torus3(x, y, z), w.abs() - h
    return L(d.clamp(min=0), e.clamp(min=0)) + torch.maximum(d, e).clamp(max=0)


# --- loft：沿 w 在两个 3D SDF 之间插值，切在不同 w 就是 A→B 的形变 ---
def _smooth(e0, e1, x):
    t = ((x - e0) / (e1 - e0)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def f_loft_sphere_torus(p, Lw=1.0):
    x, y, z, w = _cols(p)
    fa = L(x, y, z) - 0.8
    fb = _torus3(x, y, z, 0.8, 0.28)
    s = _smooth(-0.8 * Lw, 0.8 * Lw, w)
    return torch.maximum(fa * (1 - s) + fb * s, w.abs() - Lw)


def f_loft_sphere_twins(p, Lw=1.0):
    x, y, z, w = _cols(p)
    fa = L(x, y, z) - 0.85
    fb = torch.minimum(L(x - 0.62, y, z), L(x + 0.62, y, z)) - 0.45
    s = _smooth(-0.8 * Lw, 0.8 * Lw, w)
    return torch.maximum(fa * (1 - s) + fb * s, w.abs() - Lw)


# --- link：S¹ 管 与 S² 壳 在 4D 里互锁（1 + 2 = 4 − 1，链环数 1）---
def f_link(p, R=0.8, rho=0.6, r=0.16):
    x, y, z, w = _cols(p)
    circle = L(L(x, y) - R, z, w) - r               # S¹ = {x²+y² = R², z = w = 0} 加粗
    sphere = L(L(x - R, z, w) - rho, y) - r         # S² = {(x−R)²+z²+w² = ρ², y = 0} 加粗，圆心在 S¹ 上
    return torch.minimum(circle, sphere)


# --- sweep：3-球沿 4D 曲线 γ(t) = R/√2 (cos t, sin t, cos 3t, sin 3t)（Clifford 环面上的 (1,3) 曲线）扫过 ---
_SW_T = torch.linspace(0, 2 * math.pi, 241, device=DEV)[:-1]
_SW_C = torch.stack([torch.cos(_SW_T), torch.sin(_SW_T), torch.cos(3 * _SW_T), torch.sin(3 * _SW_T)], 1) * (1.25 / 2 ** 0.5)


def f_sweep(p, r=0.17):
    out = []
    for blk in torch.split(p, 1 << 18):
        out.append(torch.cdist(blk, _SW_C).min(1).values)
    return torch.cat(out) - r


# --- 正多胞体：凸 ⇒ f = max_i (n_i · p − d)，是距离的下界；棱 = 两个最大面值接近的地方 ---
def _tess_normals():
    n = torch.cat([torch.eye(4), -torch.eye(4)])
    return n


def _c24_normals():
    n = [v for v in _tess_normals()]
    for s in range(16):
        n.append(torch.tensor([0.5 if (s >> k) & 1 else -0.5 for k in range(4)]))
    return torch.stack(n)


def _even_perms():
    import itertools
    out = []
    for perm in itertools.permutations(range(4)):
        inv = sum(1 for i in range(4) for j in range(i + 1, 4) if perm[i] > perm[j])
        if inv % 2 == 0:
            out.append(perm)
    return out


def _c600_verts():
    """600-cell 的 120 个顶点（单位 3-球上）= 120-cell 的 120 个面法向"""
    v = [*torch.eye(4), *(-torch.eye(4))]
    for s in range(16):
        v.append(torch.tensor([0.5 if (s >> k) & 1 else -0.5 for k in range(4)]))
    base = [PHI / 2, 0.5, 1 / (2 * PHI), 0.0]
    for perm in _even_perms():
        for s in range(8):
            b = [base[0] * (1 if s & 1 else -1), base[1] * (1 if s & 2 else -1), base[2] * (1 if s & 4 else -1), 0.0]
            q = [0.0] * 4
            for i in range(4):
                q[perm[i]] = b[i]
            v.append(torch.tensor(q))
    return torch.stack(v)


def make_polytope(normals, d):
    N = normals.to(DEV, torch.float32)
    N = N / N.norm(dim=1, keepdim=True)

    def facets(p):
        return p @ N.T - d

    def f(p):
        return facets(p).max(1).values
    return f, facets


def _edges(V):
    V = V.double()
    D = torch.cdist(V, V, compute_mode="donot_use_mm_for_euclid_dist")
    m = D[D > 1e-6].min()
    i, j = torch.nonzero((D - m).abs() < 1e-4 * m, as_tuple=True)
    k = i < j
    return torch.stack([i[k], j[k]], 1)


_TES_V = torch.tensor([[(1 if (s >> k) & 1 else -1) for k in range(4)] for s in range(16)], dtype=torch.float32)
_C24_V = torch.tensor([v for v in __import__("itertools").product([-1, 0, 1], repeat=4)
                       if sum(abs(t) for t in v) == 2], dtype=torch.float32)

def _householder(d):
    """反射 H = I − 2uuᵀ，把 e_w 映到 d（H 对称正交；多胞体对称，反射无妨）"""
    d = torch.tensor(d, dtype=torch.float32); d = d / d.norm()
    u = torch.tensor([0.0, 0, 0, 1]) - d; u = u / u.norm()
    return (torch.eye(4) - 2 * torch.outer(u, u)).to(DEV)


_tes_f, _tes_fac = make_polytope(_tess_normals(), 0.8)
_c24_f, _c24_fac = make_polytope(_c24_normals(), 0.95)
_c120_f, _c120_fac = make_polytope(_c600_verts(), 1.05)

OBJS = {o.name: o for o in [
    Obj("duocylinder", "spin x2", "max( |(x,y)| - r, |(z,w)| - r ),  r = 0.85", f_duocylinder, 0.95),
    Obj("tiger", "spin x2", "( |(x,y)| - R1 )^2 + ( |(z,w)| - R2 )^2 = r^2,  R1 = R2 = 0.95, r = 0.38", f_tiger, 1.35, cam=8.4),
    Obj("ditorus", "spin(torus)", "| ( |( |(x,y)| - R, z )| - R2, w ) | = r,  R = 1.05, R2 = 0.5, r = 0.2", f_ditorus, 0.2, cam=6.6),
    Obj("spheritorus", "spin(circle)", "| ( |(x,y,z)| - R, w ) | = r,  R = 1.0, r = 0.38", f_spheritorus, 0.42),
    Obj("torisphere", "spin(sphere)", "| ( |(x,y)| - R, z, w ) | = r,  R = 1.0, r = 0.42", f_torisphere, 0.46),
    Obj("extrude_torus", "extrude", "max( torus(x,y,z; 0.85, 0.3), |w| - 0.7 )", f_extrude_torus, 0.75),
    Obj("loft_sphere_torus", "loft", "mix( sphere(0.8), torus(0.8, 0.28), smoothstep(-0.8, 0.8, w) ),  |w| < 1", f_loft_sphere_torus, 1.02, step=0.6),
    Obj("loft_sphere_twins", "loft", "mix( sphere(0.85), 2 spheres(+-0.62, 0.45), smoothstep(-0.8, 0.8, w) ),  |w| < 1", f_loft_sphere_twins, 1.02, step=0.6),
    Obj("link", "link S1-S2", "S1: x^2+y^2 = R^2, z=w=0  |  S2: (x-R)^2+z^2+w^2 = rho^2, y=0  (+ thickness 0.16)", f_link, 0.8),
    Obj("sweep", "sweep", "B^3 swept along gamma(t) = 1.25/sqrt2 (cos t, sin t, cos 3t, sin 3t),  r = 0.17", f_sweep, 1.0),
    Obj("tesseract", "polytope", "max_i |x_i| = 0.8   (8 cubic cells)", _tes_f, 1.62, facets=_tes_fac, verts=_TES_V * 0.8),
    Obj("tesseract_diag", "polytope", "max_i |x_i| = 0.8,  hyperplane normal = (1,1,1,1)/2  (tet -> oct -> tet)", _tes_f, 1.6,
        facets=_tes_fac, verts=_TES_V * 0.8, R0=_householder([0.5, 0.5, 0.5, 0.5])),
    Obj("24cell", "polytope", "max( |x_i|, |sum +-x_i|/2 ) = 0.95   (24 octahedral cells)", _c24_f, 1.0, facets=_c24_fac, verts=_C24_V * 0.95),
    Obj("120cell", "polytope", "max_k n_k . p = 1.05,  n_k = 600-cell vertices   (120 dodecahedral cells)", _c120_f, 1.15, facets=_c120_fac,
        verts=_c600_verts() * 1.05),
]}


def slice_fn(obj, R, c):
    """3D 截面的 SDF：g(x) = f([x, c] · R)"""
    if obj.R0 is not None:
        R = R @ obj.R0
    def g(x):
        q = torch.cat([x, torch.full_like(x[:, :1], c)], 1)
        return obj.f(q @ R)
    return g


# ---------- 球追踪 ----------
def camera(w, h, yaw=0.6, pitch=0.42, dist=5.2, fov=34.0):
    """>v<🧊四维切片-相机-透视相机绕原点；返回每个像素的 (ro, rd)"""
    eye = torch.tensor([dist * math.cos(pitch) * math.sin(yaw), dist * math.sin(pitch), dist * math.cos(pitch) * math.cos(yaw)], device=DEV)
    fwd = -eye / eye.norm()
    right = torch.linalg.cross(fwd, torch.tensor([0.0, 1.0, 0.0], device=DEV)); right = right / right.norm()
    up = torch.linalg.cross(right, fwd)
    ys, xs = torch.meshgrid(torch.arange(h, device=DEV), torch.arange(w, device=DEV), indexing="ij")
    s = math.tan(math.radians(fov) / 2)
    u = (xs + 0.5 - w / 2) / (h / 2) * s
    v = -(ys + 0.5 - h / 2) / (h / 2) * s
    rd = fwd + u[..., None] * right + v[..., None] * up
    rd = rd / rd.norm(dim=-1, keepdim=True)
    return eye, rd.reshape(-1, 3)


def trace(g, eye, rd, bound, step=0.9, iters=160, eps=6e-4):
    n = rd.shape[0]
    b = (eye[None] * rd).sum(1)
    disc = b * b - (eye @ eye - bound * bound)
    t = (-b - torch.sqrt(disc.clamp(min=0))).clamp(min=0)
    tfar = -b + torch.sqrt(disc.clamp(min=0))
    alive = disc > 0
    hit = torch.zeros(n, dtype=torch.bool, device=DEV)
    idx = torch.nonzero(alive).squeeze(1)
    for _ in range(iters):
        if idx.numel() == 0:
            break
        p = eye[None] + rd[idx] * t[idx, None]
        d = g(p)
        h = d < eps * (1 + t[idx])
        hit[idx[h]] = True
        t[idx] += torch.where(h, 0.0, d * step)
        keep = (~h) & (t[idx] < tfar[idx])
        idx = idx[keep]
    return hit, t


def grad4(obj, R, q, e=1e-3):
    """4D 梯度（世界坐标），中心差分"""
    gs = []
    for k in range(4):
        dq = torch.zeros(4, device=DEV); dq[k] = e
        gs.append((obj.f((q + dq) @ R) - obj.f((q - dq) @ R)) / (2 * e))  # R 已含 R0（由调用方合并）
    return torch.stack(gs, 1)


def _fwidth(u, ok):
    """屏幕空间导数 |du/dx| + |du/dy|（只用两边都命中的邻居）"""
    big = torch.full_like(u, 1e9)
    out = 0
    for dim in (0, 1):
        fw = []
        for sh in (1, -1):
            nu, nok = torch.roll(u, sh, dim), torch.roll(ok, sh, dim)
            fw.append(torch.where(ok & nok, (u - nu).abs(), big))
        out = out + torch.minimum(fw[0], fw[1])
    return out.clamp(min=1e-5)


def render(obj, R, c, w, h, look="line", yaw=0.6, pitch=0.42, ss=2, spacing=0.045, red_gain=1.0, dist=None, xray=True):
    dist = dist or obj.cam
    """>v<🧊四维切片-渲染-球追踪截面 → 灰度 + 红权重（hxw）。look: line（等距平面切出的等高线 + 轮廓）| surface（Lambert + rim）"""
    hw, hh = w * ss, h * ss
    g = slice_fn(obj, R, c)
    eye, rd = camera(hw, hh, yaw, pitch, dist)
    hit, t = trace(g, eye, rd, obj.bound, obj.step)
    P = eye[None] + rd * t[:, None]
    hi = torch.nonzero(hit).squeeze(1)
    ph = P[hi]
    # 3D 法向（四面体差分）与 4D 梯度
    e = 1.5e-3
    ks = torch.tensor([[1, -1, -1], [-1, -1, 1], [-1, 1, -1], [1, 1, 1]], dtype=torch.float32, device=DEV)
    nrm = sum(ks[i][None] * g(ph + e * ks[i][None])[:, None] for i in range(4))
    nrm = nrm / (nrm.norm(dim=1, keepdim=True) + 1e-9)
    q = torch.cat([ph, torch.full_like(ph[:, :1], c)], 1)
    if obj.R0 is not None:
        R = R @ obj.R0
    g4 = grad4(obj, R, q)
    tang = g4[:, :3].norm(dim=1) / (g4.norm(dim=1) + 1e-9)   # = |4D 法向在切片里的分量|；→0 即超平面相切
    vdir = -rd[hi]
    light = torch.tensor([-0.45, 0.75, 0.5], device=DEV); light = light / light.norm()
    lam = (nrm @ light).clamp(min=0)
    ndv = (nrm * vdir).sum(1).clamp(min=0)
    rim = (1 - ndv) ** 3

    def img(vals, fill=0.0):
        a = torch.full((hh * hw,), fill, device=DEV)
        a[hi] = vals
        return a.reshape(hh, hw)

    ok = hit.reshape(hh, hw)
    T = t.reshape(hh, hw)
    # 轮廓：命中/未命中边界 + 深度跳变
    edge = torch.zeros_like(T)
    for dim in (0, 1):
        for sh in (1, -1):
            n_ok, n_t = torch.roll(ok, sh, dim), torch.roll(T, sh, dim)
            edge = torch.maximum(edge, ((ok & ~n_ok) | (ok & n_ok & ((T - n_t) < -0.06 * T))).float())
    if look == "surface":
        v = img(0.05 + 0.62 * lam + 0.55 * rim ** 1.5)
    else:
        if obj.facets is not None:     # 多胞体：截面多面体的棱 = 两个最大面值相等
            top = torch.topk(obj.facets(q @ R), 2, dim=1).values
            u = img(top[:, 0] - top[:, 1], 1.0)
            fw = _fwidth(u, ok)
            lines = (1 - u / (fw * 2.4)).clamp(0, 1)
        else:                         # 等距平面 y = k·spacing 再切一刀 → 地形图式的等高线
            u = img(ph[:, 1] / spacing)
            fw = _fwidth(u, ok)
            dn = (u - torch.round(u)).abs()
            lines = (1 - dn / (fw * 0.85)).clamp(0, 1)
        shade = img(0.3 + 0.7 * lam + 0.3 * rim)
        v = lines * ok * shade * 1.05 + edge * 0.95 + img(0.035 + 0.05 * lam)
    v = v * ok + edge * 0 if look == "surface" else v
    red = img(((0.32 - tang) / 0.22).clamp(0, 1) ** 1.5) * red_gain * ok
    if xray and obj.facets is None:
        red = torch.maximum(red, _xray_tangency(obj, R, c, eye, hw, hh) * 0.75 * red_gain)
    # 超采样 → 降到输出分辨率
    v = F.avg_pool2d(v[None, None], ss)[0, 0]
    red = F.avg_pool2d(red[None, None], ss)[0, 0]
    return v.clamp(0, 1), red.clamp(0, 1)


def _xray_tangency(obj, R, c, eye, hw, hh, n=72, thr=0.32):
    """被挡住的相切点也要看得见：在 n³ 网格上找截面表面附近 |n_xyz|/|n| < thr 的点，透视投影后泼成柔光（R 已含 R0）"""
    B = obj.bound * 0.9
    ax = torch.linspace(-B, B, n, device=DEV)
    X = torch.stack(torch.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)
    q = torch.cat([X, torch.full_like(X[:, :1], c)], 1)
    hcell = 2 * B / (n - 1)
    fq = obj.f(q @ R)
    if not (fq < 0).any():          # 截面是空的：不画「快要出生」的幽灵点
        return torch.zeros(hh, hw, device=DEV)
    pts = q[fq.abs() < 0.9 * hcell]
    out = torch.zeros(hh, hw, device=DEV)
    if pts.shape[0] == 0:
        return out
    g4 = grad4(obj, R, pts)
    r = g4[:, :3].norm(dim=1) / (g4.norm(dim=1) + 1e-9)
    k = r < thr
    if not k.any():
        return out
    wgt = ((thr - r[k]) / (thr - 0.1)).clamp(0, 1) ** 1.5
    fwd = -eye / eye.norm()
    right = torch.linalg.cross(fwd, torch.tensor([0.0, 1.0, 0.0], device=DEV)); right = right / right.norm()
    up = torch.linalg.cross(right, fwd)
    rel = pts[k, :3] - eye
    z = rel @ fwd
    s = math.tan(math.radians(34) / 2)
    px = ((rel @ right) / z / s * (hh / 2) + hw / 2).long()
    py = (-(rel @ up) / z / s * (hh / 2) + hh / 2).long()
    m = (px >= 0) & (px < hw) & (py >= 0) & (py < hh)
    out.index_put_((py[m], px[m]), wgt[m], accumulate=True)
    rad = max(2, hh // 120)
    x = out[None, None]
    for _ in range(3):
        x = F.avg_pool2d(x, 2 * rad + 1, 1, rad)
    x = x[0, 0]
    return (x / (x.max() + 1e-9)).clamp(0, 1) if x.max() > 0 else x


def compose(v, red, glow_r=5, glow_k=0.45):
    """灰 + 红 → RGB（torch 上做辉光）"""
    img = v[..., None] * GREY_T * (1 - red[..., None]) + torch.maximum(v, red * 0.85)[..., None] * RED_T * red[..., None]
    x = img.permute(2, 0, 1)[None]
    k = 2 * glow_r + 1
    bl = F.avg_pool2d(F.avg_pool2d(x, k, 1, glow_r, count_include_pad=False), k, 1, glow_r, count_include_pad=False)
    return (x + glow_k * bl).clamp(0, 1)[0].permute(1, 2, 0)


# ---------- 拓扑：粗体素上的 Betti 数 ----------
def _cc_count(mask, conn26):
    """连通块数：标签传播（GPU），26 或 6 邻接"""
    n = mask.numel()
    BIG = float(n + 10)
    lab = torch.where(mask, torch.arange(n, device=DEV, dtype=torch.float32).reshape(mask.shape), torch.tensor(BIG, device=DEV))
    for it in range(4000):
        if conn26:
            m = -F.max_pool3d(-lab[None, None], 3, 1, 1)[0, 0]
        else:
            P = F.pad(lab[None, None], (1, 1, 1, 1, 1, 1), value=BIG)[0, 0]
            m = torch.minimum(lab, torch.minimum(torch.minimum(P[:-2, 1:-1, 1:-1], P[2:, 1:-1, 1:-1]),
                              torch.minimum(torch.minimum(P[1:-1, :-2, 1:-1], P[1:-1, 2:, 1:-1]),
                                            torch.minimum(P[1:-1, 1:-1, :-2], P[1:-1, 1:-1, 2:]))))
        m = torch.where(mask, m, torch.tensor(BIG, device=DEV))
        if it % 8 == 7 and torch.equal(m, lab):
            break
        lab = m
    return int(torch.unique(lab[mask]).numel())


def betti(obj, R, c, n=72):
    """>v<🧊四维切片-拓扑-截面在 n³ 体素上的 (b0, b1, b2)：b0 = 26 邻接连通块，b2 = 补集（6 邻接）有界块，
    χ = V − E + F − C（闭立方体并的立方复形），b1 = b0 + b2 − χ"""
    B = obj.bound * 0.9
    ax = torch.linspace(-B, B, n, device=DEV)
    X, Y, Z = torch.meshgrid(ax, ax, ax, indexing="ij")
    occ = (slice_fn(obj, R, c)(torch.stack([X, Y, Z], -1).reshape(-1, 3)) < 0).reshape(n, n, n)
    if not occ.any():
        return 0, 0, 0
    O = F.pad(occ.float()[None, None], (2, 2, 2, 2, 2, 2))
    mp = lambda k: F.max_pool3d(O, k, 1).sum().item()  # noqa: E731
    V = mp((2, 2, 2))
    E = mp((1, 2, 2)) + mp((2, 1, 2)) + mp((2, 2, 1))
    Fc = mp((2, 1, 1)) + mp((1, 2, 1)) + mp((1, 1, 2))
    C = O.sum().item()
    chi = int(round(V - E + Fc - C))
    b0 = _cc_count(occ, True)
    comp = ~F.pad(occ[None, None].float(), (1, 1, 1, 1, 1, 1))[0, 0].bool()
    b2 = _cc_count(comp, False) - 1
    return b0, b0 + b2 - chi, b2


# ---------- 多胞体棱的透视投影 ----------
def project_polytope(obj, R, w, h, yaw=0.6, pitch=0.42, dist4=3.2, dist=5.2, c=None):
    """>v<🧊四维切片-投影-棱的 4D→3D（w 方向透视，眼在 w = dist4）→ 2D 透视；越靠 4D 眼越亮。
    c 给了就把棱与超平面 w' = c 的交点染红（= 此刻 3D 截面的顶点）"""
    V = (obj.verts.to(DEV) @ R.T)                      # 转过的 4D 顶点
    Ed = _edges(obj.verts.to(DEV))
    k = dist4 / (dist4 - V[:, 3])
    P3 = V[:, :3] * k[:, None]
    eye, _ = camera(1, 1, yaw, pitch, dist)
    fwd = -eye / eye.norm()
    right = torch.linalg.cross(fwd, torch.tensor([0.0, 1.0, 0.0], device=DEV)); right = right / right.norm()
    up = torch.linalg.cross(right, fwd)
    rel = P3 - eye
    z = rel @ fwd
    s = math.tan(math.radians(34) / 2)
    sx = (rel @ right) / z / s * (h / 2) + w / 2
    sy = -(rel @ up) / z / s * (h / 2) + h / 2
    ss = 3
    im = Image.new("L", (w * ss, h * ss)); dr = ImageDraw.Draw(im)
    imr = Image.new("L", (w * ss, h * ss)); drr = ImageDraw.Draw(imr)
    sx, sy, wv = sx.cpu().numpy() * ss, sy.cpu().numpy() * ss, V[:, 3].cpu().numpy()
    wmin, wmax = wv.min(), wv.max()
    for i, j in Ed.cpu().numpy():
        b = 0.25 + 0.75 * (((wv[i] + wv[j]) / 2 - wmin) / (wmax - wmin + 1e-9))
        dr.line([(sx[i], sy[i]), (sx[j], sy[j])], fill=int(255 * b), width=max(1, int(1.3 * ss * h / 1080 + 0.5)))
        if c is not None and (wv[i] - c) * (wv[j] - c) < 0:
            a = (c - wv[i]) / (wv[j] - wv[i])
            px, py = sx[i] + a * (sx[j] - sx[i]), sy[i] + a * (sy[j] - sy[i])
            r = 3.2 * ss * h / 1080
            drr.ellipse((px - r, py - r, px + r, py + r), fill=255)
    v = np.asarray(im.resize((w, h), Image.LANCZOS), np.float32) / 255
    red = np.asarray(imr.resize((w, h), Image.LANCZOS), np.float32) / 255
    return torch.tensor(v, device=DEV), torch.tensor(red, device=DEV), len(Ed)


# ---------- HUD / 输出 ----------
def hud(im, lines, tag, red_line=None, scale=1.0):
    d = ImageDraw.Draw(im, "RGBA")
    f = ImageFont.truetype(FONT, int(18 * scale))
    Hh, Ww = im.height, im.width
    lh = int(26 * scale)
    for i, line in enumerate(lines):
        col = (235, 60, 66) if red_line is not None and i == red_line else (150, 160, 168)
        d.text((int(48 * scale), Hh - int(48 * scale) - lh * (len(lines) - i)), line, font=f, fill=col)
    d.text((Ww - int(48 * scale) - d.textlength(tag, font=f), int(40 * scale)), tag, font=f, fill=(110, 118, 124))


def to_pil(rgb):
    return Image.fromarray((rgb.clamp(0, 1) * 255).byte().cpu().numpy())


class Sink:
    def __init__(self, out, fps=24):
        self.out = Path(out); self.i = 0
        if self.out.suffix.lower() == ".mp4":
            import imageio_ffmpeg
            self.out.parent.mkdir(parents=True, exist_ok=True)
            self.proc = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                          "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16",
                                          "-preset", "slow", str(self.out)], stdin=subprocess.PIPE)
        else:
            self.proc = None; self.out.mkdir(parents=True, exist_ok=True)

    def add(self, im):
        if self.proc:
            self.proc.stdin.write(np.asarray(im.convert("RGB")).tobytes())
        else:
            im.save(self.out / f"{self.i:04d}.png")
        self.i += 1

    def close(self):
        if self.proc:
            self.proc.stdin.close(); self.proc.wait()


def _b_str(b):
    return f"b0 b1 b2 = {b[0]} {b[1]} {b[2]}   chi = {b[0] - b[1] + b[2]}"


def animate(name, op, frames, out, look="line", theta_max=180.0, c0=None, c1=None, tag=None, hold=10):
    """>v<🧊四维切片-动画-slice：c 从 −c_max 扫到 +c_max；rotate：θ_xw 0→θ_max 同时 c 从 c0 移到 c1；project：多胞体棱投影在 xw+yw 里转。
    拓扑（Betti）变了的那一刻 HUD 那一行变红 hold 帧，画面里的红始终是「超平面相切处」"""
    obj = OBJS[name]
    sink = Sink(out)
    tag = tag or f"4D_SLICE / {name.upper()}"
    prev, since = None, 999
    events = []
    t0 = time.time()
    for i in range(frames):
        s = i / max(1, frames - 1)
        e = 0.5 - 0.5 * math.cos(math.pi * s)            # 缓入缓出
        yaw = 0.55 + 0.35 * s
        if op == "project":
            th = math.radians(theta_max) * e
            R = rot4(xw=th, yw=0.5 * th)
            c = 0.0
            v, red, ne = project_polytope(obj, R, W, H, yaw=yaw, c=0.0, dist=8.0)
            rgb = compose(v * 0.9, red, 4, 0.5)
            lines = [f"{name}: {obj.eq}", f"edges = {ne}" + (" (drawn: its dual, the 600-cell)" if name == "120cell" else "") + "   4D -> 3D: perspective along w, eye at w = 3.2   brightness ~ w",
                     f"theta_xw = {math.degrees(th):6.1f} deg   theta_yw = {math.degrees(0.5 * th):6.1f} deg   red = edges piercing w' = 0"]
            im = to_pil(rgb); hud(im, lines, tag); sink.add(im)
            continue
        if op == "slice":
            th = 0.0
            c = -obj.cmax + 2 * obj.cmax * e
        else:
            th = math.radians(theta_max) * e
            a, b = (c0 if c0 is not None else 0.0), (c1 if c1 is not None else 0.0)
            c = a + (b - a) * e
        R = rot4(xw=th)
        v, red = render(obj, R, c, W, H, look, yaw=yaw)
        b = betti(obj, R, c)
        if prev is not None and b != prev:
            since = 0
            events.append((i, prev, b)); was = prev
        else:
            since += 1
        rgb = compose(v, red)
        lines = [f"{name} ({obj.op}): {obj.eq}",
                 f"slice: w' = c,  p' = R_xw(theta) p      theta_xw = {math.degrees(th):6.1f} deg      c = {c:+.3f}",
                 _b_str(b) + (f"      <- was {was[0]} {was[1]} {was[2]}" if since < hold and events else "") + "   (voxels 72^3)",
                 "red = hyperplane tangent to the 4D surface (|n_xyz| / |n| < 0.32): only here can the slice change topology"]
        im = to_pil(rgb); hud(im, lines, tag, red_line=2 if since < hold else None); sink.add(im)
        prev = b
        if i % 24 == 0:
            print(f"  frame {i}/{frames}  {time.time() - t0:.1f}s  theta={math.degrees(th):.1f} c={c:+.3f} b={b}", flush=True)
    sink.close()
    for ev in events:
        print("  topology event:", ev)
    print(f"{out}: {sink.i} frames, {time.time() - t0:.1f}s")


# ---------- 接触表 ----------
SHEET_ROWS = [
    ("tiger", "rotate", None), ("ditorus", "rotate", None), ("loft_sphere_torus", "slice", None),
    ("extrude_torus", "rotate", None), ("tesseract_diag", "slice", None), ("link", "rotate", None),
]


def sheet(out, rows=None, look="line"):
    """>v<🧊四维切片-总览-行 = 物体，列 = 6 个切片位置 / 6 个 xw 角度；每格标 c、θ 和真数的 Betti 数，拓扑与左边一格不同 → 标签变红"""
    rows = rows or SHEET_ROWS
    SW, SH = 2880, 1920
    lab_w, top, bot = 330, 70, 150
    tw = (SW - lab_w - 40) // 6
    th_ = (SH - top - bot) // len(rows)
    canvas = torch.zeros(SH, SW, 3, device=DEV)
    labels = []
    t0 = time.time()
    for r, (name, op, _) in enumerate(rows):
        obj = OBJS[name]
        prev = None
        for k in range(6):
            s = k / 5
            if op == "slice":
                th, c = 0.0, -obj.cmax * 0.97 + 2 * obj.cmax * 0.97 * s
            else:
                th, c = math.radians(90) * s, {"link": 0.05, "ditorus": 0.05}.get(name, 0.18)
            if name == "tesseract":
                R = rot4(xw=th, yw=th * 0.8, zw=th * 0.6)
            else:
                R = rot4(xw=th)
            v, red = render(obj, R, c, tw, th_, look, yaw=0.6, ss=3, dist=obj.cam * 0.9, red_gain=1.0)
            b = betti(obj, R, c, n=64)
            canvas[top + r * th_: top + (r + 1) * th_, lab_w + k * tw: lab_w + (k + 1) * tw] = compose(v, red, 3, 0.4)
            labels.append((r, k, f"theta={math.degrees(th):.0f}  c={c:+.2f}", f"b={b[0]}{b[1]}{b[2]}", prev is not None and b != prev))
            prev = b
        print(f"  row {name}: {time.time() - t0:.1f}s", flush=True)
    im = to_pil(canvas)
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, 20); fs = ImageFont.truetype(FONT, 17)
    for r, (name, op, _) in enumerate(rows):
        obj = OBJS[name]
        y = top + r * th_ + th_ // 2 - 40
        d.text((40, y), name, font=f, fill=(200, 208, 214))
        d.text((40, y + 30), f"op: {obj.op}", font=fs, fill=(130, 138, 145))
        d.text((40, y + 54), "sweep c along w" if op == "slice" else "rot xw, c fixed", font=fs, fill=(130, 138, 145))
    for r, k, a, b, red in labels:
        x, y = lab_w + k * tw + 14, top + (r + 1) * th_ - 30
        d.text((x, y), a, font=fs, fill=(120, 128, 134))
        d.text((x + tw - 110, y), b, font=fs, fill=(235, 60, 66) if red else (120, 128, 134))
    hud(im, ["g(x,y,z) = f( [x, y, z, c] R ),  R = R_xw(theta) ...   sphere-traced 3D cross-section of a 4D signed distance field",
             "lines: extra slicing planes y = k * 0.045 (polytopes: edges of the slice polyhedron)   b = Betti b0 b1 b2 on 64^3 voxels",
             "red = hyperplane tangent to the 4D surface (where topology can change);  red label = topology differs from left"], "20_4D_SLICE_OVERVIEW")
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out, quality=92)
    print(f"{out}: {time.time() - t0:.1f}s")


def main():
    ap = argparse.ArgumentParser(description="4D 物体 → 3D 截面")
    ap.add_argument("--obj", default="tiger", choices=list(OBJS))
    ap.add_argument("--op", default="slice", choices=["slice", "rotate", "project"])
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--out", default=None)
    ap.add_argument("--look", default="line", choices=["line", "surface"])
    ap.add_argument("--theta", type=float, default=180.0, help="rotate/project：转多少度")
    ap.add_argument("--c0", type=float, default=None); ap.add_argument("--c1", type=float, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--sheet", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for o in OBJS.values():
            print(f"{o.name:20s} {o.op:14s} {o.eq}")
        return
    if a.sheet:
        sheet(a.out or str(MV / "静帧" / "20_四维切片_总览.jpg"), look=a.look)
        return
    animate(a.obj, a.op, a.frames, a.out or f"{a.obj}_{a.op}.mp4", a.look, a.theta, a.c0, a.c1, a.tag)


if __name__ == "__main__":
    main()
