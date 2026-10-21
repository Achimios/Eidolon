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
  python tools/four_d.py --obj tiger --insert 1.6 --v0 240 --v-end 18 --tau 0.3 --tilt "yw=22,xy=10" --out x.mp4 --contact x.jpg
      # 插入镜头（1280x720）：帧数 = ceil(槽长×fps)，θ 与 c 先快后缓 v(t) = v_end + (v0−v_end)e^{−t/τ}，物体 4D 倾斜，相机匀速漂移；--plan 只打印参数
  python tools/four_d.py --obj mesh_spin --insert 3.16 --look sculpt --fov 24 --target 0,0.22,0.05 --yaw0 1.32 --yaw-rate 0.157 \
      --key=-80,48 --red-point --cold 0.5 --v0 70 --v-end 8 --tau 1.1 --song-t0 57.94 --plot x.png ...
      # GoldenAge 版（👤2026-10-22）：侧面特写、长焦、相机绕 target 匀速转，一盏硬主光 + 软阴影 + AO，只留一个红点；--plot 运动速度图
  python tools/four_d.py --emerge 5.6 --mesh 3D/她.glb --sdf-res 256 --sdf-faces 300000 --out x.mp4 --contact x.jpg --plot x.png --song-t0 161.14
      # 浮现（👤2026-10-22）：4D 测地锥 —— 超平面扫过，她从手指尖沿身体长进三维；相机贴着手开始、对数匀速拉远到虚空里很小的一个人
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
    def __init__(self, name, op, eq, f, cmax, facets=None, step=0.9, verts=None, bound=2.2, cam=5.2, R0=None, plane="xw", c_rot=None, vbox=None, cap=None):
        self.cap = cap      # cap(p) > 0：表面点落在 |w| − h 截出的平切面上（surface 模式自发光）
        self.name, self.op, self.eq, self.f, self.cmax = name, op, eq, f, cmax
        self.plane, self.c_rot = plane, c_rot
        self.vbox = vbox or bound * 0.9       # Betti / X 光相切点 用的体素盒半边长   # rotate 用哪个平面（spin(mesh) 对 xw 不变 → zw）；rotate 时固定的 c
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


# ---------- 真网格（GLB）→ 3D SDF → 4D ----------
MESH_DEFAULT = MV / "3D" / "测试_人台.glb"
MESH_B = 1.12        # 体素网格覆盖 [-B, B]^3；网格本身归一化进 [-1, 1]^3


def _mesh_load(path, max_faces=40000, lateral="auto"):
    """GLB → (V, F)：bbox 居中，侧向（T-pose 手臂）轴转到 x（glTF y 朝上），最长边缩放到 2。
    lateral：glTF 里哪根轴是侧向。auto = x、z 里更宽的那根（站立人形对；蜷缩姿势前后比左右厚，会猜错 →
    spin 的 sqrt(x² + w²) 只用到前半身再镜像。blender_pose 导出的网格侧向恒为 z：传 "z"）"""
    import trimesh
    m = trimesh.load(str(path), force="mesh")
    V = np.asarray(m.vertices, np.float64)
    lo, hi = V.min(0), V.max(0)
    V = V - (lo + hi) / 2
    ext = hi - lo
    if lateral == "z" or (lateral == "auto" and ext[2] > ext[0]):   # 绕 y 转 −90°：(x, y, z) → (−z, y, x)，行列式 +1；Tripo 的正面（+x）转到 +z，朝向默认相机
        V = np.stack([-V[:, 2], V[:, 1], V[:, 0]], 1)
    elif lateral not in ("auto", "x"):
        # 侧向斜着摆的模型（👤Tripo 蜷缩：侧向在 xz 平面 60°，x / z 都不对）：sym = 镜面对称面搜索；数字 = 侧向角 φ（度）
        off = 0.0
        if lateral == "sym":
            phi, off, err, err_x, err_z = mesh_symmetry_plane(m, V)
            print(f"  symmetry plane: lateral phi = {phi:.1f} deg, offset {off:+.4f}  (mean mirror error {err:.4f};"
                  f" x: {err_x:.4f}, z: {err_z:.4f}; units = model)", flush=True)
        else:
            phi = float(lateral)
        V = lateral_rotate(V, phi, off)
        lo, hi = V.min(0), V.max(0)
        ctr = (lo + hi) / 2
        ctr[0] = 0.0                     # x' = 0 必须留在对称面上（spin 绕它转）；其余两轴重新居中
        V = V - ctr
        ext = np.maximum(hi - lo, 2 * np.abs(V).max(0))
    V *= 2.0 / ext.max()
    Fc = np.asarray(m.faces, np.int64)
    if len(Fc) > max_faces:
        V, Fc = _cluster_decimate(V, Fc, max_faces)
    return V, Fc


def lateral_rotate(V, phi_deg, off=0.0):
    """绕 y 把 xz 平面里方向 l = (cos φ, 0, sin φ) 转到 +x：x' = V·l − off，z' = V·(−sin φ, 0, cos φ)，y' = y。
    行列式 +1（x' × y' = z'）；z' = l × ŷ = 正面方向（φ = −90 ≡ 旧的 "z" 选项）"""
    a = math.radians(phi_deg)
    c, s = math.cos(a), math.sin(a)
    return np.stack([V[:, 0] * c + V[:, 2] * s - off, V[:, 1], -V[:, 0] * s + V[:, 2] * c], 1)


def mesh_symmetry_plane(m, V, n=12000, seed=1):
    """>v<🧊四维切片-网格-找左右镜面：面积均匀采 n 点，扫竖直镜面（xz 里的法向角 φ ∈ [0, 180)、偏移），
    分数 = 镜像点到原点集的平均最近距离（越小越对称）。粗扫 3° → 细扫 0.5°。返回 (φ, 偏移, 误差, φ=0 误差, φ=90 误差)"""
    import trimesh
    P, _ = trimesh.sample.sample_surface(m, n, seed=seed)
    ctr = (np.asarray(m.vertices).min(0) + np.asarray(m.vertices).max(0)) / 2
    P = torch.tensor(P - ctr, dtype=torch.float32, device=DEV)
    span = float((P.max(0).values - P.min(0).values).max())

    def score(phi, off):
        nv = torch.tensor([math.cos(math.radians(phi)), 0.0, math.sin(math.radians(phi))], device=DEV)
        Q = P - 2 * (P @ nv - off)[:, None] * nv
        return float(torch.cat([torch.cdist(b, P).min(1).values for b in torch.split(Q, 3000)]).mean())

    def best(phis, offs):
        return min((score(p, o), p, o) for p in phis for o in offs)

    e0, p0, o0 = best(np.arange(0, 180, 3.0), np.linspace(-0.08, 0.08, 9) * span)
    e1, p1, o1 = best(np.arange(p0 - 3, p0 + 3.01, 0.5), o0 + np.linspace(-0.02, 0.02, 9) * span)
    ex = min(score(0.0, o) for o in np.linspace(-0.08, 0.08, 17) * span)
    ez = min(score(90.0, o) for o in np.linspace(-0.08, 0.08, 17) * span)
    return p1 % 180.0, o1, e1, ex, ez


def _cluster_decimate(V, Fc, target):
    """顶点聚类减面（不装新库）：把顶点量化到 k³ 格子里取均值，删掉退化 / 重复的三角形；k 从大往小试到面数 ≤ target。
    暴力距离是 O(体素 × 面)，百万面的 Tripo 模型必须先减；128³ 体素下 ~1 cm 级的格子不损失可见形状"""
    for k in (400, 320, 256, 200, 160, 128, 100, 80):
        key = np.floor((V + 1.0001) / 2.0002 * k).astype(np.int64)
        _, inv = np.unique(key[:, 0] * k * k + key[:, 1] * k + key[:, 2], return_inverse=True)
        inv = inv.reshape(-1)
        cnt = np.bincount(inv)
        V2 = np.stack([np.bincount(inv, V[:, j]) / cnt for j in range(3)], 1)
        F2 = inv[Fc]
        F2 = F2[(F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2]) & (F2[:, 2] != F2[:, 0])]
        F2 = np.unique(F2, axis=0)
        if len(F2) <= target:
            print(f"  decimate {len(Fc)} -> {len(F2)} faces (vertex clustering, {k}^3 cells)", flush=True)
            return V2, F2
    return V2, F2


def _sdf_points(pts, T):
    """点集的有符号距离：|d| = 暴力点–三角形距离（GPU 分块），符号 = 广义缠绕数 Σ立体角/4π > 0.5"""
    A, Bv, C, ba, cb, ac, nor, sba, scb, sac, iba, icb, iac, inor = T
    d2 = lambda v: (v * v).sum(-1)  # noqa: E731
    chunk = max(32, int(4e6 // A.shape[0]))
    ud, wn = [], []
    for blk in torch.split(pts, chunk):
        pa, pb, pc = blk[:, None] - A[None], blk[:, None] - Bv[None], blk[:, None] - C[None]
        s = torch.sign((sba * pa).sum(-1)) + torch.sign((scb * pb).sum(-1)) + torch.sign((sac * pc).sum(-1))
        e1 = d2(ba * ((ba * pa).sum(-1) * iba).clamp(0, 1)[..., None] - pa)
        e2 = d2(cb * ((cb * pb).sum(-1) * icb).clamp(0, 1)[..., None] - pb)
        e3 = d2(ac * ((ac * pc).sum(-1) * iac).clamp(0, 1)[..., None] - pc)
        dpl = (nor * pa).sum(-1) ** 2 * inor
        dd = torch.where(s < 2, torch.minimum(torch.minimum(e1, e2), e3), dpl)
        ud.append(dd.min(1).values.sqrt())
        la, lb, lc = pa.norm(dim=-1), pb.norm(dim=-1), pc.norm(dim=-1)
        num = -(pa * torch.linalg.cross(pb, pc)).sum(-1)            # det(A−p, B−p, C−p)
        den = la * lb * lc + (pa * pb).sum(-1) * lc + (pb * pc).sum(-1) * la + (pc * pa).sum(-1) * lb
        wn.append((2 * torch.atan2(num, den)).sum(1) / (4 * math.pi))
    ud, wn = torch.cat(ud), torch.cat(wn)
    return torch.where(wn.abs() > 0.5, -ud, ud)


SDF_COARSE = 64      # n > 128 时走窄带：先算 64³ 全格，只在表面附近（|d| < 2.5 粗格）算精确值


def mesh_sdf_volume(V, Fc, n=128, B=MESH_B, coarse=SDF_COARSE, narrow=None):
    """>v<🧊四维切片-网格SDF-n³ 网格上的有符号距离（返回 vol[iz, iy, ix]）。
    n ≤ 128：全格暴力。n > 128：窄带 —— 粗格 SDF 三线性上采样，|d| < 2.5 粗格边长的体素重算精确值
    （人体表面带只占体积的几个百分点；远处的值只用来让球追踪跨大步，三线性的误差 ≪ 一个粗格）"""
    V = torch.tensor(V, dtype=torch.float32, device=DEV)
    Fc = torch.tensor(Fc, device=DEV)
    A, Bv, C = V[Fc[:, 0]], V[Fc[:, 1]], V[Fc[:, 2]]
    ba, cb, ac = Bv - A, C - Bv, A - C
    nor = torch.linalg.cross(ba, ac)
    d2 = lambda v: (v * v).sum(-1)  # noqa: E731
    T = (A, Bv, C, ba, cb, ac, nor, torch.linalg.cross(ba, nor), torch.linalg.cross(cb, nor), torch.linalg.cross(ac, nor),
         *[1 / d2(e).clamp(min=1e-20) for e in (ba, cb, ac)], 1 / d2(nor).clamp(min=1e-30))
    ax = torch.linspace(-B, B, n, device=DEV)
    Z, Y, X = torch.meshgrid(ax, ax, ax, indexing="ij")
    pts = torch.stack([X, Y, Z], -1).reshape(-1, 3)
    if not (n > 128 if narrow is None else narrow):
        return _sdf_points(pts, T).reshape(n, n, n).cpu().numpy()
    vc = torch.tensor(mesh_sdf_volume(V.cpu().numpy(), Fc.cpu().numpy(), coarse, B), device=DEV)[None, None]
    up = F.grid_sample(vc, (pts / B).reshape(1, -1, 1, 1, 3), mode="bilinear", align_corners=True).reshape(-1)
    h = 2 * B / (coarse - 1)
    band = up.abs() < 2.5 * h
    up = torch.where(band, up, torch.sign(up) * (up.abs() - 0.5 * h))     # 带外保守：|d| 少报半个粗格 → 球追踪不会越过表面
    up[band] = _sdf_points(pts[band], T)
    print(f"  narrow band: {int(band.sum())} / {band.numel()} voxels exact ({band.float().mean().item() * 100:.1f}%)", flush=True)
    return up.reshape(n, n, n).cpu().numpy()


class MeshSDF:
    """f3(x, y, z)：缓存的 SDF 体 + grid_sample 三线性；网格外用「到 [-1,1]³ 盒的距离」与「边界值 − 走出去的距离」取大，仍是距离下界"""
    def __init__(self, path, n=128, max_faces=40000, lateral="auto"):
        self.path, self.n, self.max_faces, self.lateral, self.vol = Path(path), n, max_faces, lateral, None

    @property
    def label(self):
        """HUD 字体（consola）没有中文 → 中文名映射成 ASCII"""
        st = self.path.stem
        tr = {"测试": "test", "人台": "mannequin", "她": "her", "躯体核心": "torso_core", "蜷缩": "curled",
              "无材质": "untextured", "有材质": "textured", "低模": "lowpoly", "高模": "highpoly", "有Normal": "normalmap"}
        s = "_".join(tr.get(p, p) for p in st.split("_"))
        return s if s.isascii() else "mesh"

    def load(self):
        suf = ("" if self.max_faces == 40000 else f"_f{self.max_faces // 1000}k") + ("" if self.lateral == "auto" else f"_lat{self.lateral}")
        cache = self.path.with_name(f"{self.path.stem}.sdf{self.n}{suf}.npy")
        if cache.exists():
            vol = np.load(cache)
        else:
            t0 = time.time()
            if DEV == "cuda":
                torch.cuda.reset_peak_memory_stats()
            V, Fc = _mesh_load(self.path, self.max_faces, self.lateral)
            vol = mesh_sdf_volume(V, Fc, self.n).astype(np.float32)
            np.save(cache, vol)
            vram = f", peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB" if DEV == "cuda" else ""
            print(f"  SDF {self.n}^3 of {self.path.name} ({len(Fc)} faces): {time.time() - t0:.1f}s{vram} -> {cache.name}", flush=True)
        self.vol = torch.tensor(vol, device=DEV)[None, None]

    def __call__(self, x, y, z):
        if self.vol is None:
            self.load()
        q = torch.stack([x, y, z], -1).reshape(-1, 3)
        qc = q.clamp(-MESH_B, MESH_B)
        out = (q - qc).norm(dim=-1)
        d = F.grid_sample(self.vol, (qc / MESH_B).reshape(1, -1, 1, 1, 3), mode="bilinear",
                          padding_mode="border", align_corners=True).reshape(-1)
        box = (q.abs() - 1.0).clamp(min=0).norm(dim=-1)
        return torch.where(out > 0, torch.maximum(box, d - out), d).reshape(x.shape)


def _ext(d, e):
    """精确的「3D 距离 × w 区间」组合"""
    return L(d.clamp(min=0), e.clamp(min=0)) + torch.maximum(d, e).clamp(max=0)


def mesh_objs(msh, prefix="mesh", h=0.6, ht=1.0, k=math.pi / 2):
    """>v<🧊四维切片-网格算子-extrude / spin / loft→sphere / loft→torus / twist（f3 = 真网格的 SDF，x = 侧向轴）"""
    def ext(p):
        x, y, z, w = _cols(p)
        return _ext(msh(x, y, z), w.abs() - h)

    def spin(p):
        x, y, z, w = _cols(p)
        return msh(L(x, w), y, z)

    def loft_sphere(p, Lw=1.0):
        x, y, z, w = _cols(p)
        s = _smooth(-0.8 * Lw, 0.8 * Lw, w)
        return torch.maximum(msh(x, y, z) * (1 - s) + (L(x, y, z) - 0.62) * s, w.abs() - Lw)

    def loft_torus(p, Lw=1.0):
        x, y, z, w = _cols(p)
        s = _smooth(-0.8 * Lw, 0.8 * Lw, w)
        return torch.maximum(msh(x, y, z) * (1 - s) + (L(L(x, y) - 0.72, z) - 0.2) * s, w.abs() - Lw)

    def twist(p):
        x, y, z, w = _cols(p)
        a = k * w
        ca, sa = torch.cos(a), torch.sin(a)
        return _ext(msh(ca * x + sa * z, y, -sa * x + ca * z), w.abs() - ht)

    def ext_cap(p):                   # 平切面 = |w| − h 那一项占上风（_ext 的 max 里它更大）
        x, y, z, w = _cols(p)
        return (w.abs() - h) - msh(x, y, z)

    def twist_cap(p):
        x, y, z, w = _cols(p)
        a = k * w
        ca, sa = torch.cos(a), torch.sin(a)
        return (w.abs() - ht) - msh(ca * x + sa * z, y, -sa * x + ca * z)

    lab = msh.label
    return {o.name: o for o in [
        Obj(f"{prefix}_extrude", "extrude", f"max( f3(x,y,z), |w| - {h} ),  f3 = SDF of {lab}", ext, h + 0.05, bound=1.6, cam=4.3, vbox=1.15,
            cap=ext_cap),
        Obj(f"{prefix}_spin", "spin(mesh)", f"f3( sqrt(x^2 + w^2), y, z )   ({lab} revolved about the yz-plane)", spin, 0.95,
            bound=1.6, cam=4.3, vbox=1.15, plane="zw", c_rot=0.02),
        Obj(f"{prefix}_loft_sphere", "loft", f"mix( f3, sphere(0.62), smoothstep(-0.8, 0.8, w) ),  |w| < 1", loft_sphere, 1.02,
            step=0.6, bound=1.6, cam=4.3, vbox=1.15),
        Obj(f"{prefix}_loft_torus", "loft", f"mix( f3, torus_xy(0.72, 0.2), smoothstep(-0.8, 0.8, w) ),  |w| < 1", loft_torus, 1.02,
            step=0.6, bound=1.6, cam=4.3, vbox=1.15),
        Obj(f"{prefix}_twist", "twist", f"max( f3( Rot_y(k w) [x,y,z] ), |w| - {ht} ),  k = {k:.3f} (90 deg per unit w)", twist, ht + 0.02,
            step=0.5, bound=1.6, cam=4.3, vbox=1.15, c_rot=0.3, cap=twist_cap),
    ]}


MESH = MeshSDF(MESH_DEFAULT)
OBJS.update(mesh_objs(MESH))


# ---------- 生长：从一点（手）长进三维（👤2026-10-22「从另一个维度浮现，从手开始」） ----------
_NB26 = [(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dz, dy, dx) != (0, 0, 0)]
CHAMFER_SLACK = 1.09     # 26 邻接倒角距离最多高估欧氏距离 ~8 %（方向 (2,1,0)：(√2+1)/√5 = 1.080）→ 下界要除以它


def chamfer_relax(D, h, mask=None, max_iter=6000, check=16):
    """>v<🧊四维切片-生长-26 邻接倒角距离的 min-plus 松弛（Bellman–Ford，GPU）：D(x) ← min(D(x), D(x+o) + |o|·h)。
    mask 给了：只在 mask 里传播（体内测地距离，沿结构走，不跨缝）；没给：全空间（欧氏 1-Lipschitz 下包络）。有限步内精确收敛"""
    n0, n1, n2 = D.shape
    inf = torch.tensor(float("inf"), device=D.device)
    for it in range(max_iter):
        P = F.pad(D[None, None], (1, 1, 1, 1, 1, 1), value=float("inf"))[0, 0]
        best = D
        for dz, dy, dx in _NB26:
            best = torch.minimum(best, P[1 + dz:1 + dz + n0, 1 + dy:1 + dy + n1, 1 + dx:1 + dx + n2] + h * math.sqrt(dz * dz + dy * dy + dx * dx))
        if mask is not None:
            best = torch.where(mask, best, inf)
        if it % check == check - 1 and torch.equal(best, D):
            return best, it + 1
        D = best
    return D, max_iter


class Grow:
    """>v<🧊四维切片-生长-4D 物体 O = { (p, w) : f3(p) ≤ 0 且 D(p) ≤ K·(w − w0) }：身体 × 一根以 seed 为顶点的「测地锥」。
    超平面 w = c 切出来 = 身体里离 seed 测地距离 ≤ r = K(c − w0) 的部分 → c 往上扫，她从 seed（手指尖）沿身体结构长出来。
    D = 体内 26 邻接倒角测地距离（只在体素 f3 < 0 里传播 → 垂着的手离大腿再近也不会跳过去），轻微平滑 + 顶点处双曲抹圆
    D_s = √(D² + a²) − a（锥尖变光滑 → 刚出生那一刻超平面与 4D 表面真相切 → 红点）。
    D_L = D 的欧氏 1-Lipschitz 下包络（全空间松弛）：max(f3, (D_L − r)/1.09) 是到截面的距离下界 → 球追踪 / 阴影 / AO 用它；
    真截面（命中判定、法向、平切面）用 D。体外的 D = 法向延拓（取最近体内值，8 层外常数）。体内连不到 seed 的碎块（体素孤岛）从最近的已长部分「跳」过去，数量会打印"""
    def __init__(self, msh, seed_side=1.0, K=1.0, apex=0.03, smooth=1, seed=None, ext_layers=8):
        self.msh, self.seed_side, self.K, self.apex, self.smooth, self.seed_xyz = msh, seed_side, K, apex, smooth, seed
        self.ext_layers = ext_layers
        self.D = None

    def build(self):
        if self.D is not None:
            return
        t0 = time.time()
        if self.msh.vol is None:
            self.msh.load()
        vol = self.msh.vol[0, 0]
        n = vol.shape[0]
        h = 2 * MESH_B / (n - 1)
        self.h, self.n = h, n
        inside = vol < 0
        ax = torch.linspace(-MESH_B, MESH_B, n, device=DEV)
        Z, Y, X = torch.meshgrid(ax, ax, ax, indexing="ij")
        if self.seed_xyz is None:
            self.seed_xyz = hand_tip(inside, X, Y, Z, self.seed_side)
        s = torch.tensor(self.seed_xyz, device=DEV)
        d2 = (X - s[0]) ** 2 + (Y - s[1]) ** 2 + (Z - s[2]) ** 2
        d2 = torch.where(inside, d2, torch.tensor(float("inf"), device=DEV))
        si = int(torch.argmin(d2))
        D0 = torch.full_like(vol, float("inf")).reshape(-1)
        D0[si] = 0.0
        D, it_g = chamfer_relax(D0.reshape(vol.shape), h, inside)
        reach = torch.isfinite(D)
        self.unreached = int((inside & ~reach).sum())
        inf = torch.tensor(float("inf"), device=DEV)
        # 体外：法向延拓（一层层往外，取已有邻居的平均）—— 不能用 Lipschitz 下包络：那会把手的小值隔空灌到大腿表面，
        # 表面（体内体外体素之间插值）比体内先长 → 皮肤上一片蕾丝状的「提前生长」（v0 实测：大腿表面 0.7，体内 1.33）
        if self.unreached:
            H0, _ = chamfer_relax(torch.where(reach, D, inf), h)
            D = torch.where(inside & ~reach, H0, D)    # 孤岛：从最近的已长部分「跳」过去
        known = inside.clone()
        val = torch.where(known, D, torch.zeros_like(D))
        for _ in range(self.ext_layers):
            kf = known.float()[None, None]
            num = F.avg_pool3d((val * known.float())[None, None], 3, 1, 1)[0, 0]
            den = F.avg_pool3d(kf, 3, 1, 1)[0, 0]
            new = (~known) & (den > 0)
            val = torch.where(new, num / den.clamp(min=1e-9), val)
            known = known | new
        field = torch.where(known, val, torch.full_like(val, float(val[known].max()) + 1.0))   # 再远：常数（那里 f3 早就 ≫ 0）
        for _ in range(self.smooth):              # 倒角距离的等值面是多面体 → 轻微平滑（体内体外一起，seed 附近对称）
            field = F.avg_pool3d(F.pad(field[None, None], (1, 1, 1, 1, 1, 1), mode="replicate"), 3, 1)[0, 0]
            field[si // (n * n), (si // n) % n, si % n] = 0.0
        field = torch.sqrt(field * field + self.apex ** 2) - self.apex   # 锥尖抹圆（单调变换，不改先后顺序）
        DL, it_l = chamfer_relax(torch.where(inside, field, inf), h)  # 下界用：只认体内的值
        Dfield = field
        self.Dmax = float(Dfield[inside].max())
        self.D = Dfield[None, None].contiguous()
        self.DL = DL[None, None].contiguous()
        self.inside = inside
        self.seed_idx = si
        cent = torch.stack([X[inside].mean(), Y[inside].mean(), Z[inside].mean()])
        self.center = tuple(float(x) for x in cent)
        print(f"  grow: geodesic {n}^3 from seed {tuple(round(x, 3) for x in self.seed_xyz)}  ({it_g} relax iters; Lipschitz hull {it_l});"
              f" D_max = {self.Dmax:.3f}; unreached interior voxels {self.unreached} / {int(inside.sum())}; {time.time() - t0:.1f}s", flush=True)

    def _samp(self, G, x, y, z):
        q = torch.stack([x, y, z], -1).reshape(-1, 3).clamp(-MESH_B, MESH_B)
        return F.grid_sample(G, (q / MESH_B).reshape(1, -1, 1, 1, 3), mode="bilinear", padding_mode="border",
                             align_corners=True).reshape(x.shape)

    def dist(self, x, y, z):
        self.build()
        return self._samp(self.D, x, y, z)

    def r_of(self, c, w0=0.0):
        return self.K * (c - w0)

    def f(self, p):
        """4D：max( f3(p), (D(p) − K w) / √(1 + K²) )   （w0 = 0）"""
        x, y, z, w = _cols(p)
        return torch.maximum(self.msh(x, y, z), (self.dist(x, y, z) - self.K * w) / math.sqrt(1 + self.K ** 2))

    def cap(self, p):
        x, y, z, w = _cols(p)
        return (self.dist(x, y, z) - self.K * w) / math.sqrt(1 + self.K ** 2) - self.msh(x, y, z)

    def lb(self, c):
        """3D 截面（w = c）距离的下界：max(f3, (D_L − r) / 1.09)"""
        self.build()
        r = self.r_of(c)

        def g(x):
            return torch.maximum(self.msh(x[:, 0], x[:, 1], x[:, 2]), (self._samp(self.DL, x[:, 0], x[:, 1], x[:, 2]) - r) / CHAMFER_SLACK)
        return g

    def tracer(self, c, eye, rd, bound, iters=420, eps=6e-4):
        """球追踪：步长用下界 lb；lb 很小但真截面 g > 半个体素（缝边上下界偏松的地方）→ 不算命中，小步继续"""
        self.build()
        lb = self.lb(c)
        r = self.r_of(c)

        def true_g(x):
            return torch.maximum(self.msh(x[:, 0], x[:, 1], x[:, 2]), (self.dist(x[:, 0], x[:, 1], x[:, 2]) - r) / math.sqrt(1 + self.K ** 2))
        n = rd.shape[0]
        b = (eye[None] * rd).sum(1)
        disc = b * b - (eye @ eye - bound * bound)
        t = (-b - torch.sqrt(disc.clamp(min=0))).clamp(min=0)
        tfar = -b + torch.sqrt(disc.clamp(min=0))
        hit = torch.zeros(n, dtype=torch.bool, device=DEV)
        idx = torch.nonzero(disc > 0).squeeze(1)
        for _ in range(iters):
            if idx.numel() == 0:
                break
            p = eye[None] + rd[idx] * t[idx, None]
            d = lb(p)
            near = d < eps * (1 + t[idx])
            ok = near & (true_g(p) < 0.5 * self.h)
            hit[idx[ok]] = True
            t[idx] += torch.where(ok, torch.zeros_like(d), torch.where(near, torch.full_like(d, 0.35 * self.h), d * 0.92))
            keep = (~ok) & (t[idx] < tfar[idx])
            idx = idx[keep]
        return hit, t

    def emit(self, ph, c, width=0.015):
        """皮肤上紧挨着生长前沿的一圈亮带：exp(−(r − D)/width)，只在已长出的一侧"""
        r = self.r_of(c)
        gap = r - self.dist(ph[:, 0], ph[:, 1], ph[:, 2])
        return torch.where(gap >= -0.5 * self.h, torch.exp(-gap.clamp(min=0) / width), torch.zeros_like(gap))

    def xray_box(self, c, margin=0.03):
        """已长出部分的包围盒（中心, 半边长）+ 余量；X 光相切点在这里面找"""
        occ = self.occupancy(c)
        if not occ.any():
            return self.seed_xyz, margin
        n = self.n
        idx = torch.nonzero(occ)
        lo, hi = idx.min(0).values.float(), idx.max(0).values.float()
        to = lambda i: -MESH_B + i * 2 * MESH_B / (n - 1)  # noqa: E731
        ctr = [float(to((lo[k] + hi[k]) / 2)) for k in (2, 1, 0)]
        half = float((hi - lo).max()) * MESH_B / (n - 1) + margin
        return tuple(ctr), half

    def occupancy(self, c):
        """体素中心上的截面：f3 < 0 且 D ≤ r（测试 / HUD 体积比用）"""
        self.build()
        return self.inside & (self.D[0, 0] <= self.r_of(c))


def hand_tip(inside, X, Y, Z, side=1.0):
    """>v<🧊四维切片-生长-手指尖：膝盖以上（离脚底 > 30 % 身高）、侧向 side·x 超过最宽处 60 % 的体素（= 垂着的手臂）里最低的那个"""
    ys = Y[inside]
    above = Y > ys.min() + 0.3 * (ys.max() - ys.min())
    xs = X[inside & above] * side
    arm = inside & above & (X * side > 0.6 * xs.max())
    if not arm.any():
        raise ValueError("找不到手臂（侧向轴不对？）")
    j = int(torch.argmin(torch.where(arm, Y, torch.tensor(9.0, device=DEV))))
    return (float(X.reshape(-1)[j]), float(Y.reshape(-1)[j]), float(Z.reshape(-1)[j]))


def grow_obj(msh, seed_side=1.0, K=1.0):
    gr = Grow(msh, seed_side, K)
    o = Obj("mesh_grow", "grow (geodesic cone)", f"max( f3(p), (D(p) - K w) / sqrt(1+K^2) ),  D = geodesic distance inside {msh.label} from the hand, K = {K:g}",
            gr.f, 3.0, bound=1.6, cam=4.3, vbox=1.15, cap=gr.cap)
    o.grow = gr
    o.tracer, o.lb, o.emit, o.xray_box = gr.tracer, gr.lb, gr.emit, gr.xray_box
    return o


def slice_fn(obj, R, c):
    """3D 截面的 SDF：g(x) = f([x, c] · R)"""
    if obj.R0 is not None:
        R = R @ obj.R0
    def g(x):
        q = torch.cat([x, torch.full_like(x[:, :1], c)], 1)
        return obj.f(q @ R)
    return g


# ---------- 球追踪 ----------
def cam_basis(yaw=0.6, pitch=0.42, dist=5.2, target=(0.0, 0.0, 0.0)):
    """相机在以 target 为心的球面上：eye = target + dist·(cos p sin y, sin p, cos p cos y)；返回 (eye, fwd, right, up)"""
    tg = torch.tensor(target, dtype=torch.float32, device=DEV)
    off = torch.tensor([math.cos(pitch) * math.sin(yaw), math.sin(pitch), math.cos(pitch) * math.cos(yaw)], device=DEV)
    eye = tg + dist * off
    fwd = -off
    right = torch.linalg.cross(fwd, torch.tensor([0.0, 1.0, 0.0], device=DEV)); right = right / right.norm()
    up = torch.linalg.cross(right, fwd)
    return eye, fwd, right, up


def camera(w, h, yaw=0.6, pitch=0.42, dist=5.2, fov=34.0, target=(0.0, 0.0, 0.0)):
    """>v<🧊四维切片-相机-透视相机绕 target（默认原点）；返回每个像素的 (ro, rd)"""
    eye, fwd, right, up = cam_basis(yaw, pitch, dist, target)
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


def key_dir(yaw, pitch, az, el):
    """>v<🧊四维切片-雕塑光-主光方向（指向光源，世界坐标，固定不随相机转）：在 yaw / pitch 那台相机的水平基里定，
    az = 0 从相机这边打、+90 从画面右侧、±180 从背后；el = 仰角（度）。相机绕物体转时光不动 → 明暗交界线在形体上滑"""
    _, fwd, right, _ = cam_basis(yaw, pitch, 1.0)
    fh = torch.stack([fwd[0], torch.zeros_like(fwd[1]), fwd[2]]); fh = fh / fh.norm()
    a, e = math.radians(az), math.radians(el)
    d = math.cos(e) * (math.cos(a) * -fh + math.sin(a) * right) + math.sin(e) * torch.tensor([0.0, 1.0, 0.0], device=DEV)
    return d / d.norm()


def soft_shadow(g, p, L, k=14.0, iters=56, tmax=3.2):
    """iq 式软阴影：沿光线球追踪，res = min(k·h / t)；k 大 → 影子边硬（主光是硬光）"""
    res = torch.ones(p.shape[0], device=DEV)
    t = torch.full_like(res, 0.012)
    idx = torch.arange(p.shape[0], device=DEV)
    for _ in range(iters):
        if idx.numel() == 0:
            break
        hh = g(p[idx] + L[None] * t[idx, None])
        res[idx] = torch.minimum(res[idx], (k * hh / t[idx]).clamp(min=0))
        t[idx] += hh.clamp(0.008, 0.18)
        idx = idx[(hh > 2e-4) & (t[idx] < tmax)]      # 撞上（hh → 0 时 res 已 ≈ 0）或走出包围球就停
    return res.clamp(0, 1)


def sdf_ao(g, p, n, steps=5):
    """距离场环境光遮蔽：沿法向走 h_i，Σ (h_i − g)·衰减 → 凹处变暗（深黑里的体积感）"""
    occ, wgt = torch.zeros(p.shape[0], device=DEV), 1.0
    for i in range(steps):
        hs = 0.02 + 0.07 * i
        occ = occ + (hs - g(p + n * hs)) * wgt
        wgt *= 0.72
    return (1 - 2.2 * occ).clamp(0, 1)


def render(obj, R, c, w, h, look="line", yaw=0.6, pitch=0.42, ss=2, spacing=0.045, red_gain=1.0, dist=None, xray=True, cut_glow=0.55,
           fov=34.0, target=(0.0, 0.0, 0.0), key=None, emit=0.0):
    dist = dist or obj.cam
    """>v<🧊四维切片-渲染-球追踪截面 → 灰度 + 红权重（hxw）。look: line（等距平面切出的等高线 + 轮廓）| surface（Lambert + rim）
    | sculpt（一盏硬主光 key 斜扫 + 软阴影 + AO + 镜面高光，几乎无补光 → 深黑；平切面照样自发光）"""
    hw, hh = w * ss, h * ss
    g = slice_fn(obj, R, c)
    eye, rd = camera(hw, hh, yaw, pitch, dist, fov, target)
    tracer = getattr(obj, "tracer", None)          # 生长物体：自己的追踪器（下界步长 + 真截面判命中），只对 R = I 有效
    if tracer is not None and not torch.allclose(R, torch.eye(4, device=R.device)):
        raise ValueError(f"{obj.name} 只支持 R = I（生长锥没有定义 4D 旋转后的下界）")
    hit, t = tracer(c, eye, rd, obj.bound) if tracer is not None else trace(g, eye, rd, obj.bound, obj.step)
    gs = obj.lb(c) if getattr(obj, "lb", None) is not None else g      # 阴影 / AO 用距离下界（真截面在缝边不是 Lipschitz 的）
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
    if look == "sculpt":
        Lk = key if key is not None else key_dir(yaw, pitch, -95.0, 38.0)
        lk = (nrm @ Lk).clamp(min=0)
        sh = soft_shadow(gs, ph + nrm * 2.5e-3, Lk)
        ao = sdf_ao(gs, ph, nrm)
        hv = Lk[None] + vdir; hv = hv / (hv.norm(dim=1, keepdim=True) + 1e-9)
        spec = (nrm * hv).sum(1).clamp(min=0) ** 48
        lit = sh * (1.06 * lk ** 1.15 + 0.6 * spec)
        lam = lk * sh                                            # 切面亮度跟主光走（下面 face 用）
        v = img(0.012 + 0.035 * ao + lit * (0.55 + 0.45 * ao) + 0.10 * rim ** 2 * ao)
        if getattr(obj, "emit", None) is not None and emit > 0:      # 生长前沿在皮肤上的亮带
            v = v + emit * img(obj.emit(ph, c))
    elif look == "surface":
        v = img(0.05 + 0.62 * lam + 0.55 * rim ** 1.5)
    if look in ("surface", "sculpt"):
        if obj.cap is not None and cut_glow > 0:
            # 平切面（|w| − h 截出来的，正对相机时 Lambert + rim 都 ≈ 0 → 一块死黑）：自发光底 + 剖面线（y 等距）+ 亮边框
            cw = img((obj.cap(q @ R) / 0.01).clamp(0, 1))
            u = img(ph[:, 1] / spacing)
            dn = (u - torch.round(u)).abs()
            hatch = (1 - dn / (_fwidth(u, ok) * 0.85)).clamp(0, 1)
            face = cut_glow * (0.62 + 0.25 * img(lam)) + 0.3 * hatch
            border = torch.zeros_like(cw)
            for dim in (0, 1):
                for sh in (1, -1):
                    border = torch.maximum(border, (cw - torch.roll(cw, sh, dim)).abs() * (ok & torch.roll(ok, sh, dim)).float())
            v = torch.maximum(v * (1 - cw) + face * cw, border.clamp(0, 1) * 0.9)
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
    v = v * ok if look in ("surface", "sculpt") else v
    red = img(((0.32 - tang) / 0.22).clamp(0, 1) ** 1.5) * red_gain * ok
    if xray and obj.facets is None:
        red = torch.maximum(red, _xray_tangency(obj, R, c, cam_basis(yaw, pitch, dist, target), hw, hh, fov=fov) * 0.75 * red_gain)
    # 超采样 → 降到输出分辨率
    v = F.avg_pool2d(v[None, None], ss)[0, 0]
    red = F.avg_pool2d(red[None, None], ss)[0, 0]
    return v.clamp(0, 1), red.clamp(0, 1)


def _xray_tangency(obj, R, c, basis, hw, hh, n=72, thr=0.32, fov=34.0):
    """被挡住的相切点也要看得见：在 n³ 网格上找截面表面附近 |n_xyz|/|n| < thr 的点，透视投影后泼成柔光（R 已含 R0）"""
    B, ctr = obj.vbox, None
    if getattr(obj, "xray_box", None) is not None:      # 生长物体：网格只铺在已长出部分的包围盒上（刚出生时只有指尖那么大，72³ 的全盒子看不见它）
        ctr, B = obj.xray_box(c)
    ax = torch.linspace(-B, B, n, device=DEV)
    X = torch.stack(torch.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)
    if ctr is not None:
        X = X + torch.tensor(ctr, dtype=torch.float32, device=DEV)
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
    eye, fwd, right, up = basis           # 和 camera() 同一台相机（target / fov 都要一致，否则红点投错位置）
    rel = pts[k, :3] - eye
    z = rel @ fwd
    s = math.tan(math.radians(fov) / 2)
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


_COLD = torch.tensor([-0.10, -0.02, 0.08], device=DEV)


def compose(v, red, glow_r=5, glow_k=0.45, cold=0.0):
    """灰 + 红 → RGB（torch 上做辉光）。cold > 0：暗部 / 中间调再往冷里偏一点（亮部仍是本片的冷灰 GREY）"""
    col = GREY_T * (1 + cold * (1 - v[..., None]) * _COLD) if cold else GREY_T
    img = v[..., None] * col * (1 - red[..., None]) + torch.maximum(v, red * 0.85)[..., None] * RED_T * red[..., None]
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
    B = obj.vbox
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
    def __init__(self, out, fps=24, size=None):
        self.out = Path(out); self.i = 0
        w, h = size or (W, H)
        if self.out.suffix.lower() == ".mp4":
            import imageio_ffmpeg
            self.out.parent.mkdir(parents=True, exist_ok=True)
            self.proc = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                          "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16",
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
        R = rot4(**{obj.plane: th})
        v, red = render(obj, R, c, W, H, look, yaw=yaw)
        b = betti(obj, R, c)
        if prev is not None and b != prev:
            since = 0
            events.append((i, prev, b)); was = prev
        else:
            since += 1
        rgb = compose(v, red)
        lines = [f"{name} ({obj.op}): {obj.eq}",
                 f"slice: w' = c,  p' = R_{obj.plane}(theta) p      theta_{obj.plane} = {math.degrees(th):6.1f} deg      c = {c:+.3f}",
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


# ---------- 插入镜头：先快后缓（👤2026-10-21） ----------
def fast_slow(t, v0, v_end, tau):
    """>v<🧊四维切片-插入-速度曲线 v(t) = v_end + (v0 − v_end)·e^{−t/τ}：切进来时已在高速，指数缓下来，v_end ≠ 0 所以不停。
    返回 (位移 x(t) = ∫v = v_end·t + (v0 − v_end)·τ·(1 − e^{−t/τ}),  v(t))"""
    e = math.exp(-t / tau)
    return v_end * t + (v0 - v_end) * tau * (1 - e), v_end + (v0 - v_end) * e


def insert_frames(dur, fps=24):
    """槽长 dur 秒 → 帧数：向上取整（盖满剪辑表里的槽，多出的 < 1 帧由剪辑裁掉，不会露黑）"""
    return max(2, math.ceil(dur * fps - 1e-6))


def parse_tilt(spec):
    """'zw=20,yw=-15,xy=12'（度）→ {平面: 弧度}；物体在 4D 里的固定倾斜摆放"""
    out = {}
    for tok in filter(None, (spec or "").replace(" ", "").split(",")):
        k, _, v = tok.partition("=")
        if k not in ("xw", "yw", "zw", "xy"):
            raise ValueError(f"tilt 平面只能是 xw/yw/zw/xy：{tok}")
        out[k] = math.radians(float(v))
    return out


class InsertSpec:
    """插入镜头的全部参数（角度：度，角速度：度/秒；c 速度：单位/秒；相机 yaw 速度：弧度/秒，dolly：距离比例/秒）。
    相机：绕 target 转（默认原点），视角 fov（度）。key_az / key_el：sculpt 主光（度，相对起始相机，固定在世界里）。
    red_point：全片只留一个红点（最强的相切处，跨帧跟踪）。cold：暗部冷偏"""
    def __init__(self, dur, fps=24, size=(1280, 720), plane=None, plane2=None, ratio2=0.0, theta0=0.0, v0=240.0, v_end=25.0,
                 tau=0.35, c_start=None, cv0=0.0, cv_end=0.0, tilt="", yaw0=0.6, yaw_rate=0.10, pitch=0.42, dolly=0.04,
                 red_gain=1.0, look="line", cam_scale=1.0, hud_band=0, hud_scale=1.0, cut_glow=0.55,
                 fov=34.0, target=(0.0, 0.0, 0.0), key_az=-95.0, key_el=38.0, red_point=False, cold=0.0):
        self.dur, self.fps, self.size = dur, fps, size
        self.plane, self.plane2, self.ratio2 = plane, plane2, ratio2
        self.theta0, self.v0, self.v_end, self.tau = theta0, v0, v_end, tau
        self.c_start, self.cv0, self.cv_end = c_start, cv0, cv_end
        self.tilt, self.tilt_r = tilt, parse_tilt(tilt)
        self.yaw0, self.yaw_rate, self.pitch, self.dolly = yaw0, yaw_rate, pitch, dolly
        self.red_gain, self.look, self.cam_scale = red_gain, look, cam_scale
        self.cut_glow = cut_glow
        self.fov, self.target = fov, tuple(float(x) for x in target)
        self.key_az, self.key_el, self.red_point, self.cold = key_az, key_el, red_point, cold
        self.hud_band, self.hud_scale = hud_band, hud_scale      # hud_band > 0：物体只渲染在上面 H − band 像素里，HUD 独占底部条带（不重叠）
        if not 0 <= hud_band < size[1] // 2:
            raise ValueError("hud_band 必须在 [0, H/2)")
        if v_end <= 0 or v0 <= v_end:
            raise ValueError("先快后缓要求 v0 > v_end > 0（v_end = 0 会停住）")
        if yaw_rate == 0:
            raise ValueError("相机不能静止：yaw_rate ≠ 0")
        if not 5.0 <= fov <= 90.0:
            raise ValueError("fov 在 [5, 90] 度")
        if dolly * dur >= 0.5:
            raise ValueError("dolly × 槽长 ≥ 0.5：相机会推进物体里")


def insert_track(obj, sp):
    """>v<🧊四维切片-插入-每帧的真参数（纯数学，不渲染）：t, θ, v, c, dc/dt, yaw, dist。
    θ 与 c 共用同一条先快后缓曲线（同一个 τ），相机匀速漂移（不从静止开始、不停）"""
    n = insert_frames(sp.dur, sp.fps)
    c_start = sp.c_start if sp.c_start is not None else (obj.c_rot if obj.c_rot is not None else 0.0)
    out = []
    for i in range(n):
        t = i / sp.fps
        dth, v = fast_slow(t, sp.v0, sp.v_end, sp.tau)
        dc, cv = fast_slow(t, sp.cv0, sp.cv_end, sp.tau) if sp.cv0 != sp.cv_end else (sp.cv0 * t, sp.cv0)
        out.append(dict(i=i, t=t, th=sp.theta0 + dth, v=v, c=c_start + dc, cv=cv,
                        yaw=sp.yaw0 + sp.yaw_rate * t, dist=obj.cam * sp.cam_scale * (1 - sp.dolly * t)))
    return out


def insert_R(obj, sp, th_deg):
    """R = R_plane(θ) · R_plane2(ratio2·θ) · R_tilt（R_tilt 在最靠物体的一侧 = 物体自己的倾斜摆放）"""
    plane = sp.plane or obj.plane
    ang = {plane: math.radians(th_deg)}
    if sp.plane2:
        if sp.plane2 == plane:
            raise ValueError("plane2 必须和 plane 不同")
        ang[sp.plane2] = math.radians(th_deg * sp.ratio2)
    return rot4(**ang) @ rot4(**sp.tilt_r)


class RedPoint:
    """>v<🧊四维切片-红点-全片只留一个红点：红权重图（相切处 + X 光）模糊后取峰；有上一帧位置时按距离加权（偏向原地，不乱跳），
    位置一阶平滑，亮度 = 峰值强度的指数平滑（没有相切处就慢慢熄掉）。返回只含这一个高斯点的红图"""
    def __init__(self, h, w, sigma=None, keep=None, on=0.03):
        self.h, self.w = h, w
        self.sigma = sigma or max(1.6, h / 220)
        self.keep = keep or h / 8
        self.on = on                      # 模糊后的峰值 ≥ on 就算「有相切处」（满亮）
        self.pos, self.a, self.peak = None, 0.0, 0.0
        ys, xs = torch.meshgrid(torch.arange(h, device=DEV, dtype=torch.float32), torch.arange(w, device=DEV, dtype=torch.float32), indexing="ij")
        self.ys, self.xs = ys, xs

    def __call__(self, red):
        r = max(2, self.h // 90)
        x = red[None, None]
        for _ in range(2):
            x = F.avg_pool2d(x, 2 * r + 1, 1, r, count_include_pad=False)
        b = x[0, 0]
        score = b
        if self.pos is not None and self.a > 0.05:
            d2 = (self.xs - self.pos[0]) ** 2 + (self.ys - self.pos[1]) ** 2
            score = b * (0.3 + 0.7 * torch.exp(-d2 / (2 * self.keep ** 2)))
        j = int(torch.argmax(score))
        py, px = divmod(j, self.w)
        self.peak = float(b[py, px])
        tgt = min(1.0, self.peak / self.on)
        if self.pos is None or self.a <= 0.05 or math.hypot(px - self.pos[0], py - self.pos[1]) > 3 * self.keep:
            self.pos = (float(px), float(py))                 # 第一次 / 熄灭后 / 真跳走：直接落位
        else:
            self.pos = (self.pos[0] + 0.55 * (px - self.pos[0]), self.pos[1] + 0.55 * (py - self.pos[1]))
        self.a = tgt if self.a == 0 else 0.65 * self.a + 0.35 * tgt
        d2 = (self.xs - self.pos[0]) ** 2 + (self.ys - self.pos[1]) ** 2
        return (torch.exp(-d2 / (2 * self.sigma ** 2)) * 1.4 * self.a).clamp(0, 1)


def _insert_frame(obj, name, sp, k, prev_b, tag, rp=None):
    """一帧：渲染 + Betti + HUD。hud_band > 0 时物体画在上方 (H − band) 区域，HUD 在底部条带里 → 永不重叠。rp = RedPoint（跨帧状态）"""
    Wd, Hd = sp.size
    plane = sp.plane or obj.plane
    rot_s = f"R_{plane}(theta)" + (f" R_{sp.plane2}({sp.ratio2:g} theta)" if sp.plane2 else "")
    tilt_s = " ".join(f"{a}={math.degrees(v):+.0f}" for a, v in sp.tilt_r.items()) or "none"
    R = insert_R(obj, sp, k["th"])
    key = key_dir(sp.yaw0, sp.pitch, sp.key_az, sp.key_el) if sp.look == "sculpt" else None
    v, red = render(obj, R, k["c"], Wd, Hd - sp.hud_band, sp.look, yaw=k["yaw"], pitch=sp.pitch, red_gain=sp.red_gain, dist=k["dist"],
                    cut_glow=sp.cut_glow, fov=sp.fov, target=sp.target, key=key)
    if sp.red_point:
        rp = rp or RedPoint(*red.shape)
        red = rp(red)
    b = betti(obj, R, k["c"], n=64)
    rgb = compose(v, red, max(2, round(5 * Hd / 1080)), 0.45, cold=sp.cold)
    if sp.hud_band:
        rgb = torch.cat([rgb, torch.zeros(sp.hud_band, Wd, 3, device=rgb.device)], 0)
    changed = prev_b is not None and b != prev_b
    cam_s = (f"cam: orbit {math.degrees(sp.yaw_rate):+.1f} deg/s about ({sp.target[0]:+.2f},{sp.target[1]:+.2f},{sp.target[2]:+.2f})"
             f"  yaw {math.degrees(k['yaw']):6.1f}  pitch {math.degrees(sp.pitch):+.1f}  fov {sp.fov:g}  d {k['dist']:.2f}")
    if sp.look == "sculpt":
        cam_s += f"   key light az {sp.key_az:+.0f} el {sp.key_el:+.0f} (fixed in world)"
    lines = [f"{name} ({obj.op}): {obj.eq}",
             f"p' = {rot_s} R_tilt p    theta = {k['th']:+7.1f} deg    v(t) = {k['v']:6.1f} deg/s = {sp.v_end:g} + ({sp.v0:g} - {sp.v_end:g}) e^(-t/{sp.tau:g})",
             f"slice w' = c = {k['c']:+.3f}   dc/dt = {k['cv']:+.3f}/s     R_tilt: {tilt_s} deg     t = {k['t']:.3f} / {sp.dur:g} s",
             cam_s,
             _b_str(b) + "   (voxels 64^3)" + ("   <- topology changed" if changed else ""),
             ("red point = strongest tangency of hyperplane and 4D surface (|n_xyz| / |n| < 0.32), tracked" if sp.red_point
              else "red = hyperplane tangent to the 4D surface (|n_xyz| / |n| < 0.32)")]
    im = to_pil(rgb)
    hud(im, lines, tag, red_line=4 if changed else None, scale=Hd / 1080 * 1.15 * sp.hud_scale)
    return im, rgb, b, rp


def normal_flow(a, b):
    """图像运动速度估计（像素 / 帧）：亮度守恒 I_t + ∇I·u = 0 ⇒ |u_⊥| ≈ |I_t| / |∇I|；用 Σ|I_t| / Σ|∇I| 的比值（只在有梯度的地方算数）"""
    It = (b - a).abs()
    m = 0.5 * (a + b)
    gx = (m[:, 2:] - m[:, :-2])[1:-1] * 0.5
    gy = (m[2:] - m[:-2])[:, 1:-1] * 0.5
    gm = torch.sqrt(gx * gx + gy * gy)
    It = It[1:-1, 1:-1]
    wgt = gm > 0.01
    return float(It[wgt].sum() / (gm[wgt].sum() + 1e-9)) if wgt.any() else 0.0


def insert(name, sp, out, tag=None, contact=None, plot=None, song_t0=None):
    """>v<🧊四维切片-插入-渲染一条插入镜头（帧数 = ceil(槽长 × fps)），HUD 写每帧真参数；contact 给了就写接触表（全部帧 + 帧间差分）；
    plot 给了就写运动速度图（θ 角速度、相机角速度、实测帧间差分 / 法向光流，song_t0 给了就标歌里的重拍）"""
    obj = OBJS[name]
    Wd, Hd = sp.size
    track = insert_track(obj, sp)
    tag = tag or f"4D_INSERT / {name.upper()}"
    sink = Sink(out, sp.fps, sp.size)
    thumbs, diffs, flows, prev_arr, prev_b, rp = [], [], [], None, None, None
    t0 = time.time()
    for k in track:
        im, rgb, b, rp = _insert_frame(obj, name, sp, k, prev_b, tag, rp)
        arr = rgb[: Hd - sp.hud_band].mean(-1)             # 只量画面区（HUD 条带里的数字每帧都变，不算运动）
        diffs.append(float((arr - prev_arr).abs().mean()) if prev_arr is not None else float("nan"))
        flows.append(normal_flow(prev_arr, arr) if prev_arr is not None else float("nan"))
        prev_arr = arr
        sink.add(im)
        prev_b = b
        if contact:
            thumbs.append(im.resize((Wd // 4, Hd // 4), Image.LANCZOS))
        if k["i"] % 12 == 0:
            print(f"  frame {k['i']}/{len(track)}  {time.time() - t0:.1f}s", flush=True)
    sink.close()
    print(f"{out}: {sink.i} frames = {sink.i / sp.fps:.4f} s (slot {sp.dur:g} s), {time.time() - t0:.1f}s")
    print("  frame-diff (mean |dI|):", " ".join(f"{d:.4f}" for d in diffs[1:]))
    print("  normal flow (px/frame):", " ".join(f"{d:.2f}" for d in flows[1:]))
    if contact:
        _insert_contact(contact, thumbs, track, diffs, sp, name, tag)
    if plot:
        motion_plot(plot, track, diffs, flows, sp, name, tag, song_t0)
    return track, diffs


def insert_still(name, sp, idx, out, tag=None):
    """插入镜头里的单独一帧存成图（比较 look 用：同一参数、同一帧，只换 --look）。红点模式下跟踪状态从这一帧开始（无历史）"""
    obj = OBJS[name]
    k = insert_track(obj, sp)[idx]
    im, _, _, _ = _insert_frame(obj, name, sp, k, None, tag or f"4D_INSERT / {name.upper()}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    im.save(out, quality=92)
    print(f"{out}: frame #{idx} (t = {k['t']:.3f} s, theta = {k['th']:+.1f}, c = {k['c']:+.3f}, look = {sp.look})")


def song_downbeats(t0, t1):
    """感官.json（= 播放时间）里 [t0, t1] 内的重拍 / 乐句；没有文件就空"""
    p = MV / "感官.json"
    if not p.exists():
        return [], []
    import json
    ev = json.loads(p.read_text(encoding="utf-8")).get("events", {})
    pick = lambda xs: [x for x in xs if t0 - 1e-6 <= x <= t1 + 1e-6]  # noqa: E731
    return pick(ev.get("downbeats", [])), pick(ev.get("phrases", []))


def motion_plot(path, track, diffs, flows, sp, name, tag, song_t0=None):
    """>v<🧊四维切片-插入-运动速度图（PIL 画，不依赖 matplotlib）：上 = 理论角速度（θ 的 v(t)、相机绕转，度/秒），
    下 = 实测（法向光流 像素/帧，帧间差分 归一化）。横轴 = 镜头内秒数（song_t0 给了同时标歌里的时间 + 重拍竖线）"""
    PW, PH = 1600, 760
    im = Image.new("RGB", (PW, PH), (0, 0, 0))
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, 15); fs = ImageFont.truetype(FONT, 13)
    x0, x1 = 90, PW - 40
    n = len(track)
    T = (n - 1) / sp.fps
    tx = lambda t: x0 + (x1 - x0) * t / max(T, 1e-9)  # noqa: E731
    grey, dim, red, cam_c = (150, 160, 168), (60, 64, 68), (235, 60, 66), (200, 208, 214)
    panels = [(60, 360), (420, 700)]
    db, ph_ = song_downbeats(song_t0, song_t0 + T) if song_t0 is not None else ([], [])
    for (ya, yb) in panels:
        d.rectangle([x0, ya, x1, yb], outline=dim)
        for tb in db:
            xb = tx(tb - song_t0)
            d.line([(xb, ya), (xb, yb)], fill=(90, 40, 44) if tb in ph_ else (55, 58, 62), width=2 if tb in ph_ else 1)
    for tb in db:
        d.text((tx(tb - song_t0) + 4, panels[1][1] + 26), f"{tb:.2f}" + (" phrase" if tb in ph_ else " downbeat"), font=fs, fill=grey)
    # 横轴刻度
    for j in range(int(T / 0.5) + 1):
        t = j * 0.5
        d.line([(tx(t), panels[1][1]), (tx(t), panels[1][1] + 6)], fill=grey)
        d.text((tx(t) - 12, panels[1][1] + 8), f"{t:.1f}s", font=fs, fill=grey)
    # 上：理论角速度
    ya, yb = panels[0]
    cam_v = abs(math.degrees(sp.yaw_rate))
    vmax = max(max(k["v"] for k in track), cam_v) * 1.1
    ty = lambda v: yb - (yb - ya) * v / vmax  # noqa: E731
    for g in range(0, int(vmax) + 1, max(5, int(vmax / 6) // 5 * 5)):
        d.line([(x0 - 6, ty(g)), (x0, ty(g))], fill=grey)
        d.text((x0 - 50, ty(g) - 7), f"{g}", font=fs, fill=grey)
    d.line([(tx(k["t"]), ty(k["v"])) for k in track], fill=grey, width=3)
    d.line([(tx(0), ty(cam_v)), (tx(T), ty(cam_v))], fill=cam_c, width=1)
    d.text((x0, ya - 44), f"{tag}   {name}   theoretical angular speed (deg/s)", font=f, fill=cam_c)
    d.text((x0, ya - 24), f"thick grey: 4D slice rotation v(t) = {sp.v_end:g} + ({sp.v0:g} - {sp.v_end:g}) e^(-t/{sp.tau:g})"
           f"   (theta {track[0]['th']:+.0f} -> {track[-1]['th']:+.0f} deg)      thin white: camera orbit {cam_v:.1f} deg/s (constant, never 0)",
           font=fs, fill=grey)
    # 下：实测
    ya, yb = panels[1]
    fl = [x for x in flows[1:]]
    df = [x for x in diffs[1:]]
    fmax = max(fl) * 1.1 if fl and max(fl) > 0 else 1.0
    dmax = max(df) if df and max(df) > 0 else 1.0
    ty = lambda v: yb - (yb - ya) * v / fmax  # noqa: E731
    step = max(0.5, round(fmax / 5 * 2) / 2)
    g = 0.0
    while g <= fmax:
        d.line([(x0 - 6, ty(g)), (x0, ty(g))], fill=grey)
        d.text((x0 - 50, ty(g) - 7), f"{g:.1f}", font=fs, fill=grey)
        g += step
    d.line([(tx(k["t"]), ty(x)) for k, x in zip(track[1:], fl)], fill=red, width=3)
    d.line([(tx(k["t"]), ty(x / dmax * max(fl))) for k, x in zip(track[1:], df)], fill=grey, width=1)
    d.text((x0, ya - 24), f"measured: red = normal-flow speed |I_t| / |grad I| (px/frame, max {max(fl):.2f})"
           f"   grey = mean |frame diff| (scaled, max {dmax * 1000:.1f}e-3)", font=fs, fill=grey)
    if song_t0 is not None:
        d.text((x0, PH - 22), f"song time: {song_t0:.2f} -> {song_t0 + n / sp.fps:.2f} s ({n} frames @ {sp.fps} fps)   vertical lines = downbeats"
               " (senses json, playback time; dark red = phrase start)", font=fs, fill=grey)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    im.save(path)
    print(f"  motion plot -> {path}")


def _insert_contact(path, thumbs, track, diffs, sp, name, tag, cols=6):
    """接触表：每帧一格（标 θ、v、帧间差分），底部一条曲线：v(t) 的理论值（灰）与实测帧间差分（红，归一化）"""
    tw, th_ = thumbs[0].size
    rows = math.ceil(len(thumbs) / cols)
    pad, lab, plot_h = 8, 20, 170
    SW = cols * (tw + pad) + pad
    SH = rows * (th_ + lab + pad) + pad + plot_h + 40
    im = Image.new("RGB", (SW, SH), (0, 0, 0))
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, 13)
    for j, (tb, k) in enumerate(zip(thumbs, track)):
        x, y = pad + (j % cols) * (tw + pad), pad + (j // cols) * (th_ + lab + pad)
        im.paste(tb, (x, y))
        dd = diffs[j]
        d.text((x, y + th_ + 3), f"#{j:02d} th={k['th']:+.0f} v={k['v']:.0f}" + ("" if dd != dd else f" d={dd * 1000:.1f}"),
               font=f, fill=(150, 160, 168))
    y0 = SH - plot_h - 20
    x0, x1 = 60, SW - 30
    d.line([(x0, y0 + plot_h - 20), (x1, y0 + plot_h - 20)], fill=(70, 74, 78))
    n = len(track)
    vmax = max(k["v"] for k in track)
    dv = [x for x in diffs[1:]]
    dmax = max(dv) if dv else 1
    px = lambda j: x0 + (x1 - x0) * j / max(1, n - 1)  # noqa: E731
    py = lambda r: y0 + plot_h - 20 - (plot_h - 40) * r  # noqa: E731
    d.line([(px(j), py(k["v"] / vmax)) for j, k in enumerate(track)], fill=(150, 160, 168), width=2)
    if len(dv) > 1:
        d.line([(px(j + 1), py(x / dmax)) for j, x in enumerate(dv)], fill=(235, 60, 66), width=2)
    d.text((x0, y0 - 4), f"{tag}   {name}   grey: v(t) = {sp.v_end:g} + ({sp.v0:g} - {sp.v_end:g}) e^(-t/{sp.tau:g}) deg/s (max {vmax:.0f})"
           f"   red: measured mean |frame diff| (normalised, max {dmax * 1000:.1f}e-3)", font=f, fill=(150, 160, 168))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    im.save(path, quality=90)
    print(f"  contact sheet -> {path}")


# ---------- 浮现镜头：从手长出来（👤2026-10-22，孤独） ----------
class EmergeSpec:
    """浮现镜头参数。生长：「进度」s = α·r/D_max + (1 − α)·体积比，按 v(t) = v_end + (v0 − v_end) e^{−t/τ} 的形状走（v0 < v_end → 慢慢起来，不从 0 起），
    t = full 时 s = 1（全身长全；之后超平面继续走、什么都不再变）。α = vol_alpha：1 = 前沿测地匀速，< 1 = 体积大处放慢。
    相机：距离 d(t) = d0 (d1/d0)^{t/T}（对数匀速拉远 = 画面里恒定的「退开」感，不从静止开始、不停）；
    对准点 target = 已长出部分的重心（体素上真算）的指数平滑（τ = track_tau 秒），+ end_off·u（u = (d − d0)/(d1 − d0)）：
    镜头跟着生长走 —— 手指 → 手 → 小臂 → 肩 → 身体中心；
    yaw 匀速绕转。主光固定在世界里（key_az / key_el 相对起始相机）"""
    def __init__(self, dur, fps=24, size=(1280, 720), r0=0.006, gv_ratio=0.05, gtau=1.5, full=None, d0=0.9, d1=18.0, yaw0=0.55,
                 yaw_rate=0.05, pitch=0.12, fov=24.0, key_az=-70.0, key_el=42.0, end_off=(0.0, 0.0, 0.0), red_point=True, cold=0.5,
                 cut_glow=0.55, emit=0.5, hud_band=0, hud_scale=1.0, track_tau=0.25, vol_alpha=0.7, red_until=0.08):
        self.dur, self.fps, self.size = dur, fps, size
        self.r0, self.gv_ratio, self.gtau, self.full = r0, gv_ratio, gtau, (full if full is not None else 0.82 * dur)
        self.d0, self.d1, self.yaw0, self.yaw_rate, self.pitch, self.fov = d0, d1, yaw0, yaw_rate, pitch, fov
        self.key_az, self.key_el, self.end_off = key_az, key_el, tuple(end_off)
        self.red_point, self.cold, self.cut_glow, self.emit = red_point, cold, cut_glow, emit
        self.hud_band, self.hud_scale, self.track_tau, self.vol_alpha, self.red_until = hud_band, hud_scale, track_tau, vol_alpha, red_until
        if not 0 < vol_alpha <= 1:
            raise ValueError("vol_alpha ∈ (0, 1]（0 = 只按体积走，指尖那段会一闪而过）")
        if not 0 < gv_ratio < 1:
            raise ValueError("gv_ratio = v0 / v_end ∈ (0, 1)：起步慢、不为 0")
        if not 0 < self.full <= dur:
            raise ValueError("full（长全的时刻）∈ (0, dur]")
        if d1 <= d0 or yaw_rate == 0:
            raise ValueError("要拉远（d1 > d0），相机不能静止（yaw_rate ≠ 0）")


def emerge_track(gr, sp):
    """>v<🧊四维切片-浮现-每帧真参数（纯数学）：t, r, dr/dt, c, dist, yaw, target, u"""
    gr.build()
    n = insert_frames(sp.dur, sp.fps)
    T = (n - 1) / sp.fps
    unit = lambda t: fast_slow(t, sp.gv_ratio, 1.0, sp.gtau)  # noqa: E731   v_end = 1 的形状，再整体缩放
    # 「进度」s(r) = α·r/D_max + (1 − α)·V(r)/V_body（V = 已长出体积，单调）：s 按 unit(t) 走，再反解 r。
    # α = 1：前沿测地匀速（躯干一秒内灌满 → 画面运动在 2.3 s 鼓一个包，v1 实测法向光流峰 5.3 px/帧）；α < 1：体积大的地方前沿自动放慢
    rs = torch.linspace(sp.r0, gr.Dmax, 600).tolist()
    nb = float(gr.inside.sum())
    prog = [sp.vol_alpha * (r - sp.r0) / (gr.Dmax - sp.r0) + (1 - sp.vol_alpha) * float(gr.occupancy(r / gr.K).sum()) / nb for r in rs]
    prog = np.maximum.accumulate(np.array(prog)); prog = (prog - prog[0]) / (prog[-1] - prog[0])
    u_full = unit(sp.full)[0]

    def r_in(t):
        return float(np.interp(min(1.0, unit(t)[0] / u_full), prog, rs))
    v_last = (r_in(sp.full) - r_in(sp.full - 0.05)) / 0.05

    def r_at(t):                                   # 长全之后超平面照走（什么都不再变），按最后的速度外推
        return r_in(t) if t <= sp.full else gr.Dmax + (t - sp.full) * v_last
    ax = torch.linspace(-MESH_B, MESH_B, gr.n, device=DEV)
    Z, Y, X = torch.meshgrid(ax, ax, ax, indexing="ij")
    off = np.array(sp.end_off)
    out, tg = [], np.array(gr.seed_xyz, dtype=np.float64)
    a = 1 - math.exp(-1.0 / (sp.fps * sp.track_tau))
    for i in range(n):
        t = i / sp.fps
        r = r_at(t)
        v = (r_at(t + 0.5 / sp.fps) - r_at(max(0.0, t - 0.5 / sp.fps))) / ((t + 0.5 / sp.fps) - max(0.0, t - 0.5 / sp.fps))
        d = sp.d0 * (sp.d1 / sp.d0) ** (t / T)
        u = (d - sp.d0) / (sp.d1 - sp.d0)
        occ = gr.occupancy(r / gr.K)
        cen = np.array([float(X[occ].mean()), float(Y[occ].mean()), float(Z[occ].mean())]) if occ.any() else np.array(gr.seed_xyz)
        tg = tg + a * (cen + off * u - tg)          # 跟着「已经长出来的部分」的重心走（指数平滑，τ = track_tau）
        out.append(dict(i=i, t=t, r=r, v=v, c=r / gr.K, dist=d, u=u, yaw=sp.yaw0 + sp.yaw_rate * t,
                        target=tuple(float(q) for q in tg)))
    return out


def _emerge_frame(obj, sp, k, prev_b, tag, rp=None):
    gr = obj.grow
    Wd, Hd = sp.size
    R = torch.eye(4, device=DEV)
    key = key_dir(sp.yaw0, sp.pitch, sp.key_az, sp.key_el)
    v, red = render(obj, R, k["c"], Wd, Hd - sp.hud_band, "sculpt", yaw=k["yaw"], pitch=sp.pitch, dist=k["dist"], cut_glow=sp.cut_glow,
                    fov=sp.fov, target=k["target"], key=key, emit=sp.emit)
    if sp.red_point:
        rp = rp or RedPoint(*red.shape)
        if k["r"] > sp.red_until:          # 只留「出生」那一下：之后的相切（前沿在胸口绕回来自己合上、头顶长完）也是真的，但落在胸口像个装饰 → 不画，慢慢熄
            red = torch.zeros_like(red)
        red = rp(red)
    b = betti(obj, R, k["c"], n=64)
    occ = gr.occupancy(k["c"])
    frac = float(occ.sum()) / float(gr.inside.sum())
    rgb = compose(v, red, max(2, round(5 * Hd / 1080)), 0.45, cold=sp.cold)
    if sp.hud_band:
        rgb = torch.cat([rgb, torch.zeros(sp.hud_band, Wd, 3, device=rgb.device)], 0)
    changed = prev_b is not None and b != prev_b
    tg = k["target"]
    lines = [f"{obj.name} ({obj.op}): {obj.eq}",
             f"slice w = c = {k['c']:+.3f}  ->  r = K c = {k['r']:.3f} / D_max {gr.Dmax:.3f}    dr/dt = {k['v']:.3f}/s    volume {frac * 100:5.1f} %"
             f"    t = {k['t']:.3f} / {sp.dur:g} s",
             f"cam: pull back d = {sp.d0:g} (x{sp.d1 / sp.d0:.1f})^(t/T) = {k['dist']:5.2f}   look-at = centroid of grown part (EMA {sp.track_tau:g} s)"
             f" ({tg[0]:+.2f},{tg[1]:+.2f},{tg[2]:+.2f})   orbit {math.degrees(sp.yaw_rate):+.1f} deg/s   fov {sp.fov:g}   key light fixed",
             _b_str(b) + "   (voxels 64^3)" + ("   <- topology changed" if changed else ""),
             f"red point = hyperplane tangent to the 4D surface at birth (r < {sp.red_until:g}): the moment she first touches our space"]
    im = to_pil(rgb)
    hud(im, lines, tag, red_line=3 if changed else None, scale=Hd / 1080 * 1.15 * sp.hud_scale)
    return im, rgb, b, rp, frac


def emerge(obj, sp, out, tag=None, contact=None, plot=None, song_t0=None, still=None):
    """>v<🧊四维切片-浮现-渲染浮现镜头；still = 只出第 N 帧（看构图）"""
    track = emerge_track(obj.grow, sp)
    Wd, Hd = sp.size
    tag = tag or "4D_EMERGE / FROM THE HAND"
    if still is not None:
        im = _emerge_frame(obj, sp, track[still], None, tag)[0]
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        im.save(out, quality=92)
        print(f"{out}: frame #{still} t={track[still]['t']:.3f} r={track[still]['r']:.3f} d={track[still]['dist']:.2f}")
        return track
    sink = Sink(out, sp.fps, sp.size)
    thumbs, diffs, flows, fracs, prev_arr, prev_b, rp = [], [], [], [], None, None, None
    t0 = time.time()
    for k in track:
        im, rgb, b, rp, frac = _emerge_frame(obj, sp, k, prev_b, tag, rp)
        arr = rgb[: Hd - sp.hud_band].mean(-1)
        diffs.append(float((arr - prev_arr).abs().mean()) if prev_arr is not None else float("nan"))
        flows.append(normal_flow(prev_arr, arr) if prev_arr is not None else float("nan"))
        fracs.append(frac)
        prev_arr, prev_b = arr, b
        sink.add(im)
        if contact:
            thumbs.append(im.resize((Wd // 4, Hd // 4), Image.LANCZOS))
        if k["i"] % 12 == 0:
            print(f"  frame {k['i']}/{len(track)}  {time.time() - t0:.1f}s  r={k['r']:.3f} vol={frac * 100:.1f}% b={b}", flush=True)
    sink.close()
    print(f"{out}: {sink.i} frames = {sink.i / sp.fps:.4f} s (slot {sp.dur:g} s), {time.time() - t0:.1f}s")
    if contact:
        labs = [f"#{k['i']:03d} r={k['r']:.2f} vol={f * 100:.0f}% d={k['dist']:.1f}" for k, f in zip(track, fracs)]
        _emerge_contact(contact, thumbs[::2], labs[::2], track, fracs, diffs, tag)
    if plot:
        emerge_plot(plot, track, fracs, diffs, flows, sp, tag, song_t0)
    return track, fracs, diffs


def _emerge_contact(path, thumbs, labs, track, fracs, diffs, tag, cols=6):
    tw, th_ = thumbs[0].size
    rows = math.ceil(len(thumbs) / cols)
    pad, lab, plot_h = 8, 20, 170
    SW = cols * (tw + pad) + pad
    SH = rows * (th_ + lab + pad) + pad + plot_h + 40
    im = Image.new("RGB", (SW, SH), (0, 0, 0))
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, 13)
    for j, (tb, s) in enumerate(zip(thumbs, labs)):
        x, y = pad + (j % cols) * (tw + pad), pad + (j // cols) * (th_ + lab + pad)
        im.paste(tb, (x, y))
        d.text((x, y + th_ + 3), s, font=f, fill=(150, 160, 168))
    y0 = SH - plot_h - 20
    x0, x1 = 60, SW - 30
    n = len(track)
    px = lambda j: x0 + (x1 - x0) * j / max(1, n - 1)  # noqa: E731
    py = lambda r: y0 + plot_h - 20 - (plot_h - 40) * r  # noqa: E731
    d.line([(x0, py(0)), (x1, py(0))], fill=(70, 74, 78))
    d.line([(px(j), py(fr)) for j, fr in enumerate(fracs)], fill=(150, 160, 168), width=2)
    dv = diffs[1:]
    dmax = max(dv) if dv and max(dv) > 0 else 1
    d.line([(px(j + 1), py(x / dmax)) for j, x in enumerate(dv)], fill=(235, 60, 66), width=2)
    d.text((x0, y0 - 4), f"{tag}   grey: grown volume fraction (0 -> 100 %)   red: measured mean |frame diff| (normalised, max {dmax * 1000:.1f}e-3)"
           "   (every 2nd frame shown)", font=f, fill=(150, 160, 168))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    im.save(path, quality=90)
    print(f"  contact sheet -> {path}")


def emerge_plot(path, track, fracs, diffs, flows, sp, tag, song_t0=None):
    """>v<🧊四维切片-浮现-速度图：上 = 生长前沿测地速度 dr/dt（灰粗）+ 已长出体积比（灰细）+ 相机对数拉远速率（白，恒定）；
    下 = 实测法向光流（红）/ 帧间差分（灰）；横轴 = 镜头内秒数，song_t0 给了就标歌里的重拍"""
    PW, PH = 1600, 760
    im = Image.new("RGB", (PW, PH), (0, 0, 0))
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, 15); fs = ImageFont.truetype(FONT, 13)
    x0, x1 = 90, PW - 40
    n = len(track)
    T = (n - 1) / sp.fps
    tx = lambda t: x0 + (x1 - x0) * t / max(T, 1e-9)  # noqa: E731
    grey, dim, red, white = (150, 160, 168), (60, 64, 68), (235, 60, 66), (200, 208, 214)
    panels = [(60, 360), (420, 700)]
    db, ph_ = song_downbeats(song_t0, song_t0 + T) if song_t0 is not None else ([], [])
    for (ya, yb) in panels:
        d.rectangle([x0, ya, x1, yb], outline=dim)
        for tb in db:
            xb = tx(tb - song_t0)
            d.line([(xb, ya), (xb, yb)], fill=(90, 40, 44) if tb in ph_ else (55, 58, 62), width=2 if tb in ph_ else 1)
    for tb in db:
        d.text((tx(tb - song_t0) + 4, panels[1][1] + 26), f"{tb:.2f}" + (" phrase" if tb in ph_ else ""), font=fs, fill=grey)
    for j in range(int(T / 0.5) + 1):
        d.line([(tx(j * 0.5), panels[1][1]), (tx(j * 0.5), panels[1][1] + 6)], fill=grey)
        d.text((tx(j * 0.5) - 12, panels[1][1] + 8), f"{j * 0.5:.1f}s", font=fs, fill=grey)
    ya, yb = panels[0]
    vmax = max(k["v"] for k in track) * 1.15
    ty = lambda v: yb - (yb - ya) * v / vmax  # noqa: E731
    d.line([(tx(k["t"]), ty(k["v"])) for k in track], fill=grey, width=3)
    d.line([(tx(k["t"]), yb - (yb - ya) * fr) for k, fr in zip(track, fracs)], fill=grey, width=1)
    lr = math.log(sp.d1 / sp.d0) / T
    tfull = sp.full
    d.line([(tx(tfull), ya), (tx(tfull), yb)], fill=(80, 84, 88))
    d.text((tx(tfull) + 4, ya + 4), "fully grown", font=fs, fill=grey)
    d.text((x0, ya - 44), f"{tag}   growth front (geodesic units/s, max {vmax / 1.15:.2f}) and camera", font=f, fill=white)
    d.text((x0, ya - 24), f"thick: dr/dt  (progress s = {sp.vol_alpha:g} r/D_max + {1 - sp.vol_alpha:g} volume follows 1 + ({sp.gv_ratio:g} - 1) e^(-t/{sp.gtau:g}): slows where she is thick)   thin: grown volume 0 -> 100 %"
           f"   camera: d/dt ln(dist) = {lr:.3f}/s const, orbit {math.degrees(sp.yaw_rate):.1f} deg/s const", font=fs, fill=grey)
    ya, yb = panels[1]
    fl, df = flows[1:], diffs[1:]
    fmax = max(fl) * 1.1 if fl and max(fl) > 0 else 1.0
    dmax = max(df) if df and max(df) > 0 else 1.0
    ty = lambda v: yb - (yb - ya) * v / fmax  # noqa: E731
    d.line([(tx(k["t"]), ty(x)) for k, x in zip(track[1:], fl)], fill=red, width=3)
    d.line([(tx(k["t"]), ty(x / dmax * max(fl))) for k, x in zip(track[1:], df)], fill=grey, width=1)
    d.text((x0, ya - 24), f"measured: red = normal-flow speed (px/frame, max {max(fl):.2f})   grey = mean |frame diff| (scaled, max {dmax * 1000:.1f}e-3)",
           font=fs, fill=grey)
    if song_t0 is not None:
        d.text((x0, PH - 22), f"song time: {song_t0:.2f} -> {song_t0 + n / sp.fps:.2f} s ({n} frames @ {sp.fps} fps)   vertical lines = downbeats"
               " (senses json, playback time; dark red = phrase start)", font=fs, fill=grey)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    im.save(path)
    print(f"  motion plot -> {path}")


# ---------- 穿膜而出（--rise）：半个 4D 旋转体，超平面扫过 → 她从一张膜里先伸出手、再整个人被拉过来（👤2026-10-22） ----------
def rise_frame_of(msh, d, n=128, margin=0.06):
    """在 SDF 体上找：d 方向的最远点（= 第一个出生的部位，手）、身体重心、膜的位置 o（身体全在 s = (p − o)·d ≥ margin 一侧）。
    返回 (o, d, s_max, hand, center)：s 都相对 o"""
    if msh.vol is None:
        msh.load()
    B = MESH_B
    ax = torch.linspace(-B, B, n, device=DEV)
    X = torch.stack(torch.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)
    P = X[msh(X[:, 0], X[:, 1], X[:, 2]) < 0]
    d = torch.tensor(d, dtype=torch.float32, device=DEV); d = d / d.norm()
    s = P @ d
    hand = P[torch.topk(s, min(30, len(s))).indices].mean(0)
    o = d * (float(s.min()) - margin)
    return o, d, float(s.max() - (o @ d)), hand, P.mean(0)


def mesh_rise(msh, d, o, prefix="mesh"):
    """>v<🧊四维切片-穿膜-4D 物体 = 身体绕「膜」（过 o、垂直 d 的平面，在 4D 里是 d–w 平面的轴）转出来的半个旋转体：
        f4(p, w) = max( f3( o + p⊥ + d·sqrt(s² + w²) ),  −s ),   s = (p − o)·d，p⊥ = p − o − s d
    超平面 w = c 切：身体上离膜 s_b ≥ c 的部分出现，落在 s = sqrt(s_b² − c²)（新出生的部位贴着膜，随 c → 0 被拉到原位）；
    c 从 s_max 往 0 扫 → 最远的部位（手）先出生，c ≤ 膜到身体的最近距离时 = 完整的她。
    −s 那一刀（去掉镜像那半）= 真平切面：膜上露出的就是身体在 s_b = c 处的截面（cut-glow 打亮）。
    (p⊥, s, w) → (p⊥, sqrt(s² + w²)) 是 1-Lipschitz ⇒ 仍是距离下界，可以直接球追踪"""
    d = d / d.norm()

    def inner(p):
        x3 = p[:, :3] - o
        s = x3 @ d
        q = o + x3 - s[:, None] * d + d * L(s, p[:, 3])[:, None]
        return msh(q[:, 0], q[:, 1], q[:, 2]), s

    def f(p):
        m, s = inner(p)
        return torch.maximum(m, -s)

    def cap(p):
        m, s = inner(p)
        return -s - m
    ds = ",".join(f"{v:+.2f}" for v in d.tolist())
    return Obj(f"{prefix}_rise", "half-spin about a membrane",
               f"max( f3( o + p_perp + d sqrt(s^2 + w^2) ), -s ),  s = (p - o).d,  d = ({ds})   ({msh.label})",
               f, 1.4, bound=1.75, cam=4.3, vbox=1.2, cap=cap)


def hermite01(x, a, b):
    """[0,1] → [0,1] 的三次 Hermite：S(0)=0, S(1)=1, S'(0)=a, S'(1)=b（a、b 小 ⇒ 慢起、慢收，中间最快；a, b > 0 ⇒ 不停）"""
    return (a + b - 2) * x ** 3 + (3 - 2 * a - b) * x ** 2 + a * x, 3 * (a + b - 2) * x ** 2 + 2 * (3 - 2 * a - b) * x + a


class RiseSpec:
    """穿膜镜头参数。c：c0 → c1 用 Hermite(a0, a1) 走完 grow 秒（慢起、长），之后以同一末速度继续往 0 漂（不停，但 c 不过 0）。
    相机：距离 d0 → d1 对数匀速拉远（全程），对准点从「手此刻的位置」平滑移到身体重心；yaw 匀速绕，pitch 线性抬高（结尾俯一点 → 孤独）"""
    def __init__(self, dur, fps=24, size=(1280, 720), grow=4.0, a0=0.22, a1=0.05, c_end=0.012, d0=1.05, d1=20.0,
                 yaw0=1.45, yaw_rate=0.07, pitch0=-0.04, pitch1=0.14, fov=24.0, key_az=-80.0, key_el=48.0, cut_glow=0.55,
                 hud_band=96, hud_scale=0.75, cold=0.5, red_gain=1.0):
        self.dur, self.fps, self.size, self.grow, self.a0, self.a1, self.c_end = dur, fps, size, grow, a0, a1, c_end
        self.d0, self.d1, self.yaw0, self.yaw_rate, self.pitch0, self.pitch1, self.fov = d0, d1, yaw0, yaw_rate, pitch0, pitch1, fov
        self.key_az, self.key_el, self.cut_glow, self.hud_band, self.hud_scale, self.cold, self.red_gain = (
            key_az, key_el, cut_glow, hud_band, hud_scale, cold, red_gain)
        if not 0 < grow <= dur:
            raise ValueError("grow 在 (0, dur]")
        if not (0 < a0 < 1 and 0 < a1 < 1):
            raise ValueError("a0、a1 在 (0, 1)：大于 0 才不停，小于 1 才慢起慢收")
        if yaw_rate == 0 or d1 <= d0:
            raise ValueError("相机必须一直在动（yaw_rate ≠ 0，d1 > d0）")


def rise_track(sp, s_max, geo):
    """>v<🧊四维切片-穿膜-每帧真参数（纯数学）：手（离膜 s_h = s_max）在 d–w 平面里的角度 φ：c = s_h cos φ，手在截面里离膜 s_h sin φ。
    φ 用 Hermite(a0, a1) 从 φ0（指尖刚露出 0.03）走到 φ1（c = c_end）共 grow 秒，之后同速继续（封顶 < 90°：c 不过 0）。
    按 φ 而不是按 c 走：出生那一刻 ds/dc 是无穷大（sqrt），按 c 匀走手会弹出来；按 φ 走，手的速度 = s_h φ' 有界、慢起"""
    o, d, hand, center = geo
    hs = s_max
    ph0 = math.asin(min(0.03 / hs, 1.0))
    ph1 = math.acos(min(max(sp.c_end, 1e-4) / hs, 1.0))
    n = insert_frames(sp.dur, sp.fps)
    w_tail = sp.a1 * (ph1 - ph0) / sp.grow
    tail_T = sp.dur - sp.grow
    if ph1 + w_tail * tail_T >= math.pi / 2:
        w_tail = 0.9 * (math.pi / 2 - ph1) / max(tail_T, 1e-9)
    h_perp = hand - o - float((hand - o) @ d) * d
    out = []
    for i in range(n):
        t = i / sp.fps
        if t <= sp.grow:
            S, dS = hermite01(t / sp.grow, sp.a0, sp.a1)
            ph, w = ph0 + (ph1 - ph0) * S, (ph1 - ph0) * dS / sp.grow
        else:
            ph, w = ph1 + w_tail * (t - sp.grow), w_tail
        c, cv = hs * math.cos(ph), -hs * math.sin(ph) * w
        hp = o + h_perp + d * (hs * math.sin(ph))            # 手（第一个出生的部位）此刻在截面里的位置
        u = t / sp.dur
        dist = sp.d0 * (sp.d1 / sp.d0) ** u
        m = u * u * (3 - 2 * u)
        tg = (1 - m) * hp + m * center
        out.append(dict(i=i, t=t, c=c, cv=cv, v=abs(cv), phi=math.degrees(ph), hand_v=hs * w, th=0.0, dist=dist,
                        yaw=sp.yaw0 + sp.yaw_rate * t, pitch=sp.pitch0 + (sp.pitch1 - sp.pitch0) * u, target=tuple(float(x) for x in tg)))
    return out


def _rise_frame(obj, sp, k, prev_b, tag, rp=None):
    Wd, Hd = sp.size
    R = rot4()
    key = key_dir(sp.yaw0, sp.pitch0, sp.key_az, sp.key_el)
    v, red = render(obj, R, k["c"], Wd, Hd - sp.hud_band, "sculpt", yaw=k["yaw"], pitch=k["pitch"], red_gain=sp.red_gain, dist=k["dist"],
                    cut_glow=sp.cut_glow, fov=sp.fov, target=k["target"], key=key)
    rp = rp or RedPoint(*red.shape)
    red = rp(red)
    b = betti(obj, R, k["c"], n=64)
    rgb = compose(v, red, max(2, round(5 * Hd / 1080)), 0.45, cold=sp.cold)
    rgb = torch.cat([rgb, torch.zeros(sp.hud_band, Wd, 3, device=rgb.device)], 0)
    changed = prev_b is not None and b != prev_b
    tg = k["target"]
    lines = [f"{obj.name} ({obj.op}): {obj.eq}",
             f"slice w = c = {k['c']:+.3f} = s_hand cos(phi),  phi = {k['phi']:5.1f} deg (Hermite, slow start {sp.a0:g} / end {sp.a1:g}, {sp.grow:g} s)"
             f"   dc/dt = {k['cv']:+.3f}/s"
             f"     t = {k['t']:.3f} / {sp.dur:g} s",
             f"cam: dist {k['dist']:5.2f} (log-uniform pull-back {sp.d0:g} -> {sp.d1:g})  yaw {math.degrees(k['yaw']):6.1f}"
             f"  pitch {math.degrees(k['pitch']):+.1f}  fov {sp.fov:g}  aim ({tg[0]:+.2f},{tg[1]:+.2f},{tg[2]:+.2f})"
             f"   key light az {sp.key_az:+.0f} el {sp.key_el:+.0f}",
             _b_str(b) + "   (voxels 64^3)" + ("   <- topology changed" if changed else ""),
             "red point = strongest tangency of hyperplane and 4D surface (|n_xyz| / |n| < 0.32);  glow = cut face on the membrane (s = 0)"]
    im = to_pil(rgb)
    hud(im, lines, tag, red_line=3 if changed else None, scale=Hd / 1080 * 1.15 * sp.hud_scale)
    return im, rgb, b, rp


def rise(obj, sp, track, out, tag, contact=None, plot=None, song_t0=None):
    """>v<🧊四维切片-穿膜-渲染：视频 + 接触表（隔帧，标 c 与 Betti）+ 速度图（c 速度、相机、实测光流、b0 / b1 随时间）"""
    Wd, Hd = sp.size
    sink = Sink(out, sp.fps, sp.size)
    thumbs, diffs, flows, bs, prev_arr, prev_b, rp = [], [], [], [], None, None, None
    t0 = time.time()
    for k in track:
        im, rgb, b, rp = _rise_frame(obj, sp, k, prev_b, tag, rp)
        arr = rgb[: Hd - sp.hud_band].mean(-1)
        diffs.append(float((arr - prev_arr).abs().mean()) if prev_arr is not None else float("nan"))
        flows.append(normal_flow(prev_arr, arr) if prev_arr is not None else float("nan"))
        prev_arr, prev_b = arr, b
        bs.append(b)
        sink.add(im)
        if contact and k["i"] % 2 == 0:
            thumbs.append((im.resize((Wd // 4, Hd // 4), Image.LANCZOS), k, b))
        if k["i"] % 12 == 0:
            print(f"  frame {k['i']}/{len(track)}  c={k['c']:+.3f}  b={b}  {time.time() - t0:.1f}s", flush=True)
    sink.close()
    print(f"{out}: {sink.i} frames = {sink.i / sp.fps:.4f} s, {time.time() - t0:.1f}s")
    print("  normal flow (px/frame):", " ".join(f"{d:.2f}" for d in flows[1:]))
    ev = [(k["t"], a, b) for k, a, b in zip(track[1:], bs, bs[1:]) if a != b]
    print(f"  topology changes: {len(ev)}  first: " + "  ".join(f"t={t:.2f} {a}->{b}" for t, a, b in ev[:12]))
    if contact:
        _rise_contact(contact, thumbs, tag)
    if plot:
        rise_plot(plot, track, bs, flows, sp, tag, song_t0)
    return bs, flows


def _rise_contact(path, thumbs, tag, cols=6):
    tw, th_ = thumbs[0][0].size
    rows = math.ceil(len(thumbs) / cols)
    pad, lab = 8, 20
    im = Image.new("RGB", (cols * (tw + pad) + pad, rows * (th_ + lab + pad) + pad + 30), (0, 0, 0))
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, 13)
    prev = None
    for j, (tb, k, b) in enumerate(thumbs):
        x, y = pad + (j % cols) * (tw + pad), pad + (j // cols) * (th_ + lab + pad)
        im.paste(tb, (x, y))
        d.text((x, y + th_ + 3), f"#{k['i']:03d} t={k['t']:.2f} w=c={k['c']:+.3f} b={b[0]} {b[1]} {b[2]}", font=f,
               fill=(235, 60, 66) if prev is not None and b != prev else (150, 160, 168))
        prev = b
    d.text((pad, im.height - 24), f"{tag}   every 2nd frame   label: slice w = c, Betti b0 b1 b2 (red = changed since previous thumbnail)",
           font=f, fill=(150, 160, 168))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    im.save(path, quality=90)
    print(f"  contact sheet -> {path}")


def rise_plot(path, track, bs, flows, sp, tag, song_t0=None):
    """三栏：① |dc/dt|（4D 切片的速度）② 相机距离（对数轴）③ 实测法向光流 + b0 / b1；song_t0 给了就画重拍竖线"""
    PW, PH = 1600, 900
    im = Image.new("RGB", (PW, PH), (0, 0, 0))
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, 15); fs = ImageFont.truetype(FONT, 13)
    x0, x1 = 90, PW - 40
    T = track[-1]["t"]
    tx = lambda t: x0 + (x1 - x0) * t / max(T, 1e-9)  # noqa: E731
    grey, dim, red, white = (150, 160, 168), (60, 64, 68), (235, 60, 66), (200, 208, 214)
    panels = [(60, 300), (360, 560), (620, 840)]
    db, ph_ = song_downbeats(song_t0, song_t0 + T) if song_t0 is not None else ([], [])
    for ya, yb in panels:
        d.rectangle([x0, ya, x1, yb], outline=dim)
        for tb in db:
            d.line([(tx(tb - song_t0), ya), (tx(tb - song_t0), yb)], fill=(90, 40, 44) if tb in ph_ else (55, 58, 62))
    for j in range(int(T / 0.5) + 1):
        d.text((tx(j * 0.5) - 12, panels[2][1] + 8), f"{j * 0.5:.1f}s", font=fs, fill=grey)

    def curve(panel, vals, col, vmax, width=2, ts=None):
        ya, yb = panel
        ts = ts or [k["t"] for k in track]
        d.line([(tx(t), yb - (yb - ya) * v / vmax) for t, v in zip(ts, vals)], fill=col, width=width)

    vs = [k["v"] for k in track]
    vm = max(vs) * 1.1
    curve(panels[0], vs, white, vm, 3)
    d.line([(tx(sp.grow), panels[0][0]), (tx(sp.grow), panels[0][1])], fill=grey)
    d.text((x0, 16), f"{tag}   4D slice speed |dc/dt| (w-units/s, max {max(vs):.3f}):  c {track[0]['c']:+.3f} -> {track[-1]['c']:+.3f};"
           f"  slow start a0 = {sp.a0:g}, growth ends at {sp.grow:g} s (grey line), then drift {vs[-1]:.4f}/s (never 0)", font=fs, fill=white)
    ld = [math.log(k["dist"]) for k in track]
    lo, hi = min(ld), max(ld)
    curve(panels[1], [v - lo for v in ld], white, (hi - lo) * 1.1 or 1, 2)
    d.text((x0, panels[1][0] - 22), f"camera distance, log scale: {sp.d0:g} -> {sp.d1:g} (log-uniform = constant zoom rate);"
           f"  orbit {math.degrees(sp.yaw_rate):.1f} deg/s;  pitch {math.degrees(sp.pitch0):+.0f} -> {math.degrees(sp.pitch1):+.0f} deg", font=fs, fill=grey)
    fl = flows[1:]
    fm = max(fl) * 1.1 if fl and max(fl) > 0 else 1
    curve(panels[2], fl, red, fm, 3, [k["t"] for k in track[1:]])
    bm = max(max(b[0] for b in bs), max(b[1] for b in bs), 1) * 1.1
    curve(panels[2], [b[0] for b in bs], white, bm, 1)
    curve(panels[2], [b[1] for b in bs], grey, bm, 1)
    d.text((x0, panels[2][0] - 22), f"measured: red = normal-flow speed (px/frame, max {max(fl):.2f});  white = b0 (pieces, max {max(b[0] for b in bs)}),"
           f"  grey = b1 (tunnels, max {max(b[1] for b in bs)}) on 64^3 voxels", font=fs, fill=grey)
    if song_t0 is not None:
        d.text((x0, PH - 24), f"song time {song_t0:.2f} -> {song_t0 + len(track) / sp.fps:.2f} s; vertical lines = downbeats (dark red = phrase)",
               font=fs, fill=grey)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    im.save(path)
    print(f"  motion plot -> {path}")



# ---------- 接触表 ----------
SHEET_ROWS = [
    ("tiger", "rotate", None), ("ditorus", "rotate", None), ("loft_sphere_torus", "slice", None),
    ("extrude_torus", "rotate", None), ("tesseract_diag", "slice", None), ("link", "rotate", None),
]


MESH_SHEET_ROWS = "extrude,spin,loft_sphere,loft_torus,twist,twist:rotate"


_ROW_OP = {"extrude": "rotate", "spin": "rotate", "loft_sphere": "slice", "loft_torus": "slice", "twist": "slice"}


def _row_spec(ops, prefix, label):
    """'spin,twist:rotate' → [(prefix_spin, rotate, label/spin), ...]"""
    out = []
    for tok in ops.split(","):
        name, _, op = tok.strip().partition(":")
        out.append((f"{prefix}_{name}", op or _ROW_OP[name], f"{label} / {name}"))
    return out


def sheet(out, rows=None, look="line", tag="20_4D_SLICE_OVERVIEW", objs=None, foot=None):
    """>v<🧊四维切片-总览-行 = 物体，列 = 6 个切片位置 / 6 个 xw 角度；每格标 c、θ 和真数的 Betti 数，拓扑与左边一格不同 → 标签变红"""
    rows = rows or SHEET_ROWS
    objs = objs or OBJS
    lab_w, top, bot = 330, 70, 150 + 26 * max(0, len(foot or []) - 3)
    SW, SH = 2880, max(1920, top + bot + 285 * len(rows))
    tw = (SW - lab_w - 40) // 6
    th_ = (SH - top - bot) // len(rows)
    canvas = torch.zeros(SH, SW, 3, device=DEV)
    labels = []
    t0 = time.time()
    for r, (name, op, _) in enumerate(rows):
        obj = objs[name]
        prev = None
        for k in range(6):
            s = k / 5
            if op == "slice":
                th, c = 0.0, -obj.cmax * 0.97 + 2 * obj.cmax * 0.97 * s
            else:
                th = math.radians(90) * s
                c = obj.c_rot if obj.c_rot is not None else {"link": 0.05, "ditorus": 0.05}.get(name, 0.18)
            if name == "tesseract":
                R = rot4(xw=th, yw=th * 0.8, zw=th * 0.6)
            else:
                R = rot4(**{obj.plane: th})
            v, red = render(obj, R, c, tw, th_, look, yaw=0.6, ss=3, dist=obj.cam * 0.9, red_gain=1.0)
            b = betti(obj, R, c, n=64)
            canvas[top + r * th_: top + (r + 1) * th_, lab_w + k * tw: lab_w + (k + 1) * tw] = compose(v, red, 3, 0.4)
            bs = f"b={b[0]}{b[1]}{b[2]}" if max(b) < 10 else f"b={b[0]} {b[1]} {b[2]}"
            labels.append((r, k, f"theta={math.degrees(th):.0f}  c={c:+.2f}", bs, prev is not None and b != prev))
            prev = b
        print(f"  row {name}: {time.time() - t0:.1f}s", flush=True)
    im = to_pil(canvas)
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, 20); fs = ImageFont.truetype(FONT, 17)
    for r, (name, op, lab) in enumerate(rows):
        obj = objs[name]
        y = top + r * th_ + th_ // 2 - 40
        d.text((40, y), lab or name, font=f, fill=(200, 208, 214))
        d.text((40, y + 30), f"op: {obj.op}", font=fs, fill=(130, 138, 145))
        d.text((40, y + 54), "sweep c along w" if op == "slice" else f"rot {obj.plane}, c fixed", font=fs, fill=(130, 138, 145))
    for r, k, a, b, red in labels:
        x, y = lab_w + k * tw + 14, top + (r + 1) * th_ - 30
        d.text((x, y), a, font=fs, fill=(120, 128, 134))
        d.text((x + tw - 54 - d.textlength(b, font=fs), y), b, font=fs, fill=(235, 60, 66) if red else (120, 128, 134))
    hud(im, foot or ["g(x,y,z) = f( [x, y, z, c] R ),  R = R_xw(theta) ...   sphere-traced 3D cross-section of a 4D signed distance field",
             "lines: extra slicing planes y = k * 0.045 (polytopes: edges of the slice polyhedron)   b = Betti b0 b1 b2 on 64^3 voxels",
             "red = hyperplane tangent to the 4D surface (where topology can change);  red label = topology differs from left"], tag)
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out, quality=92)
    print(f"{out}: {time.time() - t0:.1f}s")


def _lateral_arg(s):
    if s in ("auto", "x", "z", "sym"):
        return s
    try:
        return f"{float(s):g}"
    except ValueError:
        raise argparse.ArgumentTypeError(f"--mesh-lateral 只能是 auto / x / z / sym / 角度数字：{s}")


def main():
    ap = argparse.ArgumentParser(description="4D 物体 → 3D 截面")
    ap.add_argument("--obj", default="tiger", choices=list(OBJS))
    ap.add_argument("--op", default="slice", choices=["slice", "rotate", "project"])
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--out", default=None)
    ap.add_argument("--look", default="line", choices=["line", "surface", "sculpt"], help="sculpt = 一盏硬主光斜扫 + 软阴影 + AO，深黑（--insert 用）")
    ap.add_argument("--theta", type=float, default=180.0, help="rotate/project：转多少度")
    ap.add_argument("--c0", type=float, default=None); ap.add_argument("--c1", type=float, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--sheet", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--mesh", default=None, help="GLB 路径：mesh_* 物体用这个网格（默认 3D/测试_人台.glb）；SDF 缓存在旁边 <名>.sdf<N>.npy")
    ap.add_argument("--sdf-res", type=int, default=128, help="网格 SDF 体素分辨率 N（N³）")
    ap.add_argument("--mesh-lateral", default="auto", type=_lateral_arg,
                    help="glTF 里的侧向轴：auto / x / z（blender_pose 的输出给 z）/ sym（镜面对称搜索，斜摆的 Tripo 蜷缩用这个）/ 数字 = xz 里的侧向角 φ 度"
                         "（φ 与 φ+180 都是同一镜面，差在正面朝 ±z）；非 auto 时缓存名带 _lat<值>")
    ap.add_argument("--sdf-faces", type=int, default=40000, help="建 SDF 前减面到 ≤ 这么多面（顶点聚类）；非默认值时缓存名带 _f<k>k")
    ap.add_argument("--extra-mesh", nargs="*", default=[], help="--mesh-sheet 时追加的行：'路径.glb=spin,twist:rotate,...'（算子名 extrude/spin/"
                    "loft_sphere/loft_torus/twist，:rotate / :slice 覆盖默认扫法）")
    ap.add_argument("--rows", default=None, help="--mesh-sheet 主网格的行，格式同上（默认 MESH_SHEET_ROWS）")
    ap.add_argument("--mesh-sheet", action="store_true", help="网格算子接触表（默认写 静帧/21_四维切片_人形.jpg）")
    g = ap.add_argument_group("插入镜头（--insert 槽长秒数）：θ 与 c 先快后缓 v(t) = v_end + (v0 − v_end) e^{−t/τ}")
    g.add_argument("--insert", type=float, default=None, help="槽长（秒）；帧数 = ceil(槽长 × fps)")
    g.add_argument("--fps", type=int, default=24)
    g.add_argument("--size", default="1280x720")
    g.add_argument("--plane", default=None, choices=["xw", "yw", "zw", "xy"], help="θ 转哪个平面（默认物体自己的：spin(mesh) = zw）")
    g.add_argument("--plane2", default=None, choices=["xw", "yw", "zw", "xy"], help="同时转的第二个平面（双旋转）")
    g.add_argument("--ratio2", type=float, default=0.0, help="第二个平面的角度 = ratio2 × θ")
    g.add_argument("--theta0", type=float, default=0.0, help="θ 起点（度）")
    g.add_argument("--v0", type=float, default=240.0, help="θ 初速（度/秒）")
    g.add_argument("--v-end", type=float, default=25.0, help="θ 渐近速度（度/秒，> 0：不停）")
    g.add_argument("--tau", type=float, default=0.35, help="缓下来的时间常数（秒）")
    g.add_argument("--c-start", type=float, default=None, help="超平面位置 c 起点（默认物体的 c_rot 或 0）")
    g.add_argument("--cv0", type=float, default=0.0, help="c 初速（/秒）")
    g.add_argument("--cv-end", type=float, default=0.0, help="c 渐近速度（/秒）")
    g.add_argument("--tilt", default="", help="物体在 4D 里的固定倾斜，度：'zw=20,yw=-15,xy=12'")
    g.add_argument("--yaw0", type=float, default=0.6); g.add_argument("--yaw-rate", type=float, default=0.10, help="相机匀速漂移（弧度/秒，≠ 0）")
    g.add_argument("--pitch", type=float, default=0.42); g.add_argument("--dolly", type=float, default=0.04, help="匀速推近（距离比例/秒）")
    g.add_argument("--red-gain", type=float, default=1.0)
    g.add_argument("--cam-scale", type=float, default=1.0, help="相机距离 = 物体默认距离 × 这个数（> 1 拉远，碎块不出画、不压 HUD）")
    g.add_argument("--contact", default=None, help="接触表 jpg 路径（全部帧 + 速度曲线 vs 实测帧间差分）")
    g.add_argument("--plan", action="store_true", help="只打印每帧 θ / v / c，不渲染")
    g.add_argument("--hud-band", type=int, default=0, help="底部 HUD 条带高度（像素）：物体只画在上面，HUD 不压物体（0 = 旧行为：HUD 叠在画面上）")
    g.add_argument("--hud-scale", type=float, default=1.0, help="HUD 字号倍数")
    g.add_argument("--cut-glow", type=float, default=0.55, help="surface 模式下 |w| − h 平切面的自发光强度（extrude / twist；0 = 旧的死黑）")
    g.add_argument("--still", type=int, default=None, help="只渲染第 N 帧存成图（--out 给 .jpg/.png），比较 --look 用")
    g.add_argument("--fov", type=float, default=34.0, help="竖直视角（度）；小 = 长焦，透视压平、更像雕塑特写")
    g.add_argument("--target", default="0,0,0", help="相机绕转 / 对准的点 'x,y,z'（模型坐标，x = 侧向、y = 上、z = 正面）")
    g.add_argument("--key", default="-95,38", help="sculpt 主光 'az,el'（度；负数要写成 --key=-95,38）：az 相对起始相机，0 = 从相机打，±90 = 侧光，±180 = 逆光")
    g.add_argument("--red-point", action="store_true", help="全片只留一个红点（最强的相切处，跨帧跟踪）")
    g.add_argument("--cold", type=float, default=0.0, help="暗部冷偏强度（0 = 关；0.5 ≈ 很淡）")
    g.add_argument("--plot", default=None, help="运动速度图 png 路径（理论角速度 + 实测法向光流 / 帧间差分）")
    g.add_argument("--song-t0", type=float, default=None, help="这个镜头在歌里的起点（秒，播放时间）：速度图上标重拍")
    e = ap.add_argument_group("浮现镜头（--emerge 槽长秒数）：4D 测地锥，超平面扫过 → 她从手指尖沿身体长进三维；相机从手对数匀速拉远")
    e.add_argument("--emerge", type=float, default=None, help="槽长（秒）；--mesh 给站立模型（默认 3D/她.glb）")
    e.add_argument("--seed-side", type=float, default=1.0, help="从哪只手长：+1 / −1（侧向 x 的正负）")
    e.add_argument("--grow-full", type=float, default=None, help="第几秒长全（默认 0.82 × 槽长）")
    e.add_argument("--grow-ratio", type=float, default=0.05, help="前沿起步速度 / 最终速度（< 1：慢慢起来）")
    e.add_argument("--grow-tau", type=float, default=1.5, help="前沿加速的时间常数（秒）")
    e.add_argument("--d0", type=float, default=0.9, help="起始相机距离（贴着手）"); e.add_argument("--d1", type=float, default=18.0, help="结束相机距离")
    e.add_argument("--end-off", default="0,0,0", help="结束时对准点 = 身体中心 + 这个偏移（x,y,z）：把人放到画面一侧，留更多空")
    e.add_argument("--emit", type=float, default=0.5, help="生长前沿在皮肤上的亮带强度")
    e.add_argument("--vol-alpha", type=float, default=0.7, help="生长进度里前沿距离的权重（1 = 测地匀速；< 1 = 躯干这种体积大的地方放慢）")
    r = ap.add_argument_group("穿膜镜头（--rise 秒数）：半个 4D 旋转体，超平面 w = c 从 s_max 扫到 0 → 最远的部位（手）先从膜里出生，再整个人被拉出来")
    r.add_argument("--rise", type=float, default=None, help="镜头长（秒）；--mesh 给蜷缩模型 + --mesh-lateral sym")
    r.add_argument("--rise-dir", default="0.25,-0.3,0.92", help="d：膜的法向，指向第一个出生的部位（默认 = 蜷缩模型的手）")
    r.add_argument("--rise-grow", type=float, default=4.0, help="c 走完（完整的她出现）用几秒；之后继续极慢地漂，不停")
    r.add_argument("--rise-a", default="0.22,0.05", help="Hermite 起 / 收斜率 a0,a1（小 = 慢起 / 慢收）")
    r.add_argument("--rise-cam", default="1.05,20", help="相机距离 d0,d1（对数匀速拉远）")
    r.add_argument("--rise-pitch", default="-0.04,0.14", help="pitch 起,止（弧度）")
    r.add_argument("--rise-yaw", default="1.45,0.07", help="yaw0（弧度）, 匀速绕（弧度/秒）")
    a = ap.parse_args()
    if a.mesh or a.sdf_res != 128 or a.sdf_faces != 40000 or a.mesh_lateral != "auto":
        MESH.path, MESH.n, MESH.max_faces, MESH.lateral, MESH.vol = Path(a.mesh or MESH.path), a.sdf_res, a.sdf_faces, a.mesh_lateral, None
        OBJS.update(mesh_objs(MESH))
    if a.mesh_sheet:
        objs = dict(OBJS)
        rows = _row_spec(a.rows or MESH_SHEET_ROWS, "mesh", MESH.label)
        for i, spec in enumerate(a.extra_mesh):
            path, _, ops = spec.partition("=")
            if not Path(path).exists():
                print(f"  skip (missing): {path}")
                continue
            msh = MeshSDF(path, a.sdf_res)
            objs.update(mesh_objs(msh, prefix=f"x{i}"))
            rows += _row_spec(ops or "spin", f"x{i}", msh.label)
        foot = [f"f3 = SDF of the row's GLB on {MESH.n}^3 (|d| = point-triangle distance, sign = winding number > 0.5; > 40k faces: vertex-cluster decimated)",
                "g(x,y,z) = f( [x, y, z, c] R ),  R = R_plane(theta)    lines: extra slicing planes y = k * 0.045",
                "b = Betti b0 b1 b2 on 64^3 voxels (mannequin limbs are separate pieces, torso_core has loose wires: those count)",
                "red = hyperplane tangent to the 4D surface (where topology can change);  red label = topology differs from left"]
        sheet(a.out or str(MV / "静帧" / "21_四维切片_人形.jpg"), rows, a.look, "21_4D_SLICE_MESH", objs, foot)
        return
    if a.list:
        for o in OBJS.values():
            print(f"{o.name:20s} {o.op:14s} {o.eq}")
        return
    if a.sheet:
        sheet(a.out or str(MV / "静帧" / "20_四维切片_总览.jpg"), look=a.look)
        return
    if a.rise is not None:
        msh = MeshSDF(a.mesh or MV / "3D" / "蜷缩_无材质_低模.glb", a.sdf_res, a.sdf_faces, a.mesh_lateral)
        o, d, s_max, hand, center = rise_frame_of(msh, [float(x) for x in a.rise_dir.split(",")])
        obj = mesh_rise(msh, d, o)
        w, h = (int(x) for x in a.size.lower().split("x"))
        a0, a1 = (float(x) for x in a.rise_a.split(","))
        d0, d1 = (float(x) for x in a.rise_cam.split(","))
        p0, p1 = (float(x) for x in a.rise_pitch.split(","))
        y0, yr = (float(x) for x in a.rise_yaw.split(","))
        kz, ke = (float(x) for x in a.key.split(","))
        sp = RiseSpec(a.rise, a.fps, (w, h), a.rise_grow, a0, a1, d0=d0, d1=d1, yaw0=y0, yaw_rate=yr, pitch0=p0, pitch1=p1, fov=a.fov,
                      key_az=kz, key_el=ke, cut_glow=a.cut_glow, hud_band=a.hud_band or 96, hud_scale=a.hud_scale, cold=a.cold)
        track = rise_track(sp, s_max, (o, d, hand, center))
        print(f"  membrane: o = {o.tolist()}, d = {d.tolist()}, s_max = {s_max:.3f}, first-born (hand) = {hand.tolist()}, centre = {center.tolist()}")
        if a.plan:
            for k in track:
                print(f"  #{k['i']:03d} t={k['t']:.3f} phi={k['phi']:.1f} hand_v={k['hand_v']:.3f} c={k['c']:+.4f} dc/dt={k['cv']:+.4f} dist={k['dist']:.2f} target={tuple(round(x, 3) for x in k['target'])}")
            return
        out = Path(a.out or "rise.mp4")
        if out.exists() or any(x and Path(x).exists() for x in (a.contact, a.plot)):
            sys.exit(f"已存在，不覆盖：{out} / {a.contact} / {a.plot}（换版本号）")
        tag = a.tag or "4D_RISE / MEMBRANE"
        if a.still is not None:
            im = _rise_frame(obj, sp, track[a.still], None, tag)[0]
            out.parent.mkdir(parents=True, exist_ok=True)
            im.save(out, quality=92)
            print(f"{out}: frame #{a.still}")
            return
        rise(obj, sp, track, str(out), tag, a.contact, a.plot, a.song_t0)
        return
    if a.emerge is not None:
        msh = MeshSDF(a.mesh or MV / "3D" / "她.glb", a.sdf_res, a.sdf_faces, a.mesh_lateral)
        obj = grow_obj(msh, a.seed_side)
        w, h = (int(x) for x in a.size.lower().split("x"))
        sp = EmergeSpec(a.emerge, a.fps, (w, h), gv_ratio=a.grow_ratio, gtau=a.grow_tau, full=a.grow_full, d0=a.d0, d1=a.d1, yaw0=a.yaw0,
                        yaw_rate=a.yaw_rate, pitch=a.pitch, fov=a.fov, key_az=float(a.key.split(",")[0]), key_el=float(a.key.split(",")[1]),
                        end_off=tuple(float(x) for x in a.end_off.split(",")), red_point=True, cold=a.cold, cut_glow=a.cut_glow, emit=a.emit,
                        hud_band=a.hud_band, hud_scale=a.hud_scale, vol_alpha=a.vol_alpha)
        if a.plan:
            for k in emerge_track(obj.grow, sp):
                print(f"  #{k['i']:03d} t={k['t']:.3f} r={k['r']:.3f} dr/dt={k['v']:.3f} d={k['dist']:.2f} u={k['u']:.2f} target={k['target']}")
            return
        out = Path(a.out or "emerge.mp4")
        if out.exists() or any(x and Path(x).exists() for x in (a.contact, a.plot)):
            sys.exit(f"已存在，不覆盖：{out} / {a.contact} / {a.plot}（换版本号）")
        emerge(obj, sp, str(out), a.tag, a.contact, a.plot, a.song_t0, a.still)
        return
    if a.insert is not None:
        w, h = (int(x) for x in a.size.lower().split("x"))
        sp = InsertSpec(a.insert, a.fps, (w, h), a.plane, a.plane2, a.ratio2, a.theta0, a.v0, a.v_end, a.tau, a.c_start, a.cv0, a.cv_end,
                        a.tilt, a.yaw0, a.yaw_rate, a.pitch, a.dolly, a.red_gain, a.look, a.cam_scale, a.hud_band, a.hud_scale, a.cut_glow,
                        a.fov, tuple(float(x) for x in a.target.split(",")), *(float(x) for x in a.key.split(",")), a.red_point, a.cold)
        if a.plan:
            obj = OBJS[a.obj]
            for k in insert_track(obj, sp):
                print(f"  #{k['i']:02d} t={k['t']:.3f} theta={k['th']:+7.1f} v={k['v']:6.1f} c={k['c']:+.3f} dc/dt={k['cv']:+.3f}"
                      + ("   (|c| > cmax: slice may be empty)" if abs(k["c"]) > obj.cmax else ""))
            return
        out = Path(a.out or f"{a.obj}_insert.mp4")
        if out.exists() or any(x and Path(x).exists() for x in (a.contact, a.plot)):
            sys.exit(f"已存在，不覆盖：{out} / {a.contact} / {a.plot}（换版本号）")
        if a.still is not None:
            insert_still(a.obj, sp, a.still, str(out), a.tag)
            return
        insert(a.obj, sp, str(out), a.tag, a.contact, a.plot, a.song_t0)
        return
    animate(a.obj, a.op, a.frames, a.out or f"{a.obj}_{a.op}.mp4", a.look, a.theta, a.c0, a.c1, a.tag)


if __name__ == "__main__":
    main()
