#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🎬样片-渲染运动-她抬头：没绑骨的 Tripo 网格，用「脖子枢轴 + 高度平滑权重」把头弯下 / 抬起，自己渲染动作，
再交给 Seedream 当参考（👤：动作自己渲，哪怕不只是灰模）。

渲染 = GPU 点溅射（表面按面积采样几百万点 + z-buffer），不装新库。两种风格：
  solid：灰模 + 顶光（给 Seedream 看结构）
  holo ：全息 —— 贴在身体上的等高线（随头一起动）+ 屏幕扫描线 + 肩部点云消散 + 顶光 + 轮廓光，冷灰白低饱和
运动（方法_剪辑与情绪 §四⅞¾）：头角和推镜都「首帧已经在动、尾帧还在动」，只变速，不晃。
用法：.venv/Scripts/python tools/headlift_render.py --style holo --out 路径.mp4 [--frames 120] [--w 1280 --h 720]
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

MV = Path("C:/Users/Administrator/Desktop/Projekt_Visual/MV_Cant_Kill_Us")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
FPS = 24

# 她.glb 的坐标（实测横截面）：y 朝上，脸朝 +x，侧向 = z；脖子最细在 y≈0.33–0.34，下巴前缘 y≈0.35 / x≈0.054，头顶 y≈0.49
PIVOT = (-0.015, 0.33)          # 枢轴（x, y）：寰枕关节附近，偏后
TILT = 0.4                      # 权重用 s = y + TILT·(x − x_pivot)：前面（下巴）算头，后颈算颈 → 倾斜的分界面


def smooth(e0, e1, x):
    u = np.clip((x - e0) / (e1 - e0), 0, 1) if isinstance(x, np.ndarray) else torch.clamp((x - e0) / (e1 - e0), 0, 1)
    return u * u * (3 - 2 * u)


def bend_weight(P):
    """头 = 1、颈部分担 30%、躯干 = 0；平滑过渡，避免折痕"""
    s = P[:, 1] + TILT * (P[:, 0] - PIVOT[0])
    return 0.3 * smooth(0.255, 0.335, s) + 0.7 * smooth(0.325, 0.37, s)


def head_angle(u, a0=-25.0, a1=20.0):
    """度。85% smootherstep + 15% 线性：两端速度不为零（剪进来时已经在动）"""
    se = u * u * u * (u * (6 * u - 15) + 10)
    return a0 + (a1 - a0) * (0.85 * se + 0.15 * u)


def cam_dist(u, d0=0.43, d1=0.335):
    """慢推：基础速度 + 轻微加速（只变速，不停）"""
    return d0 + (d1 - d0) * (0.7 * u + 0.3 * u * u)


def load_points(n_pts, seed=3):
    import trimesh
    m = trimesh.load(str(MV / "3D" / "她.glb"), force="mesh")
    V = np.asarray(m.vertices); Fc = np.asarray(m.faces)
    keep = V[Fc].max(1)[:, 1] > 0.18                    # 只要胸口以上（特写）
    Fc = Fc[keep]
    sub = trimesh.Trimesh(V, Fc, process=False)
    pts, fi = trimesh.sample.sample_surface(sub, n_pts, seed=seed)
    nrm = sub.face_normals[fi]
    return pts.astype(np.float32), nrm.astype(np.float32)


def rot_z(P, ang, w):
    """绕 z（侧向）轴、过 PIVOT 旋转 ang·w 弧度；ang<0 = 低头（脸 +x 往下）"""
    a = ang * w
    c, s = torch.cos(a), torch.sin(a)
    x = P[:, 0] - PIVOT[0]; y = P[:, 1] - PIVOT[1]
    return torch.stack([x * c - y * s + PIVOT[0], x * s + y * c + PIVOT[1], P[:, 2]], 1)


def render(P, N, w, ang, dist, W, H, style, rest_y, keep_mask, fov=30.0, rng_noise=None):
    P2 = rot_z(P, ang, w); N2 = rot_z(N + torch.tensor([PIVOT[0], PIVOT[1], 0.0], device=DEV), ang, w) - rot_z(
        torch.tensor([[PIVOT[0], PIVOT[1], 0.0]], device=DEV).expand_as(N), ang, w)
    N2 = F.normalize(N2, dim=1)
    # 镜头跟一点头：瞄准点 = 静止脸心与转动后脸心之间 60%（只是跟，不晃）
    fc = torch.tensor([[0.04, 0.415, 0.0]], device=DEV)
    target = (0.4 * fc + 0.6 * rot_z(fc, ang, torch.ones(1, device=DEV)))[0]
    target = target * torch.tensor([0.0, 1.0, 0.0], device=DEV)
    cam = target + torch.tensor([dist, -0.012, 0.0], device=DEV)  # 略低于眼睛，微仰
    fwd = F.normalize(target - cam, dim=0)
    right = F.normalize(torch.linalg.cross(fwd, torch.tensor([0.0, 1.0, 0.0], device=DEV)), dim=0)
    up = torch.linalg.cross(right, fwd)
    d = P2 - cam
    zc = d @ fwd; xc = d @ right; yc = d @ up
    f = (H / 2) / np.tan(np.radians(fov / 2))
    px = (xc / zc * f + W / 2).long(); py = (-yc / zc * f + H / 2).long()
    ok = (zc > 1e-3) & (px >= 0) & (px < W) & (py >= 0) & (py < H) & keep_mask
    idx = py[ok] * W + px[ok]; z = zc[ok]
    zbuf = torch.full((W * H,), 1e9, device=DEV).scatter_reduce(0, idx, z, "amin")
    vis = z <= zbuf[idx] + 0.0025
    idx = idx[vis]
    n = N2[ok][vis]; view = -F.normalize(d[ok][vis], dim=1)
    L = F.normalize(torch.tensor([0.35, 1.0, 0.0], device=DEV), dim=0)   # 顶光，略偏前
    lam = (n @ L).clamp(min=0)
    rim = (1 - (n * view).sum(1).abs()).clamp(0, 1) ** 3
    ry = rest_y[ok][vis]
    fade = smooth(0.235, 0.33, ry)                       # 肩以下隐入黑暗
    if style == "solid":
        v = (0.07 + 0.9 * lam ** 1.2 + 0.12 * rim) * (0.25 + 0.75 * fade)
    else:
        contour = torch.exp(-((torch.frac(ry * 230.0) - 0.5) / 0.09) ** 2)   # 贴在身体上的等高线
        v = (0.05 + 0.75 * lam ** 1.4 + 0.45 * rim) * (0.55 + 0.75 * contour) * (0.15 + 0.85 * fade)
    img = torch.zeros(W * H, device=DEV).scatter_reduce(0, idx, v, "amax")
    cov = torch.zeros(W * H, device=DEV).index_fill(0, idx, 1.0)
    img = img.view(1, 1, H, W); cov = cov.view(1, 1, H, W)
    # 补洞：空像素取 3×3 邻域均值（只在被覆盖的区域内）
    k = torch.ones(1, 1, 3, 3, device=DEV)
    s = F.conv2d(img, k, padding=1); c = F.conv2d(cov, k, padding=1)
    for _ in range(2):
        s = F.conv2d(img, k, padding=1); c = F.conv2d(cov, k, padding=1)
        hole = (cov == 0) & (c >= 3)
        img = torch.where(hole, s / c.clamp(min=1), img); cov = torch.where(hole, torch.ones_like(cov), cov)
    img = img[0, 0]
    if style == "holo":
        rows = torch.arange(H, device=DEV, dtype=torch.float32)
        img = img * (0.72 + 0.28 * torch.cos(rows * np.pi / 1.5)[:, None] ** 2)   # 细扫描线（3 px 周期）
        blur = F.avg_pool2d(img[None, None], 9, 1, 4)
        blur = F.avg_pool2d(blur, 9, 1, 4)[0, 0]
        img = img + 0.35 * blur
        # 顶光的光锥：画面上方中间一点冷光
        xs = (torch.arange(W, device=DEV) - W / 2) / W
        cone = torch.exp(-(xs[None, :] / (0.05 + 0.25 * rows[:, None] / H)) ** 2) * torch.exp(-rows[:, None] / (0.35 * H))
        img = img + 0.10 * cone
    img = img.clamp(0, 1)
    tint = torch.tensor([0.86, 0.92, 1.0], device=DEV) if style == "holo" else torch.tensor([0.95, 0.96, 0.97], device=DEV)
    rgb = (img[..., None] * tint).clamp(0, 1)
    return (rgb * 255).byte().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default="holo", choices=["holo", "solid"])
    ap.add_argument("--frames", type=int, default=120)    # 5 s @ 24
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--pts", type=int, default=10_000_000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames-dir", default=None, help="另存每帧 png")
    a = ap.parse_args()
    out = Path(a.out)
    assert not out.exists(), f"不覆盖已有文件：{out}"
    out.parent.mkdir(parents=True, exist_ok=True)
    pts, nrm = load_points(a.pts)
    P = torch.tensor(pts, device=DEV); N = torch.tensor(nrm, device=DEV)
    w = bend_weight(P)
    rest_y = P[:, 1].clone()
    keep = torch.ones(len(P), dtype=torch.bool, device=DEV)
    if a.style == "holo":   # 肩部点云消散：越往下越稀
        g = torch.Generator(device=DEV).manual_seed(5)
        keep = torch.rand(len(P), device=DEV, generator=g) < (0.06 + 0.94 * smooth(0.22, 0.31, rest_y))
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{a.w}x{a.h}", "-r", str(FPS),
                             "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", str(out)], stdin=subprocess.PIPE)
    fd = Path(a.frames_dir) if a.frames_dir else None
    if fd:
        fd.mkdir(parents=True, exist_ok=True)
    for i in range(a.frames):
        u = i / (a.frames - 1)
        ang = torch.tensor(np.radians(head_angle(u)), device=DEV, dtype=torch.float32)
        fr = render(P, N, w, ang, cam_dist(u), a.w, a.h, a.style, rest_y, keep)
        proc.stdin.write(fr.tobytes())
        if fd:
            Image.fromarray(fr).save(fd / f"f{i:03d}.png")
    proc.stdin.close(); proc.wait()
    print("wrote", out)


if __name__ == "__main__":
    main()
