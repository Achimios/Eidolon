#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🖼概念静帧2-入口-《Can't Kill Us》第二批概念静帧 11–18（接 stills.py 的 01–10，同一套视觉语法）。
每张都是真算出来的：Chladni 沙粒、Rosensweig 磁流体尖刺、MEMS 梳齿谐振器一阶模态、本曲频谱瀑布、Kuramoto 瀑布、
介电击穿 DBM、扫描线撕裂（本曲波形写进她的每一行）、被 threat 通道扰动的分岔图；
19 = 11 维超立方体 Petrie 投影上随本曲频带颤抖的卡拉比–丘残留意识。
风格：黑底、冷灰、只给一个有含义的红；左下 HUD = 真参数（符号说真话）；右上标签。
用法：python tools/stills_more.py [--only 11,14] [--out 目录] [--no-sheet]   默认输出到 Projekt_Visual/MV_Cant_Kill_Us/静帧/
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stills as S  # noqa: E402

W, H = S.W, S.H
GEN = S.MV / "生成"
SENS = S.MV / "感官.json"
SS = 2  # 线稿超采样倍数


# ---------- 公共：超采样线稿 ----------
def _canvas():
    im = Image.new("L", (W * SS, H * SS), 0)
    return im, ImageDraw.Draw(im)


def _down(im):
    """2x 线稿 → 画幅 0~1"""
    return np.asarray(im.resize((W, H), Image.LANCZOS), np.float64) / 255


def waterfall(rows, x0, x1, y_top, y_bot, red_idx=None, lw=1.6, bright=0.9):
    """Unknown Pleasures 式堆叠线 + 消隐：从后(上)往前(下)画，每条先用黑多边形盖住它下面的旧线。
    rows: (n, m) 每条线的抬升(像素，向上为正)。返回 (灰层, 红层)，0~1。"""
    n, m = rows.shape
    g, dg = _canvas(); r, dr = _canvas()
    xs = np.linspace(x0, x1, m) * SS
    base = np.linspace(y_top, y_bot, n) * SS
    for i in range(n):
        ys = base[i] - rows[i] * SS
        pts = list(zip(xs.tolist(), ys.tolist()))
        poly = pts + [(xs[-1], H * SS), (xs[0], H * SS)]
        dg.polygon(poly, fill=0); dr.polygon(poly, fill=0)
        (dr if i == red_idx else dg).line(pts, fill=int(255 * bright), width=int(round(lw * SS)), joint="curve")
    return _down(g), _down(r)


def _smooth1d(x, k, axis=-1):
    if k <= 1:
        return x
    ker = np.ones(k) / k
    return np.apply_along_axis(lambda v: np.convolve(np.pad(v, k, mode="edge"), ker, mode="same")[k:-k], axis, x)


def _lowpass(x, sr, fc):
    """两次盒式滤波 ≈ 低通 fc"""
    k = max(1, int(sr / fc))
    return np.convolve(np.convolve(x, np.ones(k) / k, mode="same"), np.ones(k) / k, mode="same")


# ---------- 11 Chladni ----------
def s_chladni(modes=((1, 4), (3, 5), (2, 9)), N=90000, steps=900, sig=0.03, seed=11):
    """>v<🖼概念静帧2-Chladni-方板驻波 u = cos(nπx)cos(mπy) − cos(mπx)cos(nπy)；沙粒随机跳，步长 ∝ |u|，
    在节线上停下（稳态密度 ∝ 1/u²）。三种模态并排；红 = 最后一步仍在腹点附近跳的沙"""
    rng = np.random.default_rng(seed)
    side = 560; gap = 70
    x0s = [(W - 3 * side - 2 * gap) // 2 + i * (side + gap) for i in range(3)]
    y0 = (H - side) // 2 - 30
    d = np.zeros((H, W)); acc = np.zeros((H, W)); field = np.zeros((H, W)); frame = np.zeros((H, W))
    for (n, m), x0 in zip(modes, x0s):
        u = lambda x, y: np.cos(n * np.pi * x) * np.cos(m * np.pi * y) - np.cos(m * np.pi * x) * np.cos(n * np.pi * y)
        p = rng.random((N, 2))
        for _ in range(steps):
            a = np.abs(u(p[:, 0], p[:, 1])) / 2
            p += rng.normal(0, 1, p.shape) * (sig * a)[:, None]
            p = np.abs(p); p = 1 - np.abs(1 - p)   # 板边反射
        a = np.abs(u(p[:, 0], p[:, 1])) / 2
        X = x0 + p[:, 0] * (side - 1); Y = y0 + p[:, 1] * (side - 1)
        still = a < 0.25
        d += S.splat(X[still], Y[still])
        acc += S.splat(X[~still], Y[~still])
        gy, gx = np.mgrid[0:side, 0:side] / (side - 1)
        field[y0:y0 + side, x0:x0 + side] = (np.abs(u(gx, gy)) / 2) ** 2
        frame[y0, x0:x0 + side] = frame[y0 + side - 1, x0:x0 + side] = 1
        frame[y0:y0 + side, x0] = frame[y0:y0 + side, x0 + side - 1] = 1
    v = S.tone(d, 0.5)
    img = S.colorize(np.clip(v + 0.035 * field + 0.25 * frame, 0, 1))
    a = np.clip(S.blur(acc, 2) * 6, 0, 1)
    img = img * (1 - a[..., None]) + a[..., None] * S.RED
    f = ImageFont.truetype(S.FONT, 18); im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)); dd = ImageDraw.Draw(im)
    for (n, m), x0 in zip(modes, x0s):
        dd.text((x0, y0 + side + 14), f"(n, m) = ({n}, {m})   f ~ n^2 + m^2 = {n*n + m*m}", font=f, fill=(120, 128, 134))
    img = np.asarray(im, np.float64) / 255
    moved = 100 * (1 - (acc.sum() / (acc.sum() + d.sum())))
    return S.glow(img, 2, 0.35), ["u(x,y) = cos(n pi x) cos(m pi y) - cos(m pi x) cos(n pi y)   square plate, free",
                                   f"grains = {N:,} / plate   steps = {steps}   step ~ |u|   ->  rho_inf ~ 1/u^2",
                                   f"{moved:.1f}% settled on nodal lines   red = sand still jumping at antinodes"]


# ---------- 12 磁流体 ----------
def s_ferrofluid(lam=10.0, sigma=0.030, rho=1200.0, mu_r=2.0, seed=12):
    """>v<🖼概念静帧2-磁流体-Rosensweig 失稳：临界波数 k_c = √(ρg/σ) 的三组六角模态叠加，
    振幅 ∝ √(B/B_c − 1)（超临界叉式分岔），幂次锐化成尖刺；透视高度场逐行从前往后消隐渲染。红 = 最高的那根刺"""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = 9.81
    lam_c = 2 * np.pi * np.sqrt(sigma / (rho * g)) * 1000   # mm
    lam = lam_c
    k = 2 * np.pi / lam
    mu0 = 4e-7 * np.pi; r_ = np.sqrt(mu_r)
    Mc = np.sqrt(2 / mu0 * (1 + 1 / r_) * np.sqrt(rho * g * sigma))
    zc, Rd, RB, hmax, p = 200.0, 120.0, 105.0, 1.25 * lam, 3.0
    ang = np.deg2rad(np.array([0, 120, 240]) + 7)

    def height(X, Z):
        x = X; y = Z - zc
        xw = x + 0.9 * torch.sin(y / 31 + 1.3) + 0.5 * torch.sin(y / 13); yw = y + 0.9 * torch.sin(x / 27)
        s = sum(torch.cos(k * (xw * np.cos(a) + yw * np.sin(a))) for a in ang)
        pat = ((s + 1.5) / 4.5).clamp(0, 1) ** p
        BB = 1.32 * torch.exp(-(x * x + y * y) / RB ** 2)
        amp = torch.sqrt((BB - 1).clamp(min=0) / 0.32)
        return hmax * amp * pat

    # 最高刺的位置
    gx = torch.linspace(-40, 40, 1601, device=dev); gz = torch.linspace(zc - 40, zc + 40, 1601, device=dev)
    GX, GZ = torch.meshgrid(gx, gz, indexing="xy")
    hh = height(GX, GZ); i = int(torch.argmax(hh)); Xs, Zs, Hs = float(GX.flatten()[i]), float(GZ.flatten()[i]), float(hh.max())

    ch, beta, F = 80.0, np.deg2rad(26), 2300.0
    sb, cb = np.sin(beta), np.cos(beta)
    xs = torch.arange(W, device=dev, dtype=torch.float64)
    Yi = torch.arange(H, device=dev, dtype=torch.float64)[:, None]
    img = torch.zeros((H, W), device=dev, dtype=torch.float64); red = torch.zeros_like(img)
    horizon = torch.full((W,), float(H), device=dev, dtype=torch.float64)
    # 地面 y_s 均匀采样 → Z
    ys_g = np.linspace(H + 120, 330, 2600)
    Ls = np.array([0.15, 1.0, 0.35]); Ls /= np.linalg.norm(Ls)   # 头顶偏后的条形灯：平液面反射不到，只有刺的侧面亮
    L1 = np.array([-0.45, 0.75, -0.5]); L1 /= np.linalg.norm(L1)  # 相机后左上方的主光：刺朝相机的那面起高光
    for yg in ys_g:
        # 解 y = H/2 - F*up/d, up = -ch cb + Z sb, d = ch sb + Z cb
        t = (H / 2 - yg) / F
        Z = (ch * cb + t * ch * sb) / (sb - t * cb)
        if Z <= 0 or Z > zc + Rd + 5:
            continue
        d0 = ch * sb + Z * cb
        X = (xs - W / 2) * d0 / F
        Zt = torch.full_like(X, Z)
        r = torch.sqrt(X ** 2 + (Zt - zc) ** 2)
        inside = r < Rd
        if not bool(inside.any()):
            continue
        e = 0.05
        h = height(X, Zt); hx = (height(X + e, Zt) - height(X - e, Zt)) / (2 * e); hz = (height(X, Zt + e) - height(X, Zt - e)) / (2 * e)
        yw = h - ch
        dd = -yw * sb + Zt * cb; up = yw * cb + Zt * sb
        ys = H / 2 - F * up / dd
        nrm = torch.stack([-hx, torch.ones_like(hx), -hz]); nrm = nrm / nrm.norm(dim=0)
        V = torch.stack([-X, ch - h, -Zt]); V = V / V.norm(dim=0)
        ndv = (nrm * V).sum(0)
        R = 2 * ndv * nrm - V
        L = torch.tensor(Ls, device=dev)[:, None]
        rl = (R * L).sum(0).clamp(min=0)
        fres = 0.04 + 0.96 * (1 - ndv.clamp(0, 1)) ** 5
        r1 = (R * torch.tensor(L1, device=dev)[:, None]).sum(0).clamp(min=0)
        shade = 0.006 + 1.3 * rl ** 120 + 0.25 * rl ** 16 + 0.9 * r1 ** 70 + 0.12 * r1 ** 8 + 0.08 * R[1].clamp(min=0) ** 6 * fres + 0.10 * R[2].clamp(min=0) ** 40
        rim = ((Rd - r) / 6).clamp(0, 1)
        shade = shade * rim
        spike = (((X - Xs) ** 2 + (Zt - Zs) ** 2) < (0.33 * lam) ** 2) & (h > 0.35 * Hs)
        top = torch.where(inside, ys, horizon)
        mask = (Yi >= top[None, :]) & (Yi < horizon[None, :])
        img = torch.where(mask, shade[None, :].expand(H, W), img)
        red = torch.where(mask, spike.double()[None, :].expand(H, W), red)
        horizon = torch.minimum(horizon, top)
    v = img.clamp(0, 1).cpu().numpy()
    a = red.cpu().numpy()
    out = S.colorize(v) * (1 - a[..., None]) + np.clip(v * 1.4 + 0.10, 0, 1)[..., None] * S.RED * a[..., None]
    return S.glow(np.clip(out, 0, 1), 3, 0.45), [
        f"Rosensweig instability   lambda_c = 2 pi sqrt(sigma / rho g) = {lam_c:.1f} mm   (sigma = {sigma} N/m, rho = {rho:.0f})",
        f"M_c^2 = (2/mu0)(1 + 1/r) sqrt(rho g sigma)  ->  M_c = {Mc/1000:.2f} kA/m   (mu_r = {mu_r})",
        f"h = A(B) [ (sum_3 cos(k_c n_i . x) + 3/2) / (9/2) ]^{p:.0f}   A ~ sqrt(B/B_c - 1)   red = tallest spike {Hs:.1f} mm"]


# ---------- 13 MEMS ----------
def s_mems(E=160e9, rho=2330.0, t=2e-6, wb=2.0, L=150.0, A=9.0, phases=7):
    """>v<🖼概念静帧2-MEMS-Tang 式横向梳齿谐振器的工程图：中间梭子 + 上下梳齿 + 左右折叠梁“翅膀”。
    一阶模态：梭子平动 V，桁架 V/2，每根梁是两端导向梁 u(s) = (3s² − 2s³)；多相位叠曝。
    k = 2Eh w³/L³，m_eff = m_shuttle + m_truss/4 + 12/35 m_beams（Rayleigh）。红 = 最大位移点（梳齿尖，+V 极位）"""
    um = 3.05   # px / µm
    cx, cy = W / 2 + 40, H / 2 - 30
    P = lambda u, v: ((cx + u * um) * SS, (cy - v * um) * SS)   # v 向上为正(屏幕向上)
    # --- 几何(µm)：梭子 ---
    su, sv = 60, 40               # 梭子半宽 / 半高
    notch = (36, 20)              # 左右缺口里放锚点
    holes = [(u, v) for u in np.arange(-24, 25, 12) for v in np.arange(-28, 29, 14)]
    hole = 4
    Lf, fw, pitch, gap = 34, 3, 8, 6     # 梳齿长、宽、节距、齿尖间隙
    fu = np.arange(-52, 53, pitch)       # 动齿 u
    trw = 8                               # 桁架宽
    vin, vout = 32, 12                    # 内梁 / 外梁的 v
    # 质量
    A_sh = (2 * su) * (2 * sv) - 2 * (su - notch[0]) * (2 * notch[1]) - len(holes) * hole ** 2 + 2 * len(fu) * fw * Lf
    A_tr = 2 * trw * (2 * vin + wb)
    A_b = 8 * L * wb
    m = lambda a: rho * t * a * 1e-12
    m_eff = m(A_sh) + m(A_tr) / 4 + 12 / 35 * m(A_b)
    k = 2 * E * t * (wb * 1e-6) ** 3 / (L * 1e-6) ** 3
    f0 = np.sqrt(k / m_eff) / (2 * np.pi)

    def rect(d, u0, v0, u1, v1, fill):
        (a, b), (c, e) = P(u0, v1), P(u1, v0)
        d.rectangle((a, b, c, e), outline=fill, width=2)

    def moving(d, V, fill):
        # 梭子外轮廓(带缺口)
        n0 = su - notch[0]
        poly = [(-su, sv), (su, sv), (su, notch[1]), (n0, notch[1]), (n0, -notch[1]), (su, -notch[1]), (su, -sv), (-su, -sv),
                (-su, -notch[1]), (-n0, -notch[1]), (-n0, notch[1]), (-su, notch[1])]
        d.line([P(u, v + V) for u, v in poly + poly[:1]], fill=fill, width=2)
        for u, v in holes:
            rect(d, u - hole / 2, v - hole / 2 + V, u + hole / 2, v + hole / 2 + V, fill)
        for sgn in (1, -1):      # 动齿
            for u in fu:
                a, b = sv, sv + Lf
                if sgn < 0:
                    a, b = -sv - Lf, -sv
                rect(d, u - fw / 2, a + V, u + fw / 2, b + V, fill)
        s = np.linspace(0, 1, 60); shape = 3 * s ** 2 - 2 * s ** 3
        for side in (1, -1):     # 折叠梁 + 桁架
            ut = side * (su + L)
            for vb in (vin, -vin):           # 内梁：梭子(V) → 桁架(V/2)
                us = side * (su + s * L); vv = vb + V - V / 2 * shape
                for off in (-wb / 2, wb / 2):
                    d.line([P(a_, b_ + off) for a_, b_ in zip(us, vv)], fill=fill, width=2)
            for vb in (vout, -vout):         # 外梁：锚(0) → 桁架(V/2)，锚在缺口里
                us = side * (su + 2 + s * (L - 2))   # 从锚伸出梭子外缘 2 µm 处起梁
                vv = vb + V / 2 * shape
                for off in (-wb / 2, wb / 2):
                    d.line([P(a_, b_ + off) for a_, b_ in zip(us, vv)], fill=fill, width=2)
            rect(d, min(ut, ut + side * trw), -vin - wb + V / 2, max(ut, ut + side * trw), vin + wb + V / 2, fill)

    acc = np.zeros((H, W))
    Vs = A * np.sin(np.linspace(-np.pi / 2, np.pi / 2, phases))
    for V in Vs:
        im, d = _canvas(); moving(d, V, 255)
        acc += _down(im) * (0.95 if abs(V) < 1e-9 else 0.3)
    # --- 固定件：定子梳、锚点(斜线填充) ---
    im, d = _canvas()
    n0 = su - notch[0]
    hatch = np.zeros((H * SS, W * SS), bool)
    yy, xx = np.mgrid[0:H * SS, 0:W * SS]
    hp = ((xx + yy) % 14) < 2

    def anchor(u0, v0, u1, v1):
        rect(d, u0, v0, u1, v1, 230)
        (a, b), (c, e) = P(u0, v1), P(u1, v0)
        hatch[int(b):int(e), int(a):int(c)] = True

    for sgn in (1, -1):
        bb0 = sv + Lf + gap; bb1 = bb0 + 12
        if sgn > 0:
            rect(d, -60, bb0, 60, bb1, 230); anchor(-40, bb1, 40, bb1 + 22)
            for u in fu[:-1] + pitch / 2:
                rect(d, u - fw / 2, sv + gap, u + fw / 2, bb0, 230)
        else:
            rect(d, -60, -bb1, 60, -bb0, 230); anchor(-40, -bb1 - 22, 40, -bb1)
            for u in fu[:-1] + pitch / 2:
                rect(d, u - fw / 2, -bb0, u + fw / 2, -sv - gap, 230)
    for sgn in (1, -1):          # 缺口里的锚：u 从 n0+4 到 su+2(伸出梭子外缘一点接外梁)
        u0, u1 = (n0 + 4, su + 2) if sgn > 0 else (-su - 2, -n0 - 4)
        anchor(u0, -notch[1] + 4, u1, notch[1] - 4)
    # 尺寸线与标注
    f = ImageFont.truetype(S.FONT, 15 * SS); fs = ImageFont.truetype(S.FONT, 13 * SS)
    lc = 150

    def dim_h(u0, u1, v, text):
        a, b = P(u0, v); c, _ = P(u1, v)
        d.line([(a, b), (c, b)], fill=lc, width=1 * SS)
        for xq, sg in ((a, 1), (c, -1)):
            d.line([(xq, b), (xq + sg * 14, b - 6)], fill=lc, width=SS); d.line([(xq, b), (xq + sg * 14, b + 6)], fill=lc, width=SS)
            d.line([(xq, b - 16), (xq, b + 16)], fill=lc, width=SS)
        d.text(((a + c) / 2 - d.textlength(text, font=fs) / 2, b - 34), text, font=fs, fill=lc)

    dim_h(su, su + L, -vin - 26, f"L = {L:.0f} um")
    dim_h(-su, su, -sv - Lf - gap - 12 - 22 - 18, "shuttle 120 x 80 um")
    lab = [((-su - L, vin + 20), "folded-beam flexure"), ((-24, sv + Lf + gap + 12 + 32), "comb drive (stator)"),
           ((su + L + 14, vin + 6), "truss  V/2"), ((su - 31, -notch[1] - 4), "anchor")]
    for (u, v), tx in lab:
        d.text(P(u, v), tx, font=fs, fill=lc)
    d.text(P(-su - L, vin + 32), f"w = {wb:.0f} um   h = {t*1e6:.0f} um   g = {gap - (0):.0f} um", font=fs, fill=lc)
    fixed = _down(im)
    hat = np.asarray(Image.fromarray((hatch & hp).astype(np.uint8) * 120).resize((W, H), Image.BOX), np.float64) / 255
    # 模态形状小图：沿折叠梁的 u(s)
    im2, d2 = _canvas()
    bx0, by0, bw, bh = 1480, 120, 340, 150
    d2.rectangle((bx0 * SS, by0 * SS, (bx0 + bw) * SS, (by0 + bh) * SS), outline=90, width=SS)
    s = np.linspace(0, 1, 100); shape = 3 * s ** 2 - 2 * s ** 3
    path = np.concatenate([0.5 * shape, 0.5 + 0.5 * shape])   # 锚 → 桁架 → 梭子
    qx = bx0 + 10 + np.linspace(0, bw - 20, path.size); qy = by0 + bh - 18 - path * (bh - 40)
    d2.line([(a * SS, b * SS) for a, b in zip(qx, qy)], fill=210, width=2 * SS)
    for tx, (a, b) in (("anchor", (bx0 + 8, by0 + bh + 6)), ("truss", (bx0 + bw / 2 - 20, by0 + bh + 6)), ("shuttle", (bx0 + bw - 60, by0 + bh + 6)),
                       ("mode 1: u(s)/V,  u = 3s^2 - 2s^3 per beam", (bx0, by0 - 24))):
        d2.text((a * SS, b * SS), tx, font=fs, fill=130)
    inset = _down(im2)
    v = np.clip(acc * 0.9 + fixed * 0.85 + hat * 0.6 + inset, 0, 1)
    # 红：最大位移点 = +V 极位时右端动齿的齿尖
    red = np.zeros((H, W))
    ux, vy = (cx + fu[-1] * um), (cy - (sv + Lf + A) * um)
    yy, xx = np.mgrid[0:H, 0:W]
    red += np.clip(1 - np.abs(np.hypot(xx - ux, yy - vy) - 9) / 1.5, 0, 1) + (np.hypot(xx - ux, yy - vy) < 3.5)
    red = np.clip(red, 0, 1)
    img = S.colorize(v) * (1 - red[..., None]) + red[..., None] * S.RED
    return S.glow(np.clip(img, 0, 1), 2, 0.3), [
        f"f0 = (1/2pi) sqrt(k/m_eff) = {f0/1000:.1f} kHz   k = 2 E h w^3 / L^3 = {k:.2f} N/m   (poly-Si, E = {E/1e9:.0f} GPa)",
        f"m_eff = m_shuttle + m_truss/4 + (12/35) m_beams = {m_eff*1e12:.1f} ng   rho = {rho:.0f} kg/m^3",
        f"mode 1, {phases} phases overlaid, displacement x{A/0.3:.0f} (V = 0.3 um drawn as {A:.0f} um)   red = max |u|"]


# ---------- 14 频谱瀑布 ----------
def s_waterfall(start=45.0, dur=15.0, n_lines=80, drop=51.51, nfft=4096, hop=512):
    """>v<🖼概念静帧2-频谱瀑布-本曲 45–60 s（吸气 → drop1）的 STFT，每条线 = 一个时间片的平均对数谱（log f 横轴）；
    Unknown Pleasures 式消隐。红 = drop 那一刻（51.51 s）的那条线"""
    x, sr = S.audio(sr=22050, start=start, dur=dur)
    win = np.hanning(nfft)
    nfr = 1 + (len(x) - nfft) // hop
    fr = np.stack([x[i * hop:i * hop + nfft] * win for i in range(nfr)])
    mag = np.abs(np.fft.rfft(fr, axis=1))
    tfr = (np.arange(nfr) * hop + nfft / 2) / sr
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    fx = np.geomspace(35, 11000, 700)
    slot = dur / n_lines
    ph = ((drop - start) / slot) % 1 * slot   # 时间片网格对齐 drop：红线 = drop 起的那一片
    rows = []
    for i in range(n_lines):
        sel = (tfr >= ph + i * slot) & (tfr < ph + (i + 1) * slot)
        spec = mag[sel].mean(0) if sel.any() else mag[np.argmin(np.abs(tfr - (i + 0.5) * slot))]
        rows.append(np.interp(fx, freqs, spec))
    db = 20 * np.log10(np.array(rows) + 1e-9)
    lo, hi = np.percentile(db, 45), np.percentile(db, 99.7)
    v = np.clip((db - lo) / (hi - lo), 0, 1) ** 1.4
    v = _smooth1d(v, 3)
    red_i = int(round((drop - start - ph) / slot))
    g, r = waterfall(v * 100, 420, 1500, 200, 925, red_i, lw=1.4)
    img = S.colorize(g) + r[..., None] * S.RED
    return S.glow(np.clip(img, 0, 1), 2, 0.35), [
        f"STFT of Can't Kill Us   n_fft = {nfft} ({nfft/sr*1000:.0f} ms, Hann)   hop = {hop}   sr = {sr} Hz",
        f"x: log f, 35 Hz -> 11 kHz   y: t = {start:.0f} -> {start+dur:.0f} s, {n_lines} slices of {slot*1000:.0f} ms   height = log |X|",
        f"inhale -> drop   red = the slice starting at t = {drop} s (drop 1)"]


# ---------- 15 Kuramoto 瀑布 ----------
def s_kuramoto_waterfall(N=8000, gamma=0.5, Kmax=3.0, T=320.0, dt=0.04, n_lines=96, bins=320, wmax=2.0, seed=15):
    """>v<🖼概念静帧2-Kuramoto瀑布-横轴 = 固有频率 ω（线性，±2），每一格是该频段振子的局部序参量 z(ω) = ⟨e^{iθ}⟩；
    K 从 0 缓慢线性升到 3。每条线 = R·Re(z(ω) e^{−iψ})（序参量加权、投影到平均场相位）。
    锁相区 |ω| < KR 拱成一座 √(1 − (ω/KR)²) 的山并逐行变宽；漂移的振子在两侧是噪声。红 = R 第一次越过 0.5"""
    rng = np.random.default_rng(seed)
    q = (np.arange(N) + 0.5) / N
    w = gamma * np.tan(np.pi * (q - 0.5))
    th = rng.uniform(0, 2 * np.pi, N)
    bi = np.clip(((w + wmax) / (2 * wmax) * bins).astype(int), -1, bins)
    ok = (bi >= 0) & (bi < bins)
    cnt = np.bincount(bi[ok], minlength=bins).clip(min=1)
    steps = int(T / dt)
    every = steps // n_lines
    rows, Rs, Ks = [], [], []
    for i in range(steps):
        K = Kmax * i / steps
        z = np.exp(1j * th).mean()
        th = th + dt * (w + K * np.abs(z) * np.sin(np.angle(z) - th))
        if i % every == 0 and len(rows) < n_lines:
            e = np.exp(1j * th); z = e.mean(); R, psi = np.abs(z), np.angle(z)
            zb = np.bincount(bi[ok], weights=(e[ok] * np.exp(-1j * psi)).real, minlength=bins) / cnt
            rows.append(R * zb); Rs.append(R); Ks.append(K)
    rows = np.array(rows); Rs = np.array(Rs)
    red_i = int(np.argmax(Rs > 0.5))
    g, r = waterfall(rows * 110, 420, 1500, 190, 925, red_i, lw=1.3)
    img = S.colorize(g) + r[..., None] * S.RED
    Kc = 2 * gamma
    Kth = Kc / (1 - 0.5 ** 2)
    return S.glow(np.clip(img, 0, 1), 2, 0.35), [
        f"d theta_i/dt = w_i + K R sin(psi - theta_i)   N = {N}   w ~ Lorentz(gamma = {gamma})   K: 0 -> {Kmax} over t = {T:.0f}",
        f"line = R Re( <e^(i theta)>_w e^(-i psi) ),  x = w in [-{wmax:.0f}, {wmax:.0f}], {bins} bins   K_c = 2 gamma = {Kc:.1f}",
        f"red: R crosses 0.5 at K = {Ks[red_i]:.2f}  (steady-state theory R = sqrt(1 - K_c/K) -> K = {Kth:.2f})"]


# ---------- 16 Lichtenberg ----------
def s_lichtenberg(eta=1.15, R=230, per_step=4, iters=24, seed=16):
    """>v<🖼概念静帧2-Lichtenberg-介电击穿模型 DBM（Niemeyer–Pietronero–Wiesmann 1984）：
    圆形外电极 φ=1，放电树 φ=0，解拉普拉斯（GPU 红黑 SOR，热启动），周界点生长概率 ∝ φ^η。
    线宽 ∝ log(下游节点数) = 电流。红 = 最后打到外电极的那条通路"""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator(device=dev); g.manual_seed(seed)
    S_ = 2 * R + 5; c = S_ // 2
    yy, xx = torch.meshgrid(torch.arange(S_, device=dev), torch.arange(S_, device=dev), indexing="ij")
    rr = torch.sqrt(((yy - c) ** 2 + (xx - c) ** 2).float())
    outer = rr >= R
    tree = torch.zeros((S_, S_), dtype=torch.bool, device=dev); tree[c, c] = True
    phi = (rr / R).clamp(0, 1).double()
    red_b = ((yy + xx) % 2 == 0); blk = ~red_b
    fixed = outer | tree

    def relax(n, phi, fixed):
        for _ in range(n):
            for cb in (red_b, blk):
                nb = 0.25 * (torch.roll(phi, 1, 0) + torch.roll(phi, -1, 0) + torch.roll(phi, 1, 1) + torch.roll(phi, -1, 1))
                upd = cb & ~fixed
                phi = torch.where(upd, phi + 1.9 * (nb - phi), phi)
        return phi

    phi = relax(600, phi, fixed)
    parent = {(c, c): None}; order = [(c, c)]
    hit = None
    while hit is None:
        nbt = torch.roll(tree, 1, 0) | torch.roll(tree, -1, 0) | torch.roll(tree, 1, 1) | torch.roll(tree, -1, 1)
        cand = nbt & ~tree & ~outer
        idx = torch.nonzero(cand.flatten()).squeeze(1)
        p = phi.flatten()[idx].clamp(min=0) ** eta
        if float(p.sum()) <= 0:
            break
        pick = idx[torch.multinomial(p.float(), min(per_step, idx.numel()), replacement=False, generator=g)].tolist()
        tnp = None
        for f_ in pick:
            y, x = divmod(f_, S_)
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if (y + dy, x + dx) in parent:
                    parent[(y, x)] = (y + dy, x + dx); order.append((y, x)); break
            tree[y, x] = True; phi[y, x] = 0
            if rr[y, x] >= R - 1.5:
                hit = (y, x)
        fixed = outer | tree
        phi = relax(iters, phi, fixed)
    # 下游节点数(电流)
    cnt = {n: 1 for n in order}
    for n in reversed(order):
        pa = parent[n]
        if pa is not None:
            cnt[pa] += cnt[n]
    strike = set(); n = hit
    while n is not None:
        strike.add(n); n = parent[n]
    # 盒计数分形维数
    occ = tree.cpu().numpy()
    sizes = [2, 4, 8, 16, 32]; counts = []
    for s in sizes:
        m_ = occ[:S_ // s * s, :S_ // s * s].reshape(S_ // s, s, S_ // s, s).any((1, 3)).sum(); counts.append(m_)
    Df = -np.polyfit(np.log(sizes), np.log(counts), 1)[0]
    sc = (H - 70) / (2 * R)
    tp = lambda n: ((W / 2 + (n[1] - c) * sc) * SS, (H / 2 + (n[0] - c) * sc) * SS)
    gi, gd = _canvas(); ri, rd = _canvas()
    for n in order:
        pa = parent[n]
        if pa is None:
            continue
        lw = 1.0 + 1.6 * np.log10(cnt[n])
        b = int(255 * min(1, 0.45 + 0.2 * np.log10(cnt[n])))
        (rd if n in strike else gd).line([tp(pa), tp(n)], fill=255 if n in strike else b, width=max(1, int(round(lw * SS))))
    gv = _down(gi); rv = _down(ri)
    ring = np.zeros((H, W)); yy2, xx2 = np.mgrid[0:H, 0:W]
    ring = np.clip(1 - np.abs(np.hypot(xx2 - W / 2, yy2 - H / 2) - R * sc - 2) / 1.0, 0, 1) * 0.22
    img = S.colorize(np.clip(gv + ring, 0, 1)) * (1 - rv[..., None]) + rv[..., None] * S.RED
    return S.glow(np.clip(img, 0, 1), 4, 0.7), [
        f"dielectric breakdown model (Niemeyer-Pietronero-Wiesmann)   lap(phi) = 0,  P(grow) ~ phi^eta",
        f"eta = {eta}   sites = {len(order):,}   grid R = {R}   box-counting D_f = {Df:.2f}",
        "line width ~ log(current downstream)   red = the strike that reached the electrode"]


# ---------- 17 扫描线撕裂 ----------
def s_scan_tear(t0=51.51, sr=11025, dec=1, fc=900.0, gain=320.0, dead=0.5, seed=17):
    """>v<🖼概念静帧2-扫描撕裂-把本曲 drop 起的波形像光栅扫描一样写进她的每一行：第 y 行水平位移 = gain·x(t0 + y·δ)（低通），
    只在她的遮罩里；位移方向拖出水平残影（The Crew 预告式撕裂）。红 = 位移最大的那一行"""
    img = np.asarray(Image.open(GEN / "角色_她_正面_灰_pro.jpg").convert("RGB").resize((W, H), Image.LANCZOS), np.float64) / 255
    m = np.asarray(Image.open(GEN / "角色_她_正面_灰_pro_alpha.png").convert("L").resize((W, H), Image.LANCZOS), np.float64) / 255
    x, _ = S.audio(sr=sr, start=t0 - 0.05, dur=0.6)
    x = _lowpass(x, sr, fc)[int(0.05 * sr):]
    delta = dec / sr
    rows_in = m.max(1) > 0.5
    y_top = int(np.argmax(rows_in))          # 扫描从她的第一行开始 = drop 的那一刻
    xs = np.zeros(H); seg = x[::dec][:H - y_top]; xs[y_top:y_top + len(seg)] = seg
    xn = xs / np.abs(xs).max()
    off = np.round(gain * np.sign(xn) * np.clip(np.abs(xn) - dead, 0, None) / (1 - dead)).astype(int)   # 死区：信号过阈值才撕(像丢同步)
    fig = img * m[..., None]; bg = img * (1 - m[..., None])
    out = bg.copy(); red = np.zeros((H, W))
    ymax = int(np.argmax(np.abs(off) * rows_in))
    cols = np.arange(W)
    for y in range(H):
        o = off[y]
        if not rows_in[y] or o == 0:
            out[y] += fig[y]; continue
        def sh(a, s):
            r = np.zeros_like(a)
            if s > 0: r[s:] = a[:-s]
            elif s < 0: r[:s] = a[-s:]
            else: r[:] = a
            return r
        shifted = sh(fig[y], o); ms = sh(m[y], o)
        streak = np.zeros_like(shifted)
        for j, fr in enumerate(np.linspace(0, 1, 18)):
            streak = np.maximum(streak, sh(fig[y], int(round(o * fr))) * (0.06 + 0.4 * fr ** 3))
        out[y] = out[y] * (1 - ms[:, None]) + np.maximum(shifted, streak)
        if y in (ymax, ymax + 1):
            red[y] = np.clip(ms * 1.2, 0, 1)
    v = out.mean(2, keepdims=True)
    o2 = np.clip(out * 0.55 + v * S.GREY * 0.5, 0, 1)   # 往冷灰收一点
    o2 = o2 * (1 - red[..., None]) + red[..., None] * S.RED
    return o2, [f"row y <- x(t0 + (y - {y_top}) delta)   t0 = {t0} s (drop 1)   delta = {delta*1000:.3f} ms   window = {(H - y_top)*delta*1000:.0f} ms",
                f"x lowpassed {fc:.0f} Hz   offset = {gain:.0f} px * sgn(x) max(|x| - {dead}, 0) / {1-dead:.1f}   inside her mask only, smear",
                f"red = row {ymax}, max |offset| = {abs(off[ymax])} px"]


# ---------- 18 颤抖的分岔 ----------
def s_bifurcation_tremble(eps=0.035):
    """>v<🖼概念静帧2-颤抖分岔-逻辑斯蒂分岔图，但每一列的 r 被本曲 threat 通道（感官.json, 24 fps，全曲压到横轴）推一下：
    r_col = r + ε·threat(t)。主干 dx*/dr = 1/r² 很小几乎不动，下游分支和混沌区跟着歌抖。红 = r∞ 之后（同 03）"""
    sens = json.loads(SENS.read_text(encoding="utf-8"))
    thr = np.asarray(sens["channels"]["threat"], np.float64); fps = sens["fps"]
    n = W * 2
    rs = np.linspace(2.8, 4.0, n)
    tt = np.linspace(0, len(thr) - 1, n)
    th = np.interp(tt, np.arange(len(thr)), thr)
    rp = np.clip(rs + eps * th, 0, 4.0)
    x = np.full_like(rs, 0.5)
    for _ in range(600):
        x = rp * x * (1 - x)
    d = np.zeros((H, W))
    px = ((rs - 2.8) / 1.2 * (W - 1)).astype(int)
    for _ in range(500):
        x = rp * x * (1 - x)
        py = ((1 - x) * (H * 0.8) + H * 0.1).astype(int)
        np.add.at(d, (py, px), 1)
    v = S.tone(d, 0.55)
    chaos = (np.arange(W) >= (3.5699 - 2.8) / 1.2 * W).astype(float)[None, :] * np.ones((H, 1))
    # 底部一条细细的 threat 曲线：扰动源本身
    tr = np.zeros((H, W)); ty = (H * 0.105 - th * 22).astype(int)
    np.add.at(tr, (ty, px), 1); tr = np.clip(tr, 0, 1) * 0.35
    img = S.colorize(np.clip(v + tr, 0, 1), chaos * 0.9 * (tr == 0))
    return S.glow(img, 2, 0.4), ["x_(n+1) = r_c x_n (1 - x_n),   r_c = r + eps * threat(t_c)",
                                 f"r : 2.80 -> 4.00   eps = {eps}   t_c : 0:00 -> {int(len(thr)/fps)//60}:{int(len(thr)/fps)%60:02d} (threat @ {fps} fps, trace at top)",
                                 "trunk dx*/dr = 1/r^2 barely moves; branches tremble   r_inf = 3.5699 (red)"]


# ---------- 19 十一维里的卡拉比–丘 ----------
def _line_dens(P, wts, ss=2, k=1.4):
    """折线 → 抗锯齿密度图（ss 倍超采样、双线性泼溅，再 ss×ss 盒式下采样）。
    P: (L, M, 2) 像素坐标的 L 条折线；wts: (L,) 每条的亮度。单位长度的亮度恒定，与采样密度无关"""
    a = P[:, :-1].reshape(-1, 2) * ss; b = P[:, 1:].reshape(-1, 2) * ss
    w = np.repeat(wts, P.shape[1] - 1)
    n = np.maximum(2, np.ceil(np.hypot(*(b - a).T) * k)).astype(int)
    seg = np.repeat(np.arange(len(a)), n)
    t = (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n) + 0.5) / np.repeat(n, n)
    q = a[seg] + (b[seg] - a[seg]) * t[:, None]
    ww = w[seg] * np.repeat(np.hypot(*(b - a).T) / n, n)
    Wd, Hd = W * ss, H * ss
    x0 = np.floor(q[:, 0]).astype(int); y0 = np.floor(q[:, 1]).astype(int); fx = q[:, 0] - x0; fy = q[:, 1] - y0
    d = np.zeros(Wd * Hd)
    for dx, dy, f in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xx, yy = x0 + dx, y0 + dy
        ok = (xx >= 0) & (xx < Wd) & (yy >= 0) & (yy < Hd)
        d += np.bincount(yy[ok] * Wd + xx[ok], weights=(ww * f)[ok], minlength=Wd * Hd)
    return d.reshape(H, ss, W, ss).mean((1, 3)) * ss


def _cy_quintic(n=5, m=10, s=18, ext=1.0):
    """Hanson 式卡拉比–丘五次截面：z1^n + z2^n = 1 ⊂ C²，z = a + ib，a ∈ [0, π/2]，b ∈ [−ext, ext]，
    z1 = e^{2πi k1/n} cos(z)^{2/n}，z2 = e^{2πi k2/n} sin(z)^{2/n}，25 片 (k1, k2)。
    返回 (re1, re2, im1, im2)：每个 (n²·2m, s) —— 每片 m 条等 a 线 + m 条等 b 线，每条 s 个采样。投影角 alpha 留给调用方"""
    A, B = np.linspace(0, np.pi / 2, m), np.linspace(-ext, ext, m)
    u = np.linspace(0, 1, s)
    za = (A[:, None] + 1j * (-ext + 2 * ext * u)[None, :])            # 等 a 线
    zb = ((np.pi / 2 * u)[None, :] + 1j * B[:, None])                   # 等 b 线
    Z = np.concatenate([za, zb])
    c1, c2 = np.cos(Z) ** (2 / n), np.sin(Z) ** (2 / n)
    out = []
    for k1 in range(n):
        for k2 in range(n):
            out.append((np.exp(2j * np.pi * k1 / n) * c1, np.exp(2j * np.pi * k2 / n) * c2))
    z1 = np.concatenate([o[0] for o in out]); z2 = np.concatenate([o[1] for o in out])
    return z1, z2


def s_calabi_yau_11d(dim=11, n=5, t_lo=80.0, t_hi=84.0, t_near=82.0, f_lo=40.0, f_hi=8000.0, expo=5, seed=19,
                     rings=((0.30, 2, 16.0), (0.58, 2, 18.0), (0.80, 1, 19.0), (1.0, 1, 25.0))):
    """>v<🖼概念静帧2-卡拉比丘11维-她掉进 11 维：空间 = 11 维超立方体 {±½}^11 的 Petrie 正交投影（e_k = (cos πk/11, sin πk/11)），
    2048 顶点 11264 条棱，极细；残留意识 = 挂在若干顶点轨道上的小卡拉比–丘五次截面（M 理论 11 = 4 + 1 + 6，弦的振动模式决定它是什么）。
    每个 CY 分到本曲 82 s 附近的一个对数频带，颤抖幅度 = 该频带此刻的电平相对 80–84 s 中位数的抬升；颤抖画成多重曝光（在 (Im z1, Im z2) 平面里转 δα + 平移）。
    她 = 正中一个亮点。红 = 最内层轨道里此刻最响的那一个：离她最近、正在同化她的那个"""
    rng = np.random.default_rng(seed)
    # --- 11-cube Petrie 投影 ---
    V = (((np.arange(2 ** dim)[:, None] >> np.arange(dim)) & 1) - 0.5)
    th = np.pi * np.arange(dim) / dim
    E = np.stack([np.cos(th), np.sin(th)], 1)
    P = V @ E
    lo = ((np.arange(2 ** dim)[:, None] >> np.arange(dim)) & 1) == 0
    ia, kb = np.nonzero(lo)
    ib = ia + (1 << kb)
    n_edges = len(ia)
    Rmax = np.hypot(*P.T).max()
    sc = 400 / Rmax
    cx, cy = W / 2 + 130, H / 2 - 20                              # 右移一点，给左下 HUD 让位
    XY = np.stack([cx + P[:, 0] * sc, cy - P[:, 1] * sc], 1)
    n_dist = len(np.unique(np.round(P, 9), axis=0))
    seg = np.stack([XY[ia], XY[ib]], 1)
    lat = _line_dens(seg, np.ones(n_edges))
    # --- 顶点轨道（绕中心转 π/11 的对称群，22 个一组）---
    r = np.hypot(*P.T); rk = np.round(r, 6)
    radii = np.unique(rk)
    orbit_r = np.array([q for q in radii if (rk == q).sum() == 2 * dim])   # 恰好一条 C22 轨道（不和别的轨道同半径）
    sites, ring, size = [], [], []
    for j, (fr, step, sz) in enumerate(rings):
        q = orbit_r[np.argmin(np.abs(orbit_r - fr * Rmax))]
        idx = np.nonzero(rk == q)[0]
        idx = idx[np.argsort(np.arctan2(P[idx, 1], P[idx, 0]))][::step]
        sites += idx.tolist(); ring += [j] * len(idx); size += [sz] * len(idx)
    sites = np.array(sites); ring = np.array(ring); size = np.array(size); N = len(sites)
    # --- 本曲频带电平：N 个对数频带，内层 = 低频 ---
    sr, nfft, hop = 22050, 8192, 256
    x, _ = S.audio(sr=sr, start=t_lo, dur=t_hi - t_lo)
    win = np.hanning(nfft)
    nfr = 1 + (len(x) - nfft) // hop
    pw = np.abs(np.fft.rfft(np.stack([x[i * hop:i * hop + nfft] * win for i in range(nfr)]), axis=1)) ** 2
    tf = t_lo + (np.arange(nfr) * hop + nfft / 2) / sr
    fq = np.fft.rfftfreq(nfft, 1 / sr)
    edges = np.geomspace(f_lo, f_hi, N + 1)
    fine = np.geomspace(f_lo, f_hi, N * 24)
    Pf = np.stack([np.interp(fine, fq, p) * fine for p in pw])           # ×f：在 log f 上积分
    L = 10 * np.log10(Pf.reshape(nfr, N, 24).sum(2) + 1e-12)              # (帧, 频带) dB
    near = np.abs(tf - t_near) < 0.35
    flux = np.r_[0, np.clip(np.diff(L, axis=0), 0, None).sum(1)]
    fi = np.nonzero(near)[0][np.argmax(flux[near])]                        # 82 s 附近最强的起音帧
    t_at = tf[fi]
    rise = L[fi] - np.median(L, 0)
    amp = np.clip(0.5 + rise / 12.0, 0, 1)
    # --- 卡拉比–丘 ---
    z1, z2 = _cy_quintic(n=n, m=5, s=16)
    ext = max(np.abs(z1.real).max(), np.abs(z2.real).max())
    inner = np.nonzero(ring == 0)[0]
    red_i = inner[np.argmax(amp[inner])]
    gl, gw, rl, rw = [], [], [], []
    for i in range(N):
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        ph = rng.uniform(0, 2 * np.pi, (expo, 3))
        for e in range(expo):
            a_e = amp[i] * (0 if e == 0 else 1)
            al = np.pi / 4 + a_e * 0.55 * np.sin(ph[e, 0])
            p3 = np.stack([z1.real, z2.real, np.cos(al) * z1.imag + np.sin(al) * z2.imag], -1) / ext
            p3 = p3 @ Q.T
            off = a_e * size[i] * 0.55 * np.array([np.cos(ph[e, 1]), np.sin(ph[e, 1])])
            px = XY[sites[i], 0] + off[0] + p3[..., 0] * size[i]
            py = XY[sites[i], 1] + off[1] + p3[..., 1] * size[i]
            depth = 0.25 + 0.75 * np.clip(p3[..., 2].mean(-1) * 0.6 + 0.5, 0, 1)   # 近亮远暗，读出立体
            wt = depth * (0.8 if e == 0 else 0.25 + 0.3 * amp[i])
            (rl if i == red_i else gl).append(np.stack([px, py], -1)); (rw if i == red_i else gw).append(wt)
    cyg = _line_dens(np.concatenate(gl), np.concatenate(gw))
    cyr = _line_dens(np.concatenate(rl), np.concatenate(rw))
    # --- 合成 ---
    lv = np.clip(lat / (1.6 * S.blur(S.blur(lat, 10), 10) + 0.25), 0, 1) ** 1.3 * 0.55   # 局部对比度归一：中心重叠上千条棱也还是细线，不糊成灰雾
    cv = 1 - np.exp(-cyg * 0.20)                                  # 线稿很密：指数饱和，网格内部还看得见
    rv = 1 - np.exp(-cyr * 0.22)
    yy, xx = np.mgrid[0:H, 0:W]
    dc = np.hypot(xx - cx, yy - cy)
    her = np.exp(-(dc / 2.2) ** 2) + 0.25 * np.exp(-(dc / 9) ** 2)
    hole = np.clip(np.exp(-(dc / 46) ** 2) * 0.85, 0, 1)          # 她周围压暗一点，亮点才读得出
    g = np.clip(lv * (1 - hole) + cv + her, 0, 1)
    img = S.colorize(g) * (1 - rv[..., None]) + rv[..., None] * S.RED
    img = S.glow(np.clip(img, 0, 1), 3, 0.45)
    lo_b, hi_b = edges[red_i], edges[red_i + 1]
    return img, [
        f"11-cube {{-1/2, 1/2}}^11: {2**dim} vertices, {n_edges} edges ({n_dist} distinct points)",
        f"Petrie projection e_k = (cos pi k/11, sin pi k/11), symmetry C22",
        f"Calabi-Yau quintic slice z1^5 + z2^5 = 1 in C^2, (k1,k2) in Z5 x Z5 = 25 patches",
        f"remnants N = {N} on {len(rings)} vertex orbits, one log band each {f_lo:.0f} Hz - {f_hi/1000:.0f} kHz (inner = low)",
        f"tremble: da, dx ~ a_i = 0.5 + (L_i(t) - median L_i) / 12 dB   Can't Kill Us @ {t_at:.2f} s",
        f"red = most agitated inner remnant ({lo_b:.0f}-{hi_b:.0f} Hz, a = {amp[red_i]:.2f})   centre = her"]


STILLS = {"11_chladni": s_chladni, "12_ferrofluid": s_ferrofluid, "13_mems": s_mems, "14_waterfall": s_waterfall,
          "15_kuramoto_waterfall": s_kuramoto_waterfall, "16_lichtenberg": s_lichtenberg, "17_scan_tear": s_scan_tear,
          "18_bifurcation_tremble": s_bifurcation_tremble, "19_calabi_yau_11d": s_calabi_yau_11d}


def contact_sheet(out, names, tw=640, th=360, pad=16, lab=30):
    """2 行 × 4 列总览，每格下方小字标签"""
    cols = 4; rows = (len(names) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw + (cols + 1) * pad, rows * (th + lab) + (rows + 1) * pad), (0, 0, 0))
    d = ImageDraw.Draw(sheet); f = ImageFont.truetype(S.FONT, 16)
    for i, nm in enumerate(names):
        p = out / f"{nm}.jpg"
        if not p.exists():
            continue
        x0 = pad + (i % cols) * (tw + pad); y0 = pad + (i // cols) * (th + lab + pad)
        sheet.paste(Image.open(p).resize((tw, th), Image.LANCZOS), (x0, y0))
        d.text((x0, y0 + th + 6), nm.upper(), font=f, fill=(140, 148, 154))
    p = out / "_新增_总览.jpg"
    sheet.save(p, quality=92)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=str(S.MV / "静帧"))
    ap.add_argument("--no-sheet", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for name, fn in STILLS.items():
        if a.only and not any(k in name for k in a.only.split(",")):
            continue
        t0 = time.time()
        img, hud = fn()
        p = S.save(img, name, hud, out)
        print(f"{name}: {time.time() - t0:.1f}s -> {p.name}", flush=True)
    if not a.no_sheet:
        print(contact_sheet(out, list(STILLS)), flush=True)


if __name__ == "__main__":
    main()
