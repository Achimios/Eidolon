#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🎞样片-最优传输-重组：静帧 09 的同一个源（均匀圆盘）和目标（素材图亮部 = 她），做成会动的位移插值（McCann）。
x_i(τ) = (1-τ)·x_i + τ·T(x_i)：直线、互不相撞；含义（效果库）= 重组：什么都没丢，只是重新排列。

T 怎么来：
  源 / 目标 = stills.ot_problem（和静帧 09 同 seed，逐位相同）。
  ⚠️没用静帧的切片 OT 配对当 T：实测切片配对不是循环单调（随机点对里 1.4% 有 <dx,dy> < 0，即两条路径
  会「交叉」；局部 2-opt 精修只降到 0.4%，因为违例点对在源里相隔 ~30px、在终点相隔 ~150–260px，是大尺度的）。
  改用熵 Brenier 映射：Sinkhorn（对数域、ε 退火）求势 g，T(x) = Σ softmax((g - |x-y|²)/ε)·y = ∇(凸函数)
  ⇒ 对**所有**点对单调 ⇒ 位移插值里任何两粒子不相撞，且该配对对经验测度 (μ0, T#μ0) 精确最优。
  代价：终点是目标的 √ε 尺度平滑（ε=3e-4 → ~0.017 单位 ≈ 4px）。
时间：
  全局钟 g(u) 快出慢到、两端速度都不为 0；每个粒子按距离错开出发（远的先走、所有人在同一个强拍到达）；
  一个红点（她的「心」，终点在胸口）走自己的钟，晚一拍到达。
HUD（真值，逐帧算）：W2(μ_t, ν) = sqrt(mean|x_i(t) - y_i|²)（当前配对的代价；全局同一 τ 时 = (1-τ)·W2_0），
  N、质量守恒（世界网格直方图求和 = N）。sidecar 里另有每帧子样本匈牙利 W2 —— 配对是否最优的「必须相等」核对。
用法：python tools/ot_flow.py [--start 205.11] [--dur 4.0] [--name 20_最优传输_重组_试_v1]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stills as S  # noqa: E402

FPS = 24
W, H = 1280, 720
GREY = np.array([0.80, 0.85, 0.88])
RED = np.array([1.0, 0.16, 0.20])
HEART_UV = (760 / 1536, 470 / 1024)   # 素材图里胸口的位置（相对坐标）


# ---------------------------------------------------------------- 最优传输
def problem(N=60000, iters=60, seed=5):
    """和静帧 09 同源同目标同切片：返回 src, Y（Y[i] = 切片 OT 把 src[i] 送到的位置）"""
    src, tgt, rng = S.ot_problem(N, seed)
    return src, S.sliced_ot(src, tgt, iters, rng)


def target_point(uv):
    """素材图相对坐标 → 世界坐标（和 stills.ot_problem 的目标同一套变换）"""
    return np.array([(uv[0] * 320 - 160) / 120 + 0.9, (uv[1] * 240 - 120) / 120])


def w2_coupling(P, Y):
    """给定配对（P[i] ↔ Y[i]）的传输代价 sqrt(mean|P-Y|²)。配对最优时它就是 W2"""
    return float(np.sqrt(np.mean(np.sum((P - Y) ** 2, 1))))


def hungarian(C):
    """精确指派（最小代价），O(n³) 最短增广路（e-maxx 版）。返回 col[i] = 第 i 行分到的列。只给小 n（≤ ~500）用"""
    C = np.asarray(C, np.float64)
    n = C.shape[0]
    u = np.zeros(n + 1); v = np.zeros(n + 1)
    p = np.zeros(n + 1, int); way = np.zeros(n + 1, int)
    for i in range(1, n + 1):
        p[0] = i; j0 = 0
        minv = np.full(n + 1, np.inf); used = np.zeros(n + 1, bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            free = ~used[1:]
            cur = C[i0 - 1] - u[i0] - v[1:]
            upd = free & (cur < minv[1:])
            minv[1:][upd] = cur[upd]
            way[1:][upd] = j0
            cand = np.where(free, minv[1:], np.inf)
            j1 = int(np.argmin(cand)) + 1
            delta = cand[j1 - 1]
            u[p[used]] += delta
            v[used] -= delta
            minv[1:][free] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]; p[j0] = p[j1]; j0 = j1
            if j0 == 0:
                break
    col = np.empty(n, int)
    col[p[1:] - 1] = np.arange(n)
    return col


def w2_exact(P, Y):
    """两组等权点之间的精确 W2（匈牙利）。小 n 用"""
    C = np.sum((P[:, None, :] - Y[None, :, :]) ** 2, -1)
    col = hungarian(C)
    return float(np.sqrt(C[np.arange(len(P)), col].mean())), col


def _dev(device):
    import torch
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def barycentric(x, y, g, eps, device=None, chunk=2048):
    """熵 Brenier 映射 T(x) = Σ_j softmax_j((g_j - |x-y_j|²)/ε) · y_j。
    指数里与 x 相关的只有 2<x,y_j>/ε（|x|² 在 softmax 里约掉）⇒ T = ∇φ，φ(x) = ε/2·LSE_j((2<x,y_j> + g_j - |y_j|²)/ε) 凸
    ⇒ 对任意 g（收敛与否）都单调：<x_i - x_j, T(x_i) - T(x_j)> ≥ 0 ⇒ 位移插值中任意两粒子不相撞"""
    import torch
    dev = _dev(device)
    X = torch.tensor(x, device=dev, dtype=torch.float64); Y = torch.tensor(y, device=dev, dtype=torch.float64)
    G = torch.tensor(np.asarray(g), device=dev, dtype=torch.float64)
    ly = G - (Y ** 2).sum(1)
    out = torch.empty_like(X)
    for s in range(0, len(X), chunk):
        out[s:s + chunk] = torch.softmax((2 * X[s:s + chunk] @ Y.T + ly[None, :]) / eps, 1) @ Y
    return out.cpu().numpy()


def entropic_map(x, y, eps_list=(1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001), iters=25, device=None, chunk=2048, log=None):
    """对数域 Sinkhorn（等权、ε 退火）→ 势 g → barycentric(x, y, g, ε_last)。返回 T(x)（每个源点的终点）和信息"""
    import torch
    dev = _dev(device)
    dt = torch.float32 if dev == "cuda" else torch.float64
    X = torch.tensor(x, device=dev, dtype=dt); Y = torch.tensor(y, device=dev, dtype=dt)
    N, M = len(X), len(Y)

    def pot(A, B, other, eps, logw):
        # 对数域：-ε·LSE_j((other_j - |a-b_j|²)/ε + log w)；|a|² 提出去，矩阵乘代替 cdist（快）
        a2, b2 = (A ** 2).sum(1), (B ** 2).sum(1)
        res = torch.empty(len(A), device=dev, dtype=dt)
        for s in range(0, len(A), chunk):
            z = (2 * A[s:s + chunk] @ B.T + (other - b2)[None, :]) / eps
            res[s:s + chunk] = a2[s:s + chunk] - eps * (torch.logsumexp(z, 1) + logw)
        return res

    f = torch.zeros(N, device=dev, dtype=dt); g = torch.zeros(M, device=dev, dtype=dt)
    info = []
    for eps in eps_list:
        for _ in range(iters):
            f = pot(X, Y, g, eps, -np.log(M))
            g = pot(Y, X, f, eps, -np.log(N))
        err = float((pot(X, Y, g, eps, -np.log(M)) - f).abs().max() / eps)   # 行边缘误差（以 ε 为单位的势变化）
        info.append({"eps": eps, "row_marginal_err_over_eps": err})
        if log:
            log(f"  sinkhorn eps {eps:g}  row err/eps {err:.3g}")
    return barycentric(x, y, g.double().cpu().numpy(), eps_list[-1], device=dev, chunk=chunk), info


def violation_stats(src, Y, n_pairs=2_000_000, seed=1):
    """全局随机点对里 <dx,dy> < 0 的比例，以及这些点对在 τ∈[0,1] 上的最近距离（世界单位）"""
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(src), n_pairs); j = rng.integers(0, len(src), n_pairs)
    a = src[i] - src[j]; b = Y[i] - Y[j]
    dot = np.sum(a * b, 1)
    bad = dot < 0
    return {"pairs": n_pairs, "violating_frac": float(bad.mean()),
            "min_approach_of_violators": float(min_approach(a[bad], b[bad]).min()) if bad.any() else None}


def min_approach(a, b):
    """两点相对位移 r(τ) = (1-τ)a + τb，τ∈[0,1] 上 |r| 的最小值（闭式）"""
    d = b - a
    dd = np.sum(d * d, -1)
    tau = np.clip(-np.sum(a * d, -1) / np.where(dd > 0, dd, 1), 0, 1)
    r = a + tau[..., None] * d
    return np.sqrt(np.sum(r * r, -1))


# ---------------------------------------------------------------- 时间
def ease(u, a=0.3):
    """u∈[0,1] → [0,1]，g'(0) = 2-a（快出），g'(1) = a（慢到但不停）"""
    u = np.clip(u, 0, 1)
    return a * u + (1 - a) * (1 - (1 - u) ** 2)


def c1_ramp(x, w):
    """0 → 线性 的 C¹ 软拐角（宽 2w）"""
    return np.where(x <= -w, 0.0, np.where(x >= w, x, (x + w) ** 2 / (4 * w)))


def particle_tau(g, s, w=0.08):
    """全局钟 g → 每个粒子的 τ_i：s_i 后出发、g=1 时全部到达"""
    return np.clip(c1_ramp(g - s, w) / c1_ramp(1 - s, w), 0, 1)


def stagger(src, Y, k=0.35):
    """远的先走：s_i = k·(1 - 距离的排名分位)"""
    d = np.linalg.norm(Y - src, axis=1)
    q = np.argsort(np.argsort(d)) / (len(d) - 1)
    return k * (1 - q)


# ---------------------------------------------------------------- 渲染
def splat_bilinear(x, y, w, out):
    x0 = np.floor(x).astype(np.int64); y0 = np.floor(y).astype(np.int64)
    fx = x - x0; fy = y - y0
    for dx, dy, ww in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xx, yy = x0 + dx, y0 + dy
        ok = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
        out += np.bincount(yy[ok] * W + xx[ok], weights=(w * ww)[ok], minlength=W * H).reshape(H, W)


def camera(u):
    """匀速漂移 + 匀速推近（世界 → 屏幕），一直在动、没有抖动"""
    cx = -0.45 + 1.20 * u      # 从圆盘与她之间 → 她；匀速
    cy = 0.02 + 0.10 * u       # 末帧她整体上移，不压底部 HUD
    sc = 245.0 * (1 + 0.12 * u)
    return cx, cy, sc


def to_screen(P, cam):
    cx, cy, sc = cam
    return (P[:, 0] - cx) * sc + W / 2, (P[:, 1] - cy) * sc + H / 2


def font(sz):
    return ImageFont.truetype(S.FONT, sz)


def hud(im, lines, tag):
    d = ImageDraw.Draw(im)
    f = font(14)
    for i, line in enumerate(lines):
        d.text((28, H - 30 - 19 * (len(lines) - i)), line, font=f, fill=(150, 160, 168))
    d.text((W - 28 - d.textlength(tag, font=f), 24), tag, font=f, fill=(105, 112, 118))
    return im


def outname(p):
    """不覆盖：已存在就加 _2、_3…"""
    p = Path(p)
    if not p.exists():
        return p
    k = 2
    while p.with_name(f"{p.stem}_{k}{p.suffix}").exists():
        k += 1
    return p.with_name(f"{p.stem}_{k}{p.suffix}")


def contact(frames, path, cols=4):
    tw, th = 480, 270
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * th), (8, 8, 10))
    d = ImageDraw.Draw(sheet)
    for i, (lab, im) in enumerate(frames):
        x, y = (i % cols) * tw, (i // cols) * th
        sheet.paste(im.resize((tw, th), Image.LANCZOS), (x, y))
        d.text((x + 8, y + 6), lab, font=font(13), fill=(230, 230, 230))
    sheet.save(path, quality=90)


# ---------------------------------------------------------------- 主程序
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=float, default=205.11, help="播放时间（感官.json 已是播放时间）")
    ap.add_argument("--dur", type=float, default=4.0)
    ap.add_argument("--arrive", type=float, default=208.38, help="主体到达 = drop3 第 3 小节强拍")
    ap.add_argument("--heart", type=float, default=208.76, help="红点到达 = 下一拍")
    ap.add_argument("--N", type=int, default=60000)
    ap.add_argument("--stagger", type=float, default=0.5)
    ap.add_argument("--eps", default="1,0.3,0.1,0.03,0.01,0.003,0.001,0.0003", help="Sinkhorn ε 退火表")
    ap.add_argument("--iters", type=int, default=25, help="每个 ε 的 Sinkhorn 迭代数")
    ap.add_argument("--sub", type=int, default=300, help="逐帧子样本匈牙利核对的点数")
    ap.add_argument("--name", default="20_最优传输_重组_试_v1")
    ap.add_argument("--outdir", default=str(S.MV / "样片"), help="调参草稿可指到 scratchpad")
    ap.add_argument("--tmp", required=True, help="临时目录（scratchpad）")
    a = ap.parse_args()
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    tmp = Path(a.tmp); tmp.mkdir(parents=True, exist_ok=True)
    sn = json.loads((S.MV / "感官.json").read_text("utf-8"))
    ev = sn["events"]

    t0 = time.time()
    src, tgt, rng0 = S.ot_problem(a.N, 5)
    Y0 = S.sliced_ot(src, tgt, 60, rng0)            # 静帧的切片配对，只拿来对比
    w2_sliced = w2_coupling(src, Y0)
    eps_list = tuple(float(e) for e in a.eps.split(","))
    cache = tmp / f"ot_map_N{a.N}_eps{eps_list[-1]:g}_it{a.iters}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True); Y = z["T"]; sk_info = list(z["info"])
    else:
        Y, sk_info = entropic_map(src, tgt, eps_list, a.iters, log=print)
        np.savez(cache, T=Y, info=np.array(sk_info, dtype=object))
    W2_0 = w2_coupling(src, Y)
    viol0, viol = violation_stats(src, Y0), violation_stats(src, Y)
    print(f"OT {time.time() - t0:.1f}s  W2 sliced {w2_sliced:.4f}  entropic {W2_0:.4f}  violations {viol0} -> {viol}")
    hist = sk_info

    s = stagger(src, Y, a.stagger)
    heart = int(np.argmin(np.linalg.norm(Y - target_point(HEART_UV), axis=1)))
    rng = np.random.default_rng(3)
    sub = rng.choice(a.N, a.sub, replace=False)
    sub = sub[sub != heart]

    n = int(round(a.dur * FPS))
    u_b, u_h = a.arrive - a.start, a.heart - a.start
    TR = np.arange(18) * 0.009           # 拖尾：往回 18 个子时刻（7 个时有台阶条纹）
    TW = np.exp(-np.arange(18) / 4.0); TW /= TW.sum()

    def state(tp):
        """播放时间 tp → 所有粒子位置（含红点）、全局 g"""
        u = tp - a.start
        g = ease(u / u_b)
        tau = particle_tau(g, s)
        tau[heart] = ease(u / u_h)
        return src + tau[:, None] * (Y - src), g, tau

    # 曝光固定（不逐帧归一化，否则会闪）：用最后一帧的密度定
    cam_end = camera(1.0)
    dref = np.zeros((H, W)); x, y = to_screen(Y, cam_end); splat_bilinear(x, y, np.ones(a.N), dref)
    d0 = np.percentile(dref[dref > 0], 97)
    grid_lo, grid_hi = np.array([-3.0, -2.0]), np.array([3.5, 2.0])

    vtmp = tmp / f"{a.name}_noaudio.mp4"
    pipe = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                             "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "15", str(vtmp)],
                            stdin=subprocess.PIPE)
    side = {"play_start": a.start, "dur": n / FPS, "arrive": a.arrive, "heart_arrive": a.heart,
            "time": "感官.json is in playback time (no offset)",
            "source_target": "stills.ot_problem(N, seed=5) — same source/target as 静帧/09",
            "map": "entropic Brenier map (log-domain Sinkhorn, eps annealing) — gradient of a convex fn => monotone for all pairs",
            "eps": eps_list, "sinkhorn": hist,
            "N": a.N, "W2_sliced_pairing_of_still": w2_sliced, "W2_0": W2_0,
            "violations_sliced": viol0, "violations_entropic": viol, "stagger_k": a.stagger, "heart_index": heart,
            "check": "w2_sub_exact = Hungarian W2 on a fixed subsample of particles at current positions vs their endpoints; "
                     "w2_sub_coupling = cost of our pairing on the same subsample. equal ⇔ our pairing is optimal there",
            "frames": []}
    thumbs, every = [], max(1, n // 16)
    tf = time.time()
    for i in range(n):
        tp = a.start + i / FPS
        u = i / FPS
        cam = camera(u / a.dur)
        dens = np.zeros((H, W)); hd = np.zeros((H, W))
        for dt, wt in zip(TR, TW):
            P, _, _ = state(tp - dt)
            x, y = to_screen(P, cam)
            wv = np.full(a.N, wt); wv[heart] = 0
            splat_bilinear(x, y, wv, dens)
            splat_bilinear(x[heart:heart + 1], y[heart:heart + 1], np.array([wt]), hd)
        P, g, tau = state(tp)
        # 质量守恒：世界网格直方图（与相机无关）求和
        ok = np.all((P >= grid_lo) & (P < grid_hi), 1)
        hist2, _, _ = np.histogram2d(P[ok, 0], P[ok, 1], bins=(130, 80), range=[(grid_lo[0], grid_hi[0]), (grid_lo[1], grid_hi[1])])
        mass = int(hist2.sum())
        w2 = w2_coupling(P, Y)
        w2_lin = (1 - g) * W2_0
        w2_sub_c = w2_coupling(P[sub], Y[sub])
        w2_sub_e, _ = w2_exact(P[sub], Y[sub])
        v = (1 - np.exp(-dens / d0)) ** 0.8
        img = v[..., None] * GREY
        # 红点：小核 + 到达那一拍亮一下
        flash = np.exp(-max(0.0, tp - a.heart) / 0.18) if tp >= a.heart else 0.0
        hk = np.clip(S.blur(S.blur(hd, 2), 2) * (40 + 60 * flash), 0, 1)
        img = img * (1 - hk[..., None]) + hk[..., None] * RED
        img = S.glow(np.clip(img, 0, 1), 4, 0.45)
        im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        arrived = int(np.sum(tau >= 1 - 1e-9))
        gap = abs(w2_sub_c - w2_sub_e) / max(w2_sub_e, 1e-12)
        hud(im, [f"t = {tp:7.3f} s (play)   global g = {g:.3f}   in flight {a.N - arrived:,}   arrived {arrived:,}",
                 f"W2(mu_t, nu) = {w2:.4f}   = {100 * w2 / W2_0:5.1f}% of W2_0 {W2_0:.4f}   "
                 f"check: exact Hungarian on {len(sub)} pts  gap {100 * gap:.4f}%",
                 f"N = {a.N:,}   sum in = {a.N:,}   sum out = {mass:,}   " + ("MASS CONSERVED" if mass == a.N else "MASS LOST"),
                 "displacement interpolation x(t) = (1-t) x + t T(x)   nothing was lost. only rearranged"],
            "20_OPTIMAL_TRANSPORT  /  RECOMPOSE")
        pipe.stdin.write(np.asarray(im).tobytes())
        if i % every == 0 or i == n - 1:
            thumbs.append((f"{tp:.2f}s  g {g:.2f}  W2 {w2:.3f}", im))
        side["frames"].append({"i": i, "t_play": round(tp, 4), "g": round(float(g), 5), "W2": w2, "W2_lin_(1-g)W2_0": w2_lin,
                               "w2_sub_coupling": w2_sub_c, "w2_sub_exact": w2_sub_e, "mass": mass,
                               "arrived": arrived, "tau_heart": float(tau[heart])})
        if i % 12 == 0:
            print(f"  frame {i}/{n}  {time.time() - tf:.0f}s  W2 {w2:.4f} sub {w2_sub_c:.4f}/{w2_sub_e:.4f}", flush=True)
    pipe.stdin.close(); pipe.wait()
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    out = outname(Path(a.outdir) / f"{a.name}.mp4")
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(vtmp), "-ss", f"{a.start:.3f}", "-t", f"{n / FPS:.3f}",
                    "-i", str(S.MUSIC), "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
                    "-t", f"{n / FPS:.3f}", str(out)], check=True)
    Path(str(out) + ".json").write_text(json.dumps(side, ensure_ascii=False, indent=1), "utf-8")
    contact(thumbs, out.with_name(out.stem + "_接触表.jpg"))
    gaps = [abs(f["w2_sub_coupling"] - f["w2_sub_exact"]) / max(f["w2_sub_exact"], 1e-12) for f in side["frames"]]
    print("->", out, f" sub-sample coupling/exact max rel gap {max(gaps):.4%}  total {time.time() - t0:.0f}s")
    print("downbeats in window:", [d for d in ev["downbeats"] if a.start <= d <= a.start + n / FPS])


if __name__ == "__main__":
    main()
