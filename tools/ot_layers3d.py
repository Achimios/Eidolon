#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🎞样片-最优传输-3D多层：静帧 09 的多层构图搬进 3D —— 不演传输过程，只把位移插值在 K 个 t 上切片，
每片放在不同深度的平面上；片上的粒子「活着」（沿各自的 OT 方向小幅来回，振幅跟音乐呼吸），
相机快进慢出地绕着这一摞层走（入画已在动、一路减速、不停；无滚转）—— 层间视差给深度。

数学（真值）：
  T = 熵 Brenier 映射（ot_flow.entropic_map：对数域 Sinkhorn + ε 退火；T = ∇凸函数 ⇒ 对所有点对单调
  ⇒ 配对对 (μ0, T#μ0) 精确最优）。第 k 层 = μ_{t_k} = ((1-t_k)·id + t_k·T)#μ0。
  ⇒ W2(μ_{t_k}, ν) = (1-t_k)·W2_0（最优配对下的位移插值；tests 里用精确匈牙利核对）。
  HUD 每层印的是**当前帧实测**的配对代价 sqrt(mean|x_i - T(x_i)|²)（含抖动），不是公式抄写。
  为什么不回切片 OT：用户说撞衫无所谓，但切片配对不是循环单调（~1.4% 点对违例），那时 HUD 上的数只是
  W2 的上界，不是 W2；3D 多层里熵映射的 √ε 平滑（~4px）看不出来 ⇒ 留 Sinkhorn，HUD 才说真话。
源 / 目标可插拔：--src / --dst 取 still09（默认，= 静帧 09 的圆盘 / 她）、图片路径（亮度分布采样）、
  .npy 点集（(n,2)，按世界坐标原样用；数量不等则有放回重采样到 N）。
用法：python tools/ot_layers3d.py --tmp <scratchpad> [--start 205.11] [--dur 4.0] [--src still09 --dst still09]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stills as S  # noqa: E402
import ot_flow as O  # noqa: E402

FPS = 24
W, H = 1280, 720
GREY = np.array([0.84, 0.88, 0.92])   # 冷灰白
RED = np.array([1.0, 0.16, 0.20])     # 唯一的红：她的心


# ---------------------------------------------------------------- 源 / 目标
def image_measure(path, N, rng, thresh=0.35, power=2.0, height=2.0, x_shift=0.0, grid=(320, 240)):
    """图片亮部 → N 个点（世界坐标：高 = height，居中，再右移 x_shift）。同 stills.ot_problem 的做法。
    返回 (点, uv→世界 的函数)"""
    gw, gh = grid
    g = np.asarray(Image.open(path).convert("L").resize((gw, gh)), np.float64)
    g = np.clip(g / 255 - thresh, 0, 1) ** power
    p = (g / g.sum()).ravel()
    idx = rng.choice(p.size, N, p=p)
    pts = np.stack([idx % gw + rng.random(N), idx // gw + rng.random(N)], 1)
    s = height / gh

    def uv2w(uv):
        return np.array([(uv[0] * gw - gw / 2) * s + x_shift, (uv[1] * gh - gh / 2) * s])
    return (pts - [gw / 2, gh / 2]) * s + [x_shift, 0], uv2w


def load_measure(spec, N, rng, **kw):
    """'disc' / 图片 / .npy → (N,2) 点 + uv→世界（.npy 与 disc 没有 uv，返回 None）"""
    if spec == "disc":
        ang = rng.uniform(0, 2 * np.pi, N); rr = np.sqrt(rng.random(N)) * 0.35
        return np.stack([rr * np.cos(ang) - 1.6, rr * np.sin(ang)], 1), None
    p = Path(spec)
    if p.suffix.lower() == ".npy":
        pts = np.load(p).astype(np.float64)
        assert pts.ndim == 2 and pts.shape[1] == 2, "npy 点集必须是 (n,2)"
        if len(pts) != N:
            pts = pts[rng.choice(len(pts), N, replace=len(pts) < N)]
        return pts, None
    return image_measure(p, N, rng, **kw)


def problem(src_spec, dst_spec, N, seed=5):
    """返回 src, tgt, heart_world（目标图里胸口的世界坐标；无 uv 时 None）。默认两个都是 still09 → 与静帧 09 逐位相同"""
    if src_spec == "still09" and dst_spec == "still09":
        src, tgt, _ = S.ot_problem(N, seed)
        return src, tgt, O.target_point(O.HEART_UV)
    rng = np.random.default_rng(seed)
    src, _ = load_measure("disc" if src_spec == "still09" else src_spec, N, rng)
    if dst_spec == "still09":
        tgt, uv2w = image_measure(S.SRC_IMG, N, rng, x_shift=0.9)
    else:
        tgt, uv2w = load_measure(dst_spec, N, rng, x_shift=0.0)
    return src, tgt, (uv2w(O.HEART_UV) if uv2w and dst_spec == "still09" else None)


# ---------------------------------------------------------------- 层
def layer_ts(K):
    return np.linspace(0.0, 1.0, K)


def layer_points(src, T, t, jitter=None):
    """位移插值切片：x(t) = (1-t)·x + t·T(x)；jitter = 每粒子 τ 偏移（沿自己的 OT 方向，裁到 [0,1]）"""
    tau = np.full(len(src), float(t)) if jitter is None else np.clip(t + jitter, 0, 1)
    return src + tau[:, None] * (T - src)


# ---------------------------------------------------------------- 相机
def ease(u, a=0.18):
    """快进慢出：g(0)=0, g(1)=1, g'(0)=3-2a（入画已经很快）, g'(1)=a>0（不停），单调递减的速度"""
    u = np.clip(u, 0, 1)
    return a * u + (1 - a) * (1 - (1 - u) ** 3)


def bezier(p0, p1, p2, s):
    s = np.asarray(s)[..., None]
    return (1 - s) ** 2 * p0 + 2 * (1 - s) * s * p1 + s ** 2 * p2


# 从圆盘那头（深处、左后方）沿着这一摞往前掠，收在左前方：始终在层轴的同一侧，不穿过「层全对齐」的视线（那里叠加会过曝）
CAM_C = (np.array([-4.2, -1.2, 6.0]), np.array([-4.6, -0.9, -0.6]), np.array([-1.5, -0.35, -3.1]))
CAM_L = (np.array([-1.0, 0.0, 3.2]), np.array([-0.3, 0.0, 1.8]), np.array([0.05, 0.05, 1.1]))


def camera(u):
    """u∈[0,1] → (位置 C, 注视点 L)。两条二次贝塞尔共用同一个缓动参数 ⇒ 平移和转向一起减速"""
    s = ease(u)
    return bezier(*CAM_C, s), bezier(*CAM_L, s)


def basis(C, L):
    """无滚转相机基（世界 y 朝下，OpenCV 式 x 右 y 下 z 前）：x_c 永远水平"""
    f = L - C; f = f / np.linalg.norm(f)
    x = np.cross([0.0, 1.0, 0.0], f); x /= np.linalg.norm(x)
    y = np.cross(f, x)
    return np.stack([x, y, f])


def project(P3, C, L, focal):
    """世界点 (n,3) → 屏幕 (u, v, 深度)；深度 ≤ 0.15 的给 NaN"""
    R = basis(C, L)
    Q = (P3 - C) @ R.T
    z = Q[:, 2]
    zz = np.where(z > 0.15, z, np.nan)
    return focal * Q[:, 0] / zz + W / 2, focal * Q[:, 1] / zz + H / 2, z


# ---------------------------------------------------------------- 音乐
def music_env(sn, t, beats, downbeats):
    """呼吸包络：rhythm 通道（24fps）+ 每拍一个指数衰减脉冲（强拍更大）。0~1 左右"""
    ch = np.asarray(sn["channels"]["rhythm"], np.float64)
    r = float(np.interp(t * sn["fps"], np.arange(len(ch)), ch))
    past = [b for b in beats if b <= t]
    pb = np.exp(-(t - past[-1]) / 0.12) if past else 0.0
    big = 1.0 if past and min(abs(past[-1] - d) for d in downbeats) < 0.03 else 0.6
    return r, pb * big


# ---------------------------------------------------------------- 主程序
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=float, default=205.11, help="播放时间（感官.json 已是播放时间）")
    ap.add_argument("--dur", type=float, default=4.0)
    ap.add_argument("--src", default="still09", help="still09 | disc | 图片路径 | .npy (n,2)")
    ap.add_argument("--dst", default="still09", help="still09 | 图片路径 | .npy (n,2)")
    ap.add_argument("--N", type=int, default=60000)
    ap.add_argument("--K", type=int, default=9, help="层数（t 均匀取 0..1）")
    ap.add_argument("--dz", type=float, default=0.55, help="层间距（世界单位）")
    ap.add_argument("--amp", type=float, default=0.018, help="抖动的 τ 振幅（随音乐呼吸放大）")
    ap.add_argument("--eps", default="1,0.3,0.1,0.03,0.01,0.003,0.001,0.0003")
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--frames", default="", help="只渲这些帧（逗号）出 png 预览，不出视频")
    ap.add_argument("--name", default="20_最优传输_3D多层_v1")
    ap.add_argument("--outdir", default=str(S.MV / "样片" / "最优传输"))
    ap.add_argument("--tmp", required=True)
    a = ap.parse_args()
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    tmp = Path(a.tmp); tmp.mkdir(parents=True, exist_ok=True)
    sn = json.loads((S.MV / "感官.json").read_text("utf-8"))
    beats, downbeats = sn["events"]["beats"], sn["events"]["downbeats"]

    t0 = time.time()
    src, tgt, heart_w = problem(a.src, a.dst, a.N)
    eps_list = tuple(float(e) for e in a.eps.split(","))
    key = f"{Path(a.src).stem}_{Path(a.dst).stem}_N{a.N}_eps{eps_list[-1]:g}_it{a.iters}"
    cache = tmp / f"ot_map_{key}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True); T = z["T"]; sk_info = list(z["info"])
    else:
        T, sk_info = O.entropic_map(src, tgt, eps_list, a.iters, log=print)
        np.savez(cache, T=T, info=np.array(sk_info, dtype=object))
    W2_0 = O.w2_coupling(src, T)
    print(f"OT {time.time() - t0:.1f}s  W2_0 {W2_0:.4f}")
    ts = layer_ts(a.K)
    zs = (a.K - 1 - np.arange(a.K)) * a.dz          # t=1（她）在最前 z=0，t=0（圆盘）最深
    heart = int(np.argmin(np.linalg.norm(T - heart_w, axis=1))) if heart_w is not None else None
    rng = np.random.default_rng(11)
    phase = rng.uniform(0, 2 * np.pi, (a.K, a.N))   # 每层每粒子自己的相位
    freq = rng.uniform(0.6, 1.4, (a.K, a.N))
    focal = 0.95 * W

    def layer_state(k, tp):
        r, pb = music_env(sn, tp, beats, downbeats)
        A = a.amp * (0.5 + 1.2 * r + 1.5 * pb)
        jit = A * np.sin(phase[k] + 2 * np.pi * 0.9 * freq[k] * (tp - a.start))
        return layer_points(src, T, ts[k], jit), r, pb

    # 层框：每层点的 0.5–99.5 分位外扩
    boxes, area = [], []
    for k in range(a.K):
        P = layer_points(src, T, ts[k])
        lo, hi = np.percentile(P, 0.5, 0) - 0.12, np.percentile(P, 99.5, 0) + 0.12
        boxes.append((lo, hi))
        cells = np.unique(np.floor(P / 0.05).astype(np.int64), axis=0)
        area.append(len(cells) * 0.05 ** 2)
    area = np.array(area) / area[-1]                       # 以「她」那层为 1
    lw = np.where(np.arange(a.K) == a.K - 1, 1.0, 0.6)    # 最前面的她亮一些，其余层退后

    n = int(round(a.dur * FPS))
    SUB = 4                                   # 运动模糊子帧（快门 0.6 帧）

    def render(i):
        tp = a.start + i / FPS
        dens = np.zeros((H, W)); hd = np.zeros((H, W))
        meas = []
        layers = [layer_state(k, tp) for k in range(a.K)]
        r, pb = layers[0][1], layers[0][2]
        for sub in range(SUB):
            u = (i + 0.6 * (sub / SUB - 0.5)) / FPS / a.dur
            C, L = camera(u)
            for k in range(a.K):
                P = layers[k][0]
                P3 = np.column_stack([P, np.full(len(P), zs[k])])
                x, y, z = project(P3, C, L, focal)
                ok = np.isfinite(x)
                dist = np.linalg.norm(np.array([P[:, 0].mean(), 0, zs[k]]) - C)
                # 屏幕密度 ∝ N·z²/(A·f²) ⇒ 权重 ∝ (z_ref/z)²·A_k 让每层表面亮度一致；再乘雾、层权重
                w = np.exp(-max(0.0, dist - 3.0) / 6.0) * (3.5 / z[ok]) ** 2 * area[k] * lw[k] / SUB
                ww = np.full(len(P), 0.0); ww[ok] = w
                if heart is not None and k == a.K - 1:
                    ww[heart] = 0
                    O.splat_bilinear(x[heart:heart + 1], y[heart:heart + 1], np.array([1.0 / SUB]), hd)
                O.splat_bilinear(np.nan_to_num(x[ok], nan=-9), np.nan_to_num(y[ok], nan=-9), ww[ok], dens)
        for k in range(a.K):
            meas.append(O.w2_coupling(layers[k][0], T))
        return tp, dens, hd, meas, r, pb

    # 曝光固定：用末帧定（不逐帧归一化，否则闪）
    _, dref, _, _, _, _ = render(n - 1)
    d0 = np.percentile(dref[dref > 0], 80) * 1.6
    print(f"exposure d0 {d0:.4g}")

    def compose(i):
        tp, dens, hd, meas, r, pb = render(i)
        v = (1 - np.exp(-dens / d0)) ** 0.75
        img = v[..., None] * GREY
        # 层框 + 每层小标签（2× 超采样画线）
        u = i / FPS / a.dur
        C, L = camera(u)
        ov = Image.new("L", (2 * W, 2 * H), 0); dov = ImageDraw.Draw(ov)
        fill = Image.new("L", (2 * W, 2 * H), 0); dfl = ImageDraw.Draw(fill)
        labels = []
        for k in range(a.K):
            lo, hi = boxes[k]
            cs = np.array([[lo[0], lo[1], zs[k]], [hi[0], lo[1], zs[k]], [hi[0], hi[1], zs[k]], [lo[0], hi[1], zs[k]]])
            x, y, _ = project(cs, C, L, focal)
            if not np.all(np.isfinite(x)):
                continue
            pts = [(2 * float(xx), 2 * float(yy)) for xx, yy in zip(x, y)]
            dfl.polygon(pts, fill=10)
            dov.line(pts + [pts[0]], fill=70, width=2)
            labels.append((x[0], y[0], k))
        ov = np.asarray(ov.resize((W, H), Image.LANCZOS), np.float64) / 255
        fl = np.asarray(fill.resize((W, H), Image.BILINEAR), np.float64) / 255
        img = img + (ov[..., None] * 0.55 + fl[..., None] * 0.5) * GREY
        img = img + 0.018 * GREY                        # 底不死黑
        if heart is not None:
            hk = np.clip(S.blur(S.blur(hd, 3), 3) * (60 + 140 * pb), 0, 1)
            img = img * (1 - 0.6 * hk[..., None]) + hk[..., None] * RED * (1.2 + 0.8 * pb)
        img = S.glow(np.clip(img, 0, 1), 5, 0.55)
        img = S.glow(img, 18, 0.18)
        im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        d = ImageDraw.Draw(im)
        f = O.font(11)
        placed = []
        for x0, y0, k in sorted(labels, key=lambda l: -l[2]):
            if 0 <= x0 < W - 150 and 14 <= y0 < H and all(abs(y0 - py) > 13 or abs(x0 - px) > 150 for px, py in placed):
                placed.append((x0, y0))
                d.text((x0 + 3, y0 - 14), f"t {ts[k]:.3f}  W2 {meas[k]:.3f}", font=f, fill=(150, 158, 166))
        f2 = O.font(12)
        dev = max(abs(m - (1 - t) * W2_0) for m, t in zip(meas, ts))
        lines = [f"entropic OT  eps {eps_list[-1]:g}   N = {a.N:,} / layer  x {a.K} layers   W2_0 = {W2_0:.4f}",
                 f"W2(mu_t, nu) measured per layer (live jitter)   max |W2 - (1-t) W2_0| = {dev:.4f}"]
        for j, line in enumerate(lines):
            d.text((24, H - 22 - 16 * (len(lines) - j)), line, font=f2, fill=(140, 148, 156))
        tag = "20_OPTIMAL_TRANSPORT  /  LAYERS"
        d.text((W - 24 - d.textlength(tag, font=f2), 20), tag, font=f2, fill=(110, 116, 122))
        return im, {"i": i, "t_play": round(tp, 4), "W2_layers": meas, "W2_lin": [(1 - t) * W2_0 for t in ts],
                    "rhythm": r, "beat_pulse": pb, "cam_u": u}

    if a.frames:
        for i in [int(s) for s in a.frames.split(",")]:
            im, _ = compose(i)
            im.save(tmp / f"{a.name}_f{i:03d}.png")
            print("preview", i)
        return

    vtmp = tmp / f"{a.name}_noaudio.mp4"
    pipe = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                             "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "15", str(vtmp)],
                            stdin=subprocess.PIPE)
    side = {"play_start": a.start, "dur": n / FPS, "time": "感官.json is in playback time (no offset)",
            "src": a.src, "dst": a.dst, "map": "entropic Brenier map (log-domain Sinkhorn, eps annealing)",
            "eps": eps_list, "sinkhorn": sk_info, "N": a.N, "K": a.K, "layer_t": ts.tolist(), "layer_z": zs.tolist(),
            "W2_0": W2_0, "heart_index": heart, "amp": a.amp,
            "camera": {"C_bezier": [c.tolist() for c in CAM_C], "L_bezier": [c.tolist() for c in CAM_L], "ease_a": 0.18},
            "frames": []}
    thumbs, every = [], max(1, n // 16)
    tf = time.time()
    for i in range(n):
        im, rec = compose(i)
        pipe.stdin.write(np.asarray(im).tobytes())
        side["frames"].append(rec)
        if i % every == 0 or i == n - 1:
            thumbs.append((f"{rec['t_play']:.2f}s  cam u {rec['cam_u']:.2f}", im))
        if i % 12 == 0:
            print(f"  frame {i}/{n}  {time.time() - tf:.0f}s", flush=True)
    pipe.stdin.close(); pipe.wait()
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    out = O.outname(Path(a.outdir) / f"{a.name}.mp4")
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(vtmp), "-ss", f"{a.start:.3f}", "-t", f"{n / FPS:.3f}",
                    "-i", str(S.MUSIC), "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
                    "-t", f"{n / FPS:.3f}", str(out)], check=True)
    Path(str(out) + ".json").write_text(json.dumps(side, ensure_ascii=False, indent=1), "utf-8")
    O.contact(thumbs, out.with_name(out.stem + "_接触表.jpg"))
    print("->", out, f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
