#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🎞样片-最优传输-3D多层：静帧 09 的多层构图搬进 3D —— 不演传输过程，只把位移插值在 K 个 t 上切片，
每片放在不同深度的平面上；片上的粒子「活着」（沿各自的 OT 方向小幅来回，振幅跟音乐呼吸），
相机快进慢出（入画已在动、一路减速、不停；无滚转）—— 层间视差给深度。

两种目标：
  图（默认 still09）：T = 熵 Brenier 映射（ot_flow.entropic_map：对数域 Sinkhorn + ε 退火；T = ∇凸函数
    ⇒ 对所有点对单调 ⇒ 配对对 (μ0, T#μ0) 精确最优）。第 k 层 = μ_{t_k}，W2(μ_{t_k}, ν) = (1-t_k)·W2_0。
    HUD 每层印的是当前帧实测的配对代价（含抖动），不是公式抄写。
    为什么不回切片 OT：切片配对不是循环单调（~1.4% 点对违例），那时 HUD 上的数只是 W2 的上界；
    3D 多层里熵映射的 √ε 平滑（~4px）看不出来 ⇒ 静态目标留 Sinkhorn，HUD 说真话。
  视频（--video）：每一视频帧一个目标 ν_f（亮部采样，同一 seed ⇒ 静止处采样不跳）。
    T_f = 热启动切片 OT：X_0 = sliced(圆盘 → ν_0, 60 刀)，X_f = sliced(X_{f-1} → ν_f, 20 刀)。
    粒子身份逐帧连续 ⇒ 前层的她跟着视频动而不闪；代价：配对 src→X_f 不再保证最优 ⇒ HUD 印「配对代价 c」
    （≥ W2），外加 300 点子样本的精确匈牙利核对差距 %（= 离最优多远，真值）。
    Sinkhorn 每帧重解太贵（6 万点一次 ~260 s，GPU 还被别人占着），用户也说撞衫无所谓 ⇒ 选切片。
    第 k 层再往回错 lag·(K-1-k) 个视频帧：深度 = 传输进度 + 一点时间回声（抬头的动作沿层往后传）。
源 / 目标可插拔：--src / --dst 取 still09（默认）、图片路径、.npy 点集（(n,2)，世界坐标原样用）；--video 覆盖 --dst。
用法：python tools/ot_layers3d.py --tmp <scratchpad> [--start 45.12 --dur 5] [--cam ref] [--video 素材/xx.mp4]
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
def gray_measure(g, N, rng, thresh=0.35, power=2.0, height=2.0, x_shift=0.0):
    """灰度图（0~1，(gh,gw)）亮部 → N 个点（世界坐标：高 = height，居中，再右移 x_shift）。返回 (点, uv→世界)"""
    gh, gw = g.shape
    w = np.clip(g - thresh, 0, 1) ** power
    p = (w / w.sum()).ravel()
    idx = rng.choice(p.size, N, p=p)
    pts = np.stack([idx % gw + rng.random(N), idx // gw + rng.random(N)], 1)
    s = height / gh

    def uv2w(uv):
        return np.array([(uv[0] * gw - gw / 2) * s + x_shift, (uv[1] * gh - gh / 2) * s])
    return (pts - [gw / 2, gh / 2]) * s + [x_shift, 0], uv2w


def image_measure(path, N, rng, grid=(320, 240), **kw):
    """图片亮部 → N 个点。同 stills.ot_problem 的做法"""
    g = np.asarray(Image.open(path).convert("L").resize(grid), np.float64) / 255
    return gray_measure(g, N, rng, **kw)


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


def read_video_gray(path, ff, gw=320, gh=180):
    """视频 → (F, gh, gw) 灰度 0~1"""
    raw = subprocess.run([ff, "-loglevel", "error", "-i", str(path), "-vf", f"scale={gw}:{gh}", "-f", "rawvideo",
                          "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, gh, gw).astype(np.float64) / 255


def sliced_w2(A, B, n_dir=32, seed=0):
    """切片 W2（固定方向，排序配对）：两组等权点的分布差（残差指标）"""
    th = np.random.default_rng(seed).uniform(0, np.pi, n_dir)
    D = np.stack([np.cos(th), np.sin(th)], 1)
    a, b = np.sort(A @ D.T, 0), np.sort(B @ D.T, 0)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def video_maps(src, frames, N, x_shift, iters0=60, iters=20, seed=5, thresh=0.35, power=2.0, height=2.0, log=None):
    """热启动切片 OT：返回 X (F, N, 2)（X[f][i] = 粒子 i 在第 f 帧的终点）、每帧切片残差、uv→世界"""
    X = np.empty((len(frames), N, 2))
    resid = []
    prev, uv2w = None, None
    for f, g in enumerate(frames):
        tgt, uv2w = gray_measure(g, N, np.random.default_rng(seed), thresh=thresh, power=power, height=height,
                                 x_shift=x_shift)
        rng = np.random.default_rng(1000 + f)
        prev = S.sliced_ot(src if prev is None else prev, tgt, iters0 if prev is None else iters, rng)
        X[f] = prev
        resid.append(sliced_w2(prev, tgt))
        if log and f % 24 == 0:
            log(f"  sliced frame {f}/{len(frames)}  residual SW2 {resid[-1]:.4f}")
    return X, resid, uv2w


def lag_index(f, k, K, lag, F):
    """第 k 层（k=K-1 最前）显示的视频帧：f - lag·(K-1-k)，裁到 [0, F-1]"""
    return int(np.clip(f - lag * (K - 1 - k), 0, F - 1))


# ---------------------------------------------------------------- 层
def layer_ts(K):
    return np.linspace(0.0, 1.0, K)


def layer_points(src, T, t, jitter=None):
    """位移插值切片：x(t) = (1-t)·x + t·T(x)；jitter = 每粒子 τ 偏移（沿自己的 OT 方向，裁到 [0,1]）"""
    tau = np.full(len(src), float(t)) if jitter is None else np.clip(t + jitter, 0, 1)
    return src + tau[:, None] * (T - src)


# ---------------------------------------------------------------- 相机
def ease(u, a=0.18):
    """快进慢出：g(0)=0, g(1)=1, g'(0)=3-2a, g'(1)=a>0（不停），单调递减的速度"""
    u = np.clip(u, 0, 1)
    return a * u + (1 - a) * (1 - (1 - u) ** 3)


def bezier(p0, p1, p2, s):
    s = np.asarray(s)[..., None]
    return (1 - s) ** 2 * p0 + 2 * (1 - s) * s * p1 + s ** 2 * p2


CAMS = {
    # v1：从圆盘那头（深处、左后方）沿着这一摞往前掠，收在左前方（用户：转得太多、太快，中段看到背面）
    "v1": {"C": (np.array([-4.2, -1.2, 6.0]), np.array([-4.6, -0.9, -0.6]), np.array([-1.5, -0.35, -3.1])),
           "L": (np.array([-1.0, 0.0, 3.2]), np.array([-0.3, 0.0, 1.8]), np.array([0.05, 0.05, 1.1])), "a": 0.18},
    # ref：守在静帧 09 的机位附近（她在右前、各层往左后退），只做一段轻推 + 小横移，视线几乎不转
    "ref": {"C": (np.array([-2.3, -0.62, -4.0]), np.array([-1.85, -0.48, -3.5]), np.array([-1.5, -0.38, -3.1])),
            "L": (np.array([0.12, 0.06, 1.2]), np.array([0.08, 0.05, 1.15]), np.array([0.05, 0.05, 1.1])), "a": 0.35},
}


def camera(u, preset="ref"):
    """u∈[0,1] → (位置 C, 注视点 L)。两条二次贝塞尔共用同一个缓动参数 ⇒ 平移和转向一起减速"""
    c = CAMS[preset]
    s = ease(u, c["a"])
    return bezier(*c["C"], s), bezier(*c["L"], s)


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
    ap.add_argument("--start", type=float, default=45.12, help="播放时间（感官.json 已是播放时间）")
    ap.add_argument("--dur", type=float, default=5.0)
    ap.add_argument("--src", default="still09", help="still09 | disc | 图片路径 | .npy (n,2)")
    ap.add_argument("--dst", default="still09", help="still09 | 图片路径 | .npy (n,2)")
    ap.add_argument("--video", default="", help="视频目标（相对 MV 目录或绝对路径）；第 i 输出帧 = 视频第 i 帧")
    ap.add_argument("--video_xshift", type=float, default=0.45)
    ap.add_argument("--video_thresh", type=float, default=0.15, help="视频亮度阈值（高了脸中间调没了，只剩轮廓像骷髅）")
    ap.add_argument("--video_power", type=float, default=1.5)
    ap.add_argument("--video_height", type=float, default=1.75)
    ap.add_argument("--lag", type=int, default=2, help="视频：每深一层往回错几帧")
    ap.add_argument("--heart_uv", default="0.62,0.97", help="视频：红点在画面里的固定相对位置（每帧取离它最近的粒子）")
    ap.add_argument("--cam", default="ref", choices=sorted(CAMS))
    ap.add_argument("--N", type=int, default=60000)
    ap.add_argument("--K", type=int, default=9, help="层数（t 均匀取 0..1）")
    ap.add_argument("--dz", type=float, default=0.55, help="层间距（世界单位）")
    ap.add_argument("--amp", type=float, default=0.03, help="抖动的 τ 振幅（随音乐呼吸放大）")
    ap.add_argument("--wfreq", type=float, default=0.7, help="抖动基频 Hz")
    ap.add_argument("--eps", default="1,0.3,0.1,0.03,0.01,0.003,0.001,0.0003")
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--frames", default="", help="只渲这些帧（逗号）出 png 预览，不出视频")
    ap.add_argument("--name", default="20_最优传输_3D多层_v2_参考机位")
    ap.add_argument("--outdir", default=str(S.MV / "样片" / "最优传输"))
    ap.add_argument("--tmp", required=True)
    a = ap.parse_args()
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    tmp = Path(a.tmp); tmp.mkdir(parents=True, exist_ok=True)
    sn = json.loads((S.MV / "感官.json").read_text("utf-8"))
    beats, downbeats = sn["events"]["beats"], sn["events"]["downbeats"]
    n = int(round(a.dur * FPS))
    ts = layer_ts(a.K)
    zs = (a.K - 1 - np.arange(a.K)) * a.dz          # t=1（她）在最前 z=0，t=0（圆盘）最深

    t0 = time.time()
    side = {"play_start": a.start, "dur": n / FPS, "time": "感官.json is in playback time (no offset)",
            "N": a.N, "K": a.K, "layer_t": ts.tolist(), "layer_z": zs.tolist(), "amp": a.amp, "cam": a.cam,
            "camera": {k: (v.tolist() if hasattr(v, "tolist") else [c.tolist() for c in v] if isinstance(v, tuple) else v)
                       for k, v in CAMS[a.cam].items()}}
    if a.video:
        vp = Path(a.video) if Path(a.video).is_absolute() else S.MV / a.video
        frames = read_video_gray(vp, ff)
        F = len(frames)
        src, _ = load_measure("disc", a.N, np.random.default_rng(5))
        cache = tmp / (f"vid_{vp.stem}_N{a.N}_th{a.video_thresh:g}_p{a.video_power:g}_h{a.video_height:g}"
                       f"_x{a.video_xshift:g}.npz")
        if cache.exists():
            z = np.load(cache); X = z["X"]; resid = list(z["resid"])
            _, uv2w = gray_measure(frames[0], 10, np.random.default_rng(0), thresh=a.video_thresh, power=a.video_power,
                                   height=a.video_height, x_shift=a.video_xshift)
        else:
            X, resid, uv2w = video_maps(src, frames, a.N, a.video_xshift, thresh=a.video_thresh, power=a.video_power,
                                        height=a.video_height, log=print)
            np.savez(cache, X=X, resid=np.array(resid))
        heart_w = uv2w(tuple(float(v) for v in a.heart_uv.split(",")))
        # 红点 = 每帧离画面固定位置（锁骨窝）最近的粒子：切片热启动下粒子会滑，固定一个粒子会滑到脸上
        hearts = [int(np.argmin(np.linalg.norm(Xf - heart_w, axis=1))) for Xf in X]
        heart = hearts[0]

        def T_of(k, i):
            return X[lag_index(i, k, a.K, a.lag, F)]
        cost_name = "c"
        rngs = np.random.default_rng(3)
        sub = rngs.choice(a.N, 300, replace=False)
        side.update({"video": str(vp), "video_frames": F, "lag": a.lag, "map": "warm-started sliced OT per video frame "
                     "(60 slices first frame, 20 per later frame); pairing src->X_f is NOT guaranteed optimal: "
                     "'c' = pairing cost >= W2; check = exact Hungarian on 300-pt subsample", "sliced_residual": resid,
                     "heart_index_per_frame": hearts})
        Tref = X[min(n - 1, F - 1)]
        print(f"video maps {time.time() - t0:.1f}s  F {F}  residual SW2 max {max(resid):.4f}")
    else:
        src, tgt, heart_w = problem(a.src, a.dst, a.N)
        eps_list = tuple(float(e) for e in a.eps.split(","))
        key = f"{Path(a.src).stem}_{Path(a.dst).stem}_N{a.N}_eps{eps_list[-1]:g}_it{a.iters}"
        cache = tmp / f"ot_map_{key}.npz"
        if cache.exists():
            z = np.load(cache, allow_pickle=True); T = z["T"]; sk_info = list(z["info"])
        else:
            T, sk_info = O.entropic_map(src, tgt, eps_list, a.iters, log=print)
            np.savez(cache, T=T, info=np.array(sk_info, dtype=object))
        heart = int(np.argmin(np.linalg.norm(T - heart_w, axis=1))) if heart_w is not None else None
        hearts = None

        def T_of(k, i):
            return T
        cost_name = "W2"
        Tref = T
        side.update({"src": a.src, "dst": a.dst, "map": "entropic Brenier map (log-domain Sinkhorn, eps annealing)",
                     "eps": eps_list, "sinkhorn": sk_info, "W2_0": O.w2_coupling(src, T), "heart_index": heart})
        print(f"OT {time.time() - t0:.1f}s  W2_0 {side['W2_0']:.4f}")

    heart0 = heart
    rng = np.random.default_rng(11)
    phase = rng.uniform(0, 2 * np.pi, (a.K, a.N))   # 每层每粒子自己的相位
    freq = rng.uniform(0.6, 1.4, (a.K, a.N))
    focal = 0.95 * W

    def layer_state(k, i, tp):
        r, pb = music_env(sn, tp, beats, downbeats)
        A = a.amp * (0.5 + 1.2 * r + 1.5 * pb)
        jit = A * np.sin(phase[k] + 2 * np.pi * a.wfreq * freq[k] * (tp - a.start))
        return layer_points(src, T_of(k, i), ts[k], jit), r, pb

    # 层面积（亮度补偿用）：参考映射算一次
    area = []
    for k in range(a.K):
        P = layer_points(src, Tref, ts[k])
        area.append(len(np.unique(np.floor(P / 0.05).astype(np.int64), axis=0)) * 0.05 ** 2)
    area = np.array(area) / area[-1]                       # 以「她」那层为 1
    lw = np.where(np.arange(a.K) == a.K - 1, 1.0, 0.6)    # 最前面的她亮一些，其余层退后
    SUB = 4                                               # 运动模糊子帧（快门 0.6 帧）

    def render(i):
        tp = a.start + i / FPS
        heart = hearts[min(i, len(hearts) - 1)] if hearts else heart0
        dens = np.zeros((H, W)); hd = np.zeros((H, W))
        layers = [layer_state(k, i, tp) for k in range(a.K)]
        r, pb = layers[0][1], layers[0][2]
        for sub in range(SUB):
            u = (i + 0.6 * (sub / SUB - 0.5)) / FPS / a.dur
            C, L = camera(u, a.cam)
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
                O.splat_bilinear(x[ok], y[ok], ww[ok], dens)
        meas = [O.w2_coupling(layers[k][0], T_of(k, i)) for k in range(a.K)]
        boxes = []
        for k in range(a.K):
            P = layers[k][0]
            boxes.append((np.percentile(P, 0.5, 0) - 0.12, np.percentile(P, 99.5, 0) + 0.12))
        return tp, dens, hd, meas, r, pb, boxes

    # 曝光固定：用末帧定（不逐帧归一化，否则闪）
    dref = render(n - 1)[1]
    d0 = np.percentile(dref[dref > 0], 80) * 1.6
    print(f"exposure d0 {d0:.4g}")

    def compose(i):
        tp, dens, hd, meas, r, pb, boxes = render(i)
        v = (1 - np.exp(-dens / d0)) ** 0.75
        img = v[..., None] * GREY
        u = i / FPS / a.dur
        C, L = camera(u, a.cam)
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
        if heart0 is not None:
            hk = np.clip(S.blur(S.blur(hd, 3), 3) * (60 + 140 * pb), 0, 1)
            img = img * (1 - 0.6 * hk[..., None]) + hk[..., None] * RED * (1.2 + 0.8 * pb)
        img = S.glow(np.clip(img, 0, 1), 5, 0.55)
        img = S.glow(img, 18, 0.18)
        im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        d = ImageDraw.Draw(im)
        f = O.font(11)
        placed = []
        for x0, y0, k in sorted(labels, key=lambda l: -l[2]):
            if 0 <= x0 < W - 170 and 14 <= y0 < H and all(abs(y0 - py) > 13 or abs(x0 - px) > 170 for px, py in placed):
                placed.append((x0, y0))
                extra = f"  -{a.lag * (a.K - 1 - k)}f" if a.video and k < a.K - 1 else ""
                d.text((x0 + 3, y0 - 14), f"t {ts[k]:.3f}  {cost_name} {meas[k]:.3f}{extra}", font=f, fill=(150, 158, 166))
        f2 = O.font(12)
        rec = {"i": i, "t_play": round(tp, 4), "cost_layers": meas, "rhythm": r, "beat_pulse": pb, "cam_u": u}
        if a.video:
            fi = min(i, F - 1)
            c_sub = O.w2_coupling(src[sub], X[fi][sub])
            e_sub, _ = O.w2_exact(src[sub], X[fi][sub])
            gap = (c_sub - e_sub) / e_sub
            rec.update({"video_frame": fi, "c_sub": c_sub, "w2_sub_exact": e_sub, "sliced_residual": resid[fi]})
            lines = [f"sliced OT, warm-started per video frame {fi:3d}/{F - 1}   N = {a.N:,} / layer  x {a.K} layers   "
                     f"residual SW2 {resid[fi]:.4f}",
                     f"c = pairing cost (>= W2)   check: exact Hungarian on {len(sub)} pts, pairing is {100 * gap:.2f}% above optimal"]
        else:
            W2_0 = side["W2_0"]
            dev = max(abs(m - (1 - t) * W2_0) for m, t in zip(meas, ts))
            rec["W2_lin"] = [(1 - t) * W2_0 for t in ts]
            lines = [f"entropic OT  eps {side['eps'][-1]:g}   N = {a.N:,} / layer  x {a.K} layers   W2_0 = {W2_0:.4f}",
                     f"W2(mu_t, nu) measured per layer (live jitter)   max |W2 - (1-t) W2_0| = {dev:.4f}"]
        for j, line in enumerate(lines):
            d.text((24, H - 22 - 16 * (len(lines) - j)), line, font=f2, fill=(140, 148, 156))
        tag = "20_OPTIMAL_TRANSPORT  /  LAYERS"
        d.text((W - 24 - d.textlength(tag, font=f2), 20), tag, font=f2, fill=(110, 116, 122))
        return im, rec

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
    side["frames"] = []
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
    Path(str(out) + ".json").write_text(json.dumps(side, ensure_ascii=False, indent=1, default=float), "utf-8")
    O.contact(thumbs, out.with_name(out.stem + "_接触表.jpg"))
    print("->", out, f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
