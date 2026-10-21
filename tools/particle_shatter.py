#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🎬样片-粒子-碎裂 v4（学 The Crew 3:13–3:14 的逐帧机理，见 `_设计/方法_剪辑与情绪.md` §四⅞½）：
长静 → 褪色预告 → 单帧前震 → 2 帧横向拖影 → 1 帧无关噪点 → 1 帧白色横线 → 6–8 帧爆散 → 尘埃慢飘。
碎片双尺度：1–2 px 尘埃 + 整行宽的扫描线条块（她本来就是扫描线全息），没有中等碎片；单方向（向右），机位锁死，2D。
所有时刻锚在感官层：冲击 = 用户耳朵定的 drop 拍点；前震 = 冲击前最后一个起音。点数守恒，HUD 写真数。
用法：python tools/particle_shatter.py [--start 47.5] [--dur 7] [--hit 51.51]

v5（`v5` 子命令，👤2026-10-21：3D 版太简陋，回到 v4 的 2D 机理，但源换成上一镜的视频，风格统一）：
  · 源 = 视频（默认 素材/全息女性视频运镜描述_裁肩_v1.mp4），按上一镜的时间窗（--src-play 45.12,51.52）变速映射，
    冲击帧 = 源的最后一帧 = 上一镜的最后一帧（接缝）；遮罩 = BiRefNet（tools/matte.py），没 torch 就亮度阈值。
    place_dx 跟 face_glitch 一致（她在右边）；方向默认向左（接上一镜瀑布图行往左拉的力）。
  · 没有定格：冲击前（--lead 给了才渲）每帧读当前源帧的亮度，前震 / 拖影发生在动的画面上。
  · 运动是闭式解（不是逐帧积分）：x(τ) = x0 + dir·(v·T1·(1−e^{−u/T1}) + w·ramp(u−ts))，u = τ − 释放时刻。
    → 同一个粒子场可以在任意 τ、任意速度（慢放）、任意机位下取样，余波镜头和冲击镜头是同一批点。
  · 脸最后走：脸权重高的点释放时刻 τ_r ∈ [face_t0, face_t1]，按权重 × 抖动排序（边缘先散成颗粒，中心最后）。
  · 余波镜头 = 同一场 + 2D/2.5D 机位：条块推近、跟流横摇、2.5D 拉远视差、脸的慢放回看。镜头表见 SHOTS_V5。
  · 配色跟 17_面部瓦解 瀑布图行 本片：冷灰白 + 唯一的红（轮廓右缘那一圈的点，飞出去仍是红）。
  python tools/particle_shatter.py v5 [--hit 51.52] [--end 57.92] [--lead 1.2] [--N 300000] [--tag v5]
  （在仓库 .venv 里跑才有 BiRefNet；输出不覆盖已有文件，重名就报错）

v5c（👤2026-10-22：上一镜 face_glitch v4 换源 裁肩_v2、头在右 0.64W、原速放完定格到 51.52、以亮的全幅抹除收尾）：
  · --src 裁肩_v2 --place-dx 0 --src-map real：帧 0 的底图 = 源末帧原样（接缝自检 --seam-check 上一镜.mp4 → 均差 + 并排图）
  · --layout v5c（SHOTS_V5C）：冲击帧 0 = 前闪（她在原位重现、过曝 + 抹除余像，τ = 0），帧 1 起炸；不再有噪点帧 / 白线帧
    （上一镜的抹除已经是「信号断了」）。切点 = 下拍 53.12 / 54.72 / 56.32 / 57.94，不重叠；脸只在 余波_4 回看一次，
    余波_1–3 藏起没释放的点（只剩条块 / 尘埃）；余波_1 条块纵向填满 + 只沿 x 模糊 + 黑场；余波_3 曝光跟可见粒子走。
  · 配色：红点单独一层盖在灰上（不混成粉），--red-frac 0.3（红只是轮廓一圈）。--retime 镜头:t0:t1 出换时间窗的替代版。
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


def smooth(t, a, b):
    u = np.clip((t - a) / (b - a), 0, 1)
    return u * u * (3 - 2 * u)


def no_overwrite(*paths):
    """👤规矩：绝不覆盖，重名就停（换版本号）"""
    hit = [str(p) for p in paths if Path(p).exists()]
    if hit:
        raise SystemExit(f"已存在，不覆盖（换个 --tag / --out）：{hit}")


def main_v4(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=float, default=47.5)
    ap.add_argument("--dur", type=float, default=7.0)
    ap.add_argument("--hit", type=float, default=51.51)
    ap.add_argument("--img", default=str(S.MV / "生成" / "角色_她_正面_灰_pro.jpg"))
    ap.add_argument("--N", type=int, default=420000)
    ap.add_argument("--slab", type=float, default=0.35, help="属于扫描线条块的点的比例（其余是尘埃）")
    ap.add_argument("--out", default=str(S.MV / "样片" / "08_粒子_碎裂_v4.mp4"))
    a = ap.parse_args(argv)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    no_overwrite(out)
    W, H = S.W, S.H
    rng = np.random.default_rng(23)

    im = np.asarray(Image.open(a.img).convert("RGB").resize((W, H), Image.LANCZOS)) / 255.0
    L = im.mean(2); lum = np.clip((L - 0.05) / 0.95, 0, 1)
    p = (lum ** 1.2).ravel(); p /= p.sum()
    idx = rng.choice(W * H, size=a.N, p=p)
    sy, sx = np.divmod(idx, W)
    x0 = sx + rng.random(a.N); y0 = sy + rng.random(a.N)
    b = lum.ravel()[idx]; tint = im.reshape(-1, 3)[idx]           # 原色（极淡的青）→ 冲击前褪成纯白
    ys, xs = np.nonzero(lum > 0.25); top, bot = ys.min(), ys.max(); hgt = bot - top
    face = np.exp(-(((x0 - xs.mean()) / 80) ** 2 + ((y0 - (top + 0.11 * hgt)) / 100) ** 2))
    legs = smooth(y0 - (top + 0.70 * hgt), 0, 80)
    hitw = (1 - 0.95 * face) * (1 - 0.8 * legs)                    # 受力权重：脸和腿基本留下

    # ---- 双尺度：条块 = 同一行带（高 2–4 px）的点一起走；尘埃 = 每点各走各的
    band_h = rng.integers(2, 5, size=H)                            # 每行属于多高的条带
    band = np.cumsum(band_h)[np.minimum(np.searchsorted(np.cumsum(band_h), y0.astype(int)), H - 1)] // 1
    _, band = np.unique(band, return_inverse=True); nb = band.max() + 1
    is_slab = (rng.random(nb) < a.slab)[band] & (rng.random(a.N) < 0.95)
    vb = rng.lognormal(np.log(5200), 0.55, nb)                      # 条块要横穿画面（The Crew 的条块飞满全宽）                      # 每条带一个水平速度（向右）
    vx_i = np.where(is_slab, vb[band], rng.lognormal(np.log(1500), 0.7, a.N)) * hitw
    vy_i = np.where(is_slab, rng.normal(0, 12, nb)[band], rng.normal(0, 160, a.N)) * hitw
    gone = hitw > 0.08 + 0.5 * rng.random(a.N) ** 2                # 抖动阈值：脸的边缘也碎成颗粒，不是一块光滑的椭圆（👤：「为啥留一个边缘清晰的脸」）

    sense = json.loads((S.MV / "感官.json").read_text("utf-8"))
    pre = [t for t in sense["events"]["onsets"] if a.hit - 1.6 < t < a.hit - 0.15]
    pre_t = pre[-1] if pre else a.hit - 0.6                         # 前震 = 冲击前最后一个起音

    f = ImageFont.truetype(S.FONT, 18)
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.Popen([ff, "-y", "-loglevel", "error",
                             "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                             "-ss", str(a.start), "-t", str(a.dur), "-i", str(S.MUSIC),
                             "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16",
                             "-c:a", "aac", "-b:a", "256k", "-shortest", str(out)], stdin=subprocess.PIPE)
    n = int(round(a.dur * FPS)); fh = int(round((a.hit - a.start) * FPS))   # 冲击帧号
    x, y = x0.copy(), y0.copy(); vx = np.zeros(a.N); vy = np.zeros(a.N)
    t0w = time.time(); phase = "STILL"
    for i in range(n):
        tt = a.start + i / FPS; k = i - fh
        drain = smooth(tt, a.hit - 1.0, a.hit - 0.1)               # 褪色：原色 → 纯白剪影
        g = tint.mean(1, keepdims=True) * np.array([[1.0, 1.0, 1.02]])
        col = tint * (1 - drain) + g * drain * 1.15                # 褪色：去掉那点青，只略提亮（别过曝成白块）
        img = None
        if k < -2:
            phase = "STILL"
            px, py = x0, y0
            if abs(tt - pre_t) < 0.5 / FPS:                        # 单帧前震：一条带横向错位
                phase = "PRE-SHOCK"
                yb = top + rng.random() * hgt; sel = np.abs(y0 - yb) < 26
                px = x0 + sel * rng.normal(0, 90)
        elif k in (-2, -1):                                         # 2 帧横向拖影
            phase = "SMEAR"
            sm = (k + 3) * 70.0 * hitw
            px = np.concatenate([x0 + sm * u for u in np.linspace(0, 1, 6)]); py = np.tile(y0, 6)
            col = np.tile(col, (6, 1)) if col.ndim == 2 else col
        elif k == 0:                                                # 1 帧无关噪点（信号断了）
            phase = "NOISE"
            nz = rng.random((H // 4, W // 4)) ** 3 * (rng.random((H // 4, 1)) < 0.6)
            v = np.kron(nz, np.ones((4, 4)))
            img = np.clip(v[..., None] * np.array([0.8, 0.82, 0.85]), 0, 1)
        elif k == 1:                                                # 1 帧白色横线
            phase = "FLASH"
            rows = np.zeros(H); r = rng.choice(H, 40, replace=False); rows[r] = rng.random(40)
            core = np.exp(-((np.arange(H) - (top + 0.4 * hgt)) / 30.0) ** 2)
            v = np.clip(rows[:, None] * (0.6 + 0.4 * rng.random((1, W))) + core[:, None] * 1.4, 0, 1)
            img = v[..., None] * np.ones(3)
            vx = vx_i.copy(); vy = vy_i.copy()
        else:                                                       # 爆散（6–8 帧急刹）→ 尘埃慢飘
            phase = "SHATTER" if k < 9 else "DUST"
            damp = 0.80 if k < 9 else 0.93
            vx *= np.where(gone, damp, 0); vy *= np.where(gone, damp, 0)
            if k >= 9:                                              # 尘埃：很慢的漂移 + 一点布朗
                vx += gone * rng.normal(8, 14, a.N); vy += gone * rng.normal(-3, 10, a.N)
            x = x + vx / FPS; y = y + vy / FPS
            px, py = x, y
        if img is None:
            stretch = 1.0
            if phase == "SHATTER":                                  # 条块带运动模糊：沿 x 拉成线
                px = np.concatenate([px - vx / FPS * u for u in np.linspace(0, 1, 4)]); py = np.tile(py, 4)
                stretch = 4.0
            w = np.tile(b, len(px) // a.N) / stretch
            cc = col if col.ndim == 2 and len(col) == len(px) else np.tile(col if col.ndim == 2 else np.ones((a.N, 3)), (len(px) // a.N, 1))
            img = np.stack([S.splat(px, py, weight=w * cc[:, c]) for c in range(3)], -1)
            img = np.clip(img / 1.5, 0, 1) ** 0.8
            img = S.glow(img, 3, 0.25 if phase != "SHATTER" else 0.5)
        fr = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        dr = ImageDraw.Draw(fr)
        moved = int(((np.hypot(x - x0, y - y0) > 6) & gone).sum()) if k > 1 else 0
        hud = [f"t = {tt:6.2f} s   {phase:<9}   pre-shock @ {pre_t:.2f}   impact @ {a.hit:.2f}",
               f"N = {a.N:,}   body {a.N - moved:7,}   dust {int((moved and (~is_slab & gone).sum()) or 0):7,}   slabs {int((moved and (is_slab & gone).sum()) or 0):7,}   mass in = mass out"]
        if phase not in ("NOISE", "FLASH"):
            for jj, line in enumerate(hud):
                dr.text((48, H - 48 - 26 * (len(hud) - jj)), line, font=f, fill=(150, 160, 168))
        proc.stdin.write(fr.tobytes())
        if i % 48 == 0:
            print(f"frame {i}/{n}  {time.time() - t0w:.0f}s  {phase}", flush=True)
    proc.stdin.close(); proc.wait()
    print("->", out)


# ======================================================================================
# v5：视频源 + 闭式运动的粒子场 + 余波机位
# ======================================================================================
T1 = 1.0 / (FPS * (1 - 0.80))         # 爆散急刹（v4：每帧 ×0.80）。取「总抛距相等」：v·T1 = v/FPS·Σ0.8^k → T1 ≈ 0.208 s
T2 = 1.0 / (FPS * (1 - 0.93))         # 尘埃漂移（v4：每帧 ×0.93），同样按总量相等 ≈ 0.595 s
TS = 9.0 / FPS                         # 释放后多久开始漂（v4：k ≥ 9）

# 镜头表（播放时间，秒）。冲击 = 剪辑点；余波每镜 0.4–1.6 s，切点全在 0.4 s 拍格上（下拍 53.12 / 54.72 / 56.32）。
# tau0 / speed：这一镜从粒子场的哪个 τ 开始、以几倍速走；cam：机位（2D 平移缩放 + 可选 2.5D 视差）
SHOTS_V5 = [
    dict(name="冲击", t0=51.52, t1=53.12, tau0=0.0, speed=1.0, cam="lock", events=True),
    dict(name="余波_1", t0=53.12, t1=53.52, tau0=0.05, speed=1.0, cam="strips"),     # 条块推近，重看一次爆散
    dict(name="余波_2", t0=53.52, t1=54.72, tau0=0.10, speed=1.0, cam="pan"),        # 跟着流横摇
    dict(name="余波_3", t0=54.72, t1=56.32, tau0=0.20, speed=1.0, cam="wide25d"),    # 拉远 + 2.5D 视差：条块拖出画外多远
    dict(name="余波_4", t0=56.32, t1=57.92, tau0=0.28, speed=0.66, cam="face"),      # 脸最后走的慢放回看
]

# v5c（👤2026-10-22）：切点 = 下拍 53.12 / 54.72 / 56.32 / 57.94，互不重叠（v5b 的 余波_3 延长 54.72–57.82 压在 余波_4 上）。
# 脸只回看一次（余波_4）；余波_1–3 藏起还没释放的点（= 脸）→ 只剩条块 / 尘埃。冲击帧 0 = 前闪（见 V5.frame）。
_ST = dict(red_sep=True, bp=0.03)
SHOTS_V5C = [
    dict(name="冲击", t0=51.52, t1=53.12, tau0=0.0, speed=1.0, cam="lock", events="preflash", style=dict(_ST)),
    dict(name="余波_1", t0=53.12, t1=53.52, tau0=0.05, speed=1.0, cam="strips", hide_held=True, shutter_k=0.6,
         style=dict(_ST, bp=0.06, dust_keep=0.12, dust_gain=0.7, slab_gain=2.2, vfill=True, hblur=True, red_k=0.5)),
    dict(name="余波_2", t0=53.52, t1=54.72, tau0=0.10, speed=1.0, cam="pan", hide_held=True,
         style=dict(_ST, bp=0.04, dust_keep=0.5, slab_gain=1.5, hblur=True, red_k=0.5)),
    dict(name="余波_3", t0=54.72, t1=56.32, tau0=0.20, speed=1.0, cam="wide25d", hide_held=True,
         style=dict(_ST, auto=(0.85, 3.5), red_k=0.5)),
    dict(name="余波_4", t0=56.32, t1=57.94, tau0=0.28, speed=0.66, cam="face", style=dict(_ST)),
]


def n_frames(t0, t1):
    return int(round((t1 - t0) * FPS))


def src_index(t, play0, play1, n_src, mode="stretch"):
    """播放时间 → 源帧号。stretch：源整段线性映射到 [play0, play1)；real（v5c）：原速从 play0 放，放完定格在最后一帧
    （= face_glitch v4 的「原速 + 结尾定格到 51.52」）。t ≥ play1 → 最后一帧（冲击帧 = 上一镜的末帧）"""
    if t >= play1 - 1e-9:
        return n_src - 1
    if mode == "real":
        return int(np.clip(np.floor((t - play0) * FPS + 1e-6), 0, n_src - 1))
    u = (t - play0) / (play1 - play0)
    return int(np.clip(np.floor(u * n_src + 1e-6), 0, n_src - 1))


def read_frames(path, idxs, crop=None, size=None):
    """读源视频里指定的帧（uint8 RGB），可选静态裁切 (x, y, w, h) + 缩放到 size"""
    import imageio_ffmpeg
    want = set(idxs); out = {}
    rd = imageio_ffmpeg.read_frames(str(path))
    meta = next(rd); w, h = meta["size"]
    try:
        for i, buf in enumerate(rd):
            if i in want:
                im = Image.frombytes("RGB", (w, h), buf)
                if crop:
                    x, y, cw, ch = crop; im = im.crop((x, y, x + cw, y + ch))
                if size and im.size != tuple(size):
                    im = im.resize(tuple(size), Image.LANCZOS)
                out[i] = np.asarray(im)
            if i >= max(want):
                break
    finally:
        rd.close()
    return out


def count_frames(path):
    import imageio_ffmpeg
    return imageio_ffmpeg.count_frames_and_secs(str(path))[0]


def get_mask(img_u8, use_birefnet=True):
    """BiRefNet → 0/1 遮罩；没有 torch 就亮度阈值（face_glitch.lum_mask）"""
    import face_glitch as F
    if use_birefnet:
        try:
            import matte
            al = np.asarray(matte.Matter()(Image.fromarray(img_u8)), np.float32) / 255
            return (al > 0.5).astype(np.float32), "BiRefNet"
        except Exception as e:   # noqa: BLE001  —— 说出来，不静默
            print(f"BiRefNet 不可用（{type(e).__name__}: {e}），退回亮度阈值", flush=True)
    return F.lum_mask(img_u8.astype(np.float32) / 255), "luminance"


def face_auto(mask):
    """胸像里脸的中心与尺度（粗估，--face 可覆盖）：头顶 top；头宽 hw = 头顶往下 1/3 段的行宽中位（含头发）；
    头高 ≈ 1.2·hw（下巴 = top + 1.2·hw）；脸心 ≈ 鼻子 = top + 0.62·头高、x = 头部质心。
    （v5 第一版用「行宽突增 = 肩线」找下巴，被蓬的头发骗了，脸心落在额头 → 嘴先散）"""
    H, W = mask.shape
    rows = np.nonzero(mask.sum(1) > 2)[0]
    top = int(rows.min()) if len(rows) else 0
    wid = (mask > 0.5).sum(1)
    head = wid[top:top + max(8, (H - top) // 3)]
    hw = float(np.median(head[head > 0])) if (head > 0).any() else W / 4
    chin = int(min(H - 1, top + 1.2 * hw))
    yy, xx = np.nonzero(mask[top:chin] > 0.5)
    cx = float(xx.mean()) if len(xx) else W / 2
    cy = top + 0.62 * (chin - top)
    return dict(cx=cx, cy=cy, sx=0.30 * hw, sy=0.42 * hw, top=top, chin=chin, head_w=hw)


class Field:
    """一次采样、闭式运动的粒子场。所有随机量在构造时抽完 → pos(τ) 是确定的纯函数（慢放 / 回看 = 同一批点）。"""

    def __init__(self, img, mask, rng, N=300000, slab=0.35, direction=-1, face=None,
                 face_t=(0.30, 1.25), scale=None, red_ring=5, red_frac=0.7):
        import face_glitch as F
        H, W = mask.shape
        self.H, self.W, self.N, self.dir = H, W, N, direction
        k = scale if scale is not None else W / 1920.0                 # v4 的速度是 1920 宽下定的
        self.k = k
        L = F.lum(img.astype(np.float32))
        lum = np.clip((L - 0.03) / 0.97, 0, 1)
        p = (lum ** 1.2 * (mask > 0.5)).ravel().astype(np.float64)
        p /= p.sum()
        self.idx = rng.choice(W * H, size=N, p=p)
        sy, sx = np.divmod(self.idx, W)
        self.sx, self.sy = sx, sy
        self.x0 = sx + rng.random(N); self.y0 = sy + rng.random(N)
        self.b = lum.ravel()[self.idx]
        # 唯一的红：轮廓右缘内侧 red_ring px 那一圈（17 的红边在她右外侧；这里取内侧的点，飞出去仍是红）
        nx, _, _ = F.mask_normal(mask, blur=15)
        from PIL import ImageFilter
        er = np.asarray(Image.fromarray(((mask > 0.5) * 255).astype(np.uint8)).filter(
            ImageFilter.MinFilter(2 * red_ring + 1)), np.float32) / 255
        rim = ((mask > 0.5) & (er < 0.5) & (nx > 0.25)).ravel()[self.idx]
        self.red = rim & (rng.random(N) < red_frac)                    # 同一次抽样，只改阈值（v5c 0.3：红只是轮廓的一圈，不是满屏红点）
        # 脸
        self.face_info = face or face_auto(mask)
        fi = self.face_info
        self.face = np.exp(-(((self.x0 - fi["cx"]) / fi["sx"]) ** 2 + ((self.y0 - fi["cy"]) / fi["sy"]) ** 2))
        hitw = 1 - 0.95 * self.face
        # 双尺度：条块 = 同一行带（2–4 px 高）一起走；尘埃各走各的（v4 原样）
        band_h = rng.integers(2, 5, size=H)
        edges = np.cumsum(band_h)
        band = np.minimum(np.searchsorted(edges, self.y0.astype(int), side="right"), H - 1)
        _, band = np.unique(band, return_inverse=True); nb = band.max() + 1
        self.band, self.nb = band, nb
        self.is_slab = (rng.random(nb) < slab)[band] & (rng.random(N) < 0.95)
        vb = rng.lognormal(np.log(5200 * k), 0.55, nb)
        self.vx = np.where(self.is_slab, vb[band], rng.lognormal(np.log(1500 * k), 0.7, N))
        self.vy = np.where(self.is_slab, rng.normal(0, 12 * k, nb)[band], rng.normal(0, 160 * k, N))
        # 释放时刻：脸以外 = 冲击那一刻；脸 = 抖动阈值（边缘先散成颗粒，中心最后）
        f0, f1 = face_t
        is_face = self.face > 0.10 + 0.35 * rng.random(N) ** 2
        q = np.clip(self.face ** 0.8 * (0.7 + 0.6 * rng.random(N)), 0, 1)
        self.tr = np.where(is_face, f0 + (f1 - f0) * q, 0.0)
        self.is_face = is_face
        sc = np.where(is_face, np.maximum(hitw, 0.45), hitw)            # 脸上的点走得慢一点（剥落，不是炸）
        self.vx *= sc; self.vy *= sc
        # 漂移（v4：vx += N(8,14)·×0.93/帧 → 稳态 ≈ 106 px/s；vy += N(−3,10) → ≈ −43 px/s 上飘）
        self.wx = rng.normal(106, 38, N) * k
        self.wy = rng.normal(-43, 27, N) * k
        # 留在原地的脸：冲击后整行的余震（衰减）
        self.jit_a = rng.normal(0, 4.0 * W / 1280, nb)
        self.jit_ph = rng.random(nb) * 2 * np.pi
        # 2.5D 用的深度：条块按行带、尘埃各自
        self.z = np.where(self.is_slab, rng.uniform(0.75, 1.5, nb)[band], rng.uniform(0.6, 1.8, N))

    # ---- 闭式运动 ----
    def released(self, tau):
        return tau >= self.tr

    def pos(self, tau):
        """τ = 冲击后秒数 → (x, y, alpha)。τ < 0：原位。"""
        u = np.maximum(tau - self.tr, 0.0)
        e1 = 1 - np.exp(-u / T1)
        q = np.maximum(u - TS, 0.0)
        r2 = q - T2 * (1 - np.exp(-q / T2))
        x = self.x0 + self.dir * (self.vx * T1 * e1 + self.wx * r2)
        y = self.y0 + self.vy * T1 * e1 + self.wy * r2
        if tau > 0:                                                    # 还没走的点：整行余震（2 帧内最大，0.3 s 衰减）
            hold = u <= 0
            j = self.jit_a[self.band] * np.exp(-tau / 0.3) * np.cos(2 * np.pi * 9 * tau + self.jit_ph[self.band])
            x = x + hold * j
        alpha = np.where(u > 1.2, np.exp(-(u - 1.2) / 2.5), 1.0)        # 尘埃慢慢暗下去
        return x, y, alpha

    def displacement_median(self, tau, which="slab"):
        sel = self.is_slab if which == "slab" else ~self.is_slab
        x, _, _ = self.pos(tau)
        return float(np.median((x - self.x0)[sel & (self.tr == 0)]))


def splat_bilinear(x, y, w, W, H):
    """点 → 密度图（双线性分到 4 个像素，np.bincount；越界的点丢掉）。返回 (img, 落在画内的权重和)"""
    ok = (x >= 0) & (x < W - 1) & (y >= 0) & (y < H - 1)
    x, y, w = x[ok], y[ok], w[ok]
    ix = x.astype(np.int64); iy = y.astype(np.int64)
    fx = x - ix; fy = y - iy
    i0 = iy * W + ix
    idx = np.concatenate([i0, i0 + 1, i0 + W, i0 + W + 1])
    ww = np.concatenate([w * (1 - fx) * (1 - fy), w * fx * (1 - fy), w * (1 - fx) * fy, w * fx * fy])
    return np.bincount(idx, ww, minlength=W * H).reshape(H, W), float(w.sum())


class Camera:
    """世界（源画面像素）→ 屏幕。2D：中心 (cx, cy) + 缩放 zoom；2.5D：par > 0 时按粒子深度 z 做视差
    （刚释放的点都在 z = 1 平面，飞得越久越分层 → 接缝处和 2D 完全一样）。"""

    def __init__(self, W, H, cx=None, cy=None, zoom=1.0, par=0.0, zc=0.0):
        self.W, self.H = W, H
        self.cx = W / 2 if cx is None else cx
        self.cy = H / 2 if cy is None else cy
        self.zoom, self.par, self.zc = zoom, par, zc

    def project(self, x, y, field=None, u=None):
        s = np.full(len(x), self.zoom)
        self.vis = None
        if self.par and field is not None:
            ze = 1 + (field.z - 1) * np.clip(u / 0.6, 0, 1) * self.par
            s = self.zoom / np.maximum(ze - self.zc, 0.2)
            # 机位推过去的点：ze − zc < 0.2 剔除，[0.2, 0.25] 渐隐（原版 余波_3 的 zc ≤ 0.35、ze ≥ 0.6 → 永远 ≥ 0.25，不受影响；
            # 只有「延长」把推轨继续往前时才用得上 = 穿过尘埃）
            self.vis = np.clip((ze - self.zc - 0.2) / 0.05, 0, 1)
        X = self.W / 2 + (x - self.cx) * s                               # 2.5D：深度不同 → 同样的机位移动、屏幕位移不同（视差）
        Y = self.H / 2 + (y - self.cy) * s
        return X, Y, s


def grey_rgb(b, red):
    g = np.asarray(S.GREY) / max(S.GREY)
    col = b[:, None] * g[None]
    return np.where(red[:, None], b[:, None] * np.asarray(S.RED)[None] * 1.1, col)


def hblur(img, r):
    """只沿 x 的盒式模糊（扫描线：行与行之间保持锐利）"""
    if r < 1:
        return img
    p = np.pad(img, [(0, 0), (r + 1, r)] + [(0, 0)] * (img.ndim - 2), mode="edge")
    acc = np.cumsum(p, axis=1)
    return (acc[:, 2 * r + 1:] - acc[:, :-2 * r - 1]) / (2 * r + 1)


def render_field(field, tau, cam, shutter=0.0, gain=1.0, plate=None, plate_a=0.0, cur_b=None, extra_dx=None, style=None):
    """一帧。shutter = 快门（τ 秒）：按屏幕位移自适应取 K 个子样本拉出运动模糊（条块被拉成线）。
    style（v5c，缺省 = v5/v5b 原样）：
      sel        bool[N]：只画这些点（余波里藏起还没释放的脸 → 只剩条块 / 尘埃）
      dust_gain / slab_gain：两种尺度分别的亮度；vfill：推近时条块按屏幕缩放纵向填满行带（条块是实的，不是稀疏针点）
      hblur      推近时只沿 x 模糊（行锐利）；bp：黑场（压掉灰雾）
      red_sep    红点单独一层、按覆盖度盖在灰上（不和灰混成粉）
      auto       (target, gmax, state)：按画内可见粒子的 99.5 分位把曝光拉回 target（倍率 ∈ [1, gmax]，EMA 平滑）"""
    st = style or {}
    W, H = cam.W, cam.H
    b = field.b if cur_b is None else cur_b
    x, y, al = field.pos(tau)
    if extra_dx is not None:
        x = x + extra_dx
    u = np.maximum(tau - field.tr, 0)
    X, Y, s = cam.project(x, y, field, u)
    vis = cam.vis                                                       # 2.5D 剔除（取当前时刻，不取快门子样本）
    K = 1
    if shutter > 0:
        xp, yp, _ = field.pos(tau - shutter)
        Xp, Yp, _ = cam.project(xp, yp, field, np.maximum(tau - shutter - field.tr, 0))
        d = np.hypot(X - Xp, Y - Yp)
        K = int(np.clip(np.ceil(np.percentile(d, 99) / 2.0), 1, 12))
    red_sep = st.get("red_sep", False)
    col = grey_rgb(b, field.red) if not red_sep else grey_rgb(b, np.zeros(field.N, bool))
    wt = al * gain * s ** 2 / K                                        # 推近时保持表面亮度（点数密度 ∝ 1/zoom²）
    if vis is not None:
        wt = wt * vis
    if st.get("wmul") is not None:
        wt = wt * st["wmul"]
    if "dust_gain" in st or "slab_gain" in st:
        wt = wt * np.where(field.is_slab, st.get("slab_gain", 1.0), st.get("dust_gain", 1.0))
    sel = st.get("sel")
    zm = float(np.median(s))
    nv = int(np.clip(np.ceil(zm), 1, 6)) if st.get("vfill") and zm > 1.3 else 1
    acc = np.zeros((H, W, 3)); racc = np.zeros((H, W))
    for kk in range(K):
        if K > 1:
            ts_ = tau - shutter * kk / (K - 1)
            xs, ys, _ = field.pos(ts_)
            if extra_dx is not None:
                xs = xs + extra_dx
            Xs, Ys, _ = cam.project(xs, ys, field, np.maximum(ts_ - field.tr, 0))
        else:
            Xs, Ys = X, Y
        if sel is not None:
            Xs, Ys, w_k, c_k, r_k, sl_k = Xs[sel], Ys[sel], wt[sel], col[sel], field.red[sel], field.is_slab[sel]
        else:
            w_k, c_k, r_k, sl_k = wt, col, field.red, field.is_slab
        if nv > 1:                                                      # 条块纵向填满（屏幕上一行源像素 = zm 行）
            offs = (np.arange(nv) - (nv - 1) / 2) * (zm / nv)
            Xs = np.concatenate([Xs] + [Xs[sl_k]] * nv); Ys = np.concatenate([Ys] + [Ys[sl_k] + o for o in offs])
            w_k = np.concatenate([np.where(sl_k, 0.0, w_k)] + [w_k[sl_k] / nv] * nv)
            c_k = np.concatenate([c_k] + [c_k[sl_k]] * nv); r_k = np.concatenate([r_k] + [r_k[sl_k]] * nv)
        for c in range(3):
            ww = w_k * c_k[:, c] * (~r_k if red_sep else 1)
            d_, w_in = splat_bilinear(Xs, Ys, ww, W, H)
            acc[..., c] += d_
        if red_sep:
            d_, _ = splat_bilinear(Xs[r_k], Ys[r_k], w_k[r_k] * np.maximum(c_k[r_k].max(1), 0.35), W, H) if r_k.any() else (0, 0)
            racc += d_
    inside = int(((X >= 0) & (X < W - 1) & (Y >= 0) & (Y < H - 1)).sum())
    if zm > 1.3:                                                       # 推近：点长成小斑，不是稀疏的针点
        r = int(round(min(4, (zm - 1) * 0.8)))
        if st.get("hblur"):
            acc = hblur(acc, 2 * r); racc = hblur(racc, 2 * r)
        else:
            acc = S.blur(acc, r); racc = S.blur(racc, r) if red_sep else racc
    img = np.clip(acc, 0, None)
    au = st.get("auto")
    if au:                                                             # 曝光跟着可见粒子走（尘埃散开 / 变暗时不掉进黑）
        target, gmax, state = au
        L = img.mean(2); nz = L[L > 1e-4]
        p = float(np.percentile(nz, 99.5)) if nz.size > 100 else target
        m = float(np.clip(target / max(p, 1e-6), 1.0, gmax))
        m = m if state.get("m") is None else state["m"] + 0.3 * (m - state["m"])
        state["m"] = m; img = img * m; racc = racc * m
    if plate is not None and plate_a > 0:
        img = img + plate * plate_a
    bp = st.get("bp", 0.0)
    img = np.clip((np.clip(img, 0, 1) - bp) / (1 - bp), 0, 1) ** 0.8
    if red_sep:                                                        # 唯一的红：盖在灰上（覆盖度 a），保持纯红
        a = np.clip(racc * 6.0 * st.get("red_k", 1.0), 0, 1)[..., None]
        img = img * (1 - a) + np.asarray(S.RED)[None, None] * a
    img = S.glow(img, 3, 0.25 + 0.25 * (K > 3))
    return img, inside


def noise_frame(W, H, rng):
    nz = rng.random((H // 4, W // 4)) ** 3 * (rng.random((H // 4, 1)) < 0.6)
    v = np.kron(nz, np.ones((4, 4)))[:H, :W]
    return np.clip(v[..., None] * np.array([0.8, 0.82, 0.85]), 0, 1)


def flash_frame(W, H, rng, yc):
    rows = np.zeros(H); r = rng.choice(H, int(40 * H / 1080), replace=False); rows[r] = rng.random(len(r))
    core = np.exp(-((np.arange(H) - yc) / (30.0 * H / 1080)) ** 2)
    v = np.clip(rows[:, None] * (0.6 + 0.4 * rng.random((1, W))) + core[:, None] * 1.4, 0, 1)
    return v[..., None] * (np.asarray(S.GREY) / max(S.GREY))[None, None]


def streak_echo(img, L, direction, rng):
    """上一镜「抹除」的余像（v5c 冲击前闪）：每行把亮的往流的方向拖，指数衰减 exp(−d/L)（跟 face_glitch even 抹除同一规律），
    行增益 U(0.55, 1) 做出扫描线条理。返回拖影层（不含原图）。"""
    dec = np.exp(-1.0 / max(L, 1.0))
    src = img if direction < 0 else img[:, ::-1]
    out = np.zeros_like(src); run = np.zeros_like(src[:, 0])
    for c in range(src.shape[1] - 1, -1, -1):                          # 向左拖：从右往左扫
        run = np.maximum(src[:, c], run * dec); out[:, c] = run
    out = out if direction < 0 else out[:, ::-1]
    return out * (0.55 + 0.45 * rng.random((img.shape[0], 1, 1)))


def shot_camera(shot, field, tau, frac):
    """镜头表里的机位名 → Camera（frac = 镜头内进度 0..1）"""
    W, H, fi = field.W, field.H, field.face_info
    c = shot["cam"]
    if c == "lock":
        return Camera(W, H)
    if c == "strips":                                                  # 推近到条块：脸左侧（流的方向）、颈肩高度
        cx = fi["cx"] + field.dir * 0.55 * fi["head_w"] * 1.6
        cy = fi["chin"] + 0.08 * H
        return Camera(W, H, cx - field.dir * 18 * frac, cy, zoom=2.6 + 0.25 * frac)
    if c == "pan":                                                     # 跟流：机位追条块中位位移的 55%
        d = field.displacement_median(tau, "slab")
        return Camera(W, H, fi["cx"] + 0.55 * d, H / 2 + 0.05 * H, zoom=1.45)
    if c == "wide25d":                                                 # 换景别：拉远 + 2.5D 推轨（深度分层 → 视差），机位往流的方向挪
        return Camera(W, H, fi["cx"] + field.dir * (0.20 + 0.25 * frac) * W, fi["cy"] + 0.05 * H,
                      zoom=0.80, par=1.0, zc=-0.10 + 0.45 * frac)
    if c == "face":                                                    # 脸的慢放：推近并缓推
        return Camera(W, H, fi["cx"] + field.dir * 0.12 * fi["head_w"], fi["cy"], zoom=1.75 + 0.3 * frac)
    raise ValueError(c)


def contact_sheet(thumbs, path, cols=4, tw=480):
    """thumbs = [(PIL 图, 标签)] → 拼一张接触表"""
    th = int(tw * thumbs[0][0].size[1] / thumbs[0][0].size[0])
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * th), (10, 10, 12))
    f = ImageFont.truetype(S.FONT, 13); d = ImageDraw.Draw(sheet)
    for i, (im, lab) in enumerate(thumbs):
        x, y = (i % cols) * tw, (i // cols) * th
        sheet.paste(im.resize((tw, th), Image.LANCZOS), (x, y))
        d.text((x + 6, y + 4), lab, font=f, fill=(220, 220, 220))
    sheet.save(path, quality=90)


class V5:
    """把源、遮罩、粒子场、时间轴装在一起；render_range 渲任意播放时间段（冲击 / 余波 / 连续 / 引入）"""

    def __init__(self, a):
        import face_glitch as F
        self.a = a
        src = Path(a.src)
        n_src = count_frames(src)
        self.n_src = n_src
        self.play0, self.play1 = a.src_play
        self.hit = a.hit
        crop = tuple(a.crop) if a.crop else None
        size = (a.W, a.H)
        lead_t = [self.hit - a.lead + i / FPS for i in range(n_frames(self.hit - a.lead, self.hit))] if a.lead > 0 else []
        self.src_map = getattr(a, "src_map", "stretch")
        idxs = sorted({src_index(t, self.play0, self.play1, n_src, self.src_map) for t in lead_t} | {n_src - 1})
        fr = read_frames(src, idxs, crop, size)
        self.frames = {i: F.place(fr[i].astype(np.float32) / 255, np.zeros(fr[i].shape[:2], np.float32), a.place_dx)[0]
                       for i in idxs}
        hit_u8 = fr[n_src - 1]
        self.hit_u8 = hit_u8                                           # 接缝自检用：干净的源末帧（= 上一镜定格的那一帧）
        mask, self.mask_kind = get_mask(hit_u8, not a.no_birefnet)
        img, mask = F.place(hit_u8.astype(np.float32) / 255, mask, a.place_dx)
        self.img, self.mask = img, mask
        rng = np.random.default_rng(a.seed)
        self.field = Field(img, mask, rng, N=a.N, slab=a.slab, direction=a.dir, face_t=tuple(a.face_t),
                           red_frac=getattr(a, "red_frac", 0.7))
        self.W, self.H = mask.shape[1], mask.shape[0]
        # 背景（顶光锥）：遮罩外的源画面，灰化；冲击后 0.2 s 熄掉
        g = F.lum(img)[..., None] * (np.asarray(S.GREY) / max(S.GREY))[None, None]
        self.plate = g * (1 - mask[..., None]) * 0.9
        # 曝光：冲击帧（静止场）的 99.5 分位 = 1（全程固定，不逐帧自动）
        cam = Camera(self.W, self.H)
        x, y, _ = self.field.pos(0.0)
        d, _ = splat_bilinear(x, y, self.field.b, self.W, self.H)
        self.gain = 1.0 / max(np.percentile(d[mask > 0.5], 99.5), 1e-6)
        sense = json.loads((S.MV / "感官.json").read_text("utf-8"))      # 播放时间，不加偏移
        pre = [t for t in sense["events"]["onsets"] if self.hit - 1.6 < t < self.hit - 0.15]
        self.pre_t = pre[-1] if pre else self.hit - 0.6
        self.font = ImageFont.truetype(S.FONT, 12)
        self.flash_rng_seed = a.seed + 7
        self.ru = np.random.default_rng(a.seed + 99).random(self.field.N)   # 尘埃稀疏化用（确定的，不同镜头同一批点）
        self.auto_state = {}; self._shot_name = None
        del cam

    def cur_b(self, t):
        """冲击前：粒子位置不动，亮度读当前源帧（她还在动）"""
        import face_glitch as F
        i = src_index(t, self.play0, self.play1, self.n_src, self.src_map)
        L = F.lum(self.frames[i])
        return np.clip((L - 0.03) / 0.97, 0, 1).ravel()[self.field.idx]

    def frame(self, t, shot):
        """播放时间 t、所在镜头 → (RGB float 图, 阶段名, τ, 画内点数)"""
        fld, W, H = self.field, self.W, self.H
        if shot is None:                                               # 冲击前的引入段
            import face_glitch as F
            k = int(np.floor((t - self.hit) * FPS + 1e-6))
            i = src_index(t, self.play0, self.play1, self.n_src, self.src_map)
            cb = self.cur_b(t)
            plate = F.lum(self.frames[i])[..., None] * (np.asarray(S.GREY) / max(S.GREY))[None, None] * (1 - self.mask[..., None]) * 0.9
            cam = Camera(W, H)
            dx = None; phase = "LIVE"
            if abs(t - self.pre_t) < 0.5 / FPS:                         # 单帧前震：一条带横向错位（在动的画面上）
                rng = np.random.default_rng(int(t * 1000))
                yb = fld.face_info["top"] + rng.random() * (H - fld.face_info["top"])
                dx = (np.abs(fld.y0 - yb) < 26 * H / 1080) * rng.normal(0, 90 * W / 1920); phase = "PRE-SHOCK"
            if k in (-2, -1):                                          # 2 帧横向拖影
                phase = "SMEAR"; hitw = 1 - 0.95 * fld.face
                acc = None
                for uu in np.linspace(0, 1, 6):
                    im, n_in = render_field(fld, -1.0, cam, gain=self.gain / 6 * 1.6, cur_b=cb,
                                            extra_dx=fld.dir * (k + 3) * 70 * fld.k * hitw * uu)
                    acc = im if acc is None else np.maximum(acc, im)
                return np.clip(acc + plate * 0.6, 0, 1), phase, t - self.hit, N_in(fld, n_in)
            im, n_in = render_field(fld, -1.0, cam, gain=self.gain, plate=plate, plate_a=1.0, cur_b=cb, extra_dx=dx)
            return im, phase, t - self.hit, N_in(fld, n_in)
        tau = shot["tau0"] + (t - shot["t0"]) * shot["speed"]
        frac = (t - shot["t0"]) / shot.get("dur0", shot["t1"] - shot["t0"])   # 延长时 frac > 1：机位按原速度继续走，不归零
        st = dict(shot.get("style") or {})
        if shot["name"] != self._shot_name:                            # 自动曝光的 EMA 每镜重来（单镜渲 = 连续渲）
            self.auto_state = {}; self._shot_name = shot["name"]
        if "auto" in st:
            st["auto"] = (st["auto"][0], st["auto"][1], self.auto_state)
        sel = None
        if shot.get("hide_held"):                                      # 只剩条块 / 尘埃：还没释放的点（脸）不画
            sel = fld.released(tau) & (tau > 0)
            u = np.maximum(tau - fld.tr, 0)                            # 刚释放的脸点还排成脸形 → 飞开之后才渐显
            st["wmul"] = np.where(fld.is_face, smooth(u, 0.15, 0.6), 1.0)
        if "dust_keep" in st:
            keep = fld.is_slab | (self.ru < st.pop("dust_keep"))
            sel = keep if sel is None else sel & keep
        if sel is not None:
            st["sel"] = sel
        ev = shot.get("events")
        if ev:
            k = int(round((t - self.hit) * FPS))
            rng = np.random.default_rng(self.flash_rng_seed + k)
            if ev == "preflash" and k in (0, 1):
                # v5c 前闪：上一镜以亮的全幅抹除结束 → 帧 0 她在原位重现（τ = 0，和定格帧同一位置），过曝 + 抹除余像；
                # 帧 1 已经开始炸（τ = 1/24），余像减半。冲击 = 51.52 不变。
                g, pa, L, ea = ((1.9, 1.0, 520, 0.55), (1.35, 0.35, 160, 0.25))[k]
                im, n_in = render_field(fld, tau, Camera(W, H), shutter=0.0 if k == 0 else 1.0 / FPS, gain=self.gain * g,
                                        plate=self.plate, plate_a=pa * g, style=st)
                im = np.clip(im + ea * streak_echo(im, L * W / 1280, fld.dir, rng), 0, 1)
                return im, ("PRE-FLASH" if k == 0 else "BURST"), tau, N_in(fld, n_in)
            if ev is True and k == 0:
                return noise_frame(W, H, rng), "NOISE", tau, None
            if ev is True and k == 1:
                return flash_frame(W, H, rng, fld.face_info["chin"]), "FLASH", tau, None
        cam = shot_camera(shot, fld, tau, frac)
        phase = "SHATTER" if tau < 9 / FPS else ("FACE" if (fld.is_face & (fld.tr > tau)).any() else "DUST")
        if shot.get("hide_held") and phase == "FACE":
            phase = "STRIPS"
        pa = 0.0                                                       # 背景顶光随信号一起断（留着会露出她的剪影洞）
        shutter = 0.5 / FPS * shot["speed"] * (2.0 if tau < 9 / FPS else 1.0) * shot.get("shutter_k", 1.0)
        im, n_in = render_field(fld, tau, cam, shutter=shutter, gain=self.gain, plate=self.plate, plate_a=pa, style=st or None)
        return im, phase, tau, N_in(fld, n_in)

    def frame0_clean(self):
        """接缝自检：冲击帧 0 去掉一切效果 = 源末帧原样 place（plate 视角）+ 静止粒子场（τ = 0、不加前闪）"""
        import face_glitch as F
        plate_src, _ = F.place(self.hit_u8.astype(np.float32) / 255, np.zeros(self.hit_u8.shape[:2], np.float32), self.a.place_dx)
        im, _ = render_field(self.field, 0.0, Camera(self.W, self.H), gain=self.gain, plate=self.plate, plate_a=1.0)
        return plate_src, self.img, im

    def hud(self, fr, t, shot, phase, tau, n_in):
        fld = self.field
        d = ImageDraw.Draw(fr)
        nm = shot["name"] if shot else "引入"
        sp = shot["speed"] if shot else 1.0
        rel = int(fld.released(tau).sum()) if tau >= 0 else 0
        lines = [f"t = {t:6.2f} s  {phase:<9}  tau = {tau:+.2f} s  x{sp:.2f}  impact @ {self.hit:.2f}  pre-shock @ {self.pre_t:.2f}",
                 f"N = {fld.N:,}  held {fld.N - rel:,}  released {rel:,}  in frame {n_in if n_in is not None else '-'}  mask {self.mask_kind}"]
        for j, line in enumerate(lines):
            d.text((14, self.H - 14 - 16 * (len(lines) - j)), line, font=self.font, fill=(120, 128, 134))

    def render_range(self, t0, t1, shots, out, thumbs_n=12, hud=True):
        """[t0, t1) 的帧（t0 + i/FPS），每帧找所在镜头；out = mp4（带歌的对应段）+ 接触表"""
        import imageio_ffmpeg
        no_overwrite(out, out.with_name(out.stem + "_接触表.jpg"))
        n = n_frames(t0, t1)
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        proc = subprocess.Popen([ff, "-y", "-loglevel", "error",
                                 "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{self.W}x{self.H}", "-r", str(FPS), "-i", "-",
                                 "-ss", f"{t0:.4f}", "-t", f"{n / FPS:.4f}", "-i", str(S.MUSIC),
                                 "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16",
                                 "-c:a", "aac", "-b:a", "256k", "-t", f"{n / FPS:.4f}", str(out)], stdin=subprocess.PIPE)
        # ⚠️不用 -shortest：AAC 末帧对齐会让音轨略短，-shortest 就把视频最后一帧切掉（v5 第一轮每镜少 1 帧）
        pick = set(np.linspace(0, n - 1, min(thumbs_n, n)).round().astype(int).tolist())
        thumbs = []; t_w = time.time()
        for i in range(n):
            t = t0 + i / FPS
            shot = next((s for s in shots if s["t0"] - 1e-9 <= t < s["t1"] - 1e-9), None)
            if shot is None and t >= self.hit - 1e-9:
                shot = shots[-1]
            img, phase, tau, n_in = self.frame(t, shot)
            fr = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
            if hud and phase not in ("NOISE", "FLASH"):
                self.hud(fr, t, shot, phase, tau, n_in)
            proc.stdin.write(fr.tobytes())
            if i in pick:
                thumbs.append((fr.copy(), f"{t:.2f}s  {shot['name'] if shot else '引入'}  {phase}  tau {tau:+.2f}"))
            if i % 24 == 0:
                print(f"  {out.name}  frame {i}/{n}  {time.time() - t_w:.0f}s  {phase}", flush=True)
        proc.stdin.close(); proc.wait()
        contact_sheet(thumbs, out.with_name(out.stem + "_接触表.jpg"))
        print("->", out, flush=True)


def N_in(fld, n_in):
    """画内点数（运动模糊子样本平均后取整）"""
    return int(round(n_in))


def parser_v5():
    ap = argparse.ArgumentParser(prog="particle_shatter.py v5")
    ap.add_argument("--src", default=str(S.MV / "素材" / "全息女性视频运镜描述_裁肩_v1.mp4"))
    ap.add_argument("--src-play", type=float, nargs=2, default=[45.12, 51.52], help="上一镜里源视频占的播放时间窗（变速映射）")
    ap.add_argument("--crop", type=int, nargs=4, default=None, help="x y w h：源还没裁肩时用原片 + 静态裁切")
    ap.add_argument("--W", type=int, default=1280); ap.add_argument("--H", type=int, default=720)
    ap.add_argument("--place-dx", type=int, default=180, help="跟 face_glitch 的 place_dx 一致（她在右边）")
    ap.add_argument("--hit", type=float, default=51.52)
    ap.add_argument("--end", type=float, default=57.92)
    ap.add_argument("--lead", type=float, default=1.2, help="另渲一段冲击前的引入（秒，0 = 不渲）")
    ap.add_argument("--N", type=int, default=300000)
    ap.add_argument("--slab", type=float, default=0.35)
    ap.add_argument("--dir", type=int, default=-1, choices=[-1, 1], help="碎片方向：-1 向左（接上一镜瀑布图行的拉力）")
    ap.add_argument("--face-t", type=float, nargs=2, default=[0.30, 1.25], help="脸的释放时间窗（冲击后秒数）")
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--red-frac", type=float, default=0.7, help="轮廓右缘那圈点里染红的比例（v5c 0.3）")
    ap.add_argument("--no-birefnet", action="store_true")
    ap.add_argument("--no-hud", action="store_true")
    ap.add_argument("--tag", default="v5")
    ap.add_argument("--only", default=None, help="逗号分隔：冲击,余波_1,…,连续,引入（默认全渲）")
    ap.add_argument("--outdir", default=str(S.MV / "样片"))
    ap.add_argument("--src-map", default="stretch", choices=["stretch", "real"],
                    help="冲击前源帧怎么对时间：stretch = 整段拉满窗口（v5/v5b）；real = 原速放完定格（face_glitch v4，v5c）")
    ap.add_argument("--layout", default="v5", choices=["v5", "v5c"], help="镜头表：v5 = SHOTS_V5；v5c = SHOTS_V5C（前闪 + 下拍切点 + 脸只回看一次）")
    ap.add_argument("--retime", default=None, help="镜头名:t0:t1，例 余波_3:53.52:56.32 —— 只渲这一镜换时间窗的版本（τ 起点不变，机位运动铺满新时长）")
    ap.add_argument("--seam-check", default=None, help="上一镜的视频（定格末帧）：出帧 0 接缝对比图 + 均差")
    ap.add_argument("--extend", default=None, help="镜头名:秒，例 余波_3:1.5 —— 只渲这一镜的延长版（同一机位运动继续走，整镜 = 原镜 + 延长）")
    return ap


def seam_check(v, prev_video, out):
    """帧 0 接缝：① 干净源末帧 vs 本镜帧 0 去效果的底图（同一裁切 / 同一位置 → 应 = 0）；
    ② 上一镜定格末帧 / 源末帧 / 本镜帧 0 去效果的粒子渲染 / 本镜真实帧 0（前闪）并排；
    ③ 位置：源末帧 vs 粒子渲染的亮度质心与相关系数（粒子是点采样，像素均差本来就不会是 0，看位置对不对齐）"""
    import face_glitch as F
    n_prev = count_frames(prev_video)
    prev = read_frames(prev_video, [n_prev - 1])[n_prev - 1].astype(np.float32) / 255
    plate_src, base, parts = v.frame0_clean()
    d_base = float(np.abs(plate_src - base).mean() * 255)
    Ls, Lp = F.lum(plate_src), parts.mean(2)
    def cen(L):
        yy, xx = np.mgrid[:L.shape[0], :L.shape[1]]; w = L / L.sum()
        return float((xx * w).sum()), float((yy * w).sum())
    cs, cp = cen(Ls), cen(Lp)
    corr = float(np.corrcoef(S.blur(Ls, 2).ravel(), S.blur(Lp, 2).ravel())[0, 1])
    d_parts = float(np.abs(Ls - Lp).mean() * 255)
    shot = dict(SHOTS_V5C[0]); dt = v.hit - shot["t0"]; shot["t0"] += dt; shot["t1"] += dt
    real0, *_ = v.frame(v.hit, shot)
    tiles = [(prev, f"prev shot held last frame (#{n_prev - 1})"), (plate_src, f"clean source last frame (#{v.n_src - 1})"),   # 字体没有中文 → 英文标签
             (parts, "frame 0 before effects (tau=0 particles + plate)"), (real0, "frame 0 actual (pre-flash)"),
             (np.clip(np.abs(Ls - Lp) * 3, 0, 1)[..., None] * np.ones(3), "|source - particles| x3")]
    contact_sheet([(Image.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8)), lab) for x, lab in tiles], out, cols=3, tw=640)
    return dict(mean_abs_diff_base_255=round(d_base, 3), mean_abs_diff_particles_vs_src_lum_255=round(d_parts, 2),
                corr_particles_vs_src=round(corr, 4), centroid_src=[round(c, 1) for c in cs], centroid_particles=[round(c, 1) for c in cp],
                prev_last_frame=str(prev_video.name), image=str(out.name))


def main_v5(argv):
    a = parser_v5().parse_args(argv)
    od = Path(a.outdir)
    v = V5(a)
    shots = [dict(s) for s in (SHOTS_V5C if a.layout == "v5c" else SHOTS_V5)]
    dt = a.hit - shots[0]["t0"]
    for s in shots:                                                    # 冲击时刻改了 → 整张表平移
        s["t0"] += dt; s["t1"] += dt
    print(f"源 {Path(a.src).name}  {v.n_src} 帧  遮罩 {v.mask_kind}  face {({k: round(x, 1) for k, x in v.field.face_info.items()})}"
          f"  pre-shock {v.pre_t:.3f}  gain {v.gain:.3f}", flush=True)
    only = set(a.only.split(",")) if a.only else None
    want = lambda k: only is None or k in only  # noqa: E731
    side = {"source": Path(a.src).name, "src_play": a.src_play, "crop": a.crop, "place_dx": a.place_dx, "mask": v.mask_kind,
            "hit": a.hit, "pre_shock": v.pre_t, "N": a.N, "slab": a.slab, "dir": a.dir, "face_t": a.face_t, "seed": a.seed,
            "face": {k: round(float(x), 2) for k, x in v.field.face_info.items()}, "T1": T1, "T2": T2, "TS": TS,
            "src_map": a.src_map, "red_frac": a.red_frac, "layout": a.layout,
            "shots": shots, "motion": "x = x0 + dir*(v*T1*(1-exp(-u/T1)) + w*ramp(u-TS)), u = tau - t_release; ramp(q) = q - T2*(1-exp(-q/T2))"}
    if a.retime:
        nm, r0, r1 = a.retime.split(":"); r0, r1 = float(r0), float(r1)
        sp = od / f"19_碎裂_{a.tag}_{nm}_{r0:.2f}-{r1:.2f}.json"
    else:
        sp = od / (f"19_碎裂_{a.tag}.json" if not a.extend else f"19_碎裂_{a.tag}_{a.extend.split(':')[0]}_延长{float(a.extend.split(':')[1]):g}s.json")
    no_overwrite(sp)
    side["extend"] = a.extend; side["retime"] = a.retime
    if a.seam_check:
        side["seam"] = seam_check(v, Path(a.seam_check), od / f"19_碎裂_{a.tag}_帧0接缝.jpg")
        print("seam", side["seam"], flush=True)
    sp.write_text(json.dumps(side, ensure_ascii=False, indent=1), "utf-8")
    hud = not a.no_hud
    if a.retime:
        s = dict(next(x for x in shots if x["name"] == nm))
        s["t0"], s["t1"] = r0, r1
        v.render_range(r0, r1, [s], od / f"19_碎裂_{a.tag}_{nm}_{r0:.2f}-{r1:.2f}.mp4", hud=hud)
        return
    if a.extend:
        nm, sec = a.extend.split(":"); sec = float(sec)
        s = dict(next(x for x in shots if x["name"] == nm))
        s["dur0"] = s["t1"] - s["t0"]; s["t1"] += sec
        v.render_range(s["t0"], s["t1"], [s], od / f"19_碎裂_{a.tag}_{nm}_延长{sec:g}s.mp4", hud=hud)
        return
    if a.lead > 0 and want("引入"):
        v.render_range(a.hit - a.lead, a.hit + 0.5, shots, od / f"19_碎裂_{a.tag}_引入.mp4", hud=hud)
    for s in shots:
        if want(s["name"]):
            v.render_range(s["t0"], s["t1"], [s], od / f"19_碎裂_{a.tag}_{s['name']}.mp4", hud=hud)
    if want("连续"):
        v.render_range(a.hit, a.end, shots, od / f"19_碎裂_{a.tag}_连续.mp4", thumbs_n=24, hud=hud)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "v5":
        main_v5(sys.argv[2:])
    else:
        main_v4(sys.argv[1:])
