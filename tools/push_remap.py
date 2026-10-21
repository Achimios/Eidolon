# -*- coding: utf-8 -*-
""">v<🎥推进曲线-镜头「进 / 缓」后期烘焙：速度 = 基础漂移 + 每个声音点一次「进」（2–3 帧加速）接「缓」（指数衰减，永不归零）。

方法见 _设计/方法_剪辑与情绪.md §四⅞¾（镜头运动不从静止开始、不停在静止；按声音点一次一次地进；只改速度，不晃）。

两种素材：
  still：一张图 → 推近（缩放 = exp(位移)），可加极慢的匀速滚转（漂移，不是晃）
  video：一段视频 → 时间重映射（源时间 ∝ 位移，速度曲线决定播放快慢），可再叠推近
两种运动（--mode）：
  push（默认）：平滑「进 / 缓」速度曲线
  jump：跳切式推近 —— 基础慢漂移 + 在声音点上瞬间放大（0 帧，或 ≤3 帧过渡），video 还可同时跳掉一段时间（--drop）
输出严格按歌曲时间：第 k 帧 = t0 + k/fps，共 n 帧；另出配乐版、接触表、逐帧速度图。

例：
  python tools/push_remap.py still 关键帧.jpg out.mp4 --t0 38.70 --n 154 --push 40.00:1 --push 41.52:1.3 --music 原曲.mp4
  python tools/push_remap.py video seedance.mp4 out.mp4 --t0 38.70 --n 154 --push 40.00:1 --push 41.52:1.3
  python tools/push_remap.py video seedance.mp4 out.mp4 --mode jump --t0 38.70 --n 154 --jump 40.352:1.25 --jump 41.789:1.3 --drop 0.15

⚠️歌曲时间：原曲 mp4 的音频第一个样本 pts = 0.042 s。感官.json 的时间是「从第一个样本数」的，
   比播放器 / -ss 截出来的时间早 42 ms（1 帧）。--snap 时用 --sense-offset 0.042 换到播放时间。
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

SUB = 32  # 每帧积分细分


def velocity(t, pushes, base, ramp_s, tau):
    """>v<🎥推进曲线-速度 v(t)：base + Σ a·g(t-ti)。g：ramp_s 内 smoothstep 升到 1，然后 exp(-(x-ramp)/tau)。"""
    t = np.asarray(t, float)
    v = np.full_like(t, float(base))
    for ti, a in pushes:
        x = t - ti
        r = np.clip(x / ramp_s, 0, 1)
        g = np.where(x < 0, 0.0, np.where(x < ramp_s, r * r * (3 - 2 * r), np.exp(-(x - ramp_s) / tau)))
        v += a * g
    return v


def curve(n, fps, t0, pushes, base=0.04, gain=0.6, ramp_frames=2.5, tau=0.30):
    """>v<🎥推进曲线-逐帧：返回 (帧时间 [n], 帧速度 [n], 帧位移 [n])。pushes = [(歌曲时间, 相对强度)]，gain × 强度 = 峰值附加速度。
    位移从 0 开始，是 v 的精确积分（细分 SUB）。"""
    ts = t0 + np.arange(n) / fps
    rel = [(ti - t0, gain * a) for ti, a in pushes]
    fine = np.arange(n * SUB + 1) / (fps * SUB)
    vf = velocity(fine, rel, base, ramp_frames / fps, tau)
    pos = np.concatenate([[0.0], np.cumsum((vf[1:] + vf[:-1]) / 2 * np.diff(fine))])
    return ts, vf[::SUB][:n], pos[::SUB][:n]


def snap(t, events, tol=0.060):
    """>v<🎥推进曲线-吸附：events 里离 t 最近且在 ±tol 内的点；没有就原样返回 t。返回 (时间, 是否吸附)"""
    ev = np.asarray(list(events), float)
    if ev.size == 0:
        return t, False
    i = int(np.argmin(np.abs(ev - t)))
    return (float(ev[i]), True) if abs(ev[i] - t) <= tol else (float(t), False)


def jump_plan(n, fps, t0, jumps, base=0.03, trans_frames=0, drop=0.0, src_dur=None):
    """>v<🎥推进曲线-跳切：返回 (帧时间, log 缩放 [n], 源时间 [n], 跳点帧号)。
    jumps = [(歌曲时间, 放大倍数)]；跳点帧 = 离该时间最近的帧。log 缩放 = base·t + Σ log(倍数)·s，
    s 在 trans_frames=0 时是阶跃（那一帧直接变大），否则 smoothstep 在 trans_frames 帧内走完。
    源时间 = rate·t + Σ drop·[k ≥ 跳点]（跳掉中间时间）；rate=1，源不够长时自动压慢到刚好用完（src_dur 给定时）。"""
    k = np.arange(n)
    ts = t0 + k / fps
    rel = k / fps
    logz = base * rel
    src = np.zeros(n)
    kj = []
    for tj, f in jumps:
        j = int(round((tj - t0) * fps))
        kj.append(j)
        if trans_frames <= 0:
            st = (k >= j).astype(float)
        else:
            r = np.clip((k - j + 1) / trans_frames, 0, 1)
            st = r * r * (3 - 2 * r)
        logz = logz + np.log(f) * st
        src = src + drop * (k >= j)
    rate = 1.0
    if src_dur is not None and rel[-1] > 0:
        rate = min(1.0, (src_dur - 1e-6 - drop * sum(1 for j in kj if j < n)) / rel[-1])
        if rate <= 0:
            raise ValueError("源视频太短：跳掉的时间比源还长")
    src = src + rate * rel
    return ts, logz, src, kj


# ---------------------------------------------------------------- 画面

def _ff():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _transform(im, zoom, roll_deg, focus, size):
    """以 focus（0–1 相对坐标）为中心放大 zoom 倍、滚转 roll_deg，输出 size"""
    from PIL import Image
    W, H = im.size
    ow, oh = size
    cx, cy = focus[0] * W, focus[1] * H
    s = (W / ow) / zoom  # 输出 1 像素 = 源 s 像素
    c, sn = np.cos(np.radians(roll_deg)), np.sin(np.radians(roll_deg))
    # 输出 (x,y) → 源：中心对齐，旋转 + 缩放
    a, b = s * c, -s * sn
    d, e = s * sn, s * c
    ox, oy = ow / 2, oh / 2
    coeffs = (a, b, cx - a * ox - b * oy, d, e, cy - d * ox - e * oy)
    return im.transform(size, Image.AFFINE, coeffs, resample=Image.BICUBIC)


def _read_video(path):
    import imageio_ffmpeg
    rd = imageio_ffmpeg.read_frames(str(path))
    meta = next(rd)
    w, h = meta["size"]
    frames = [np.frombuffer(f, np.uint8).reshape(h, w, 3) for f in rd]
    return frames, meta.get("fps", 24.0)


def render(kind, src, out, ts, fps, zoom, sidx=None, size=(1920, 1080), roll=0.0, focus=(0.5, 0.5), frames=None):
    """>v<🎥推进曲线-渲染：每帧按 zoom[k] 推近、roll 度/秒匀速滚转；video 按 sidx[k]（源帧号，可小数 → 相邻两帧混合）取帧"""
    import imageio_ffmpeg
    from PIL import Image
    n = len(ts)
    rel_t = ts - ts[0]
    if kind == "still":
        base_im = Image.open(src).convert("RGB")
        get = lambda k: base_im  # noqa: E731
    else:
        if frames is None:
            frames, _ = _read_video(src)
        span = len(frames) - 1

        def get(k):
            x = float(np.clip(sidx[k], 0, span))
            i = int(np.floor(x)); f = x - i
            j = min(i + 1, span)
            if f < 1e-3:
                return Image.fromarray(frames[i])
            a = frames[i].astype(np.float32) * (1 - f) + frames[j].astype(np.float32) * f
            return Image.fromarray(np.clip(a + 0.5, 0, 255).astype(np.uint8))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    w = imageio_ffmpeg.write_frames(str(out), size, fps=fps, codec="libx264", quality=None,
                                    output_params=["-crf", "16", "-preset", "slow"], macro_block_size=8)
    w.send(None)
    thumbs = []
    for k in range(n):
        fr = _transform(get(k), float(zoom[k]), roll * rel_t[k], focus, size)
        w.send(np.ascontiguousarray(np.asarray(fr)))
        thumbs.append(fr)
    w.close()
    return thumbs, zoom


def mux(video, music, t0, dur, out):
    """配乐版：原曲按播放时间截 t0..t0+dur（-ss 在 -i 前 = 容器时间，和剪辑表 / edl 一致）"""
    subprocess.run([_ff(), "-loglevel", "error", "-y", "-i", str(video), "-ss", f"{t0:.3f}", "-t", f"{dur:.4f}", "-i", str(music),
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-shortest", str(out)], check=True)


def contact_sheet(thumbs, ts, out, cols=6, every=8, w=320):
    from PIL import Image, ImageDraw
    pick = list(range(0, len(thumbs), every))
    h = int(w * thumbs[0].size[1] / thumbs[0].size[0])
    rows = (len(pick) + cols - 1) // cols
    sh = Image.new("RGB", (cols * w, rows * (h + 18)), (10, 10, 10))
    d = ImageDraw.Draw(sh)
    for i, k in enumerate(pick):
        x, y = (i % cols) * w, (i // cols) * (h + 18)
        sh.paste(thumbs[k].resize((w, h)), (x, y))
        d.text((x + 4, y + h + 2), f"f{k}  {ts[k]:.3f}s", fill=(200, 200, 200))
    sh.save(out, quality=88)


def speed_plot(ts, v, pushes, out, W=1400, H=380):
    """逐帧速度图（PIL，不依赖 matplotlib）：每帧一根竖条，红线 = 推进点"""
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (W, H), (14, 14, 14)); d = ImageDraw.Draw(im)
    L, R, T, B = 60, 20, 20, 40
    t0, t1 = ts[0], ts[-1] + (ts[1] - ts[0])
    vmax = float(v.max()) * 1.1
    X = lambda t: L + (t - t0) / (t1 - t0) * (W - L - R)  # noqa: E731
    Y = lambda y: H - B - y / vmax * (H - T - B)  # noqa: E731
    bw = max(1, int((W - L - R) / len(ts)) - 1)
    for t, y in zip(ts, v):
        d.rectangle([X(t), Y(y), X(t) + bw, H - B], fill=(170, 175, 180))
    for ti, a in pushes:
        d.line([X(ti), T, X(ti), H - B], fill=(220, 40, 40), width=1)
        d.text((X(ti) + 3, T), f"{ti:.2f}s x{a}", fill=(220, 80, 80))
    for s in np.arange(np.ceil(t0), t1, 0.5):
        d.line([X(s), H - B, X(s), H - B + 5], fill=(150, 150, 150))
        d.text((X(s) - 12, H - B + 8), f"{s:.1f}", fill=(150, 150, 150))
    d.text((4, T), f"v max {v.max():.3f}", fill=(150, 150, 150))
    d.text((4, H - B - 12), f"min {v.min():.3f}", fill=(150, 150, 150))
    im.save(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["still", "video"])
    ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--t0", type=float, required=True, help="第一帧的歌曲时间（秒）")
    ap.add_argument("--n", type=int, required=True, help="帧数")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--push", action="append", default=[], help="时间:强度，如 40.00:1")
    ap.add_argument("--snap", help="感官.json：每个 push 吸附到 ±tol 内最近的 onsets")
    ap.add_argument("--snap-tol", type=float, default=0.060)
    ap.add_argument("--sense-offset", type=float, default=0.042, help="感官.json 时间 + 这个 = 播放时间")
    ap.add_argument("--base", type=float, default=0.04); ap.add_argument("--gain", type=float, default=0.6)
    ap.add_argument("--ramp", type=float, default=2.5, help="进：帧"); ap.add_argument("--tau", type=float, default=0.30, help="缓：秒")
    ap.add_argument("--zoom-k", type=float, default=1.0); ap.add_argument("--zoom0", type=float, default=1.0)
    ap.add_argument("--roll", type=float, default=0.0, help="匀速滚转 度/秒（漂移，不晃）")
    ap.add_argument("--focus", default="0.5,0.5")
    ap.add_argument("--music", help="原曲（出 _配乐 版）")
    ap.add_argument("--mode", choices=["push", "jump"], default="push")
    ap.add_argument("--jump", action="append", default=[], help="jump 模式：时间:放大倍数，如 40.352:1.25")
    ap.add_argument("--trans", type=int, default=0, help="jump：过渡帧数（0 = 硬跳，≤3）")
    ap.add_argument("--drop", type=float, default=0.0, help="jump + video：每个跳点跳掉的源时间（秒）")
    ap.add_argument("--drift", type=float, default=0.03, help="jump：基础推近漂移（log 缩放 / 秒）")
    a = ap.parse_args()

    pushes = []
    for p in a.push:
        t, s = p.split(":")
        pushes.append([float(t), float(s)])
    log = {"pushes_in": [list(p) for p in pushes]}
    if a.snap:
        ev = json.loads(Path(a.snap).read_text("utf-8"))["events"]
        cand = sorted(ev.get("onsets", []))  # 只吸附到起音（强声音），不吸附到拍格
        cand = [c + a.sense_offset for c in cand]
        for p in pushes:
            p[0], ok = snap(p[0], cand, a.snap_tol)
            p.append("snapped" if ok else "kept")
        log["snap"] = [list(p) for p in pushes]
        pushes = [(p[0], p[1]) for p in pushes]
    focus = tuple(float(x) for x in a.focus.split(","))
    out = Path(a.out)
    frames = None
    if a.mode == "push":
        ts, v, pos = curve(a.n, a.fps, a.t0, pushes, a.base, a.gain, a.ramp, a.tau)
        zoom = a.zoom0 * np.exp(a.zoom_k * pos)
        sidx = None
        if a.kind == "video":
            frames, _ = _read_video(a.src)
            sidx = pos / max(pos[-1], 1e-9) * (len(frames) - 1)
        marks, plot_v = pushes, v
    else:
        jumps = [(float(t), float(f)) for t, f in (x.split(":") for x in a.jump)]
        src_dur = None
        if a.kind == "video":
            frames, sfps = _read_video(a.src)
            src_dur = (len(frames) - 1) / sfps
            a.drop = round(a.drop * sfps) / sfps  # 跳掉整数个源帧，否则之后每帧都是两帧混合（变软）
        ts, logz, src_t, kj = jump_plan(a.n, a.fps, a.t0, jumps, a.drift, a.trans, a.drop if a.kind == "video" else 0.0, src_dur)
        zoom = a.zoom0 * np.exp(logz)
        sidx = src_t * sfps if a.kind == "video" else None
        plot_v = np.diff(logz, prepend=logz[0] - a.drift / a.fps) * a.fps  # 每帧推近速度（log 缩放 / 秒）
        marks = jumps
        log.update({"jumps": jumps, "jump_frames": kj, "jump_frame_times": [round(a.t0 + j / a.fps, 4) for j in kj],
                    "trans": a.trans, "drop": a.drop, "drift": a.drift,
                    "src_time_first_last": [float(src_t[0]), float(src_t[-1])] if a.kind == "video" else None})
        v = plot_v
    thumbs, _ = render(a.kind, a.src, a.out, ts, a.fps, zoom, sidx, roll=a.roll, focus=focus, frames=frames)
    contact_sheet(thumbs, ts, out.with_name(out.stem + "_接触表.jpg"))
    speed_plot(ts, plot_v, marks, out.with_name(out.stem + "_速度.png"))
    if a.music:
        mux(out, a.music, a.t0, a.n / a.fps, out.with_name(out.stem + "_配乐.mp4"))
    log.update({k: getattr(a, k) for k in ("mode", "kind", "src", "t0", "n", "fps", "base", "gain", "ramp", "tau", "zoom_k", "zoom0", "roll", "focus")})
    log.update({"t_end": a.t0 + a.n / a.fps, "v_min": float(v.min()), "v_max": float(v.max()), "zoom_end": float(zoom[-1])})
    Path(str(out) + ".json").write_text(json.dumps(log, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps(log, ensure_ascii=False))


if __name__ == "__main__":
    main()
