#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🎬运动参考-MEMS 结构匹配（给 Seedance 网页端）：把静帧 13_mems（stills_more.s_mems）的 2D 版图原样挤出成 3D，
在 Blender 里按真实力学动起来，镜头取 A2_暗_v1a 的斜俯视构图。画面里**没有任何文字**（Seedance 会照抄）。

力学（与 s_mems 同一套几何，单位 µm）：
  一阶模态 —— 梭子沿梳齿轴（v）平动 V，桁架 V/2；每根折叠梁两端导向，u(s) = 3s² − 2s³：
    内梁（梭子 → 桁架）：v = v0 + V − (V/2)·u(s)       外梁（锚 → 桁架）：v = v0 + (V/2)·u(s)
  动齿随梭子插入 / 退出定子齿间隙；|V| ≤ VMAX < 齿尖间隙 6 µm，所以永不相撞（自检里查）。
驱动 = 「other」分轨（抽搐音所在层）：动机_抽搐.json 的命中点给一次冲击（强度 ∝ 模板窗内包络峰），
  再叠一点包络正向通量（小冲击）；对模态脉冲响应 h(t) = e^(−t/τ)·sin(2π f_vis t) 卷积 → 命中处幅值跳起、再振铃衰减。
  ⚠️时间缩放：f0 = 27.1 kHz 的振动画面上读不出来，改成 f_vis（默认 3 Hz，约 9000× 慢放）；位移放大到 VMAX（真实 ~0.3 µm）。
镜头：push_remap.curve 的「进 / 缓」—— 第 1 帧已在动，命中处加速，之后指数减速但永不停；直线推向红点，不晃。

两段运行：系统侧（.venv）算驱动 + 镜头 → plan.json → 调 Blender（-b --python 本文件 -- --plan）逐帧渲染 → ffmpeg 合成 + 接触表 + 逐帧 json。
用法：
  python tools/mems_motion_ref.py --hit 62.624 --out 生成/给网页端_运动参考_v1/MEMS_参考视频_结构匹配_v1.mp4 --work <临时目录>
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

# ---------- 几何（µm），与 stills_more.s_mems 一致 ----------
SU, SV = 60, 40                       # 梭子半宽 / 半高
NOTCH = (36, 20)                      # 左右缺口（锚在里面）
HOLES = [(u, v) for u in range(-24, 25, 12) for v in range(-28, 29, 14)]
HOLE = 4
LF, FW, PITCH, GAP = 34, 3, 8, 6      # 梳齿长、宽、节距、齿尖间隙
FU = list(range(-52, 53, PITCH))      # 动齿 u
FS = [u + PITCH / 2 for u in FU[:-1]]  # 定子齿 u
TRW = 8                               # 桁架宽
VIN, VOUT = 32, 12                    # 内梁 / 外梁 v
L, WB = 150, 2                        # 梁长、梁宽
BAR = 12                              # 定子横梁厚
ANC_H = 22                            # 定子锚长
F0 = 27.1e3                           # 真实一阶频率（s_mems 算出）
VMAX = 5.0                            # 画面最大位移 µm（< GAP）
H_ST = 6.0                            # 结构层厚（真实 2 µm，画面加厚 3×，好读）
Z_GAP = 3.0                           # 结构层离基底的牺牲层间隙
RED_LOCAL = (FU[-1], SV + LF)         # 红点 = 右端上动齿的齿尖（s_mems 里的最大位移点），随梭子动

MV = Path("C:/Users/Administrator/Desktop/Projekt_Visual/MV_Cant_Kill_Us")
BLENDER = Path("E:/2_Wares/Blender Foundation/Blender 5.2/blender.exe")


def mode_shape(s):
    return 3 * s ** 2 - 2 * s ** 3


def beam_v(kind, v0, V, s):
    """>v<🎬运动参考-MEMS-梁中线 v(s)：kind='inner' 梭子(s=0, V) → 桁架(s=1, V/2)；'outer' 锚(0) → 桁架(V/2)"""
    s = np.asarray(s, float)
    if kind == "inner":
        return v0 + V - V / 2 * mode_shape(s)
    return v0 + V / 2 * mode_shape(s)


def beams():
    """8 根梁：(名字, kind, side, v0, u_start, u_end)，u 从梭子 / 锚那端走向桁架"""
    out = []
    for side in (1, -1):
        for vb in (VIN, -VIN):
            out.append((f"beam_in_{side:+d}_{vb:+d}", "inner", side, vb, side * SU, side * (SU + L)))
        for vb in (VOUT, -VOUT):
            out.append((f"beam_out_{side:+d}_{vb:+d}", "outer", side, vb, side * (SU + 2), side * (SU + L)))
    return out


def rects():
    """>v<🎬运动参考-MEMS-刚体矩形：{组: [(u0, v0, u1, v1, z0, z1)]}。组 = shuttle(动 V) / truss(动 V/2) / fixed / anchor"""
    z0, z1 = Z_GAP, Z_GAP + H_ST
    g = {"shuttle": [], "truss": [], "fixed": [], "anchor": []}
    for u in FU:                                   # 动齿（梭子本体单独做带缺口 + 孔的多边形）
        g["shuttle"].append((u - FW / 2, SV, u + FW / 2, SV + LF, z0, z1))
        g["shuttle"].append((u - FW / 2, -SV - LF, u + FW / 2, -SV, z0, z1))
    for side in (1, -1):                           # 桁架
        ut = side * (SU + L)
        g["truss"].append((min(ut, ut + side * TRW), -VIN - WB, max(ut, ut + side * TRW), VIN + WB, z0, z1))
    bb0 = SV + LF + GAP
    for sg in (1, -1):                             # 定子：横梁 + 齿 + 锚
        g["fixed"].append((-60, min(sg * bb0, sg * (bb0 + BAR)), 60, max(sg * bb0, sg * (bb0 + BAR)), z0, z1))
        for u in FS:
            a, b = sorted((sg * (SV + GAP), sg * bb0))
            g["fixed"].append((u - FW / 2, a, u + FW / 2, b, z0, z1))
        a, b = sorted((sg * (bb0 + BAR), sg * (bb0 + BAR + ANC_H)))
        g["anchor"].append((-40, a, 40, b, 0.0, z1))
    n0 = SU - NOTCH[0]
    g["anchor"].append((n0 + 4, -NOTCH[1] + 4, SU + 2, NOTCH[1] - 4, 0.0, z1))
    g["anchor"].append((-SU - 2, -NOTCH[1] + 4, -n0 - 4, NOTCH[1] - 4, 0.0, z1))
    return g


def shuttle_poly():
    n0 = SU - NOTCH[0]
    return [(-SU, SV), (SU, SV), (SU, NOTCH[1]), (n0, NOTCH[1]), (n0, -NOTCH[1]), (SU, -NOTCH[1]), (SU, -SV), (-SU, -SV),
            (-SU, -NOTCH[1]), (-n0, -NOTCH[1]), (-n0, NOTCH[1]), (-SU, NOTCH[1])]


# ---------- 驱动 ----------
def envelope(x, sr, rate=480, win=0.010):
    """>v<🎬运动参考-MEMS-包络：RMS（win 秒窗），按 rate Hz 取样。返回 (t, e)"""
    hop = int(sr / rate); k = max(1, int(sr * win))
    e = np.sqrt(np.convolve(x ** 2, np.ones(k) / k, mode="same"))[::hop]
    return np.arange(e.size) * hop / sr, e


def drive(t_env, e, hits, frame_t, f_vis=3.0, tau=1.1, flux_gain=0.12, template=1.04, vmax=VMAX):
    """>v<🎬运动参考-MEMS-位移 V(帧)：力 = 命中冲击（∝ 模板窗内包络峰）+ flux_gain × 包络正向通量；
    V = 力 ⊛ e^(−t/τ) sin(2π f_vis t)，再按片段内峰值归一到 vmax。返回 (V[帧], 信息)"""
    dt = t_env[1] - t_env[0]
    lo, hi = frame_t[0] - 4 * tau, frame_t[-1]
    m = (t_env >= lo) & (t_env <= hi)
    tt, ee = t_env[m], e[m]
    force = np.zeros_like(tt)
    flux = np.maximum(np.diff(ee, prepend=ee[0]), 0)
    ref = max(ee.max(), 1e-9)
    force += flux_gain * flux / ref                         # 通量：每个包络采样的正向增量当小冲击
    hit_amp = {}
    for h in hits:
        if lo <= h <= hi:
            w = (t_env >= h) & (t_env < h + template)
            a = float(e[w].max() / ref) if w.any() else 1.0
            force[np.argmin(np.abs(tt - h))] += a
            hit_amp[h] = a
    th = np.arange(0, 6 * tau, dt)
    resp = np.exp(-th / tau) * np.sin(2 * np.pi * f_vis * th)
    V = np.convolve(force, resp)[:tt.size]
    Vf = np.interp(frame_t, tt, V)
    peak = np.abs(Vf).max() + 1e-12
    Vf = Vf / peak * vmax
    return Vf, {"hit_amp": hit_amp, "f_vis": f_vis, "tau": tau, "flux_gain": flux_gain, "time_scale": F0 / f_vis}


# ---------- 镜头 ----------
CAM0 = np.array([0.0, -940.0, 560.0])     # µm；斜俯视（约 31° 俯角），A2 构图
TGT0 = np.array([0.0, 10.0, 0.0])


def camera(n, fps, t0, hits, travel=0.42, aim=0.55):
    """>v<🎥运动参考-MEMS-镜头：push_remap.curve 的「进/缓」位移 → 归一 → 相机沿 CAM0→红点 直线推进 travel 比例，
    注视点从 TGT0 向红点移 aim 比例。返回 (cam[n,3], tgt[n,3], 速度[n])"""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import push_remap as P
    _, v, pos = P.curve(n, fps, t0, [(h, 1.0) for h in hits], base=0.05, gain=0.7, ramp_frames=2.5, tau=0.45)
    p = pos / pos[-1]
    red = np.array([RED_LOCAL[0], RED_LOCAL[1], Z_GAP + H_ST])
    cam = CAM0[None] + (red - CAM0)[None] * (travel * p)[:, None]
    tgt = TGT0[None] + (red - TGT0)[None] * (aim * p)[:, None]
    return cam, tgt, v


# ---------- 系统侧 ----------
def plan(a):
    import soundfile as sf
    x, sr = sf.read(MV / "分轨" / "other.wav"); x = x.mean(1) if x.ndim > 1 else x
    motif = json.loads((MV / "动机_抽搐.json").read_text("utf-8"))
    tpl = motif["template"][1] - motif["template"][0]
    n = int(round(a.dur * a.fps))
    t0 = a.hit - a.pre
    ft = t0 + np.arange(n) / a.fps
    te, e = envelope(x, sr)
    hits = [h["t"] for h in motif["hits"]]
    V, info = drive(te, e, hits, ft, f_vis=a.f_vis, tau=a.tau, template=tpl)
    in_clip = [h for h in hits if ft[0] <= h <= ft[-1]]
    cam, tgt, vel = camera(n, a.fps, t0, in_clip)
    return {"t0": t0, "fps": a.fps, "n": n, "w": a.w, "h": a.h, "samples": a.samples, "engine": a.engine,
            "frames_dir": str(Path(a.work) / "frames"), "hits_in_clip": in_clip, "info": {k: v for k, v in info.items() if k != "hit_amp"},
            "hit_amp": {f"{k:.3f}": v for k, v in info["hit_amp"].items()},
            "frames": [{"k": k, "t": float(ft[k]), "V": float(V[k]), "cam": cam[k].tolist(), "tgt": tgt[k].tolist(), "cam_speed": float(vel[k])}
                       for k in range(n)]}


def encode(frames_dir, out, fps):
    import imageio_ffmpeg
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(Path(frames_dir) / "f_%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "15", "-preset", "slow", "-movflags", "+faststart", "-an", str(out)], check=True)


def contact(frames_dir, pl, out, ref=None, cols=4, rows=3):
    from PIL import Image, ImageDraw, ImageFont
    n = pl["n"]
    ks = np.linspace(0, n - 1, cols * rows - (1 if ref else 0)).round().astype(int)
    tw, th = 480, 270
    sheet = Image.new("RGB", (cols * tw, rows * (th + 22)), (12, 12, 14))
    d = ImageDraw.Draw(sheet); f = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 15)
    tiles = ([("ref A2_暗_v1a", Image.open(ref))] if ref else []) + \
            [(f"#{k:03d}  t={pl['frames'][k]['t']:.3f}  V={pl['frames'][k]['V']:+.2f}um", Image.open(Path(frames_dir) / f"f_{k:04d}.png")) for k in ks]
    fz = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 15)
    for i, (lab, im) in enumerate(tiles):
        x, y = (i % cols) * tw, (i // cols) * (th + 22)
        sheet.paste(im.convert("RGB").resize((tw, th), Image.LANCZOS), (x, y + 22))
        d.text((x + 6, y + 3), lab, font=fz if not lab.startswith("#") else f, fill=(200, 200, 205))
    sheet.save(out, quality=92)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hit", type=float, default=62.624)
    ap.add_argument("--pre", type=float, default=1.0, help="命中前留几秒")
    ap.add_argument("--dur", type=float, default=4.5)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--w", type=int, default=1280); ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--f-vis", type=float, default=3.0); ap.add_argument("--tau", type=float, default=1.1)
    ap.add_argument("--samples", type=int, default=48); ap.add_argument("--engine", default="EEVEE")
    ap.add_argument("--out", required=True); ap.add_argument("--work", required=True)
    ap.add_argument("--only", type=str, default="", help="只渲这些帧号（逗号分隔），调构图用；不合成视频")
    a = ap.parse_args()
    out = Path(a.out); out = out if out.is_absolute() else MV / out
    if out.exists() and not a.only:
        sys.exit(f"已存在，不覆盖：{out}（换版本号）")
    Path(a.work, "frames").mkdir(parents=True, exist_ok=True)
    pl = plan(a)
    if a.only:
        pl["only"] = [int(k) for k in a.only.split(",")]
    pj = Path(a.work) / "plan.json"; pj.write_text(json.dumps(pl, ensure_ascii=False), "utf-8")
    Vs = np.array([f["V"] for f in pl["frames"]])
    kh = int(round((a.hit - pl["t0"]) * a.fps))
    print(f"t0={pl['t0']:.3f}  n={pl['n']}  hit 帧 {kh}  命中前 max|V|={np.abs(Vs[:kh]).max():.2f}  命中后 max|V|={np.abs(Vs[kh:]).max():.2f}  "
          f"末 1 s max|V|={np.abs(Vs[-a.fps:]).max():.2f} µm  hit_amp={pl['hit_amp']}")
    subprocess.run([str(BLENDER), "-b", "--factory-startup", "--python", str(Path(__file__).resolve()), "--", "--plan", str(pj)], check=True)
    if a.only:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    encode(pl["frames_dir"], out, a.fps)
    contact(pl["frames_dir"], pl, out.with_name(out.stem + "_接触表.jpg"), ref=MV / "生成" / "风格渲染" / "A2_暗_v1a.jpg")
    side = {k: v for k, v in pl.items() if k not in ("frames_dir",)}
    side["说明"] = (f"无 HUD。时间缩放：真实 f0 = {F0/1000:.1f} kHz → 画面 f_vis = {a.f_vis} Hz（{F0/a.f_vis:.0f}× 慢放）；"
                   f"位移峰值 {VMAX} µm（真实 ~0.3 µm，约 {VMAX/0.3:.0f}× 放大）；结构层厚画面加厚 3×。第 k 帧 = 歌曲时间 t0 + k/fps（播放时间）")
    out.with_suffix(".json").write_text(json.dumps(side, ensure_ascii=False, indent=1), "utf-8")
    print("写出", out)


# ---------- Blender 侧 ----------
def bl_main(pj):
    import bpy
    import bmesh
    from mathutils import Vector
    pl = json.loads(Path(pj).read_text("utf-8"))
    S = 0.1                                        # 1 BU = 10 µm
    sc = bpy.context.scene
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)

    def mat(name, col, metal, rough, emit=None):
        m = bpy.data.materials.new(name); m.use_nodes = True
        b = m.node_tree.nodes["Principled BSDF"]
        b.inputs["Base Color"].default_value = (*col, 1); b.inputs["Metallic"].default_value = metal; b.inputs["Roughness"].default_value = rough
        if emit:
            b.inputs["Emission Color"].default_value = (*emit[0], 1); b.inputs["Emission Strength"].default_value = emit[1]
        return m

    M_SI = mat("polySi", (0.62, 0.65, 0.69), 0.55, 0.28)
    M_SUB = mat("substrate", (0.012, 0.013, 0.016), 0.3, 0.6)
    M_FR = mat("frame", (0.12, 0.13, 0.15), 0.9, 0.35)
    M_RED = mat("red", (1, 0.05, 0.03), 0, 0.4, emit=((1.0, 0.04, 0.02), 9.0))

    def box_mesh(name, rs, m, parent=None):
        bm = bmesh.new()
        for (u0, v0, u1, v1, z0, z1) in rs:
            r = bmesh.ops.create_cube(bm, size=1.0)
            for v in r["verts"]:
                v.co.x = (u0 if v.co.x < 0 else u1) * S; v.co.y = (v0 if v.co.y < 0 else v1) * S; v.co.z = (z0 if v.co.z < 0 else z1) * S
        me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
        ob = bpy.data.objects.new(name, me); sc.collection.objects.link(ob); ob.data.materials.append(m)
        if parent:
            ob.parent = parent
        return ob

    def empty(name):
        e = bpy.data.objects.new(name, None); sc.collection.objects.link(e); return e

    G = rects()
    E_sh, E_tr = empty("shuttle_motion"), empty("truss_motion")
    box_mesh("fingers_moving", G["shuttle"], M_SI, E_sh)
    box_mesh("trusses", G["truss"], M_SI, E_tr)
    box_mesh("stators", G["fixed"], M_SI)
    box_mesh("anchors", G["anchor"], M_SI)
    # 梭子本体：带缺口的多边形挤出，再布尔挖孔
    bm = bmesh.new()
    vs = [bm.verts.new((u * S, v * S, Z_GAP * S)) for u, v in shuttle_poly()]
    f = bm.faces.new(vs)
    ext = bmesh.ops.extrude_face_region(bm, geom=[f])
    for v in ext["geom"]:
        if isinstance(v, bmesh.types.BMVert):
            v.co.z = (Z_GAP + H_ST) * S
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new("shuttle"); bm.to_mesh(me); bm.free()
    sh = bpy.data.objects.new("shuttle", me); sc.collection.objects.link(sh); sh.data.materials.append(M_SI); sh.parent = E_sh
    holes = box_mesh("holes", [(u - HOLE / 2, v - HOLE / 2, u + HOLE / 2, v + HOLE / 2, Z_GAP - 1, Z_GAP + H_ST + 1) for u, v in HOLES], M_SI, E_sh)
    holes.hide_render = True; holes.display_type = "WIRE"
    mod = sh.modifiers.new("holes", "BOOLEAN"); mod.operation = "DIFFERENCE"; mod.object = holes
    # 红点
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.9 * S, location=(RED_LOCAL[0] * S, (RED_LOCAL[1] - 0.8) * S, (Z_GAP + H_ST + 1.0) * S))
    rd = bpy.context.active_object; rd.data.materials.append(M_RED); rd.parent = E_sh
    # 梁：沿 u 细分的长方体，每帧改 y
    NS = 48
    beam_obs = []
    for name, kind, side, v0, ua, ub in beams():
        s = np.linspace(0, 1, NS + 1)
        us = ua + (ub - ua) * s
        bm = bmesh.new()
        ring = []
        for i in range(NS + 1):
            ring.append([bm.verts.new((us[i] * S, (v0 + dv) * S, z * S)) for dv, z in ((-WB / 2, Z_GAP), (WB / 2, Z_GAP), (WB / 2, Z_GAP + H_ST), (-WB / 2, Z_GAP + H_ST))])
        for i in range(NS):
            a, b = ring[i], ring[i + 1]
            for j in range(4):
                bm.faces.new((a[j], a[(j + 1) % 4], b[(j + 1) % 4], b[j]))
        bm.faces.new(ring[0][::-1]); bm.faces.new(ring[-1])
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
        ob = bpy.data.objects.new(name, me); sc.collection.objects.link(ob); ob.data.materials.append(M_SI)
        # 顶点对应的 s 与横向偏移
        co = np.zeros(len(me.vertices) * 3); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
        sv_ = (co[:, 0] / S - ua) / (ub - ua)
        dv_ = co[:, 1] / S - v0
        beam_obs.append((ob, kind, v0, sv_, dv_, co))
    # 基底 + 夹具框（A2 里的大方框）
    box_mesh("substrate", [(-900, -700, 900, 700, -6, 0)], M_SUB)
    fr = []
    xi, yi, wr = 270, 250, 46
    fr += [(-xi - wr, -yi - wr, xi + wr, -yi, 0, 26), (-xi - wr, yi, xi + wr, yi + wr, 0, 26),
           (-xi - wr, -yi, -xi, yi, 0, 26), (xi, -yi, xi + wr, yi, 0, 26)]
    fr += [(-xi - wr - 60, -yi - wr - 60, xi + wr + 60, -yi - wr - 40, 0, 14), (-xi - wr - 60, yi + wr + 40, xi + wr + 60, yi + wr + 60, 0, 14)]
    box_mesh("frame", fr, M_FR)
    # 灯：冷白顶光（从上方略靠后斜切）+ 很弱的背侧勾边
    def area(name, loc, rot, size, size_y, energy, col):
        ld = bpy.data.lights.new(name, "AREA"); ld.shape = "RECTANGLE"; ld.size = size; ld.size_y = size_y; ld.energy = energy; ld.color = col
        lo = bpy.data.objects.new(name, ld); sc.collection.objects.link(lo); lo.location = loc; lo.rotation_euler = rot
        return lo
    import math
    sp = bpy.data.lights.new("key_top", "SPOT"); sp.energy = 140000; sp.color = (0.8, 0.9, 1.0); sp.spot_size = math.radians(52)
    sp.spot_blend = 0.7; sp.shadow_soft_size = 6
    so = bpy.data.objects.new("key_top", sp); sc.collection.objects.link(so); so.location = (0, 18, 60); so.rotation_euler = (math.radians(-16), 0, 0)
    area("sheen_back", (0, 52, 32), (math.radians(-59), 0, 0), 26, 8, 4000, (0.78, 0.88, 1.0))   # 镜面反射方向：顶面泛冷光
    area("rim_back", (0, 75, 10), (math.radians(-80), 0, 0), 90, 3, 1800, (0.7, 0.82, 1.0))
    area("fill_far", (0, 0, 80), (0, 0, 0), 200, 200, 900, (0.6, 0.7, 0.85))
    w = bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (0.004, 0.005, 0.007, 1)
    # 相机
    cd = bpy.data.cameras.new("cam"); cd.lens = 50; cd.sensor_width = 36; cd.clip_start = 0.5; cd.clip_end = 2000
    cam = bpy.data.objects.new("cam", cd); sc.collection.objects.link(cam); sc.camera = cam
    tgt = empty("cam_target")
    con = cam.constraints.new("TRACK_TO"); con.target = tgt; con.track_axis = "TRACK_NEGATIVE_Z"; con.up_axis = "UP_Y"
    cd.dof.use_dof = True; cd.dof.focus_object = rd; cd.dof.aperture_fstop = 0.3
    # 渲染设置
    r = sc.render
    try:
        r.engine = "BLENDER_EEVEE" if pl["engine"] == "EEVEE" else "BLENDER_WORKBENCH"
    except TypeError:
        r.engine = "BLENDER_EEVEE_NEXT"
    r.resolution_x, r.resolution_y, r.resolution_percentage = pl["w"], pl["h"], 100
    r.image_settings.file_format = "PNG"; r.film_transparent = False
    try:
        sc.eevee.taa_render_samples = pl["samples"]
    except AttributeError:
        pass
    try:
        sc.view_settings.view_transform = "AgX"; sc.view_settings.look = "AgX - Medium High Contrast"
    except TypeError:
        pass
    sc.view_settings.exposure = 0.0
    ks = pl.get("only") or range(pl["n"])
    for k in ks:
        fr_ = pl["frames"][k]; V = fr_["V"]
        E_sh.location.y = V * S; E_tr.location.y = V / 2 * S
        for ob, kind, v0, s_, dv_, co in beam_obs:
            c = co.copy(); c[:, 1] = (beam_v(kind, v0, V, np.clip(s_, 0, 1)) + dv_) * S
            ob.data.vertices.foreach_set("co", c.ravel()); ob.data.update()
        cam.location = Vector(fr_["cam"]) * S; tgt.location = Vector(fr_["tgt"]) * S
        r.filepath = str(Path(pl["frames_dir"]) / f"f_{k:04d}.png")
        bpy.ops.render.render(write_still=True)
        print(f"[mems_motion_ref] 帧 {k}", flush=True)


if __name__ == "__main__":
    if "--plan" in sys.argv:
        bl_main(sys.argv[sys.argv.index("--plan") + 1])
    else:
        main()
