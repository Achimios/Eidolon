# -*- coding: utf-8 -*-
""">v<🧊四维切片-Blender-把 four_d.py --export-mesh 导出的网格序列在 Blender 里渲染（真实光照 / 软阴影 / 景深），场景全由本脚本搭，可复现。
用法（Blender 5.2，后台）：
  blender -b --factory-startup -P tools/blender_four_d.py -- --seq <cache/四维序列/xxx> --out <帧目录> [--engine cycles|eevee]
      [--samples 96] [--frames 0-75] [--save-blend <seq>/scene.blend] [--fstop 4]
每帧：读 fNNNN.ply（numpy 直接建网格，法线 = SDF 梯度 → 自定义法线，全平滑）→ 渲染 → 删网格（内存不随帧数涨）。
相机（eye / 视线 / 上方向含 roll / fov）、主光（世界固定）、轮廓光（跟相机 roll）、唯一红点（4D 相切处）全部来自 track.json，和 four_d 的预览同一套参数。
输出 = 物体区（W × (H − HUD 条带)）的 PNG；HUD 由 four_d.py --compose-blender 加。
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine", default="cycles", choices=["cycles", "eevee"])
    ap.add_argument("--samples", type=int, default=96)
    ap.add_argument("--frames", default=None, help="a-b（含两端）")
    ap.add_argument("--save-blend", default=None)
    ap.add_argument("--fstop", type=float, default=4.0, help="景深光圈（大 = 景深浅得少）")
    ap.add_argument("--key", type=float, default=4.2, help="主光强度（sun，W/m²）")
    ap.add_argument("--rim", type=float, default=10.0, help="轮廓光强度（× track 里的 rim 系数）")
    ap.add_argument("--rim2", default="3.5,140,10", help="右侧第二轮廓 / 反光 '强度,az,el'（度，相对相机含 roll；0 = 关）：让背 / 臀一侧外形读得出")
    ap.add_argument("--red-size", type=float, default=0.012, help="红点半径（Blender 单位 = 模型单位，身高 ≈ 2）")
    ap.add_argument("--hires", default=None, help="高模 PLY（four_d --export-hires，已对齐）：--sweep 期间一个平面从头往下扫，扫过处换成高模")
    ap.add_argument("--sweep", default="46,70", help="扫的起止帧 a,b（之前全是切片，之后全是高模）")
    ap.add_argument("--seam", type=float, default=5.0, help="扫描线的自发光强度")
    ap.add_argument("--backdrop", default=None, help="背景图（可选）：贴在跟着相机的背景板上；不给 = 纯黑（板仍在场景里，以后换图即可）")
    return ap.parse_args(argv)


def read_ply(path):
    """four_d.write_ply 的格式：binary LE，顶点 xyz + nxnynz，面全是 k 边形（uchar k + int×k；切片 k = 4，高模 k = 3）"""
    data = Path(path).read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    hdr = data[:end].decode("ascii").split("\n")
    nv = int(next(l for l in hdr if l.startswith("element vertex")).split()[-1])
    nf = int(next(l for l in hdr if l.startswith("element face")).split()[-1])
    vert = np.frombuffer(data, dtype="<f4", count=nv * 6, offset=end).reshape(nv, 6)
    k = data[end + nv * 24]
    face = np.frombuffer(data, dtype=np.dtype([("n", "u1"), ("v", "<i4", (k,))]), count=nf, offset=end + nv * 24)
    return vert[:, :3].copy(), vert[:, 3:].copy(), face["v"].copy()


def build_mesh(name, V, N, Q, mat):
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(V))
    me.vertices.foreach_set("co", V.ravel())
    me.loops.add(Q.size)
    me.loops.foreach_set("vertex_index", Q.ravel().astype(np.int32))
    me.polygons.add(len(Q))
    me.polygons.foreach_set("loop_start", (np.arange(len(Q)) * Q.shape[1]).astype(np.int32))
    me.polygons.foreach_set("use_smooth", np.ones(len(Q), dtype=bool))
    me.update(calc_edges=True)
    me.validate(clean_customdata=False)
    me.normals_split_custom_set_from_vertices([tuple(n) for n in N])      # SDF 梯度法线 → 平滑（不靠面法线）
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def look_quat(fwd, up):
    """Blender 相机 / 灯沿本地 −Z 看、本地 +Y 朝上"""
    f = Vector(fwd).normalized()
    u = Vector(up).normalized()
    r = f.cross(u).normalized()
    u = r.cross(f).normalized()
    M = Matrix((r, u, -f)).transposed()
    return M.to_quaternion()


def sun_toward(L):
    """sun 沿本地 −Z 照；光从 L 方向来 → 本地 +Z = L"""
    L = Vector(L).normalized()
    ref = Vector((0, 0, 1)) if abs(L.z) < 0.95 else Vector((0, 1, 0))
    return look_quat(-L, ref)


def scene_setup(meta, a):
    sc = bpy.context.scene
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    W, H = meta["size"]
    sc.render.resolution_x, sc.render.resolution_y = W, H - meta["hud_band"]
    sc.render.resolution_percentage = 100
    sc.render.fps = meta["fps"]
    sc.render.image_settings.file_format = "PNG"
    sc.render.film_transparent = False
    if a.engine == "cycles":
        sc.render.engine = "CYCLES"
        sc.cycles.samples = a.samples
        sc.cycles.use_denoising = True
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            for dev in ("OPTIX", "CUDA"):
                try:
                    prefs.compute_device_type = dev
                    prefs.get_devices()
                    if any(d.type == dev for d in prefs.devices):
                        for d in prefs.devices:
                            d.use = d.type == dev
                        sc.cycles.device = "GPU"
                        print("  cycles device:", dev)
                        break
                except TypeError:
                    continue
        except Exception as e:  # noqa: BLE001
            print("  GPU setup failed, CPU:", e)
    else:
        sc.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items] else "BLENDER_EEVEE"
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.view_settings.exposure = 0.0
    world = bpy.data.worlds.new("void")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0, 0, 0, 1)
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.0
    sc.world = world
    # 材质：冷灰石膏，略带光泽；切片 / 高模各一份，都带「扫描平面」：z > h 的一侧是高模、z < h 的一侧是切片，交界一条冷白发光线
    mat = plaster_material("plaster_slice", a, side=-1)
    mat_hi = plaster_material("plaster_hires", a, side=+1)
    # 相机
    cd = bpy.data.cameras.new("cam")
    cd.sensor_fit = "VERTICAL"
    cd.angle_y = math.radians(meta["fov"])
    cd.clip_start, cd.clip_end = 0.05, 100
    cd.dof.use_dof = a.fstop > 0
    cd.dof.aperture_fstop = max(a.fstop, 0.1)
    cam = bpy.data.objects.new("cam", cd)
    cam.rotation_mode = "QUATERNION"
    sc.collection.objects.link(cam)
    sc.camera = cam
    backdrop_layer(cam, meta, a)
    # 主光（世界固定，硬）
    kd = bpy.data.lights.new("key", "SUN")
    kd.energy, kd.angle, kd.color = a.key, math.radians(1.5), (0.97, 0.98, 1.0)
    key = bpy.data.objects.new("key", kd)
    key.rotation_mode = "QUATERNION"
    key.rotation_quaternion = sun_toward(meta["key"])
    sc.collection.objects.link(key)
    # 轮廓光（跟相机 roll，逐帧）+ 右侧第二轮廓（背 / 臀那一侧，弱）
    rims = []
    r2 = [float(x) for x in a.rim2.split(",")]
    for nm, energy, col, ang in (("rim", a.rim * meta["rim"], (0.86, 0.92, 1.0), 3.0), ("rim_right", r2[0], (0.80, 0.86, 0.95), 6.0)):
        rd = bpy.data.lights.new(nm, "SUN")
        rd.energy, rd.angle, rd.color = energy, math.radians(ang), col
        ob = bpy.data.objects.new(nm, rd)
        ob.rotation_mode = "QUATERNION"
        sc.collection.objects.link(ob)
        rims.append(ob)
    # 唯一红点：自发光小球（+ 极弱的红点光，只让它自己亮，不往身上泼红）
    rm = bpy.data.materials.new("red")
    rm.use_nodes = True
    nt = rm.node_tree
    for nd in list(nt.nodes):
        nt.nodes.remove(nd)
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs[0].default_value = (1.0, 0.16, 0.20, 1)
    em.inputs[1].default_value = 40.0
    outn = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(em.outputs[0], outn.inputs[0])
    bpy.ops.mesh.primitive_uv_sphere_add(radius=a.red_size, segments=24, ring_count=12)
    red = bpy.context.active_object
    red.name = "red_point"
    red.data.materials.append(rm)
    red.visible_shadow = False
    pd = bpy.data.lights.new("red_light", "POINT")
    pd.color, pd.shadow_soft_size = (1.0, 0.16, 0.20), 0.004
    redl = bpy.data.objects.new("red_light", pd)
    sc.collection.objects.link(redl)
    return dict(cam=cam, rims=rims, rim2=r2, red=red, redl=redl, em=em, mat=mat, mat_hi=mat_hi)


def plaster_material(name, a, side):
    """Principled 石膏 × 扫描可见性：side = −1 → z < h 可见（切片），+1 → z > h 可见（高模）；|z − h| < 带宽 → 加冷白自发光（扫描线）。
    h 存在名为 sweep_h 的 Value 节点里，逐帧改"""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bs = nt.nodes["Principled BSDF"]
    bs.inputs["Base Color"].default_value = (0.70, 0.74, 0.78, 1)
    bs.inputs["Roughness"].default_value = 0.42
    out = nt.nodes["Material Output"]
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geo.outputs["Position"], sep.inputs[0])
    hv = nt.nodes.new("ShaderNodeValue")
    hv.name = hv.label = "sweep_h"
    hv.outputs[0].default_value = 1e3                                    # 初值：扫描线远在头顶之上 → 切片全可见、高模全隐
    cmp = nt.nodes.new("ShaderNodeMath")
    cmp.operation = "LESS_THAN" if side < 0 else "GREATER_THAN"
    nt.links.new(sep.outputs["Z"], cmp.inputs[0])
    nt.links.new(hv.outputs[0], cmp.inputs[1])
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(cmp.outputs[0], mix.inputs[0])
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(bs.outputs[0], mix.inputs[2])
    sub = nt.nodes.new("ShaderNodeMath"); sub.operation = "SUBTRACT"
    nt.links.new(sep.outputs["Z"], sub.inputs[0]); nt.links.new(hv.outputs[0], sub.inputs[1])
    ab = nt.nodes.new("ShaderNodeMath"); ab.operation = "ABSOLUTE"
    nt.links.new(sub.outputs[0], ab.inputs[0])
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = 0.0, 0.006
    mr.inputs["To Min"].default_value, mr.inputs["To Max"].default_value = 1.0, 0.0
    nt.links.new(ab.outputs[0], mr.inputs["Value"])
    emi = nt.nodes.new("ShaderNodeEmission")
    emi.inputs[0].default_value = (0.85, 0.92, 1.0, 1)
    mul = nt.nodes.new("ShaderNodeMath"); mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = a.seam
    nt.links.new(mr.outputs[0], mul.inputs[0])
    nt.links.new(mul.outputs[0], emi.inputs[1])
    add = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(mix.outputs[0], add.inputs[0])
    nt.links.new(emi.outputs[0], add.inputs[1])
    nt.links.new(add.outputs[0], out.inputs["Surface"])
    return mat


def backdrop_layer(cam, meta, a):
    """>v<🧊四维切片-Blender-可替换背景：一块跟着相机走的板（相机子物体，30 单位外，略大于画框），材质 = 自发光图（不受灯光影响）。
    --backdrop 不给 → 颜色纯黑；以后换参考图：改 backdrop 材质里的 Image 节点，或重跑带 --backdrop"""
    W, H = meta["size"]
    h = H - meta["hud_band"]
    dist = 30.0
    ph = 2 * dist * math.tan(math.radians(meta["fov"]) / 2) * 1.15
    bpy.ops.mesh.primitive_plane_add(size=1.0)
    bd = bpy.context.active_object
    bd.name = "backdrop"
    bd.scale = (ph * W / h, ph, 1.0)
    bd.parent = cam
    bd.location = (0, 0, -dist)
    bd.visible_shadow = False
    bd.visible_diffuse = bd.visible_glossy = False                   # 不往身上反光，只给相机看
    m = bpy.data.materials.new("backdrop")
    m.use_nodes = True
    nt = m.node_tree
    for nd in list(nt.nodes):
        nt.nodes.remove(nd)
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs[0].default_value = (0, 0, 0, 1)
    em.inputs[1].default_value = 1.0
    if a.backdrop:
        img = nt.nodes.new("ShaderNodeTexImage")
        img.image = bpy.data.images.load(str(Path(a.backdrop).resolve()))
        nt.links.new(img.outputs[0], em.inputs[0])
    outn = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(em.outputs[0], outn.inputs[0])
    bd.data.materials.append(m)


def cam_dir(fr, az, el):
    """相机（含 roll）基里的方向：az 0 = 从相机打、180 = 正后方、+90 = 画面右；el 沿画面上方"""
    f, u = Vector(fr["fwd"]).normalized(), Vector(fr["up"]).normalized()
    r = f.cross(u).normalized()
    A, E = math.radians(az), math.radians(el)
    return (math.cos(E) * (math.cos(A) * -f + math.sin(A) * r) + math.sin(E) * u).normalized()


def set_frame(fr, S, sweep_h=None):
    cam = S["cam"]
    cam.location = fr["eye"]
    cam.rotation_quaternion = look_quat(fr["fwd"], fr["up"])
    cam.data.dof.focus_distance = fr["focus"]
    if fr["rim"] is not None:
        S["rims"][0].rotation_quaternion = sun_toward(fr["rim"])
    S["rims"][1].rotation_quaternion = sun_toward(cam_dir(fr, S["rim2"][1], S["rim2"][2]))
    s = fr["red_strength"]
    S["red"].location = S["redl"].location = Vector(fr["red"]) - 1.2 * S["red"].dimensions.x / 2 * Vector(fr["fwd"]).normalized()   # 往相机挪半个球：别埋进高模表面
    S["red"].hide_render = s <= 0.02
    S["em"].inputs[1].default_value = 40.0 * s
    S["redl"].data.energy = 0.03 * s
    if sweep_h is not None:
        for m in (S["mat"], S["mat_hi"]):
            m.node_tree.nodes["sweep_h"].outputs[0].default_value = sweep_h


def sweep_height(i, a, zlo, zhi):
    """扫描线高度：a 帧之前在头顶之上（全切片），b 帧之后在脚下（全高模），中间 smoothstep（慢起慢收）"""
    f0, f1 = (float(x) for x in a.sweep.split(","))
    u = min(max((i - f0) / max(f1 - f0, 1e-9), 0.0), 1.0)
    u = u * u * (3 - 2 * u)
    return (zhi + 0.02) - (zhi - zlo + 0.04) * u


def main():
    a = args()
    seq = Path(a.seq)
    meta = json.loads((seq / "track.json").read_text(encoding="utf-8"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    S = scene_setup(meta, a)
    hires, zlo, zhi = None, None, None
    if a.hires:
        V, N, Q = read_ply(a.hires)
        hires = build_mesh("hires", V, N, Q, S["mat_hi"])
        zlo, zhi = float(V[:, 2].min()), float(V[:, 2].max())
        print(f"  hires: {len(V)} verts, {len(Q)} faces, z {zlo:.3f}..{zhi:.3f}")
    frames = meta["frames"]
    if a.frames:
        lo, hi = (int(x) for x in a.frames.split("-"))
        frames = [f for f in frames if lo <= f["i"] <= hi]
    sc = bpy.context.scene
    t0 = time.time()
    for n, fr in enumerate(frames):
        t1 = time.time()
        V, N, Q = read_ply(seq / fr["mesh"])
        ob = build_mesh(f"slice_{fr['i']:04d}", V, N, Q, S["mat"])
        h = sweep_height(fr["i"], a, zlo, zhi) if hires is not None else None
        set_frame(fr, S, h)
        if a.save_blend and n == 0:
            bpy.ops.wm.save_as_mainfile(filepath=str(Path(a.save_blend).resolve()), copy=True)
            print("  saved", a.save_blend)
        sc.render.filepath = str((out / f"f{fr['i']:04d}.png").resolve())
        bpy.ops.render.render(write_still=True)
        me = ob.data
        bpy.data.objects.remove(ob, do_unlink=True)
        bpy.data.meshes.remove(me)
        print(f"  frame {fr['i']}: {len(Q)} quads" + ("" if h is None else f"  sweep z = {h:+.3f}") + f"  {time.time() - t1:.2f}s", flush=True)
    print(f"done: {len(frames)} frames, {time.time() - t0:.1f}s ({(time.time() - t0) / max(1, len(frames)):.2f} s/frame)")


main()
