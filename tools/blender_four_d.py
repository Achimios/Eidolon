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
    return ap.parse_args(argv)


def read_ply(path):
    """four_d.write_ply 的格式：binary LE，顶点 xyz + nxnynz，面全是四边形（uchar 4 + int×4）"""
    data = Path(path).read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    hdr = data[:end].decode("ascii").split("\n")
    nv = int(next(l for l in hdr if l.startswith("element vertex")).split()[-1])
    nf = int(next(l for l in hdr if l.startswith("element face")).split()[-1])
    vert = np.frombuffer(data, dtype="<f4", count=nv * 6, offset=end).reshape(nv, 6)
    face = np.frombuffer(data, dtype=np.dtype([("n", "u1"), ("v", "<i4", (4,))]), count=nf, offset=end + nv * 24)
    return vert[:, :3].copy(), vert[:, 3:].copy(), face["v"].copy()


def build_mesh(name, V, N, Q, mat):
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(V))
    me.vertices.foreach_set("co", V.ravel())
    me.loops.add(Q.size)
    me.loops.foreach_set("vertex_index", Q.ravel().astype(np.int32))
    me.polygons.add(len(Q))
    me.polygons.foreach_set("loop_start", (np.arange(len(Q)) * 4).astype(np.int32))
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
    # 材质：冷灰石膏，略带光泽
    mat = bpy.data.materials.new("plaster")
    mat.use_nodes = True
    bs = mat.node_tree.nodes["Principled BSDF"]
    bs.inputs["Base Color"].default_value = (0.70, 0.74, 0.78, 1)
    bs.inputs["Roughness"].default_value = 0.42
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
    # 主光（世界固定，硬）
    kd = bpy.data.lights.new("key", "SUN")
    kd.energy, kd.angle, kd.color = a.key, math.radians(1.5), (0.97, 0.98, 1.0)
    key = bpy.data.objects.new("key", kd)
    key.rotation_mode = "QUATERNION"
    key.rotation_quaternion = sun_toward(meta["key"])
    sc.collection.objects.link(key)
    # 轮廓光（跟相机 roll，逐帧）
    rd = bpy.data.lights.new("rim", "SUN")
    rd.energy, rd.angle, rd.color = a.rim * meta["rim"], math.radians(3.0), (0.86, 0.92, 1.0)
    rim = bpy.data.objects.new("rim", rd)
    rim.rotation_mode = "QUATERNION"
    sc.collection.objects.link(rim)
    # 唯一红点：自发光小球 + 红色点光
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
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.006, segments=16, ring_count=8)
    red = bpy.context.active_object
    red.name = "red_point"
    red.data.materials.append(rm)
    red.visible_shadow = False
    pd = bpy.data.lights.new("red_light", "POINT")
    pd.color, pd.shadow_soft_size = (1.0, 0.16, 0.20), 0.004
    redl = bpy.data.objects.new("red_light", pd)
    sc.collection.objects.link(redl)
    return cam, rim, red, redl, em, mat


def set_frame(fr, meta, cam, rim, red, redl, em):
    cam.location = fr["eye"]
    cam.rotation_quaternion = look_quat(fr["fwd"], fr["up"])
    cam.data.dof.focus_distance = fr["focus"]
    if fr["rim"] is not None:
        rim.rotation_quaternion = sun_toward(fr["rim"])
    s = fr["red_strength"]
    red.location = redl.location = fr["red"]
    red.hide_render = s <= 0.02
    em.inputs[1].default_value = 40.0 * s
    redl.data.energy = 0.015 * s                 # 只让红点自己亮，不往身上泼红（👤单红点）


def main():
    a = args()
    seq = Path(a.seq)
    meta = json.loads((seq / "track.json").read_text(encoding="utf-8"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cam, rim, red, redl, em, mat = scene_setup(meta, a)
    frames = meta["frames"]
    if a.frames:
        lo, hi = (int(x) for x in a.frames.split("-"))
        frames = [f for f in frames if lo <= f["i"] <= hi]
    sc = bpy.context.scene
    t0 = time.time()
    for n, fr in enumerate(frames):
        t1 = time.time()
        V, N, Q = read_ply(seq / fr["mesh"])
        ob = build_mesh(f"slice_{fr['i']:04d}", V, N, Q, mat)
        set_frame(fr, meta, cam, rim, red, redl, em)
        if a.save_blend and n == 0:
            bpy.ops.wm.save_as_mainfile(filepath=str(Path(a.save_blend).resolve()), copy=True)
            print("  saved", a.save_blend)
        sc.render.filepath = str((out / f"f{fr['i']:04d}.png").resolve())
        bpy.ops.render.render(write_still=True)
        me = ob.data
        bpy.data.objects.remove(ob, do_unlink=True)
        bpy.data.meshes.remove(me)
        print(f"  frame {fr['i']}: {len(Q)} quads  {time.time() - t1:.2f}s", flush=True)
    print(f"done: {len(frames)} frames, {time.time() - t0:.1f}s ({(time.time() - t0) / max(1, len(frames)):.2f} s/frame)")


main()
