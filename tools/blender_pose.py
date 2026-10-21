# -*- coding: utf-8 -*-
""">v<🧍她-摆姿势：Tripo 网格（没绑骨）→ 横截面拟合关节 → 程序化人形骨架 → 体素代理上算热扩散权重 → 迁到渲染网格 →
按「方向」摆蜷缩姿势（攻壳机动队 1995 片头「诞生」：膝盖收到胸前、手臂环抱、低头、脊柱弯成弓、漂浮）→ 导出 GLB + 灰模预览。

跑法（Blender 自带 python，不装新库）：
  "E:/2_Wares/Blender Foundation/Blender 5.2/blender.exe" -b --python tools/blender_pose.py -- [--variants 蜷缩_v1,蜷缩_v2] [--joints-only]
输出（默认在 MV/3D/，已存在则拒绝 → 换版本号）：
  她_<变体>.glb（静态摆好的网格）、她_rig.blend（骨架 + 蒙皮 + 每个变体一条 Action）、
  她_蜷缩_预览_v1.jpg（每变体一行：正面 / 左侧 / 四分之三俯 / 背后拉伸热图 —— 红 = 边长变了 ≥ 2 倍）

⚠️导出的 GLB 侧向轴恒为 glTF z；蜷缩后前后比左右厚，four_d 的自动判断会猜错 → 用 four_d 时必须加 --mesh-lateral z。
坐标（Blender，glTF 导入后）：X = 她的正面，Y = 她的左侧，Z = 上。
关节拟合 = 每 1% 身高切一片水平截面，2D 栅格连通块：
  腋下 = 从腰往上第一片只剩 1 块；手臂 = 腋下以下最外侧两块（到指尖）；裆 = 中间开始分成 2 块；
  腕 = 指尖往上 0.085–0.15 H 窗口里手臂最细（宽×厚最小）；肘 = 肩→腕折线弧长 0.56 处（上臂:前臂 ≈ 0.186:0.146 H）；
  脖子 = 腋下以上最窄的一片；踝 = 腿底部 3–15% 身高内最细；膝 = 髋→踝中点；脚尖 = 最低一片里最靠前的点。
摆姿势 = 每根骨给目标方向（身体系），从父骨继承扭转后做最短弧摆动（不会凭空扭麻花）；
变形 = Armature（保体积 / 对偶四元数）+ Corrective Smooth；自检 = 边长拉伸比分布 + 零权重顶点数。
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector, kdtree

MV = Path("C:/Users/Administrator/Desktop/Projekt_Visual/MV_Cant_Kill_Us")

# ---------- 姿势：目标方向（身体系 F=+X 前，L=+Y 左，U=+Z 上；.R 没写就镜像 .L）+ 可选自扭转（度） ----------
POSES = {
    # 紧：髋屈 ~125°，膝屈 ~150°（脚跟收到臀下），脊柱弓起，头埋到交叠的前臂上，两臂环抱膝盖（左臂在上）
    "蜷缩_v1": {
        "pelvis": (-0.30, 0, 1), "spine_01": (0.35, 0, 1), "spine_02": (0.85, 0, 1), "spine_03": (1.6, 0, 1),
        "neck": (1.5, 0, 0.45), "head": (1.0, 0, -0.30),
        "clavicle.L": (0.55, 1, 0.05),
        "upper_arm.L": (0.75, 0.10, -0.65), "forearm.L": (0.35, -0.90, 0.20), "hand.L": (0.10, -0.85, -0.45),
        "upper_arm.R": (0.80, -0.10, -0.60), "forearm.R": (0.30, 0.92, -0.05), "hand.R": (0.05, 0.85, -0.50),
        "thigh.L": (0.85, 0.10, 0.60), "shin.L": (-0.55, 0.02, -0.84), "foot.L": (0.05, 0, -1),
        "thigh.R": (0.85, -0.08, 0.52), "shin.R": (-0.60, -0.02, -0.80), "foot.R": (0.0, 0, -1),
    },
    # 松：漂浮，脊柱轻弓，头微微后仰（下巴略抬、脸朝前上），两臂交叠收在胸前，膝盖收起但没贴胸，左右不对称
    "蜷缩_v2": {
        "pelvis": (-0.20, 0, 1), "spine_01": (0.25, 0, 1), "spine_02": (0.55, 0, 1), "spine_03": (0.85, 0, 1),
        "neck": (0.55, 0, 1), "head": (-0.10, 0, 1),
        "clavicle.L": (0.40, 1, 0.08),
        "upper_arm.L": (0.55, 0.15, -0.85), "forearm.L": (0.30, -0.80, 0.55), "hand.L": (0.20, -0.75, 0.60),
        "upper_arm.R": (0.45, -0.18, -0.88), "forearm.R": (0.45, 0.75, 0.40), "hand.R": (0.35, 0.70, 0.45),
        "thigh.L": (0.85, 0.14, 0.42), "shin.L": (-0.35, 0.06, -0.94), "foot.L": (0.15, 0, -1),
        "thigh.R": (0.85, -0.18, 0.22), "shin.R": (-0.15, -0.08, -1), "foot.R": (0.30, 0, -1),
    },
}
TWIST = {}  # {变体: {骨: 度}}，需要时再加


def log(*a):
    print("[pose]", *a, flush=True)


# ---------- 横截面 ----------
def _comps2d(pts, cell):
    """平面点 → 栅格连通块（8 邻接，手写 flood fill：Blender 自带 python 没有 scipy）。返回每块的统计，按 y 排序"""
    ij = np.floor(pts / cell).astype(np.int64)
    ij -= ij.min(0)
    sh = tuple(ij.max(0) + 3)
    occ = np.zeros(sh, bool)
    occ[ij[:, 0] + 1, ij[:, 1] + 1] = True
    lab = np.zeros(sh, np.int32)
    n = 0
    for s in zip(*np.nonzero(occ)):
        if lab[s]:
            continue
        n += 1
        st = [s]
        lab[s] = n
        while st:
            a, b = st.pop()
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    q = (a + da, b + db)
                    if occ[q] and not lab[q]:
                        lab[q] = n
                        st.append(q)
    L = lab[ij[:, 0] + 1, ij[:, 1] + 1]
    out = []
    for k in range(1, n + 1):
        q = pts[L == k]
        if len(q) < 30:
            continue
        out.append(dict(x=q[:, 0].mean(), y=q[:, 1].mean(), w=np.ptp(q[:, 1]), d=np.ptp(q[:, 0]),
                        xmax=q[:, 0].max(), xmin=q[:, 0].min(), n=len(q)))
    return sorted(out, key=lambda c: c["y"])


def fit_joints(P):
    """>v<🧍她-关节拟合-横截面分析。P = 顶点 (N,3)，Blender 坐标。返回 {名字: Vector} 和诊断"""
    z0, z1 = P[:, 2].min(), P[:, 2].max()
    H = z1 - z0
    th, cell = 0.0031 * H, 0.0061 * H
    S = []
    for f in np.arange(0.01, 0.995, 0.01):
        z = z0 + f * H
        sel = P[np.abs(P[:, 2] - z) < th]
        S.append(dict(f=f, z=z, c=_comps2d(sel[:, :2], cell) if len(sel) > 50 else []))
    mid = next(i for i, s in enumerate(S) if s["f"] >= 0.5)
    arm_i = next(i for i in range(mid, len(S)) if len(S[i]["c"]) == 1)          # 腋下：往上第一片只剩一块
    armpit = S[arm_i]["z"]
    arms = {+1: [], -1: []}
    crotch_i = None
    i = arm_i - 1
    arm_on = True
    while i > 0:
        cs = S[i]["c"]
        if arm_on and len(cs) >= 3:
            arms[-1].append((S[i]["z"], cs[0]))
            arms[+1].append((S[i]["z"], cs[-1]))
            inner = cs[1:-1]
        else:
            arm_on = False
            inner = cs
        if crotch_i is None and len(inner) >= 2:
            crotch_i = i
        if not arm_on and crotch_i is not None:
            break
        i -= 1
    step = 0.01 * H
    crotch = S[crotch_i]["z"] + step / 2
    tip = min(z for z, _ in arms[+1]) - step / 2

    J = {}
    torso_at = lambda z: min((s for s in S if len(s["c"]) >= 1), key=lambda s: abs(s["z"] - z))  # noqa: E731

    def torso_c(z):
        cs = torso_at(z)["c"]
        return min(cs, key=lambda c: abs(c["y"]))

    # 腿
    legs = {+1: [], -1: []}
    for s in S:
        if s["z"] < crotch - step:
            inner = [c for c in s["c"] if abs(c["y"]) < 0.13 * H] if s["z"] < tip else s["c"][1:-1]
            if len(inner) >= 2:
                legs[-1].append((s["z"], inner[0]))
                legs[+1].append((s["z"], inner[-1]))
    for sd, nm in ((+1, "L"), (-1, "R")):
        lg = legs[sd]
        upper = [c for z, c in lg if crotch - 0.07 * H < z < crotch - 0.015 * H]
        hy = float(np.mean([c["y"] for c in upper]))
        hz = crotch + 0.045 * H
        J[f"hip.{nm}"] = Vector((float(torso_c(crotch + step)["x"]), hy, hz))
        low = [(z, c) for z, c in lg if z0 + 0.03 * H < z < z0 + 0.15 * H]
        za, ca = min(low, key=lambda t: t[1]["w"] * t[1]["d"])
        J[f"ankle.{nm}"] = Vector((float(ca["x"]), float(ca["y"]), float(za) - 0.01 * H))
        kz = (hz + J[f"ankle.{nm}"].z) / 2
        zk, ck = min(lg, key=lambda t: abs(t[0] - kz))
        J[f"knee.{nm}"] = Vector((float(ck["x"]) + 0.1 * float(ck["d"]), float(ck["y"]), float(kz)))
        sel = P[(P[:, 2] < z0 + 0.03 * H) & (np.sign(P[:, 1]) == sd)]
        J[f"toe.{nm}"] = Vector((float(sel[:, 0].max()) - 0.01 * H, float(ca["y"]), z0 + 0.02 * H))
    # 手臂
    for sd, nm in ((+1, "L"), (-1, "R")):
        pl = arms[sd]                                  # 从腋下往下
        top = pl[0][1]
        sh = Vector((float(top["x"]), float(top["y"]) - sd * 0.012 * H, armpit + 0.055 * H))
        cand = [t for t in pl if tip + 0.085 * H < t[0] < tip + 0.15 * H]     # 手长 ≈ 0.108 H：腕只在这个窗口里找（扁手掌更细，别被它骗）
        zw, cw = min(cand, key=lambda t: t[1]["w"] * t[1]["d"])
        wr = Vector((float(cw["x"]), float(cw["y"]), float(zw)))
        poly = [sh] + [Vector((float(c["x"]), float(c["y"]), float(z))) for z, c in pl if z > zw] + [wr]
        seg = [(b - a).length for a, b in zip(poly, poly[1:])]
        tgt, acc = 0.56 * sum(seg), 0.0
        el = poly[-1]
        for a, b, L in zip(poly, poly[1:], seg):
            if acc + L >= tgt:
                el = a.lerp(b, (tgt - acc) / L)
                break
            acc += L
        el = el + Vector((-0.012 * H, 0, 0))           # 肘关节在前臂中轴偏后
        zt, ct = pl[-1]
        J[f"shoulder.{nm}"], J[f"elbow.{nm}"], J[f"wrist.{nm}"] = sh, el, wr
        J[f"fingertip.{nm}"] = Vector((float(ct["x"]), float(ct["y"]), tip))
    # 躯干 / 脖子 / 头
    up = [s for s in S if armpit < s["z"] < z1 - 0.1 * H and len(s["c"]) == 1]
    nk = min(up, key=lambda s: s["c"][0]["w"])
    neck_min = nk["z"]
    neck_base = neck_min - 0.045 * H
    hip_z = J["hip.L"].z

    def spine_pt(z):
        c = torso_c(z)
        return Vector((float(c["x"]) - 0.15 * float(c["d"]), 0.0, float(z)))

    for k, fr in (("pelvis", 0.0), ("spine_01", 0.12), ("spine_02", 0.42), ("spine_03", 0.72)):
        J[k] = spine_pt(hip_z + fr * (neck_base - hip_z))
    J["neck"] = spine_pt(neck_base)
    J["head"] = Vector((float(nk["c"][0]["x"]) - 0.15 * float(nk["c"][0]["d"]), 0.0, float(neck_min)))
    J["head_top"] = Vector((float(spine_pt(z1 - 0.06 * H).x) + 0.02 * H, 0.0, float(z1)))
    for sd, nm in ((+1, "L"), (-1, "R")):
        J[f"clavicle.{nm}"] = Vector((J["neck"].x + 0.02 * H, sd * 0.015 * H, neck_base - 0.025 * H))
    diag = dict(H=float(H), armpit=float(armpit), crotch=float(crotch), fingertip=float(tip), neck_min=float(neck_min),
                slabs=[(round(float(s["f"]), 2), len(s["c"])) for s in S])
    return J, diag


# ---------- 骨架 ----------
BONES = [  # 名字, 头, 尾, 父, 相连, 滚转对齐
    ("pelvis", "pelvis", "spine_01", None, False, "X"),
    ("spine_01", "spine_01", "spine_02", "pelvis", True, "X"),
    ("spine_02", "spine_02", "spine_03", "spine_01", True, "X"),
    ("spine_03", "spine_03", "neck", "spine_02", True, "X"),
    ("neck", "neck", "head", "spine_03", True, "X"),
    ("head", "head", "head_top", "neck", True, "X"),
]
for _s in "LR":
    BONES += [
        (f"clavicle.{_s}", f"clavicle.{_s}", f"shoulder.{_s}", "spine_03", False, "X"),
        (f"upper_arm.{_s}", f"shoulder.{_s}", f"elbow.{_s}", f"clavicle.{_s}", True, "X"),
        (f"forearm.{_s}", f"elbow.{_s}", f"wrist.{_s}", f"upper_arm.{_s}", True, "X"),
        (f"hand.{_s}", f"wrist.{_s}", f"fingertip.{_s}", f"forearm.{_s}", True, "X"),
        (f"thigh.{_s}", f"hip.{_s}", f"knee.{_s}", "pelvis", False, "X"),
        (f"shin.{_s}", f"knee.{_s}", f"ankle.{_s}", f"thigh.{_s}", True, "X"),
        (f"foot.{_s}", f"ankle.{_s}", f"toe.{_s}", f"shin.{_s}", True, "Z"),
    ]


def build_armature(J):
    arm = bpy.data.armatures.new("她_骨架")
    ob = bpy.data.objects.new("她_骨架", arm)
    bpy.context.scene.collection.objects.link(ob)
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm.edit_bones
    for name, h, t, par, conn, roll in BONES:
        b = eb.new(name)
        b.head, b.tail = J[h], J[t]
        b.align_roll(Vector((1, 0, 0)) if roll == "X" else Vector((0, 0, 1)))
        if par:
            b.parent = eb[par]
            b.use_connect = conn
    bpy.ops.object.mode_set(mode="OBJECT")
    arm.display_type = "STICK"
    return ob


# ---------- 网格 ----------
def import_mesh(src):
    bpy.ops.import_scene.gltf(filepath=str(src))
    obs = [o for o in bpy.context.selected_objects if o.type == "MESH"]
    bpy.ops.object.select_all(action="DESELECT")
    for o in obs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = obs[0]
    if len(obs) > 1:
        bpy.ops.object.join()
    ob = bpy.context.view_layer.objects.active
    for o in list(bpy.context.scene.objects):
        if o is not ob and o.type != "MESH":
            bpy.data.objects.remove(o)
    ob.parent = None
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    ob.data.materials.clear()
    for im in list(bpy.data.images):
        bpy.data.images.remove(im)
    ob.name = "她"
    return ob


def verts(ob):
    a = np.empty(len(ob.data.vertices) * 3, np.float32)
    ob.data.vertices.foreach_get("co", a)
    return a.reshape(-1, 3).astype(np.float64)


def apply_mod(ob, mod):
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.modifier_apply(modifier=mod.name)


def decimate(ob, faces):
    n = len(ob.data.polygons)
    if n > faces:
        m = ob.modifiers.new("dec", "DECIMATE")
        m.ratio = faces / n
        apply_mod(ob, m)
    log(f"{ob.name}: {n} -> {len(ob.data.polygons)} faces")


def make_proxy(ob, voxel, faces):
    """体素重建的水密代理：Tripo 网格 93 个岛、非水密，热扩散权重直接上会失败或漏"""
    px = ob.copy()
    px.data = ob.data.copy()
    px.name = "她_代理"
    bpy.context.scene.collection.objects.link(px)
    m = px.modifiers.new("vox", "REMESH")
    m.mode, m.voxel_size = "VOXEL", voxel
    apply_mod(px, m)
    decimate(px, faces)
    return px


def heat_weights(px, rig):
    bpy.ops.object.select_all(action="DESELECT")
    px.select_set(True)
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    W = weight_matrix(px, [b.name for b in rig.data.bones])
    return W


def weight_matrix(ob, names):
    W = np.zeros((len(ob.data.vertices), len(names)), np.float32)
    gi = {g.index: names.index(g.name) for g in ob.vertex_groups if g.name in names}
    for v in ob.data.vertices:
        for g in v.groups:
            if g.group in gi:
                W[v.index, gi[g.group]] = g.weight
    return W


def bone_distance_weights(P, rig, names, falloff):
    """兜底：到骨段的距离 → 软权重（热扩散没覆盖到的顶点用）"""
    D = np.zeros((len(P), len(names)))
    for j, nm in enumerate(names):
        b = rig.data.bones[nm]
        a, c = np.array(b.head_local), np.array(b.tail_local)
        ab = c - a
        t = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)
        D[:, j] = np.linalg.norm(P - (a + t[:, None] * ab), axis=1)
    W = np.exp(-(D / falloff) ** 2) * (D <= D.min(1, keepdims=True) * 1.6 + 1e-9)
    return W / W.sum(1, keepdims=True)


def transfer_weights(ob, Pp, Wp, k=6):
    """代理 → 渲染网格：k 近邻反距离加权（自带平滑）"""
    kd = kdtree.KDTree(len(Pp))
    for i, p in enumerate(Pp):
        kd.insert(p, i)
    kd.balance()
    P = verts(ob)
    W = np.zeros((len(P), Wp.shape[1]), np.float32)
    for i, p in enumerate(P):
        hits = kd.find_n(p, k)
        ws = np.array([1 / (d + 1e-4) for _, _, d in hits])
        W[i] = (ws[:, None] * Wp[[j for _, j, _ in hits]]).sum(0) / ws.sum()
    return W


def set_weights(ob, rig, W, names, cs_iters=6):
    ob.vertex_groups.clear()
    groups = [ob.vertex_groups.new(name=n) for n in names]
    W = W / np.maximum(W.sum(1, keepdims=True), 1e-9)
    for j, g in enumerate(groups):
        col = W[:, j]
        for w in np.unique(np.round(col[col > 0.004], 3)):
            idx = np.nonzero(np.abs(np.round(col, 3) - w) < 1e-6)[0]
            g.add(idx.tolist(), float(w), "REPLACE")
    ob.parent = rig
    am = ob.modifiers.new("Armature", "ARMATURE")
    am.object = rig
    am.use_deform_preserve_volume = True
    if cs_iters > 0:
        cs = ob.modifiers.new("CorrectiveSmooth", "CORRECTIVE_SMOOTH")
        cs.iterations, cs.smooth_type, cs.rest_source, cs.use_only_smooth = cs_iters, "LENGTH_WEIGHTED", "ORCO", False


# ---------- 摆姿势 ----------
def pose(rig, spec, twist=None):
    """>v<🧍她-摆姿势-按方向：每根骨先继承父骨的旋转（含扭转），再最短弧摆到目标方向。直接写 pose_bone.matrix（骨架空间）"""
    twist = twist or {}
    pbs = rig.pose.bones
    for pb in pbs:
        pb.rotation_mode = "QUATERNION"
        pb.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()
    Rp, Mp = {}, {}

    def tgt(name):
        if name in spec:
            return Vector(spec[name])
        if name.endswith(".R") and name[:-2] + ".L" in spec:
            x, y, z = spec[name[:-2] + ".L"]
            return Vector((x, -y, z))
        return None

    def visit(b):
        rest = b.matrix_local.to_3x3()
        if b.parent:
            pr = b.parent.matrix_local
            Rin = Rp[b.parent.name] @ pr.to_3x3().inverted() @ rest
            head = Mp[b.parent.name] @ (pr.inverted() @ b.head_local)
        else:
            Rin, head = rest.copy(), b.head_local.copy()
        d = tgt(b.name)
        R = Rin
        if d is not None:
            cur = Rin @ Vector((0, 1, 0))
            R = cur.rotation_difference(d.normalized()).to_matrix() @ Rin
        if b.name in twist:
            R = R @ Matrix.Rotation(math.radians(twist[b.name]), 3, "Y")
        M = Matrix.Translation(head) @ R.to_4x4()
        Rp[b.name], Mp[b.name] = R, M
        pbs[b.name].matrix = M
        bpy.context.view_layer.update()
        for c in b.children:
            visit(c)

    for b in rig.data.bones:
        if b.parent is None:
            visit(b)


def key_action(rig, name):
    rig.animation_data_create()
    rig.animation_data.action = None
    for pb in rig.pose.bones:
        pb.keyframe_insert("rotation_quaternion", frame=1)
        pb.keyframe_insert("location", frame=1)
    act = rig.animation_data.action
    act.name = name
    act.use_fake_user = True
    return act


def evaluated_copy(ob, name):
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    o = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(o)
    return o


def edge_stretch(rest_ob, posed_ob):
    """自检：每条边 摆后长度 / 静止长度。>v< 必须相等的量：拓扑相同，边一一对应"""
    e = np.empty(len(rest_ob.data.edges) * 2, np.int64)
    rest_ob.data.edges.foreach_get("vertices", e)
    e = e.reshape(-1, 2)
    A, B = verts(rest_ob), verts(posed_ob)
    assert len(A) == len(B)
    l0 = np.linalg.norm(A[e[:, 0]] - A[e[:, 1]], axis=1)
    r = np.linalg.norm(B[e[:, 0]] - B[e[:, 1]], axis=1) / np.maximum(l0, 1e-9)
    lr = np.abs(np.log(np.maximum(r, 1e-6)))
    # 每个顶点：相邻边 |log 拉伸| 的「按静止边长加权」平均（减面留下的极短碎边比值会乱跳，不该让它们主导）
    # → 顶点色：灰 = 变化 ≤ 1.25 倍，红 = 拉长 / 压扁 ≥ 2 倍；预览第 4 列看哪里坏
    num, den = np.zeros(len(A)), np.zeros(len(A))
    for j in (0, 1):
        np.add.at(num, e[:, j], lr * l0)
        np.add.at(den, e[:, j], l0)
    pv = num / np.maximum(den, 1e-12)
    k = np.clip((pv - math.log(1.25)) / (math.log(2) - math.log(1.25)), 0, 1)[:, None]
    col = np.concatenate([(1 - k) * 0.6 + k * np.array([0.9, 0.12, 0.1]), np.ones((len(A), 1))], 1).astype(np.float32)
    at = posed_ob.data.color_attributes.new("stretch", "FLOAT_COLOR", "POINT")
    at.data.foreach_set("color", col.ravel())
    posed_ob.data.color_attributes.active_color = at
    lg = l0 >= np.median(l0)
    return dict(p01=float(np.percentile(r, 1)), p50=float(np.median(r)), p99=float(np.percentile(r, 99)), max=float(r.max()),
                frac_gt2=float((r > 2).mean()), frac_lt05=float((r < 0.5).mean()),
                long_p99=float(np.percentile(r[lg], 99)), long_frac_gt2=float((r[lg] > 2).mean()), long_frac_lt05=float((r[lg] < 0.5).mean()),
                verts_bad=float((pv > math.log(2)).mean()), verts_warn=float((pv > math.log(1.25)).mean()),
                max_disp=float(np.linalg.norm(B - A, axis=1).max()))


# ---------- 预览 ----------
def render_views(ob, out_png_prefix, size=560):
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sh = sc.display.shading
    sh.light, sh.color_type = "STUDIO", "SINGLE"
    sh.single_color = (0.62, 0.62, 0.62)
    sh.show_cavity = True
    sc.render.resolution_x = sc.render.resolution_y = size
    sc.render.film_transparent = False
    sc.world = sc.world or bpy.data.worlds.new("w")
    sc.render.image_settings.file_format = "PNG"
    P = verts(ob) @ np.array(ob.matrix_world.to_3x3()).T + np.array(ob.matrix_world.translation)
    c = Vector(((P.min(0) + P.max(0)) / 2).tolist())
    ext = float(np.ptp(P, 0).max())
    cam = bpy.data.objects.get("预览相机") or bpy.data.objects.new("预览相机", bpy.data.cameras.new("预览相机"))
    if cam.name not in sc.collection.objects:
        sc.collection.objects.link(cam)
    cam.data.type, cam.data.ortho_scale = "ORTHO", ext * 1.18
    sc.camera = cam
    outs = []
    for k, (az, el) in enumerate(((0, 8), (90, 8), (40, 25), (215, 15))):   # 正面 / 左侧 / 四分之三俯 / 背后（拉伸热图）
        sh.color_type = "VERTEX" if k == 3 else "SINGLE"
        a, e = math.radians(az), math.radians(el)
        d = Vector((math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)))
        cam.location = c + d * ext * 3
        cam.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
        p = f"{out_png_prefix}_{k}.png"
        sc.render.filepath = p
        bpy.ops.render.render(write_still=True)
        outs.append(p)
    return outs


def contact(rows, out_jpg, size=560):
    """每行一个变体 × 3 角度，拼成一张 jpg（用 Blender 的图像 API，不依赖 PIL）"""
    R, C = len(rows), len(rows[0])
    big = np.zeros((R * size, C * size, 4), np.float32)
    for r, pngs in enumerate(rows):
        for cidx, p in enumerate(pngs):
            im = bpy.data.images.load(p)
            a = np.empty(size * size * 4, np.float32)
            im.pixels.foreach_get(a)
            y0 = (R - 1 - r) * size                       # Blender 像素原点在左下
            big[y0:y0 + size, cidx * size:(cidx + 1) * size] = a.reshape(size, size, 4)
            bpy.data.images.remove(im)
    img = bpy.data.images.new("preview", C * size, R * size, alpha=False)
    img.pixels.foreach_set(big.ravel())
    img.filepath_raw = str(out_jpg)
    img.file_format = "JPEG"
    img.save()


# ---------- 主流程 ----------
def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(MV / "3D" / "她.glb"))
    ap.add_argument("--out-dir", default=str(MV / "3D"))
    ap.add_argument("--variants", default="蜷缩_v1,蜷缩_v2")
    ap.add_argument("--rig", default="她_rig.blend")
    ap.add_argument("--preview", default="她_蜷缩_预览_v1.jpg")
    ap.add_argument("--faces", type=int, default=200000, help="渲染网格减面目标")
    ap.add_argument("--voxel", type=float, default=0.004, help="代理体素边长（模型单位；她身高 ≈ 0.98）")
    ap.add_argument("--proxy-faces", type=int, default=60000)
    ap.add_argument("--cs-iters", type=int, default=6, help="Corrective Smooth 迭代数（0 = 不加）")
    ap.add_argument("--joints-only", action="store_true", help="只打印拟合的关节")
    ap.add_argument("--tmp", default=None, help="中间 png 目录（默认系统临时目录/blender_pose_tmp）")
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    variants = [v for v in a.variants.split(",") if v]
    stem = Path(a.src).stem
    targets = [out / f"{stem}_{v}.glb" for v in variants] + [out / a.rig, out / a.preview]
    if not a.joints_only:
        ex = [str(p) for p in targets if p.exists()]
        if ex:
            sys.exit(f"已存在，不覆盖：{ex}（换版本号）")
    t0 = time.time()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    ob = import_mesh(a.src)
    log(f"imported {len(ob.data.vertices)} verts / {len(ob.data.polygons)} faces, {time.time() - t0:.1f}s")
    J, diag = fit_joints(verts(ob))
    log("diag", json.dumps({k: v for k, v in diag.items() if k != "slabs"}))
    for k, v in J.items():
        log(f"  {k:14s} x={v.x:+.3f} y={v.y:+.3f} z={v.z:+.3f}")
    if a.joints_only:
        return
    decimate(ob, a.faces)
    rig = build_armature(J)
    names = [b.name for b in rig.data.bones]
    px = make_proxy(ob, a.voxel, a.proxy_faces)
    t1 = time.time()
    Wp = heat_weights(px, rig)
    Pp = verts(px)
    zero = Wp.sum(1) < 1e-4
    log(f"heat weights on proxy ({len(Pp)} verts): {time.time() - t1:.1f}s, unweighted {zero.sum()} ({zero.mean() * 100:.2f}%)")
    if zero.any():
        H = diag["H"]
        Wp[zero] = bone_distance_weights(Pp[zero], rig, names, 0.05 * H)
    t1 = time.time()
    W = transfer_weights(ob, Pp, Wp)
    log(f"weights -> render mesh ({len(W)} verts): {time.time() - t1:.1f}s, unweighted {(W.sum(1) < 1e-4).sum()}")
    set_weights(ob, rig, W, names, a.cs_iters)
    px.hide_render = True
    px.hide_set(True)

    import tempfile
    tmp = Path(a.tmp or Path(tempfile.gettempdir()) / "blender_pose_tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    rows, report = [], {"joints": {k: list(v) for k, v in J.items()}, "diag": diag}
    rest_copy = evaluated_copy(ob, "rest_eval")          # 静止姿势经过同样的修改器 → 边一一对应
    for v in variants:
        pose(rig, POSES[v], TWIST.get(v))
        act = key_action(rig, v)
        bpy.context.view_layer.update()
        po = evaluated_copy(ob, f"{stem}_{v}")
        st = edge_stretch(rest_copy, po)
        report[v] = st
        log(f"{v}: edge stretch {json.dumps({k: round(x, 4) for k, x in st.items()})}")
        ob.hide_render = True
        rest_copy.hide_render = True
        po.hide_render = False
        rows.append(render_views(po, str(tmp / v)))
        bpy.ops.object.select_all(action="DESELECT")
        po.select_set(True)
        bpy.context.view_layer.objects.active = po
        glb = out / f"{stem}_{v}.glb"
        bpy.ops.export_scene.gltf(filepath=str(glb), use_selection=True, export_format="GLB", export_materials="NONE",
                                  export_yup=True, export_animations=False, export_skins=False)
        log(f"-> {glb} ({glb.stat().st_size / 1e6:.1f} MB)")
        po.hide_render = True
        po.hide_set(True)
        ob.hide_render = False
        _ = act
    contact(rows, out / a.preview)
    log(f"-> {out / a.preview}")
    # 存 rig：骨架回到静止姿势（Action 都留着、带 fake user），删掉临时对象
    for o in [rest_copy] + [bpy.data.objects[f"{stem}_{v}"] for v in variants]:
        bpy.data.objects.remove(o)
    rig.animation_data.action = None
    for pb in rig.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
    bpy.ops.wm.save_as_mainfile(filepath=str(out / a.rig), compress=True)
    (out / a.rig).with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8") \
        if not (out / a.rig).with_suffix(".json").exists() else None
    log(f"-> {out / a.rig}   total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
