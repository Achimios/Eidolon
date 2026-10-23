#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""四维插入镜头的速度曲线自检（纯数学，不渲染）。python -X utf8 -m unittest discover -s tests -q"""
import importlib.util
import math
import sys
import unittest
from pathlib import Path

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import four_d as F  # noqa: E402


class _Obj:
    cam, c_rot, plane, cmax = 5.0, None, "xw", 1.0


@unittest.skipUnless(HAS_TORCH, "four_d 需要 torch（.venv）")
class TestFastSlow(unittest.TestCase):
    def test_velocity_endpoints(self):
        x, v = F.fast_slow(0.0, 300.0, 15.0, 0.22)
        self.assertEqual(x, 0.0)
        self.assertAlmostEqual(v, 300.0)
        _, v_inf = F.fast_slow(50.0, 300.0, 15.0, 0.22)
        self.assertAlmostEqual(v_inf, 15.0, places=6)

    def test_position_is_integral_of_velocity(self):
        """必须相等的量：数值微分 dx/dt = v(t)"""
        h = 1e-5
        for t in (0.01, 0.2, 0.7, 1.5):
            x1, _ = F.fast_slow(t - h, 240, 20, 0.3)
            x2, _ = F.fast_slow(t + h, 240, 20, 0.3)
            _, v = F.fast_slow(t, 240, 20, 0.3)
            self.assertAlmostEqual((x2 - x1) / (2 * h), v, places=4)

    def test_monotone_decreasing_never_stops(self):
        vs = [F.fast_slow(i / 24, 420, 30, 0.25)[1] for i in range(40)]
        self.assertTrue(all(a > b for a, b in zip(vs, vs[1:])))
        self.assertGreater(min(vs), 30)

    def test_frames_cover_slot(self):
        for dur, n in ((1.58, 38), (1.60, 39), (1.10, 27), (1.56, 38), (1.0, 24)):
            self.assertEqual(F.insert_frames(dur, 24), n)
            self.assertGreaterEqual(n / 24, dur - 1e-9)
            self.assertLess(n / 24 - dur, 1 / 24)

    def test_spec_rejects_stop(self):
        with self.assertRaises(ValueError):
            F.InsertSpec(1.0, v0=100, v_end=0)
        with self.assertRaises(ValueError):
            F.InsertSpec(1.0, yaw_rate=0)

    def test_tilt_parse(self):
        r = F.parse_tilt("zw=20, yw=-15,xy=12")
        self.assertAlmostEqual(r["zw"], math.radians(20))
        self.assertAlmostEqual(r["yw"], math.radians(-15))
        with self.assertRaises(ValueError):
            F.parse_tilt("ab=3")

    def test_track(self):
        sp = F.InsertSpec(1.1, theta0=72, v0=220, v_end=15, tau=0.18, c_start=0.35, cv0=-1.0, cv_end=-0.04, dolly=0.03, cam_scale=1.25)
        tr = F.insert_track(_Obj, sp)
        self.assertEqual(len(tr), 27)
        self.assertAlmostEqual(tr[0]["th"], 72)
        self.assertAlmostEqual(tr[0]["c"], 0.35)
        self.assertAlmostEqual(tr[0]["dist"], 5.0 * 1.25)
        steps = [b["th"] - a["th"] for a, b in zip(tr, tr[1:])]
        self.assertTrue(all(s1 > s2 > 0 for s1, s2 in zip(steps, steps[1:])))   # 先快后缓，不停
        yaws = [k["yaw"] for k in tr]
        self.assertTrue(all(b != a for a, b in zip(yaws, yaws[1:])))             # 相机始终在动


@unittest.skipUnless(HAS_TORCH, "four_d 需要 torch（.venv）")
class TestNarrowBandSDF(unittest.TestCase):
    def test_band_equals_full_and_never_overestimates(self):
        """必须相等的量：表面带内 = 全格暴力；带外 |d| 只能少报（球追踪才不会越过表面）"""
        import numpy as np
        V = np.array([[0.6, 0, 0], [-0.6, 0, 0], [0, 0.7, 0], [0, -0.7, 0], [0, 0, 0.5], [0, 0, -0.5]], float)
        Fc = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
        full = F.mesh_sdf_volume(V, Fc, 40, narrow=False)
        nb = F.mesh_sdf_volume(V, Fc, 40, coarse=16, narrow=True)
        h = 2 * F.MESH_B / 15
        band = np.abs(nb) < 2.5 * h - 0.5 * h
        self.assertLess(np.abs(nb - full)[np.abs(full) < 1.5 * h].max(), 1e-5)
        self.assertLessEqual((np.abs(nb) - np.abs(full)).max(), 1e-5)
        self.assertTrue((np.sign(nb) == np.sign(full)).all())
        self.assertGreater(band.sum(), 0)

    def test_hud_band_spec(self):
        F.InsertSpec(1.0, size=(640, 360), hud_band=100)
        with self.assertRaises(ValueError):
            F.InsertSpec(1.0, size=(640, 360), hud_band=200)

    def test_mesh_label_ascii(self):
        self.assertEqual(F.MeshSDF("她_蜷缩_v1.glb").label, "her_curled_v1")
        self.assertEqual(F.MeshSDF("她.glb").label, "her")
        self.assertEqual(F.MeshSDF("测试_人台.glb").label, "test_mannequin")
        self.assertEqual(F.MeshSDF("蜷缩_无材质_低模.glb").label, "curled_untextured_lowpoly")


class _Sphere:
    label = "sphere"

    def __call__(self, x, y, z):
        return F.L(x, y, z) - 0.5


@unittest.skipUnless(HAS_TORCH, "four_d 需要 torch（.venv）")
class TestLateralAndCut(unittest.TestCase):
    def test_lateral_rotate(self):
        """φ 方向转到 +x、行列式 +1；φ = −90 ≡ 旧的 "z" 选项 (x, y, z) → (−z, y, x)"""
        import numpy as np
        rng = np.random.default_rng(0)
        V = rng.normal(size=(50, 3))
        for phi in (0.0, 37.0, 60.0, 151.0):
            l = np.array([math.cos(math.radians(phi)), 0, math.sin(math.radians(phi))])
            self.assertTrue(np.allclose(F.lateral_rotate(l[None], phi), [[1, 0, 0]], atol=1e-12))
            A = F.lateral_rotate(np.eye(3), phi)          # 线性映射的矩阵（行 = 基向量的像）
            self.assertAlmostEqual(np.linalg.det(A), 1.0, places=12)
        self.assertTrue(np.allclose(F.lateral_rotate(V, -90.0), np.stack([-V[:, 2], V[:, 1], V[:, 0]], 1)))

    def test_symmetry_plane_recovers_known_yaw(self):
        """左右对称、前后不对称的形体斜摆 φ = 35°（侧向）后，镜面搜索要找回 35°（mod 180）；x / z 两个旧选项的误差都更大"""
        import numpy as np
        import trimesh
        parts = [trimesh.creation.box((0.3, 0.8, 0.3)).apply_translation(t) for t in ([0.35, 0, 0], [-0.35, 0, 0])]
        parts.append(trimesh.creation.box((0.4, 0.3, 0.6)).apply_translation([0, 0.3, 0.35]))   # 只在前面（+z）→ 前后不对称
        m = trimesh.util.concatenate(parts)
        a = math.radians(35.0)
        Rm = np.array([[math.cos(a), 0, -math.sin(a)], [0, 1, 0], [math.sin(a), 0, math.cos(a)]])   # x 轴 → (cos35, 0, sin35)
        m.vertices = m.vertices @ Rm.T
        phi, off, err, ex, ez = F.mesh_symmetry_plane(m, np.asarray(m.vertices), n=4000)
        d = (phi - 35.0 + 90) % 180 - 90
        self.assertLess(abs(d), 1.01)
        self.assertLess(err, 0.5 * min(ex, ez))

    def test_cut_face_marker(self):
        """extrude 的 cap(p)：|w| = h 的平切面上 > 0，3D 曲面上 < 0"""
        import torch
        o = F.mesh_objs(_Sphere(), prefix="t")
        ext, tw = o["t_extrude"], o["t_twist"]
        p = torch.tensor([[0.0, 0.0, 0.0, 0.6], [0.5, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.5, 0.0, 0.3]], device=F.DEV)
        self.assertTrue((ext.f(p[:2]).abs() < 1e-5).all())          # 两点都在表面上（L 带 1e-12 → 1e-6 级偏差）
        self.assertGreater(float(ext.cap(p[:1])), 0)
        self.assertLess(float(ext.cap(p[1:2])), 0)
        self.assertGreater(float(tw.cap(p[2:3])), 0)
        self.assertLess(float(tw.cap(p[3:4])), 0)

    def test_cut_glow_lights_only_cut_faces(self):
        """surface：平切面从死黑变亮；cut_glow = 0 时与旧行为一致（像素逐一相等），只有平切面附近的像素会变"""
        import torch
        o = F.mesh_objs(_Sphere(), prefix="t")["t_extrude"]
        R = F.rot4(xw=math.pi / 2)
        v0, _ = F.render(o, R, 0.1, 96, 64, "surface", yaw=1.4, pitch=0.2, dist=3.0, xray=False, cut_glow=0.0)
        v1, _ = F.render(o, R, 0.1, 96, 64, "surface", yaw=1.4, pitch=0.2, dist=3.0, xray=False, cut_glow=0.55)
        changed = (v1 - v0).abs() > 1e-4
        self.assertGreater(int(changed.sum()), 50)
        self.assertGreater(float(v1[changed].mean()), float(v0[changed].mean()) + 0.2)
        no_cap = F.mesh_objs(_Sphere(), prefix="u")["u_spin"]          # spin 没有平切面：开不开都一样
        a, _ = F.render(no_cap, F.rot4(), 0.0, 64, 48, "surface", xray=False, cut_glow=0.0)
        b, _ = F.render(no_cap, F.rot4(), 0.0, 64, 48, "surface", xray=False, cut_glow=0.55)
        self.assertTrue(torch.equal(a, b))


@unittest.skipUnless(HAS_TORCH, "four_d 需要 torch（.venv）")
class TestGoldenAge(unittest.TestCase):
    """sculpt 光 / 绕 target 的相机 / 单个红点 / 法向光流（2026-10-22 GoldenAge 版）"""

    def test_target_default_matches_old_camera(self):
        """target = 原点时与旧相机逐像素相等；平移 target = 整台相机平移（射线方向不变）"""
        import torch
        e0, r0 = F.camera(32, 18, 0.7, 0.3, 4.0)
        e1, r1 = F.camera(32, 18, 0.7, 0.3, 4.0, target=(0.2, -0.1, 0.3))
        self.assertTrue(torch.allclose(r0, r1))
        self.assertTrue(torch.allclose(e1 - e0, torch.tensor([0.2, -0.1, 0.3], device=F.DEV), atol=1e-6))
        self.assertAlmostEqual(float(e0.norm()), 4.0, places=5)

    def test_xray_projection_uses_same_camera(self):
        """必须相等的量：一个 3D 点用 X 光的投影公式落在哪个像素，相机那个像素的射线就要穿过它（target / fov 非默认）"""
        import torch
        hw, hh, fov, tg = 160, 90, 24.0, (0.1, 0.25, -0.05)
        eye, fwd, right, up = F.cam_basis(1.4, -0.2, 4.2, tg)
        p = torch.tensor([0.15, 0.4, 0.1], device=F.DEV)
        rel = p - eye
        z = rel @ fwd
        s = math.tan(math.radians(fov) / 2)
        px = float((rel @ right) / z / s * (hh / 2) + hw / 2)
        py = float(-(rel @ up) / z / s * (hh / 2) + hh / 2)
        e2, rd = F.camera(hw, hh, 1.4, -0.2, 4.2, fov, tg)
        r = rd.reshape(hh, hw, 3)[int(py), int(px)]
        dvec = rel / rel.norm()
        self.assertGreater(float(r @ dvec), math.cos(math.radians(fov / hh * 1.5)))

    def test_key_dir(self):
        """az = 0 从相机那边打（与视线反向）、az = 90 从画面右侧、el = 90 正上方；单位长"""
        import torch
        _, fwd, right, _ = F.cam_basis(0.9, 0.0, 1.0)
        self.assertTrue(torch.allclose(F.key_dir(0.9, 0.0, 0, 0), -fwd, atol=1e-6))
        self.assertTrue(torch.allclose(F.key_dir(0.9, 0.0, 90, 0), right, atol=1e-6))
        self.assertTrue(torch.allclose(F.key_dir(0.9, 0.3, 37, 90), torch.tensor([0.0, 1.0, 0.0], device=F.DEV), atol=1e-6))
        self.assertAlmostEqual(float(F.key_dir(0.9, 0.3, -80, 48).norm()), 1.0, places=6)

    def test_sculpt_shadow_darkens(self):
        """球挡住光的那一侧地面（这里用第二个球）要比没挡的暗：soft_shadow 在被遮挡点 ≈ 0，开阔处 = 1"""
        import torch
        g = lambda p: torch.minimum(F.L(p[:, 0], p[:, 1] - 0.6, p[:, 2]) - 0.3, F.L(p[:, 0], p[:, 1] + 0.6, p[:, 2]) - 0.3)  # noqa: E731
        Ld = torch.tensor([0.0, 1.0, 0.0], device=F.DEV)
        p = torch.tensor([[0.0, -0.29, 0.0], [0.0, 0.91, 0.0]], device=F.DEV)   # 下球顶（被上球挡）/ 上球顶（开阔）
        s = F.soft_shadow(g, p, Ld)
        self.assertLess(float(s[0]), 0.05)
        self.assertGreater(float(s[1]), 0.95)

    def test_red_point_single_and_sticky(self):
        """红图里两个斑：只出一个点；第二帧另一个斑略强，也不跳过去（跟踪偏向原地）；没有红 → 慢慢熄"""
        import torch
        H_, W_ = 90, 160
        rp = F.RedPoint(H_, W_)
        red = torch.zeros(H_, W_, device=F.DEV)
        red[20:24, 30:34] = 1.0
        red[60:64, 120:124] = 0.9
        out = rp(red)
        y, x = divmod(int(torch.argmax(out)), W_)
        self.assertLess(abs(x - 31.5) + abs(y - 21.5), 3)
        self.assertLess(float(out[60:64, 118:126].max()), 0.01)      # 只有一个点
        red2 = red.clone(); red2[60:64, 120:124] = 1.1
        y2, x2 = divmod(int(torch.argmax(rp(red2))), W_)
        self.assertLess(abs(x2 - 31.5) + abs(y2 - 21.5), 3)
        a = [float(rp(torch.zeros_like(red)).max()) for _ in range(4)]
        self.assertTrue(all(p > q for p, q in zip(a, a[1:])))

    def test_normal_flow_measures_shift(self):
        """必须相等的量：平滑图案整体平移 1 像素，法向光流 ≈ 1 px/帧（斜向梯度 → |u_⊥| ≤ 1）"""
        import torch
        ys, xs = torch.meshgrid(torch.arange(120.0), torch.arange(160.0), indexing="ij")
        f = lambda dx: 0.5 + 0.4 * torch.sin((xs - dx) / 6.0)  # noqa: E731
        self.assertAlmostEqual(F.normal_flow(f(0), f(1)), 1.0, delta=0.08)
        self.assertAlmostEqual(F.normal_flow(f(0), f(0)), 0.0, places=6)

    def test_rise_operator(self):
        """穿膜半旋转体（球 r = 0.5，膜在 s = −0.6，d = +z）：c = 0 截面 = 原球；c 里只出现离膜 ≥ c 的部分，落在 sqrt(s_b² − c²)；
        距离下界：|f(p) − f(q)| ≤ |p − q|（4D 随机点对）；cap > 0 只在膜上"""
        import torch
        d = torch.tensor([0.0, 0.0, 1.0], device=F.DEV)
        o = torch.tensor([0.0, 0.0, -0.6], device=F.DEV)
        ob = F.mesh_rise(_Sphere(), d, o, prefix="t")
        g = torch.Generator(device="cpu").manual_seed(0)
        P3 = (torch.rand(4000, 3, generator=g) * 2.4 - 1.2).to(F.DEV)
        f0 = ob.f(torch.cat([P3, torch.zeros_like(P3[:, :1])], 1))
        ins = F.L(P3[:, 0], P3[:, 1], P3[:, 2]) < 0.5
        self.assertTrue(torch.equal(f0 < 0, ins))
        c = 0.9                                                    # 球上离膜 ≥ 0.9 的是 z ≥ 0.3 的帽
        q = torch.tensor([[0.0, 0.0, -0.6 + math.sqrt(1.05 ** 2 - c * c), c]], device=F.DEV)   # s_b = 1.05（z = 0.45）落点
        self.assertLess(float(ob.f(q).abs()), 0.06)
        # 贴膜（s = 0.05）的截面 = 球在 s_b ≈ 0.9014（z ≈ 0.30）处的圆，半径 sqrt(0.25 − 0.30²) ≈ 0.40
        self.assertLess(float(ob.f(torch.tensor([[0.35, 0.0, -0.55, c]], device=F.DEV))), 0)
        self.assertGreater(float(ob.f(torch.tensor([[0.45, 0.0, -0.55, c]], device=F.DEV))), 0)
        A = (torch.rand(3000, 4, generator=g) * 2.4 - 1.2).to(F.DEV)
        B = A + (torch.rand(3000, 4, generator=g) - 0.5).to(F.DEV) * 0.2
        self.assertTrue(((ob.f(A) - ob.f(B)).abs() <= (A - B).norm(dim=1) + 1e-4).all())
        mem = torch.tensor([[0.0, 0.0, -0.6 + 1e-3, 1.0]], device=F.DEV)    # 膜上、身体截面里（s_b = 1 处是球内）
        self.assertGreater(float(ob.cap(mem)), 0)

    def test_rise_track(self):
        """φ 慢起（首帧手速最小的一档）、c 单调降、永不停、永不过 0；相机距离对数匀速"""
        import torch
        sp = F.RiseSpec(5.6, grow=4.0)
        geo = (torch.zeros(3, device=F.DEV), torch.tensor([0.0, 0.0, 1.0], device=F.DEV),
               torch.tensor([0.0, 0.0, 1.3], device=F.DEV), torch.tensor([0.0, 0.0, 0.6], device=F.DEV))
        tr = F.rise_track(sp, 1.3, geo)
        cs = [k["c"] for k in tr]
        self.assertEqual(len(tr), 135)
        self.assertTrue(all(a > b for a, b in zip(cs, cs[1:])))
        self.assertGreater(min(cs), 0)
        self.assertLess(tr[0]["hand_v"], 0.25 * max(k["hand_v"] for k in tr))
        lr = [math.log(b["dist"] / a["dist"]) for a, b in zip(tr, tr[1:])]
        self.assertLess(max(lr) - min(lr), 1e-9)
        S, dS = F.hermite01(0.0, 0.2, 0.05)
        self.assertEqual((S, dS), (0.0, 0.2))
        S, dS = F.hermite01(1.0, 0.2, 0.05)
        self.assertAlmostEqual(S, 1.0); self.assertAlmostEqual(dS, 0.05)

    def test_camera_roll(self):
        """roll = 0 与旧相机相同；roll 只绕视线转画框：eye / fwd 不变，right、up 仍正交单位，转了 roll 角"""
        import torch
        a = F.cam_basis(1.2, -0.1, 3.0, (0.1, 0.2, 0.0))
        b = F.cam_basis(1.2, -0.1, 3.0, (0.1, 0.2, 0.0), roll=math.radians(25))
        self.assertTrue(torch.allclose(a[0], b[0]) and torch.allclose(a[1], b[1]))
        self.assertAlmostEqual(float(b[2] @ b[3]), 0.0, places=6)
        self.assertAlmostEqual(float(b[2].norm()), 1.0, places=6)
        self.assertAlmostEqual(float(a[2] @ b[2]), math.cos(math.radians(25)), places=6)
        self.assertTrue(torch.allclose(F.camera(8, 6, 1.2, -0.1, 3.0)[1], F.camera(8, 6, 1.2, -0.1, 3.0, roll=0.0)[1]))

    def test_body_rotation(self):
        """body_rate = 0 时 R 与旧的一样；≠ 0 时 = 左乘 yz 平面旋转（只动 y、z，不碰 x 与 w），R 仍正交"""
        import torch
        sp0 = F.InsertSpec(1.0, theta0=10)
        sp1 = F.InsertSpec(1.0, theta0=10, body_rate=-8.0)
        R0, R1 = F.insert_R(_Obj, sp0, 30.0, 1.5), F.insert_R(_Obj, sp1, 30.0, 1.5)
        self.assertTrue(torch.allclose(R0, F.insert_R(_Obj, sp0, 30.0)))
        B = F.rot4(yz=math.radians(-12.0))
        self.assertTrue(torch.allclose(R1, B @ R0, atol=1e-6))
        self.assertTrue(torch.allclose(B[[0, 3]][:, [0, 3]], torch.eye(2, device=F.DEV)))
        self.assertTrue(torch.allclose(R1 @ R1.T, torch.eye(4, device=F.DEV), atol=1e-6))

    def test_rim_follows_camera_roll(self):
        """轮廓光在相机（含 roll）的基里定：az = 180、el = 0 = 正后方（= 视线方向）；roll 改变它的世界方向（el ≠ 0 时）"""
        import torch
        sp = F.InsertSpec(2.0, rim=1.0, rim_az=180.0, rim_el=0.0)
        k = dict(t=0.0, yaw=1.1)
        _, fwd, _, _ = F.cam_basis(1.1, sp.pitch, 1.0)
        self.assertTrue(torch.allclose(F.rim_dir(sp, k)[0], fwd, atol=1e-6))
        sp2 = F.InsertSpec(2.0, rim=1.0, rim_az=-160.0, rim_el=45.0, roll_rate=-5.0)
        d0, d1 = F.rim_dir(sp2, dict(t=0.0, yaw=1.1))[0], F.rim_dir(sp2, dict(t=2.0, yaw=1.1))[0]
        self.assertLess(float(d0 @ d1), 0.999)

    def test_surface_nets_sphere(self):
        """必须相等的量：球 r = 0.5 → 顶点都在球面上、法线 = 径向、每条棱正好两个四边形（水密）、χ = V − E + F = 2、面都朝外；
        PLY 写出再读回逐值相等；Blender 坐标变换行列式 +1"""
        import numpy as np
        import tempfile
        g = lambda p: F.L(p[:, 0], p[:, 1], p[:, 2]) - 0.5  # noqa: E731
        V, Q, N = F.surface_nets(g, 1.0, 48)
        r = np.linalg.norm(V, axis=1)
        self.assertLess(np.abs(r - 0.5).max(), 1e-4)
        self.assertLess(np.abs((N * V / r[:, None]).sum(1) - 1).max(), 1e-4)
        E = np.sort(np.stack([Q, np.roll(Q, -1, 1)], -1).reshape(-1, 2), 1)
        _, cnt = np.unique(E, axis=0, return_counts=True)
        self.assertTrue((cnt == 2).all())
        self.assertEqual(len(V) - len(cnt) + len(Q), 2)
        P = V[Q]
        self.assertTrue(((np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 1]) * P.mean(1)).sum(1) > 0).all())
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "s.ply"
            F.write_ply(path, V, Q, N)
            data = path.read_bytes()
            end = data.index(b"end_header\n") + 11
            vert = np.frombuffer(data, "<f4", len(V) * 6, end).reshape(-1, 6)
            self.assertTrue(np.array_equal(vert[:, :3], V.astype(np.float32)))
        self.assertAlmostEqual(float(np.linalg.det(F.to_blender(np.eye(3)))), 1.0)

    def test_rigid_fit_recovers_motion(self):
        """必须相等的量：椭球表面点被转 6° + 平移后，对齐到原椭球 SDF → 中位残差回到 ≈ 0（椭球不对称，旋转可辨）"""
        import numpy as np

        class _Ell:
            def __call__(self, x, y, z):
                return (F.L(x / 0.6, y / 0.4, z / 0.3) - 1) * 0.3          # 椭球的近似距离（零面准确即可）
        rng = np.random.default_rng(1)
        u = rng.normal(size=(3000, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
        P = u * np.array([0.6, 0.4, 0.3])
        a = math.radians(6)
        Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        Pm = P @ Rz.T + np.array([0.03, -0.02, 0.01])
        R, t, m0, m1 = F.rigid_fit_to_sdf(Pm, _Ell(), clip=0.2)
        self.assertGreater(m0, 0.005)
        self.assertLess(m1, 0.1 * m0)

    def test_seam_normals_fallback(self):
        """|x| 镜像接缝（g = 球(|x| + 0.2 …) 的 V 形折痕）上中心差分把法线抵消 → 退回网格法线：法线与网格法线夹角都 < 45°"""
        import numpy as np
        g = lambda p: F.L(p[:, 0].abs() + 0.15, p[:, 1], p[:, 2]) - 0.5  # noqa: E731
        V, Q, N = F.surface_nets(g, 1.0, 64)
        self.assertLess(F.surface_nets.last_bad, 0.05)                 # 只在折痕附近退回
        P = V[Q]
        fn = np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 1])
        Nm = np.zeros_like(V)
        np.add.at(Nm, Q.reshape(-1), np.repeat(fn, 4, 0))
        Nm /= np.linalg.norm(Nm, axis=1, keepdims=True)
        self.assertGreater(float((N * Nm).sum(1).min()), math.cos(math.radians(45)))

    def test_grow4_ends_as_body_and_starts_empty(self):
        """真四维生长（球 r = 0.5 当身体，心口 = 球心）：c = w_end、θ = 0 → 截面 = 原球（扭转角 0，深度项不起作用）；
        c 比 W 的最小值（≈ −0.5 − 噪声）还小 → 空；中间先出现的是离心口最远的外壳"""
        import torch
        ob = F.mesh_grow4(_Sphere(), heart=(0.0, 0.0, 0.0), prefix="t")
        g = torch.Generator(device="cpu").manual_seed(2)
        P = (torch.rand(5000, 3, generator=g) * 1.4 - 0.7).to(F.DEV)
        inside = F.L(P[:, 0], P[:, 1], P[:, 2]) < 0.5
        f_end = F.slice_fn(ob, F.rot4(), 0.3)(P)
        self.assertTrue(torch.equal(f_end < 0, inside))
        self.assertFalse(bool((F.slice_fn(ob, F.rot4(), -0.7)(P) < 0).any()))
        mid = F.slice_fn(ob, F.rot4(), -0.42)(P) < 0                   # 只剩离心口 ≥ ~0.42 的外壳
        r = F.L(P[:, 0], P[:, 1], P[:, 2])
        self.assertTrue(bool(mid.any()))
        self.assertGreater(float(r[mid].min()), 0.3)

    def test_ply_cap_roundtrip(self):
        import numpy as np
        import tempfile
        V = np.random.default_rng(0).normal(size=(10, 3)).astype(np.float32)
        Q = np.arange(8, dtype=np.int32).reshape(2, 4)
        cap = np.linspace(0, 1, 10).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.ply"
            F.write_ply(p, V, Q, V, cap)
            data = p.read_bytes()
            self.assertIn(b"property float cap\n", data)
            end = data.index(b"end_header\n") + 11
            vert = np.frombuffer(data, "<f4", 70, end).reshape(10, 7)
            self.assertTrue(np.array_equal(vert[:, 6], cap))
            F.write_ply(p, np.zeros((0, 3), np.float32), np.zeros((0, 4), np.int32), np.zeros((0, 3), np.float32))   # 空截面也能写

    def test_spec_guards(self):
        with self.assertRaises(ValueError):
            F.InsertSpec(1.0, fov=2.0)
        with self.assertRaises(ValueError):
            F.InsertSpec(3.0, dolly=0.2)


if __name__ == "__main__":
    unittest.main()
