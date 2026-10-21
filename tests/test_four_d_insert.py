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


if __name__ == "__main__":
    unittest.main()
