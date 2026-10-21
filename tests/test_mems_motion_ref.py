#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MEMS 运动参考自检：python -X utf8 -m unittest discover -s tests -q（不调 Blender，不读音频）"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import mems_motion_ref as M  # noqa: E402


class TestMemsMotionRef(unittest.TestCase):
    def test_beam_ends_follow_mode(self):
        # 内梁：梭子端 = V，桁架端 = V/2；外梁：锚端 = 0，桁架端 = V/2（折叠梁桁架走一半）
        V = 4.0
        self.assertAlmostEqual(float(M.beam_v("inner", 0, V, 0.0)), V)
        self.assertAlmostEqual(float(M.beam_v("inner", 0, V, 1.0)), V / 2)
        self.assertAlmostEqual(float(M.beam_v("outer", 0, V, 0.0)), 0.0)
        self.assertAlmostEqual(float(M.beam_v("outer", 0, V, 1.0)), V / 2)
        # 两端导向：斜率在两端为 0
        s = np.array([0.0, 1e-4, 1 - 1e-4, 1.0])
        u = M.mode_shape(s)
        self.assertLess(abs(u[1] - u[0]) / 1e-4, 1e-2); self.assertLess(abs(u[3] - u[2]) / 1e-4, 1e-2)

    def test_combs_never_collide(self):
        # 齿尖间隙 > 最大位移；动齿和定子齿横向不重叠
        self.assertGreater(M.GAP - M.VMAX, 0)
        d = np.abs(np.subtract.outer(np.array(M.FU, float), np.array(M.FS, float))).min()
        self.assertGreater(d, M.FW)

    def test_drive_jumps_at_hit_and_rings_down(self):
        rate = 480.0
        t = np.arange(0, 10, 1 / rate)
        e = np.full_like(t, 0.2)
        hit = 5.0
        e[(t >= hit) & (t < hit + 0.3)] = 1.0
        ft = 4.0 + np.arange(108) / 24
        V, info = M.drive(t, e, [hit], ft, f_vis=3.0, tau=1.1, flux_gain=0.0)
        kh = int(round((hit - ft[0]) * 24))
        self.assertAlmostEqual(np.abs(V).max(), M.VMAX, places=6)
        self.assertLess(np.abs(V[:kh]).max(), 1e-6)                 # 无通量时命中前静止
        early = np.abs(V[kh:kh + 24]).max(); late = np.abs(V[-24:]).max()
        self.assertLess(late, early * 0.4)                           # 振铃衰减
        # 过零次数 ≈ 2 f_vis × 时长
        zc = np.sum(np.diff(np.sign(V[kh + 1:])) != 0)
        self.assertTrue(abs(zc - 2 * 3.0 * (len(V) - kh - 1) / 24) <= 2, zc)

    def test_camera_never_stops_and_pushes_at_hit(self):
        cam, tgt, v = M.camera(108, 24, 61.624, [62.624])
        self.assertGreater(v.min(), 0)
        self.assertIn(int(np.argmax(v)), range(24, 30))
        step = np.linalg.norm(np.diff(cam, axis=0), axis=1)
        self.assertGreater(step.min(), 0)                            # 每帧都在动
        # 直线推进（不晃）：所有位置共线
        d = cam - cam[0]; u = (cam[-1] - cam[0]) / np.linalg.norm(cam[-1] - cam[0])
        self.assertLess(np.abs(d - np.outer(d @ u, u)).max(), 1e-9)


if __name__ == "__main__":
    unittest.main()
