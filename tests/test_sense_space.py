#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""感官空间（tools/sense_space.py）自检：鱼眼时间可逆、投影正确、前馈镜头的屏幕速度峰值贴着事件。"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import sense_space as S  # noqa: E402
import info_graph as G  # noqa: E402


class TestSenseSpace(unittest.TestCase):
    def test_fisheye_roundtrip(self):
        for t_now in (5.0, 60.0, 128.0):
            dt = np.array([0.0, 0.3, 1.5, 0.5 * t_now, t_now])
            x = S.fisheye_x(dt, t_now)
            np.testing.assert_allclose(S.fisheye_inv(x, t_now), dt, atol=1e-9)
            self.assertAlmostEqual(float(x[0]), 0.0)
            self.assertAlmostEqual(float(x[-1]), -S.X1, places=6)        # 曲首 = 平面最左
            self.assertTrue(np.all(np.diff(x) < 0))

    def test_projection_center_and_handedness(self):
        st = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1000.0])                  # yaw 0：相机在 −z 看 +z
        q, z = S.project(np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 100.0, 0.0]]), st)
        np.testing.assert_allclose(q[0], (S.W / 2, S.H / 2), atol=1e-6)
        self.assertGreater(q[1, 0], q[0, 0])                               # +x 在右
        self.assertLess(q[2, 1], q[0, 1])                                  # +y 在上
        self.assertAlmostEqual(float(z[0]), 1000.0, places=6)

    def test_camera_peaks_on_events(self):
        if not G.SENSE.exists():
            self.skipTest("没有 感官.json")
        seg = (115.51, 128.36)
        mem = G.build_memory(seg=seg)
        sch = G.schedule_v4(mem, seg, john_hop=0)
        T = int(round((seg[1] - seg[0]) * S.FPS))
        npos = np.array([[S.fisheye_x(121.5 - (nd["t0"] + nd["t1"]) / 2, 121.5), 400.0, S.Z_MEM] for nd in mem.nodes])
        base = np.array([-760.0, 330.0, 300.0, 0.82, 0.17, 3000.0])
        cams, ev = S.camera_schedule(mem, sch, seg, T, npos, base)
        refs = np.array([[x, y, z] for x in np.linspace(-S.X1, 0, 7) for y in (0, S.HH / 2, S.HH) for z in (S.Z_MEM, 0, 1260.0)])
        sp = S.screen_speed(cams, refs)
        al = S.alignment(sp, ev, T)
        self.assertGreater(len(al), 10)
        for t, k, d in al:
            self.assertLessEqual(abs(d), 1, (t + seg[0], k, d))


if __name__ == "__main__":
    unittest.main()
