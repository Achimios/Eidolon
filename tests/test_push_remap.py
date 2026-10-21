#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""推进曲线自检：python -X utf8 -m unittest discover -s tests -q"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import push_remap as P  # noqa: E402

FPS = 24.0


class TestPushRemap(unittest.TestCase):
    def setUp(self):
        self.t0, self.n = 38.70, 154
        self.pushes = [(40.00, 1.0), (41.52, 1.3)]
        self.ts, self.v, self.pos = P.curve(self.n, FPS, self.t0, self.pushes, base=0.04, gain=0.6, ramp_frames=2.5, tau=0.3)

    def test_frame_times_cover_segment(self):
        self.assertEqual(len(self.ts), 154)
        self.assertAlmostEqual(self.ts[0], 38.70)
        self.assertAlmostEqual(self.ts[-1] + 1 / FPS, 45.1167, places=3)

    def test_never_rest(self):
        # 不从静止开始、不停在静止：每帧速度 ≥ 基础速度，开头结尾都 > 0
        self.assertGreaterEqual(self.v.min(), 0.04 - 1e-12)
        self.assertGreater(self.v[0], 0); self.assertGreater(self.v[-1], 0)
        self.assertTrue(np.all(np.diff(self.pos) > 0))

    def test_push_peaks_within_3_frames(self):
        for ti, a in self.pushes:
            k0 = int(np.ceil((ti - self.t0) * FPS - 1e-9))
            win = self.v[k0:k0 + 6]
            kp = k0 + int(np.argmax(win))
            self.assertLessEqual(kp - k0, 3, ti)                # 进：2–3 帧
            self.assertAlmostEqual(self.v[kp], 0.04 + 0.6 * a, delta=0.6 * a * 0.15)
            self.assertLess(self.v[k0 - 1], 0.04 + 0.6 * a * 0.2)  # 推之前没有提前动

    def test_decay_is_exponential_not_zero(self):
        k = int(round((41.52 - self.t0) * FPS))
        after = self.v[k - 12:k] - 0.04       # 第一推后的缓（第二推之前）
        self.assertTrue(np.all(np.diff(after) < 0))
        r = after[1:] / after[:-1]
        self.assertLess(np.std(r), 1e-3)       # 等比 = 指数
        self.assertAlmostEqual(r.mean(), np.exp(-1 / (FPS * 0.3)), places=3)

    def test_position_is_integral(self):
        # 必须相等的量：位移的帧差 ≈ 速度的梯形积分
        fine_t = np.linspace(self.ts[0], self.ts[-1], 20001)
        v = P.velocity(fine_t - self.t0, [(t - self.t0, 0.6 * a) for t, a in self.pushes], 0.04, 2.5 / FPS, 0.3)
        total = np.trapezoid(v, fine_t) if hasattr(np, "trapezoid") else np.trapz(v, fine_t)
        self.assertAlmostEqual(self.pos[-1], total, places=4)

    def test_snap(self):
        ev = [39.942, 40.352, 41.562, 41.789]
        self.assertEqual(P.snap(40.00, ev, 0.06), (39.942, True))
        self.assertEqual(P.snap(41.50, ev, 0.06), (41.50, False))
        self.assertEqual(P.snap(41.50, [], 0.06), (41.50, False))


if __name__ == "__main__":
    unittest.main()
