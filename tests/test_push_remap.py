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


class TestJump(unittest.TestCase):
    J = [(40.352, 1.25), (41.789, 1.3)]

    def plan(self, **kw):
        return P.jump_plan(154, FPS, 38.70, self.J, **kw)

    def test_jump_frames_nearest_to_onsets(self):
        ts, logz, src, kj = self.plan()
        self.assertEqual(kj, [40, 74])
        for (t, _), j in zip(self.J, kj):
            self.assertLessEqual(abs(ts[j] - t), 0.5 / FPS + 1e-9)

    def test_hard_jump_is_one_frame_and_exact_factor(self):
        ts, logz, src, kj = self.plan(base=0.03, trans_frames=0)
        d = np.diff(logz)
        drift = 0.03 / FPS
        for (_, f), j in zip(self.J, kj):
            self.assertAlmostEqual(np.exp(d[j - 1] - drift), f, places=9)   # 跳点那一帧整倍数放大
        others = np.delete(d, [j - 1 for j in kj])
        self.assertTrue(np.allclose(others, drift))                           # 其余每帧都在漂（不静止）
        self.assertAlmostEqual(np.exp(logz[-1] - 0.03 * 153 / FPS), 1.25 * 1.3, places=9)  # 必须相等：总倍数

    def test_transition_within_frames(self):
        ts, logz, src, kj = self.plan(base=0.0, trans_frames=3)
        j = kj[0]
        self.assertAlmostEqual(logz[j - 1], 0.0)
        self.assertAlmostEqual(logz[j + 2], np.log(1.25))
        self.assertTrue(0 < logz[j] < logz[j + 1] < logz[j + 2])

    def test_drop_skips_source_time(self):
        ts, logz, src, kj = self.plan(drop=0.15)
        d = np.diff(src)
        for j in kj:
            self.assertAlmostEqual(d[j - 1], 1 / FPS + 0.15)
        self.assertTrue(np.all(d > 0))
        self.assertAlmostEqual(src[-1], 153 / FPS + 0.30)

    def test_short_source_slows_to_fit(self):
        ts, logz, src, kj = self.plan(drop=0.15, src_dur=6.0)
        self.assertLessEqual(src[-1], 6.0)
        self.assertGreater(src[-1], 5.99)


if __name__ == "__main__":
    unittest.main()
