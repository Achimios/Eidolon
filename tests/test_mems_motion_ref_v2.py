#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MEMS 运动参考 v2 自检：python -X utf8 -m unittest tests.test_mems_motion_ref_v2 -q（不读音频，不出视频）"""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import mems_motion_ref_v2 as M  # noqa: E402


class TestRaster(unittest.TestCase):
    def test_depth_test_front_wins_and_coverage(self):
        # 两个全屏外的大三角形：近的（z=5，id 1）必须盖住远的（z=10，id 0）；覆盖像素数 ≈ 三角形面积
        H = W = 64
        far = [[0, 0, 10], [64, 0, 10], [0, 64, 10]]
        near = [[0, 0, 5], [32, 0, 5], [0, 32, 5]]
        V = torch.tensor([far, near], dtype=torch.float32, device=M.dev)
        key = M.raster(V, H, W)
        valid, pix, tid, bp, dep = M.resolve(key, V, W)
        n_near = int((tid == 1).sum()); n_far = int((tid == 0).sum())
        self.assertLess(abs(n_near - 32 * 32 / 2), 40)
        self.assertLess(abs(n_near + n_far - 64 * 64 / 2), 80)
        self.assertTrue(torch.allclose(dep[tid == 1], torch.tensor(5.0, device=M.dev), atol=1e-3))
        self.assertTrue(torch.allclose(bp.sum(1), torch.ones(len(bp), device=M.dev), atol=1e-4))


class TestPhysics(unittest.TestCase):
    def test_band_layout_interleaved(self):
        # 上排偶数带、下排奇数带，左低右高，60 根齿一一对应
        seen = {M.band_of(k) for k in range(M.NF)}
        self.assertEqual(len(seen), M.NF)
        for k in range(M.NF):
            b, i = M.band_of(k)
            self.assertEqual(M.finger_band(b, i), k)
        self.assertTrue(np.all(np.diff(M.f_band) > 0) and np.all(np.diff(M.FV) > 0))

    def test_resonance_is_band_selective(self):
        # 只给第 34 带（≈ 1.35 kHz，抽搐基频）驱动 → 它的齿摆得最大；同排邻齿只经脊耦合被带动（看得见但 < 25%）
        fs = 2400.0
        ts = np.arange(0, 3, 1 / fs)
        tt = np.arange(0, 3.2, 0.005)
        D = np.zeros((M.NF, tt.size)); D[34, (tt > 0.5)] = 10.0
        X, Q, Wsp = M.simulate(tt, D, ts, np.zeros_like(ts))
        pk = np.abs(X).max(1)
        self.assertEqual(int(np.argmax(pk)), 34)
        second = np.sort(pk)[-2]
        self.assertLess(second, 0.25 * pk[34])
        self.assertGreater(pk[32], 0.05 * pk[34])            # 同排左邻（第 32 带）被脊带动 = 振动沿梳排传播
        self.assertLess(pk[33], 0.05 * pk[34])               # 另一排（奇数带）几乎不动：两排的脊不相连
        self.assertGreater(np.abs(Wsp).max(), 0)            # 振动传进了梳排的脊

    def test_display_monotone_and_clipped(self):
        x = np.linspace(-5, 5, 101)
        d = M.disp(x, 1.0, M.PEAK_X)
        self.assertTrue(np.all(np.diff(d) >= 0))
        self.assertLess(np.abs(d).max(), 3.8 + 1e-9)
        self.assertAlmostEqual(float(M.disp(np.array([1.0]), 1.0, 2.0, clip=1e9)[0]), 2.0)


class TestCamera(unittest.TestCase):
    def test_moving_from_frame_one_push_at_hit_straight(self):
        eye, tgt, v = M.camera(108, 24, 61.624, [62.624])
        self.assertGreater(v.min(), 0)                      # 从第 1 帧起就在动，永不停
        self.assertIn(int(np.argmax(v)), range(24, 30))     # 「进」在命中处
        self.assertLess(v[-1], 0.5 * v.max())               # 之后「缓」
        step = np.linalg.norm(np.diff(eye, axis=0), axis=1)
        self.assertGreater(step.min(), 0)
        for P_ in (eye, tgt):                               # 相机 / 注视点各自共线：不晃
            d = P_ - P_[0]; u = (P_[-1] - P_[0]) / np.linalg.norm(P_[-1] - P_[0])
            self.assertLess(np.abs(d - np.outer(d @ u, u)).max(), 1e-9)


if __name__ == "__main__":
    unittest.main()
