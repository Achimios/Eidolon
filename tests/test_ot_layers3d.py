#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最优传输 3D 多层（tools/ot_layers3d.py）自检：找「必须相等」的量。小规模、CPU 即可。
1. ⭐层 t 上的 W2：最优配对下 μ_t 到 ν 的精确 W2（重新解匈牙利）== (1-t)·W2_0；熵映射同样成立（单调 ⇒ 最优）。
2. 抖动沿 OT 方向、裁在 [0,1]：抖动后的点仍在各自的 OT 线段上。
3. 相机：速度（平移 + 视线转向）处处 > 0、快进慢出（单调减速）、无滚转（相机 x 轴永远水平）、基正交。
4. 投影：注视点投到画面正中；身后的点给 NaN。
5. 可插拔源 / 目标：.npy 点集原样读入（数量不等时重采样）；默认 still09 与静帧 09 逐位相同。
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import ot_flow as O  # noqa: E402
import ot_layers3d as L3  # noqa: E402
import stills as S  # noqa: E402


def cloud(n, seed):
    r = np.random.default_rng(seed)
    a = np.stack([r.normal(-1.5, 0.3, n), r.normal(0, 0.3, n)], 1)
    b = np.stack([r.uniform(0.5, 2.0, n), r.uniform(-1, 1, n) ** 3], 1)
    return a, b


class TestLayerW2(unittest.TestCase):
    def test_w2_at_layer_t_is_linear_hungarian(self):
        x, y = cloud(150, 1)
        w2, col = O.w2_exact(x, y)
        T = y[col]
        for t in L3.layer_ts(9):
            P = L3.layer_points(x, T, t)
            self.assertAlmostEqual(O.w2_exact(P, T)[0], (1 - t) * w2, places=9)

    def test_w2_at_layer_t_entropic_map(self):
        x, y = cloud(300, 2)
        T, _ = O.entropic_map(x, y, eps_list=(1.0, 0.1, 0.02), iters=30, device="cpu")
        w2c = O.w2_coupling(x, T)
        for t in L3.layer_ts(5):
            P = L3.layer_points(x, T, t)
            self.assertAlmostEqual(O.w2_exact(P, T)[0], (1 - t) * w2c, places=7)

    def test_jitter_stays_on_ot_segment(self):
        x, y = cloud(200, 3)
        jit = np.random.default_rng(4).normal(0, 0.3, len(x))
        for t in (0.0, 0.5, 1.0):
            P = L3.layer_points(x, y, t, jit)
            d = y - x
            tau = np.sum((P - x) * d, 1) / np.sum(d * d, 1)
            self.assertTrue(np.all((tau >= -1e-12) & (tau <= 1 + 1e-12)))
            np.testing.assert_allclose(P, x + tau[:, None] * d, atol=1e-12)   # 在线段上（没有横向偏）


class TestCamera(unittest.TestCase):
    def setUp(self):
        self.u = np.linspace(0, 1, 20001)
        C, Lk = zip(*[L3.camera(v) for v in self.u])
        self.C, self.L = np.array(C), np.array(Lk)

    def test_speed_never_zero_and_decelerates(self):
        v = np.linalg.norm(np.gradient(self.C, self.u, axis=0), axis=1)
        self.assertGreater(v.min(), 0.5, "相机平移速度不能为 0")
        f = self.L - self.C; f /= np.linalg.norm(f, axis=1, keepdims=True)
        w = np.linalg.norm(np.gradient(f, self.u, axis=0), axis=1)
        self.assertGreater(w.min(), 0.0, "视线转向速度不能为 0")
        g = np.gradient(L3.ease(self.u), self.u)
        self.assertGreater(g.min(), 0.15)
        self.assertTrue(np.all(np.diff(g) <= 1e-9), "缓动速度必须单调减（快进慢出）")
        self.assertGreater(v[0], 3 * v[-1], "入画要明显快于出画")

    def test_no_roll_orthonormal(self):
        for i in range(0, len(self.u), 997):
            R = L3.basis(self.C[i], self.L[i])
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(R[0, 1], 0.0, places=12)   # 相机 x 轴无 y 分量 = 地平线水平
            self.assertGreater(R[1, 1], 0.0)                   # 相机 y 朝下（不倒置）

    def test_projection_center_and_behind(self):
        C, Lk = L3.camera(0.4)
        x, y, z = L3.project(np.array([Lk, 2 * C - Lk]), C, Lk, 1000.0)
        self.assertAlmostEqual(x[0], L3.W / 2, places=9)
        self.assertAlmostEqual(y[0], L3.H / 2, places=9)
        self.assertTrue(np.isnan(x[1]) and z[1] < 0)


class TestMeasures(unittest.TestCase):
    def test_npy_roundtrip_and_resample(self):
        pts = np.random.default_rng(5).normal(size=(500, 2))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "pts.npy"; np.save(p, pts)
            got, uv = L3.load_measure(str(p), 500, np.random.default_rng(0))
            np.testing.assert_array_equal(got, pts)
            self.assertIsNone(uv)
            got2, _ = L3.load_measure(str(p), 800, np.random.default_rng(0))
            self.assertEqual(got2.shape, (800, 2))
            self.assertTrue(np.all(np.isin(got2[:, 0], pts[:, 0])))

    def test_default_is_still09(self):
        src, tgt, heart = L3.problem("still09", "still09", 2000)
        s2, t2, _ = S.ot_problem(2000, 5)
        np.testing.assert_array_equal(src, s2); np.testing.assert_array_equal(tgt, t2)
        np.testing.assert_allclose(heart, O.target_point(O.HEART_UV))


if __name__ == "__main__":
    unittest.main()
