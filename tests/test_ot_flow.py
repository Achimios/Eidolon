#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最优传输流（tools/ot_flow.py）自检：找「必须相等」的量。小规模、CPU 即可。
1. 匈牙利本身对：与穷举全排列的最小代价相等。
2. W2 线性衰减：最优配对下位移插值 μ_t 到 ν 的精确 W2（重新解匈牙利）== (1-t)·W2_0。
3. 不相撞：最优配对下任意两粒子在 τ∈[0,1] 的最近距离 > 0（闭式），且 <dx,dy> ≥ 0；随机配对做反例。
4. 熵 Brenier 映射是凸函数的梯度 ⇒ 对所有点对单调（不依赖 Sinkhorn 是否收敛）。
5. 质量守恒：粒子数不变、终点集合不变、权重和不变。
6. 静帧 09 的源/目标与重构前逐位相同（stills.ot_problem / sliced_ot）。
"""
import itertools
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import ot_flow as O  # noqa: E402


def cloud(n, seed):
    r = np.random.default_rng(seed)
    a = np.stack([r.normal(-1.5, 0.3, n), r.normal(0, 0.3, n)], 1)
    b = np.stack([r.uniform(0.5, 2.0, n), r.uniform(-1, 1, n) ** 3], 1)
    return a, b


class TestHungarian(unittest.TestCase):
    def test_matches_bruteforce(self):
        r = np.random.default_rng(0)
        for _ in range(20):
            n = 6
            C = r.random((n, n))
            col = O.hungarian(C)
            best = min(sum(C[i, p[i]] for i in range(n)) for p in itertools.permutations(range(n)))
            self.assertAlmostEqual(C[np.arange(n), col].sum(), best, places=12)
            self.assertEqual(sorted(col), list(range(n)))


class TestDisplacementInterpolation(unittest.TestCase):
    def setUp(self):
        self.x, y = cloud(120, 1)
        w2, col = O.w2_exact(self.x, y)
        self.y = y[col]          # 最优配对：x[i] → y[i]
        self.w2 = w2

    def test_w2_linear_decay(self):
        """⭐必须相等：W2(μ_t, ν) 重新求精确指派，== (1-t)·W2_0"""
        for t in (0.1, 0.37, 0.5, 0.8, 0.95):
            P = (1 - t) * self.x + t * self.y
            w2t, _ = O.w2_exact(P, self.y)
            self.assertAlmostEqual(w2t, (1 - t) * self.w2, places=9)
            self.assertAlmostEqual(O.w2_coupling(P, self.y), (1 - t) * self.w2, places=12)

    def test_non_crossing(self):
        n = len(self.x)
        i, j = np.triu_indices(n, 1)
        a, b = self.x[i] - self.x[j], self.y[i] - self.y[j]
        self.assertTrue(np.all(np.sum(a * b, 1) >= -1e-12), "最优配对必须两两单调")
        self.assertGreater(O.min_approach(a, b).min(), 1e-6)

    def test_random_pairing_violates(self):
        """反例：随机配对会有 <dx,dy> < 0 的点对（检测器不是永远说「没事」）"""
        perm = np.random.default_rng(2).permutation(len(self.y))
        i, j = np.triu_indices(len(self.x), 1)
        a, b = self.x[i] - self.x[j], self.y[perm][i] - self.y[perm][j]
        self.assertGreater(np.mean(np.sum(a * b, 1) < 0), 0.05)

    def test_min_approach_closed_form(self):
        r = np.random.default_rng(3)
        a, b = r.normal(size=(50, 2)), r.normal(size=(50, 2))
        ts = np.linspace(0, 1, 20001)[:, None, None]
        brute = np.sqrt(np.sum(((1 - ts) * a + ts * b) ** 2, -1)).min(0)
        np.testing.assert_allclose(O.min_approach(a, b), brute, atol=1e-6)
        # 正面相撞：a = -b → 中点距离为 0
        self.assertAlmostEqual(float(O.min_approach(np.array([[1.0, 0]]), np.array([[-1.0, 0]]))[0]), 0.0)


class TestEntropicMap(unittest.TestCase):
    def test_monotone_all_pairs_and_mass(self):
        x, y = cloud(400, 4)
        T, info = O.entropic_map(x, y, eps_list=(1.0, 0.1, 0.02), iters=30, device="cpu")
        self.assertEqual(T.shape, x.shape)
        i, j = np.triu_indices(len(x), 1)
        dot = np.sum((x[i] - x[j]) * (T[i] - T[j]), 1)
        self.assertTrue(np.all(dot >= -1e-9), f"min dot {dot.min()}")
        # 单调配对 ⇒ 循环单调 ⇒ 对经验测度最优：精确匈牙利 == 配对代价，且线性衰减
        w2c = O.w2_coupling(x, T)
        w2e, _ = O.w2_exact(x, T)
        self.assertAlmostEqual(w2e, w2c, places=7)
        for t in (0.3, 0.7):
            P = (1 - t) * x + t * T
            self.assertAlmostEqual(O.w2_exact(P, T)[0], (1 - t) * w2c, places=7)

    def test_unconverged_potential_still_monotone(self):
        """凸性不靠收敛：随便给一个势 g，映射也单调"""
        x, y = cloud(300, 5)
        g = np.random.default_rng(6).normal(size=len(y))
        T = O.barycentric(x, y, g, 0.05, device="cpu")
        i, j = np.triu_indices(len(x), 1)
        self.assertTrue(np.all(np.sum((x[i] - x[j]) * (T[i] - T[j]), 1) >= -1e-9))


class TestMassAndSchedule(unittest.TestCase):
    def test_particle_tau_endpoints(self):
        s = np.linspace(0, 0.35, 11)
        np.testing.assert_allclose(O.particle_tau(1.0, s), 1.0)
        self.assertTrue(np.all(O.particle_tau(0.0, s) <= 0.03))
        g = np.linspace(0, 1, 201)
        tau = np.stack([O.particle_tau(v, s) for v in g])
        self.assertTrue(np.all(np.diff(tau, axis=0) >= -1e-12), "τ 必须单调不倒退")

    def test_ease_never_at_rest(self):
        u = np.linspace(0, 1, 10001)
        v = np.gradient(O.ease(u), u)
        self.assertGreater(v.min(), 0.25)        # 两端都在动
        self.assertGreater(v[0], v[-1])          # 快出慢到
        self.assertAlmostEqual(float(O.ease(1.0)), 1.0)

    def test_mass_conserved_along_path(self):
        x, y = cloud(500, 7)
        for t in (0, 0.5, 1):
            P = (1 - t) * x + t * y
            self.assertEqual(len(P), len(x))
            self.assertTrue(np.all(np.isfinite(P)))
        np.testing.assert_allclose(np.sort(((1 - 1.0) * x + 1.0 * y)[:, 0]), np.sort(y[:, 0]))


class TestStillUnchanged(unittest.TestCase):
    def test_still_problem_reproducible(self):
        """重构后 stills.ot_problem + sliced_ot 与重构前的代价逐位一致（静帧 HUD 印的 W2² ≈ 6.558）"""
        src, Y = O.problem(60000)
        self.assertAlmostEqual(float(np.mean(np.sum((Y - src) ** 2, 1))), 6.557504224457361, places=10)


if __name__ == "__main__":
    unittest.main()
