#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""透镜重影（tools/lens_ghost.py）自检：找「必须相等」的量。纯 numpy，系统 python 也能跑。"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import lens_ghost as L  # noqa: E402


def rand_img(H=72, W=120, seed=0):
    return np.random.default_rng(seed).random((H, W, 3)).astype(np.float32)


def lens_only(H, W, **kw):
    p = L.Params(A=0, **kw)
    cells, lens, _ = L.displacements(H, W, p)
    return cells, lens


class TestIdentity(unittest.TestCase):
    def test_zero_k_zero_A_is_identity(self):
        """k = 0、bow = 0、A = 0 → 位移场处处为 0；色散（任意 dk）、多重像平均都不改变图（逐像素）"""
        img = rand_img()
        p = L.Params(k_x=0, k_y=0, bow=0, A=0, dk=0.12, n_ghost=3)
        _, lens, cols = L.displacements(*img.shape[:2], p)
        for cx, cy in cols:
            np.testing.assert_allclose(lens[0] + cx, 0, atol=1e-12)
            np.testing.assert_allclose(lens[1] + cy, 0, atol=1e-12)
        np.testing.assert_allclose(L.lens_stage(img, lens, cols, p, p.dk), img, atol=1e-6)

    def test_strength_zero_is_identity(self):
        img = rand_img(seed=1)
        p = L.Params(strength=0.0)
        _, lens, cols = L.displacements(*img.shape[:2], p)
        np.testing.assert_allclose(L.lens_stage(img, lens, cols, p, 0.1), img, atol=1e-6)

    def test_white_in_white_out(self):
        """光谱权重按通道归一化：平坦白图在任意色散下仍是白（彩边只能出现在明暗边界上）"""
        img = np.ones((40, 60, 3), np.float32)
        dy = np.full((40, 60), 3.0)
        out = L.disperse_sample(img, np.zeros_like(dy), dy, 0.1, 9)
        np.testing.assert_allclose(out[:30], 1.0, atol=1e-5)    # 底部几行采到界外（黑）

    def test_blur_sigma_zero_identity_and_mean_preserved(self):
        img = rand_img(seed=2)
        np.testing.assert_array_equal(L.gaussian_blur(img, 0), img)
        self.assertAlmostEqual(float(L.gaussian_blur(img, 2.5).mean()), float(img.mean()), places=2)


class TestDispersion(unittest.TestCase):
    def test_channel_ordering(self):
        """一个白点、均匀位移 D = (0, 40)：每个通道的质心位移 = k_c·40，k_R < k_G < k_B，k_G = 1（光谱对称）"""
        H, W = 160, 40
        img = np.zeros((H, W, 3), np.float32)
        img[100, 20] = 1.0
        dy = np.full((H, W), 40.0)
        dk = 0.1
        out = L.disperse_sample(img, np.zeros_like(dy), dy, dk, 7)
        ys = np.arange(H)[:, None]
        cen = [(out[..., c] * ys).sum() / out[..., c].sum() for c in range(3)]
        shift = [100 - c for c in cen]               # 采样 p = y + s·40 → 输出里的点在 100 − s·40
        kc = L.channel_scale(dk, 7)
        self.assertLess(kc[0], kc[1])
        self.assertLess(kc[1], kc[2])
        self.assertAlmostEqual(kc[1], 1.0, places=6)
        for c in range(3):
            self.assertAlmostEqual(shift[c], 40 * kc[c], delta=0.05)
        self.assertLess(shift[0], shift[1])
        self.assertLess(shift[1], shift[2])

    def test_sense_mapping(self):
        p0 = L.params_from_sense(0, 0, 0, 720)
        p1 = L.params_from_sense(1, 1, 1, 720)
        self.assertEqual((p0.A, p1.A), (25, 130))
        self.assertAlmostEqual(p0.dk, 0.04); self.assertAlmostEqual(p1.dk, 0.16)
        self.assertAlmostEqual(p0.defocus, 0.5); self.assertAlmostEqual(p1.defocus, 3.0)
        self.assertAlmostEqual(L.params_from_sense(1, 0, 0, 1440).A, 260)


class TestLensGrid(unittest.TestCase):
    def test_texture_period_equals_set_period(self):
        """⭐必须相等：透镜项作用在斜坡图上，FFT 量到的纹理周期 == 设定周期（整数倍画幅 → 正好落在 bin 上）"""
        for px, py in ((12, 8), (16, 10), (9, 6), (14, 9)):
            H, W = py * 24, px * 20
            _, (lx, ly) = lens_only(H, W, period_x=px, period_y=py, k_x=1.5, k_y=1.5, bow=0.0)
            X = np.arange(W)[None, :] + lx
            Y = np.arange(H)[:, None] + ly
            ramp = (0.3 * X / W + 0.7 * Y / H)[..., None].repeat(3, -1).astype(np.float32)
            m = L.measure_period(ramp, min_period=4, max_period=40)
            self.assertAlmostEqual(m["x"], px, delta=1e-6, msg=(px, py, m))
            self.assertAlmostEqual(m["y"], py, delta=1e-6, msg=(px, py, m))

    def test_copy_spacing_law(self):
        """⭐必须相等：真走一遍双线性采样，随机图（内容无周期）。p = y + k(y − c) → 相邻两格采到同一源像素的输出行相差
        Δ = P·k/(1+k)（P = 9, k = 2 → 6 行）：重影间距 ≠ 网格周期。out[y + Δ] == out[y]，只要 y + Δ 落在下一格"""
        P, k = 9, 2.0
        H, W = P * 12, 60
        img = rand_img(H, W, 5)
        _, (lx, ly) = lens_only(H, W, period_x=15, period_y=P, k_x=k, k_y=k, bow=0.0)
        out = L.disperse_sample(img, lx, ly, 0)
        d = int(P * k / (1 + k))
        rows = [y for y in range(2 * P, H - 2 * P) if (y % P) >= P - d]
        self.assertTrue(rows)
        for y in rows:
            np.testing.assert_allclose(out[y + d, 20:40], out[y, 20:40], atol=1e-6)
        self.assertGreater(np.abs(out[2 * P + 1, 20:40] - out[2 * P + 2, 20:40]).max(), 1e-3)   # 不是平凡常数

    def test_stagger_offsets_odd_rows(self):
        c = L.lens_cells(40, 100, 10, 8, stagger=0.5)
        self.assertTrue(np.all(np.mod(c["cx"][4] - 5, 10) == 0))          # 行 0：中心在 5, 15, …
        self.assertTrue(np.all(np.mod(c["cx"][12] - 10, 10) == 0))        # 行 1：错开半格 → 10, 20, …
        self.assertTrue(np.all(np.abs(c["u"]) <= 1) and np.all(np.abs(c["v"]) <= 1))

    def test_column_displacement_constant_per_column(self):
        """大位移在每一整列透镜里是常数（stagger = 0）：同列不同行完全相同 → 离散复制、竖边不动；列与列之间不同"""
        p = L.Params(period_x=10, period_y=8, A=60, scatter=0.5, n_ghost=2)
        cells, _, cols = L.displacements(64, 100, p)
        for dx, dy in cols:
            np.testing.assert_array_equal(dy, np.broadcast_to(dy[:1], dy.shape))
            for i in np.unique(cells["i"]):
                m = cells["i"][0] == i
                self.assertLess(np.ptp(dy[0, m]), 1e-9)
            self.assertGreater(np.ptp(dy[0]), 10)
            self.assertLess(np.abs(dx).max(), np.abs(dy).max() * 0.2)      # 各向异性：横向很小
        self.assertGreater(np.abs(cols[0][1] - cols[1][1]).max(), 1)        # 两套像不一样

    def test_vertical_edge_survives_column_shift(self):
        """竖直的明暗边（左黑右白）在纯按列竖移下逐像素不变（只有横向的东西被打散）"""
        H, W = 80, 120
        img = np.zeros((H, W, 3), np.float32)
        img[:, 60:] = 1
        p = L.Params(period_x=10, period_y=8, k_x=0, k_y=0, bow=0, A=30, aniso=0, n_ghost=1)
        _, lens, cols = L.displacements(H, W, p)
        out = L.lens_stage(img, lens, cols, p, 0)
        np.testing.assert_allclose(out[35:45], img[35:45], atol=1e-6)      # 中间几行（A=30 不会采到界外）

    def test_time_evolution_slow_and_bursts(self):
        """噪声随时间连续（一帧的变化 ≪ 幅度）；脉冲在起音处 = 1、之后按 tau 衰减；未达阈值的不触发"""
        x = np.linspace(0, 500, 200)
        a, b = L.fourier_noise(x, 0.0), L.fourier_noise(x, 1 / 24)
        self.assertLess(np.abs(a - b).max(), 0.25)
        t = np.arange(48) / 24
        th = np.zeros(100); th[12] = 0.9; th[30] = 0.1
        e, fired = L.bursts(t, [0.5, 1.25], th, thr=0.3, tau=0.25)
        self.assertEqual(fired, [0.5])
        self.assertAlmostEqual(e[12], 1.0)
        self.assertAlmostEqual(e[18], np.exp(-0.25 / 0.25))
        self.assertEqual(e[:12].max(), 0.0)
        q = L.evolve_params(L.Params(A=50, dk=0.1), 2.0, 1.0, 1, drift_x=4.0, burst_gain=0.8)
        self.assertEqual((q.origin_x, q.A, q.seed), (8.0, 90.0, 1003))
        self.assertAlmostEqual(q.dk, 0.15)


if __name__ == "__main__":
    unittest.main()
