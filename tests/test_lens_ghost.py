#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""透镜重影（tools/lens_ghost.py）自检：找「必须相等」的量。纯 numpy，系统 python 也能跑（不开 torch）。"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import lens_ghost as L  # noqa: E402


def rand_img(H=72, W=120, seed=0):
    return np.random.default_rng(seed).random((H, W, 3)).astype(np.float32)


def ramp_through(lx, ly, H, W):
    X = np.arange(W)[None, :] + lx
    Y = np.arange(H)[:, None] + ly
    return (0.3 * X / W + 0.7 * Y / H)[..., None].repeat(3, -1).astype(np.float32)


class TestIdentity(unittest.TestCase):
    def test_zero_refraction_zero_ghost_is_identity(self):
        """micro = 0、A = 0、stretch = 0 → 位移场处处为 0；色散（任意 dk）、多重像平均都不改变图（逐像素）"""
        img = rand_img()
        p = L.Params(micro=0, A=0, stretch=0, dk=0.12, n_ghost=3)
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
        np.testing.assert_allclose(out[:30], 1.0, atol=1e-5)

    def test_supersample_roundtrip(self):
        """2× 上采样再 2×2 平均下采样：平滑图回到自己（误差 < 1e-3）"""
        img = L.gaussian_blur(rand_img(40, 60, 3), 3.0)
        np.testing.assert_allclose(L.downsample(L.upsample(img, 2), 2)[2:-2, 2:-2], img[2:-2, 2:-2], atol=2e-3)

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
        shift = [100 - (out[..., c] * ys).sum() / out[..., c].sum() for c in range(3)]
        kc = L.channel_scale(dk, 7)
        self.assertLess(kc[0], kc[1])
        self.assertLess(kc[1], kc[2])
        self.assertAlmostEqual(kc[1], 1.0, places=6)
        for c in range(3):
            self.assertAlmostEqual(shift[c], 40 * kc[c], delta=0.05)

    def test_sense_mapping(self):
        p0 = L.params_from_sense(0, 0, 0, 720)
        p1 = L.params_from_sense(1, 1, 1, 720)
        self.assertEqual((p0.A, p1.A), (4, 24))
        self.assertAlmostEqual(p0.stretch, 0.15); self.assertAlmostEqual(p1.stretch, 0.70)
        self.assertAlmostEqual(p0.dk, 0.04); self.assertAlmostEqual(p1.dk, 0.16)
        self.assertAlmostEqual(p0.defocus, 0.5); self.assertAlmostEqual(p1.defocus, 3.0)
        self.assertAlmostEqual(L.params_from_sense(1, 0, 0, 1440).A, 48)


class TestMicroLens(unittest.TestCase):
    def test_texture_period_equals_set_period(self):
        """⭐必须相等：C¹ 圆顶微透镜作用在斜坡图上，FFT 量到的纹理周期 == 设定周期（整数倍画幅 → 正好落在 bin 上）"""
        for px, py in ((12, 8), (16, 10), (14, 9)):
            H, W = py * 24, px * 20
            _, (lx, ly), _ = L.displacements(H, W, L.Params(period_x=px, period_y=py, micro=-0.6, A=0, stretch=0))
            m = L.measure_period(ramp_through(lx, ly, H, W), min_period=4, max_period=40)
            self.assertAlmostEqual(m["x"], px, delta=1e-6, msg=(px, py, m))
            self.assertAlmostEqual(m["y"], py, delta=1e-6, msg=(px, py, m))

    def test_micro_offset_is_c1_no_seams(self):
        """⭐v2 没有接缝：微透镜偏移在格边上 = 0，且相邻像素的变化不超过解析上界 |micro|·π·(P/2)·(2/P)·…（Lipschitz）"""
        px, py, mic = 14.0, 9.0, -0.6
        cells, g = L.cap_field(90, 140, px, py)
        dx, dy = L.micro_offset(cells, g, mic, px, py)
        edge = np.isclose(np.abs(cells["u"]), 1.0) | np.isclose(cells["u"], -1.0)
        self.assertLess(np.abs(dx[edge]).max(), 1e-9)
        # ∂D_x/∂x = micro·(P/2)·π·cos(πu)·cos²(πv/2)·(2/P) → |·| ≤ |micro|·π；∂D_y/∂x 同理 ≤ |micro|·(py/2)·(π/2)·(2/px)
        bx = abs(mic) * np.pi + 1e-9
        by = abs(mic) * (py / 2) * (np.pi / 2) * (2 / px) + 1e-9
        self.assertLessEqual(np.abs(np.diff(dx, axis=1)).max(), bx)
        self.assertLessEqual(np.abs(np.diff(dy, axis=1)).max(), by)

    def test_ghost_field_smooth_across_x(self):
        """⭐v2 没有硬边竖条：竖向重影位移沿 x 是 C¹ 的 —— 二阶差分远小于一阶差分（平滑曲线）；
        v1 按列分段常数，二阶差分和一阶差分一样大（台阶）"""
        H, W = 200, 400

        def ratio(dy):
            dy = np.broadcast_to(dy, (H, W))
            return np.abs(np.diff(dy, 2, axis=1)).max() / np.abs(np.diff(dy, axis=1)).max()

        _, _, cols = L.displacements(H, W, L.Params(A=40, stretch=0.5, corr=60, n_ghost=1))
        _, _, cols1 = L.displacements_v1(H, W, L.Params(A=40, scatter=0.3, corr=35, n_ghost=1))
        r2, r1 = ratio(cols[0][1]), ratio(cols1[0][1])
        self.assertLess(r2, 0.35, msg=(r1, r2))
        self.assertGreaterEqual(r1, 0.99, msg=(r1, r2))      # 台阶：二阶差分 = 一阶差分

    def test_no_peak_at_old_column_period(self):
        """⭐渲染一张平滑随机图（只开竖向重影，不开微透镜）：输出横向谱在旧列周期 14 px 处没有峰（v1 有）"""
        H, W = 160, 14 * 24
        img = L.gaussian_blur(rand_img(H, W, 7), 6.0) * 4

        def spec_at(out, period):
            g = np.abs(np.diff(L.lum(out)[20:-20], axis=1))[:, :W - 2]       # 横向梯度幅值：接缝 = 周期性尖峰
            g = np.pad(g, ((0, 0), (0, 2)))
            g = g - g.mean(1, keepdims=True)
            F = np.abs(np.fft.rfft(g * np.hanning(W)[None, :], axis=1)).mean(0)
            k = int(round(W / period))
            return F[k] / np.median(F[k - 8:k + 9])

        p2 = L.Params(micro=0, A=30, stretch=0.4, corr=60, n_ghost=1)
        _, lens, cols = L.displacements(H, W, p2)
        r2 = spec_at(L.lens_stage(img, lens, cols, p2, 0), 14)
        p1 = L.Params(A=30, scatter=0.5, corr=35, n_ghost=1)
        _, lens1, cols1 = L.displacements_v1(H, W, p1, k=0.0, bow=0.0)
        r1 = spec_at(L.lens_stage(img, lens1, cols1, p1, 0), 14)
        self.assertLess(r2, 2.0, msg=(r1, r2))
        self.assertGreater(r1, r2, msg=(r1, r2))

    def test_copy_spacing_law_v1(self):
        """（v1 抛物面透镜，保留）p = y + k(y − c) → 重影间距 Δ = P·k/(1+k)，逐像素相等"""
        P, k = 9, 2.0
        H, W = P * 12, 60
        img = rand_img(H, W, 5)
        cells = L.lens_cells(H, W, 15, P)
        lx, ly = L.lens_offset(cells, k, k, 0.0, P)
        out = L.disperse_sample(img, lx, ly, 0)
        d = int(P * k / (1 + k))
        for y in [y for y in range(2 * P, H - 2 * P) if (y % P) >= P - d]:
            np.testing.assert_allclose(out[y + d, 20:40], out[y, 20:40], atol=1e-6)

    def test_stagger_offsets_odd_rows(self):
        c = L.lens_cells(40, 100, 10, 8, stagger=0.5)
        self.assertTrue(np.all(np.mod(c["cx"][4] - 5, 10) == 0))
        self.assertTrue(np.all(np.mod(c["cx"][12] - 10, 10) == 0))

    def test_vertical_edge_survives_pure_shear(self):
        """竖直的明暗边（左黑右白）在纯竖向位移（无拉伸外的横向分量）下逐像素不变"""
        H, W = 80, 120
        img = np.zeros((H, W, 3), np.float32)
        img[:, 60:] = 1
        p = L.Params(micro=0, A=20, stretch=0.3, n_ghost=1)
        _, lens, cols = L.displacements(H, W, p)
        out = L.lens_stage(img, lens, cols, p, 0)
        np.testing.assert_allclose(out[35:45], img[35:45], atol=1e-6)


class TestTime(unittest.TestCase):
    def test_time_evolution_slow_and_bursts(self):
        x = np.linspace(0, 500, 200)
        a, b = L.fourier_noise(x, 0.0), L.fourier_noise(x, 1 / 24)
        self.assertLess(np.abs(a - b).max(), 0.25)
        t = np.arange(48) / 24
        th = np.zeros(100); th[12] = 0.9; th[30] = 0.1
        e, fired = L.bursts(t, [0.5, 1.25], th, thr=0.5, tau=0.25)
        self.assertEqual(fired, [0.5])
        self.assertAlmostEqual(e[12], 1.0)
        self.assertAlmostEqual(e[18], np.exp(-0.25 / 0.25))
        self.assertEqual(e[:12].max(), 0.0)
        q = L.evolve_params(L.Params(A=50, stretch=0.4, dk=0.1), 2.0, 1.0, 1, drift_x=4.0, burst_gain=0.8)
        self.assertEqual((q.origin_x, q.A, q.seed), (8.0, 90.0, 1003))
        self.assertAlmostEqual(q.dk, 0.15)
        self.assertAlmostEqual(q.stretch, 0.56)


if __name__ == "__main__":
    unittest.main()
