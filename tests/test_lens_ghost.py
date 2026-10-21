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


class TestV4(unittest.TestCase):
    def test_vertical_ghost_is_purely_vertical(self):
        """⭐v4 大重影 dx ≡ 0：竖直的明暗边逐像素不变（任意 S、d、色散）；w_s = 0、d = 0 → 恒等"""
        H, W = 80, 120
        img = np.zeros((H, W, 3), np.float32)
        img[:, 60:] = 1
        S, d = L.vertical_fields(W, 30, 10, 200, seed=2)
        out = L.vertical_ghost(img, S, d, K=7, w_s=0.6, dk=0.15, n_lambda=5)
        np.testing.assert_allclose(out[35:45], img[35:45], atol=1e-6)
        rnd = rand_img(H, W, 4)
        np.testing.assert_allclose(L.vertical_ghost(rnd, 0 * S, 0 * d, w_s=0, dk=0.1), rnd, atol=1e-6)

    def test_spread_equals_mean_of_shifted_copies(self):
        """⭐必须相等：拉开分量 = K 个整数竖移副本的平均（S 取整数、K 使抽头落在整数上 → 逐像素相等）"""
        H, W = 60, 30
        img = rand_img(H, W, 6)
        S = np.full((1, W), 6.0)
        d = np.zeros((1, W))
        out = L.vertical_ghost(img, S, d, K=7, w_s=1.0, dk=0)
        ref = np.zeros_like(img)
        for k in range(-3, 4):
            sh = np.zeros_like(img)
            if k >= 0:
                sh[:H - 2 * k] = img[2 * k:]
            else:
                sh[-2 * k:] = img[:H + 2 * k]
            ref += sh / 7
        np.testing.assert_allclose(out[8:-8], ref[8:-8], atol=1e-5)

    def test_fields_low_frequency_in_x(self):
        """S(x)、d(x) 只随 x 低频变化：一阶差分 ≤ 幅度 × 2π/corr × 4（4σ 余量）→ 横线不会弯成波浪"""
        W, corr = 1280, 500.0
        S, d = L.vertical_fields(W, 240, 40, corr, seed=3)
        self.assertLessEqual(np.abs(np.diff(S)).max(), 120 * 2 * np.pi / corr * 4)
        self.assertLessEqual(np.abs(np.diff(d)).max(), 40 * 2 * np.pi / corr * 4)

    def test_dispersion_on_y_orders_channels(self):
        """色散作用在竖向位移上：一条横向亮线竖移 30 px，R/G/B 的质心位移 = k_c·30，R < G < B（横边上出彩色色块）"""
        H, W = 120, 20
        img = np.zeros((H, W, 3), np.float32)
        img[80] = 1
        S, d = np.zeros((1, W)), np.full((1, W), 30.0)
        z = np.zeros((1, W))
        out = L.disperse_sample(img, z, d, 0.2, 5)
        ys = np.arange(H)[:, None]
        sh = [80 - (out[..., c] * ys).sum() / out[..., c].sum() for c in range(3)]
        kc = L.channel_scale(0.2, 5)
        for c in range(3):
            self.assertAlmostEqual(sh[c], 30 * kc[c], delta=0.05)
        self.assertLess(sh[0], sh[1]); self.assertLess(sh[1], sh[2])

    def test_soft_shoulder(self):
        """软肩：拐点以下恒等；单调；永远 < 1；拐点处一阶导连续（= 1）"""
        x = np.linspace(0, 3, 3001, dtype=np.float32)
        y = L.soft_shoulder(x, 0.75)
        np.testing.assert_array_equal(y[x < 0.75], x[x < 0.75])
        self.assertTrue(np.all(np.diff(y) >= 0))
        self.assertLessEqual(y.max(), 1.0)
        self.assertLess(y[x <= 1.5].max(), 1.0)
        i = np.searchsorted(x, 0.75)
        self.assertAlmostEqual(float((y[i + 1] - y[i]) / (x[i + 1] - x[i])), 1.0, places=2)

    def test_film_layer_zero_is_identity_and_black_stays_black(self):
        img = rand_img(50, 70, 8)
        tex = np.random.default_rng(1).normal(size=(50, 70)).astype(np.float32)
        np.testing.assert_allclose(L.film_layer(img, np.zeros_like(tex), 1.5, 0, 0), img, atol=1e-6)
        blk = np.zeros_like(img)
        self.assertEqual(float(np.abs(L.film_layer(blk, tex, 1.5, 0.35, 1.5)).max()), 0.0)   # 颗粒和闪点都乘亮度

    def test_quilt_texture_unit_stats_and_seamless(self):
        """补丁拼贴：均值 0、标准差 1；行 / 列相邻差的统计在补丁边界处和补丁内部一样（没有接缝）"""
        rng = np.random.default_rng(0)
        tex = L.gaussian_blur(rng.normal(size=(300, 300)).astype(np.float32), 1.0)
        tex = (tex - tex.mean()) / tex.std()
        cands = [(int(y), int(x)) for y, x in rng.integers(60, 240, size=(50, 2))]
        t = L.tile_texture(tex, 200, 320, 0, 0, cands, patch=48, seed=1)
        self.assertAlmostEqual(float(t.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(t.std()), 1.0, places=5)
        dcol = np.abs(np.diff(t, axis=1)).mean(0)
        at_seam = dcol[[c - 1 for c in range(24, 319, 24) if 0 < c - 1 < len(dcol)]].mean()
        self.assertLess(at_seam, dcol.mean() * 1.3)

    @unittest.skipUnless(L.REF.exists(), "参考图不在这台机器上")
    def test_ref_texture_keeps_reference_period(self):
        """⭐必须相等（低熵注入的检验）：纹理取自参考图，拼贴后 FFT 竖向周期仍 = 参考图本身的 9.2 px（±1 个 bin）"""
        tex, cands = L.texture_candidates(1.0, 96)
        t = L.tile_texture(tex, 414, 736, 0, 0, cands, 96, seed=3)
        m = L.measure_period(t[..., None].repeat(3, -1) * 0.2 + 0.5)
        ref = L.measure_period(L.load(L.REF)[550:1000, 380:700])
        self.assertAlmostEqual(m["y"], ref["y"], delta=0.25)


class TestTime(unittest.TestCase):
    def test_time_evolution_slow_and_bursts(self):
        x = np.linspace(0, 500, 200)
        a, b = L.fourier_noise(x, 0.0), L.fourier_noise(x, 1 / 24)
        self.assertLess(np.abs(a - b).max(), 0.25)
        t = np.arange(48) / 24
        th = np.zeros(100); th[12] = 0.9; th[30] = 0.1
        e, fired = L.bursts(t, [0.5, 1.25], th, thr=0.5, tau=0.25)
        # 每隔一个：4 个强起音（威胁 0.55）→ 第 0、2 个触发；威胁 0.7 的一定触发
        th2 = np.zeros(200); th2[[24, 48, 72, 96]] = 0.55; th2[120] = 0.7
        _, f2 = L.bursts(np.arange(200) / 24, [1, 2, 3, 4, 5], th2, thr=0.5, every=2, always_thr=0.6)
        self.assertEqual(f2, [1.0, 3.0, 5.0])
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
