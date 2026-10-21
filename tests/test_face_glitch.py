#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""面部瓦解（tools/face_glitch.py）每一级的数值自检：找「必须相等」的量。"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import face_glitch as F  # noqa: E402


def rand_img(H=64, W=96, seed=0):
    return np.random.default_rng(seed).random((H, W, 3)).astype(np.float32)


def disk_mask(H=64, W=96, cx=60, cy=32, r=20):
    yy, xx = np.mgrid[0:H, 0:W]
    return ((xx - cx) ** 2 + (yy - cy) ** 2 <= r * r).astype(np.float32)


class TestStages(unittest.TestCase):
    def test_band_displace_rows_are_rolls(self):
        """每一行 = 原行循环平移 dx（同一切片 dx 相同），mask 跟着同样平移"""
        img, m = rand_img(), disk_mask()
        out, om, bands = F.band_displace(img, m, np.random.default_rng(3), n_bands=8, max_dx=40, p_shift=1.0)
        self.assertEqual(sum(y1 - y0 for y0, y1, _ in bands), img.shape[0])
        self.assertTrue(any(dx for *_, dx in bands))
        for y0, y1, dx in bands:
            np.testing.assert_array_equal(out[y0:y1], np.roll(img[y0:y1], dx, axis=1))
            np.testing.assert_array_equal(om[y0:y1], np.roll(m[y0:y1], dx, axis=1))

    def test_row_smear_identity_limit(self):
        """L = ∞、不断点、增益 1、win = 1 → 边界左边每个像素 = 边界内侧 inset 处的像素"""
        img = np.zeros((40, 80, 3), np.float32)
        m = disk_mask(40, 80, 55, 20, 12)
        img[m > 0] = np.random.default_rng(1).random(((m > 0).sum(), 3))
        xr = F.leftmost(m)
        out = F.row_smear(img, xr, xr, np.random.default_rng(0), length=np.inf, inset=2, dropout=0,
                          row_gain=(1.0, 1.0), win=1)
        for y in np.flatnonzero(xr > 0):
            np.testing.assert_allclose(out[y, :xr[y]], np.broadcast_to(img[y, xr[y] + 2], (xr[y], 3)))

    def test_row_smear_decay(self):
        """x_b 左边 d 像素处强度 = exp(-d/L)·c（单行、L 的 lognormal 抽到 0 时）：检查单调衰减 + 台阶内满强度"""
        img = np.zeros((1, 100, 3), np.float32); img[0, 60:] = 1.0
        xr = np.array([60]); xb = np.array([50])
        out = F.row_smear(img, xr, xb, np.random.default_rng(0), length=20, inset=0, dropout=0, row_gain=(1, 1), win=1)
        np.testing.assert_allclose(out[0, 50:60], 1.0)
        v = out[0, :50, 0]
        self.assertTrue(np.all(np.diff(v) > 0))
        # 相邻像素之比恒为 exp(1/L') —— 指数
        r = v[1:] / v[:-1]
        self.assertLess(np.ptp(r), 1e-5)

    def test_waterfall_rows_time_to_x(self):
        """离边界 d 像素 = 第 d // px 帧之前的边界色"""
        H, W, T, px = 4, 50, 5, 3
        img = np.zeros((H, W, 3), np.float32)
        xr = np.array([40, 30, -1, 20])
        hist = np.zeros((T, H, 3), np.float32)
        for k in range(T):
            hist[k] = 0.1 * (k + 1)
        out = F.waterfall_rows(img, xr, hist, px_per_frame=px)
        for y in (0, 1, 3):
            for d in range(xr[y]):
                k = d // px
                want = 0.1 * (k + 1) if k < T else 0.0
                self.assertAlmostEqual(float(out[y, xr[y] - 1 - d, 0]), want, places=6)
        np.testing.assert_array_equal(out[2], 0)

    def test_pixel_sort_preserves_rows_and_sorts_runs(self):
        img = rand_img(16, 64, 2)
        region = np.ones((16, 64), bool)
        out = F.pixel_sort_rows(img, region, lo=0.2, hi=0.8)
        for y in range(16):
            a = np.sort(img[y].view([("r", "f4"), ("g", "f4"), ("b", "f4")]).ravel(), order=["r", "g", "b"])
            b = np.sort(out[y].view([("r", "f4"), ("g", "f4"), ("b", "f4")]).ravel(), order=["r", "g", "b"])
            np.testing.assert_array_equal(a, b)
        L, L0 = F.lum(out), F.lum(img)
        sel = (L0 >= 0.2) & (L0 <= 0.8)
        for y in range(16):
            d = np.diff(np.concatenate([[0], sel[y].astype(np.int8), [0]]))
            for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
                self.assertTrue(np.all(np.diff(L[y, s:e]) >= -1e-7))
            np.testing.assert_array_equal(out[y][~sel[y]], img[y][~sel[y]])

    def test_channel_split_global_exact(self):
        img = rand_img()
        out = F.channel_split(img, None, d=(5, 0, -3), mode="global")
        np.testing.assert_array_equal(out[:, 5:, 0], img[:, :-5, 0])
        np.testing.assert_array_equal(out[:, :5, 0], 0)
        np.testing.assert_array_equal(out[..., 1], img[..., 1])
        np.testing.assert_array_equal(out[:, :-3, 2], img[:, 3:, 2])

    def test_channel_split_normal_zero_is_identity(self):
        img, m = rand_img(), disk_mask()
        np.testing.assert_array_equal(F.channel_split(img, m, d=(0, 0, 0), mode="normal"), img)

    def test_column_jitter_columns_are_rolls(self):
        img = rand_img(32, 40, 4)
        for pc in (False, True):
            out = F.column_jitter(img, np.ones((32, 40), np.float32), np.random.default_rng(5), amp=4, gain=0.0,
                                  per_channel=pc)
            for x in range(40):
                for c in range(3):
                    col, o = img[:, x, c], out[:, x, c]
                    self.assertTrue(any(np.array_equal(np.roll(col, s), o) for s in range(-4, 5)))

    def test_column_jitter_outside_mask_untouched(self):
        img, m = rand_img(), disk_mask()
        out = F.column_jitter(img, m, np.random.default_rng(1), amp=5, gain=0.5)
        np.testing.assert_array_equal(out[m == 0], img[m == 0])

    def test_macroblocks_dc(self):
        img = rand_img(64, 96, 6)
        region = np.ones((64, 96), bool)
        out, blocks = F.macroblocks(img, region, np.random.default_rng(2), grid=16, n_triplets=0, n_dashes=0, n_blocks=5)
        self.assertEqual(len(blocks), 5)
        for y, x in blocks:
            blk = out[y:y + 16, x:x + 16]
            np.testing.assert_allclose(blk, np.broadcast_to(img[y:y + 16, x:x + 16].reshape(-1, 3).mean(0), blk.shape),
                                       atol=1e-6)

    def test_posterize_levels(self):
        out = F.posterize(rand_img(), levels=5)
        self.assertTrue(np.all(np.isin(np.round(out * 4, 6), np.arange(5))))

    def test_sense_mapping(self):
        p = F.params_from_sense(0.5, 0.25, 1.0)
        self.assertEqual(p.n_bands, 15)
        self.assertEqual(p.max_dx, 166)
        self.assertAlmostEqual(p.p_shift, 0.5)
        self.assertEqual(p.split_d, (6, -3, -6))
        self.assertAlmostEqual(p.smear_len, 560)

    def test_micro_fringe_grey_away_from_edge(self):
        """微彩 A：轮廓圈外（离边界 > ring）的像素没有颜色 —— 三个通道之比 = 冷灰的比例"""
        img, m = rand_img(64, 96, 3), disk_mask()
        out = F.palette_micro(img, m, "fringe", ring=4)
        far = F._ring(m, 4) == 0
        g = np.asarray(F.S.GREY, np.float32) / max(F.S.GREY)
        ratio = out[far] / np.maximum(out[far][:, :1], 1e-6) * g[0]
        np.testing.assert_allclose(ratio[out[far][:, 0] > 1e-3], np.broadcast_to(g, ratio[out[far][:, 0] > 1e-3].shape),
                                   rtol=1e-4)

    def test_hue_and_false_colour_anchors(self):
        np.testing.assert_allclose(F.hue_rgb([0.0, 1 / 3, 2 / 3]), [[1, 0, 0], [0, 1, 0], [0, 0, 1]], atol=1e-6)
        np.testing.assert_allclose(F.false_colour(np.array([0.0, 1.0])), [[0, 0, 0], [1, 1, 1]])

    def test_all_palettes_shape_range(self):
        img, m = rand_img(), disk_mask()
        for key, (_, fn) in F.PALETTES.items():
            out = fn(img, m, 3)
            self.assertEqual(out.shape, img.shape, key)
            self.assertTrue(np.all((out >= 0) & (out <= 1)), key)

    def test_envelope_bursts_and_final(self):
        """脉冲：只在威胁峰 ≥ thr 的起音处，满幅 hold 帧后指数衰减；收尾：吸到小节首拍，ramp 后恒 = 1"""
        fps = 24
        t = 10 + np.arange(120) / fps                       # 10.000 … 14.958
        threat = np.zeros(20 * fps)
        threat[int(round(10.5 * fps))] = 0.9                # 强
        threat[int(round(11.5 * fps))] = 0.3                # 弱，不触发
        g, info = F.envelope(t, [10.5, 11.5], threat, fps=fps, thr=0.5, hold=2, tau=1.5, a_min=0.6,
                             sustain=1.4, ramp=0.4, downbeats=[13.25])
        a = 0.6 + 0.4 * (0.9 - 0.5) / 0.5
        j0 = int(np.searchsorted(t, 10.5 - 0.5 / fps))
        np.testing.assert_allclose(g[j0:j0 + 2], a)
        self.assertAlmostEqual(g[j0 + 2], a * np.exp(-1 / 1.5), places=9)
        self.assertEqual(g[j0 - 1], 0.0)
        self.assertEqual(len(info["bursts"]), 1)
        # 收尾：t_end = 15.0 → r0 = 13.2 → 吸到 13.25
        self.assertAlmostEqual(info["ramp_start"], 13.25)
        np.testing.assert_array_equal(g[t >= 13.65], 1.0)
        self.assertTrue(np.all(np.diff(g[(t >= 13.25) & (t <= 13.65)]) >= 0))
        # 稀疏：收尾之前门开（> 0.05）的帧不到 15 %
        self.assertLess(np.mean(g[t < 13.25] > 0.05), 0.15)

    def test_scale_params_zero_is_calm(self):
        p = F.scale_params(F.params_from_sense(0.9, 0.9, 0.9), 0.0)
        self.assertEqual(p.max_dx, 0)
        self.assertEqual(p.split_d, (0, 0, 0))
        self.assertEqual(p.n_triplets + p.n_dashes + p.n_blocks, 0)
        self.assertEqual(p.p_swap + p.p_invert + p.p_shift, 0)

    def test_strength_gate_zero_is_identity(self):
        """强度门控（不是透明度）：g = 0 → 整条管线 ≈ 恒等（只剩挪位 + 256 级量化，误差 ≤ 0.5/255），瀑布图行也一样"""
        img, m = rand_img(72, 128, 11), disk_mask(72, 128, 80, 36, 25)
        base = F.params_from_sense(0.9, 0.9, 0.9, F.Params(place_dx=10, grid=8))
        for wf in (False, True):
            p = F.scale_params(F.Params(**{**F.asdict(base), "waterfall": wf}), 0.0)
            hist = F.BoundaryHistory()
            for _ in range(3):
                out, _ = F.run(img, m, p, hist=hist)
            ref, _ = F.place(img, m, 10)
            self.assertLessEqual(float(np.abs(out - ref).max()), 0.5 / 255 + 1e-6, wf)

    def test_strength_gate_one_is_unchanged(self):
        p = F.params_from_sense(0.4, 0.3, 0.2)
        self.assertEqual(F.asdict(F.scale_params(p, 1.0)), F.asdict(p))

    def test_wipe_endpoints(self):
        """抹除：u = 0 → 恒等；u = 1、不压暗、最右列是黑 → 每一行都被扫掉，整张全黑（脸一个像素都不剩）"""
        img = rand_img(48, 80, 12)
        img[:, -1] = 0
        xr = F.leftmost(disk_mask(48, 80, 50, 24, 15))
        np.testing.assert_array_equal(F.wipe(img, xr, 0.0), img)
        np.testing.assert_array_equal(F.wipe(img, xr, 1.0, dim=0.0), 0)

    def test_wipe_front_monotonic(self):
        """前沿只往右走：u 越大，被替换（≠ 原图）的像素只多不少（不压暗时）"""
        img = rand_img(48, 80, 13) * 0.5 + 0.5
        xr = F.leftmost(disk_mask(48, 80, 50, 24, 15))
        prev = 0
        for u in np.linspace(0.1, 1.0, 6):
            k = int((F.wipe(img, xr, u, dim=0.0) != img).any(-1).sum())
            self.assertGreaterEqual(k, prev)
            prev = k

    def test_wipe_even_bright_and_never_darker(self):
        """抹除 v2：任何 u 都不压暗（逐像素 ≥ 原图）；u = 0 恒等；u = 1 每个像素 ≥ 该行最亮像素 × 0.75 × exp(−W/6000)"""
        img = rand_img(48, 80, 14)
        img[:, -10:] = 0                                    # 右边一条黑背景：前沿走进去也不能变黑
        xr = F.leftmost(disk_mask(48, 80, 50, 24, 15))
        np.testing.assert_array_equal(F.wipe_even(img, xr, 0.0), img)
        for u in (0.3, 0.7, 1.0):
            self.assertTrue(np.all(F.wipe_even(img, xr, u) >= img - 1e-7))
        out = F.wipe_even(img, xr, 1.0)
        L, L0 = F.lum(out), F.lum(img)
        x0 = np.maximum(xr, 0)
        rowmax = np.array([L0[y, x0[y]:].max() for y in range(48)])[:, None]   # 前沿扫过的路上最亮
        self.assertTrue(np.all(L >= rowmax * 0.75 * np.exp(-80 / 6000) - 1e-6))

    def test_soft_knee(self):
        x = np.linspace(0, 5, 2001)
        y = F.soft_knee(x)
        np.testing.assert_array_equal(y[x <= 0.8], x[x <= 0.8])
        self.assertTrue(np.all(np.diff(y) > 0))             # 单调
        self.assertTrue(np.all(y < 1))                      # 永不到死白
        dy = np.diff(y) / np.diff(x)
        i = np.searchsorted(x, 0.8)
        self.assertAlmostEqual(dy[i - 1], dy[i + 1], delta=0.05)   # 在 knee 处导数连续（=1）

    def test_soft_exposure_no_hard_white_block(self):
        """横带错位后的矩形遮罩：硬截断会出成片的 1.0；软化后没有任何像素到 1.0，且竖直方向不再有一步到位的硬边"""
        img = np.full((100, 60, 3), 0.6, np.float32)
        m = np.zeros((100, 60), np.float32)
        m[80:, 10:50] = 1                                    # 底部一块矩形（横带移出来的）
        hard = F.exposure_clip(img, m, 0.5, 3.0, 0.25, soft=0.0)
        soft = F.exposure_clip(img, m, 0.5, 3.0, 0.25, soft=6.0)
        self.assertTrue((hard >= 1.0).any())
        self.assertFalse((soft >= 1.0).any())
        step_hard = np.abs(np.diff(hard[:, 30, 0])).max()
        step_soft = np.abs(np.diff(soft[:, 30, 0])).max()
        self.assertLess(step_soft, 0.5 * step_hard)

    def test_red_gate_zero_no_red(self):
        """红边门控 red_k = 0 → 本片配色里没有红：每个像素三通道之比 = 冷灰"""
        img, m = rand_img(), disk_mask()
        img[..., 0] = np.clip(img[..., 0] + 0.5, 0, 1)      # 故意很红
        out = F.palette_ours(img, m, red_k=0.0)
        g = np.asarray(F.S.GREY, np.float32) / max(F.S.GREY)
        np.testing.assert_allclose(out / g, np.repeat((out / g)[..., :1], 3, -1), atol=1e-5)

    def test_pipeline_runs_deterministic(self):
        img, m = rand_img(72, 128, 9), disk_mask(72, 128, 80, 36, 25)
        p = F.Params(place_dx=10, n_dust=50, grid=8)
        a, _ = F.run(img, m, p)
        b, _ = F.run(img, m, p)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, img.shape)


if __name__ == "__main__":
    unittest.main()
