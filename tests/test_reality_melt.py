#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""现实瓦解（tools/reality_melt.py）的数值自检：找「必须相等」的量。"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import reality_melt as R  # noqa: E402


def rand_img(H=64, W=80, seed=0):
    return np.random.default_rng(seed).random((H, W, 3)).astype(np.float32)


class TestStretch(unittest.TestCase):
    def test_zero_length_identity(self):
        """L = 0 ⇒ 输出逐像素 == 输入"""
        img = rand_img()
        W = img.shape[1]
        out, a = R.stretch(img, np.full(W, 10.0), np.zeros(W), np.full(W, 8.0))
        np.testing.assert_array_equal(out, img)
        self.assertEqual(float(a.max()), 0.0)

    def test_ramp_slope(self):
        """竖向线性渐变的源 → 条内仍是线性渐变，斜率 = 1/S（采样 s = b + (y − b)/S，线性插值精确）"""
        H, W = 120, 6
        ramp = (np.arange(H, dtype=np.float32) / H)[:, None, None] * np.ones((1, W, 3), np.float32)
        b, L, S_ = np.full(W, 20.0), np.full(W, 60.0), np.array([1, 2, 4, 8, 10, 30], float)
        out, a = R.stretch(ramp, b, L, S_)
        for x in range(W):
            y = np.arange(21, 79)
            np.testing.assert_allclose(out[y, x, 0], (20 + (y - 20) / S_[x]) / H, atol=1e-5)
            np.testing.assert_array_equal(out[90:, x], ramp[90:, x])        # 条下方不动
            np.testing.assert_array_equal(out[:19, x], ramp[:19, x])        # 断点上方不动

    def test_tip_antialias(self):
        """尖端：L 的小数部分 = 最后一个像素的覆盖度"""
        H, W = 50, 1
        img = np.zeros((H, W, 3), np.float32)
        img[10] = 1.0
        out, a = R.stretch(img, np.array([10.0]), np.array([5.25]), np.array([100.0]))
        self.assertAlmostEqual(float(a[15, 0]), 0.25, places=5)
        self.assertAlmostEqual(float(a[14, 0]), 1.0, places=5)


class TestSort(unittest.TestCase):
    def test_outside_untouched_multiset_kept(self):
        """不可排序像素逐像素不变；每段像素多重集不变；段内亮度单调不增（亮的在上）"""
        img = rand_img(40, 7, seed=3)
        rng = np.random.default_rng(5)
        mask = rng.random((40, 7)) < 0.7
        out = R.sort_runs(img, mask)
        np.testing.assert_array_equal(out[~mask], img[~mask])
        L = R.lum(out)
        for x in range(7):
            y = 0
            while y < 40:
                if not mask[y, x]:
                    y += 1
                    continue
                e = y
                while e < 40 and mask[e, x]:
                    e += 1
                seg_in = sorted(map(tuple, img[y:e, x].tolist()))
                seg_out = sorted(map(tuple, out[y:e, x].tolist()))
                self.assertEqual(seg_in, seg_out)
                self.assertTrue(np.all(np.diff(L[y:e, x]) <= 1e-6))
                y = e

    def test_order_statistics_lipschitz(self):
        """⭐闪烁的根据：段固定时，排序输出对输入 1-Lipschitz（输入每像素变 ≤ ε ⇒ 排序后每个位置变 ≤ ε）。
        用灰图（亮度 = 值）检验。"""
        rng = np.random.default_rng(1)
        g = rng.random((60, 9)).astype(np.float32)
        img = np.repeat(g[..., None], 3, -1)
        eps = 0.01
        img2 = np.clip(img + rng.uniform(-eps, eps, (60, 9, 1)).astype(np.float32), 0, 1)
        mask = np.ones((60, 9), bool)
        d = np.abs(R.sort_runs(img, mask) - R.sort_runs(img2, mask)).max()
        self.assertLessEqual(float(d), eps + 1e-6)


class TestFields(unittest.TestCase):
    def test_field_smooth_in_time(self):
        """低频场：相邻两帧（1/24 s）每列最多变 ~ 0.03，没有跳变；同参数结果确定"""
        F = R.Field1D(400, 150, 7.0, 6.0, seed=3)
        prev = F(0.0)
        worst = 0.0
        for k in range(1, 24 * 15):
            cur = F(k / 24)
            worst = max(worst, float(np.abs(cur - prev).max()))
            prev = cur
        self.assertLess(worst, 0.03)
        np.testing.assert_array_equal(R.Field1D(400, 150, 7.0, 6.0, seed=3)(3.3), F(3.3))

    def test_breakpoint_quantile(self):
        """断点 = 可碎区累积质量的 q 分位：一列只有 [20, 60) 可碎 → q = 0.5 时 b = 40，质量 = 40"""
        H, W = 100, 3
        z = np.zeros((H, W), np.float32)
        z[20:60] = 1
        b, mass = R.breakpoints(z, np.array([0.0, 0.5, 1.0]))
        np.testing.assert_allclose(mass, 40)
        np.testing.assert_allclose(b, [20.0, 40.0, 60.0], atol=1e-6)

    def test_burst_is_local(self):
        """鼓点下坠是局部的：离落点 > 4·burst_w 的列，多出的长度 == 0（不推全局）"""
        m = R.Melter(360, 640, R.Params(), 6.0)
        x0 = m.add_burst(1.0, 1.0, np.ones(m.Wc))
        add, fresh = m.burst_len(1.4)
        far = np.abs(np.arange(m.Wc) - x0) > 4 * m.p.burst_w * m.u
        self.assertGreater(float(add.max()), 10.0)
        self.assertLess(float(add[far].max()), 1e-3)
        self.assertLess((add > 1).mean(), 0.1)


class TestPipeline(unittest.TestCase):
    def test_gate_zero_identity(self):
        """gate = 0 ⇒ 整条管线（排序 + 拉伸 + 合成）逐像素 == 输入"""
        img = rand_img(72, 96, seed=2)
        m = R.Melter(72, 96, R.Params(), 5.0)
        out, _ = m.step(img, None, 1.0, 1.0, gate=0.0)
        np.testing.assert_array_equal(out, img)

    def test_protected_untouched(self):
        """保护遮罩 = 1 的像素逐像素不变（她不碎）"""
        img = rand_img(72, 96, seed=4)
        pm = np.zeros((72, 96), np.float32)
        pm[20:50, 30:60] = 1
        m = R.Melter(72, 96, R.Params(length=0.8, melt0=1.0), 5.0)
        out, inf = m.step(img, pm, 2.0)
        self.assertGreater(float(inf["alpha"].mean()), 0.01)         # 别处确实在碎
        np.testing.assert_array_equal(out[20:50, 30:60], img[20:50, 30:60])

    def test_angle_180_equals_flip(self):
        """方向：angle = 180（向上）== 上下翻转 → 向下处理 → 翻回（画布映射在 90° 倍数上是整数像素）"""
        img = rand_img(48, 64, seed=6)
        p0 = R.Params(angle=0, melt0=1.0, length=0.6)
        p1 = R.Params(angle=180, melt0=1.0, length=0.6)
        a, _ = R.Melter(48, 64, p1, 5.0).step(img, None, 2.0)
        b, _ = R.Melter(48, 64, p0, 5.0).step(img[::-1, ::-1].copy(), None, 2.0)
        np.testing.assert_allclose(a, b[::-1, ::-1], atol=1e-5)

    def test_canvas_roundtrip(self):
        """任意角画布：屏幕 → 画布 → 屏幕，内部像素误差小（双线性两次）"""
        rng = np.random.default_rng(0)
        from scipy.ndimage import gaussian_filter
        img = gaussian_filter(rng.random((60, 80)), 3).astype(np.float32)
        cv = R.Canvas(60, 80, 30.0)
        back = cv.inv(cv.fwd(img))
        self.assertLess(float(np.abs(back - img)[8:-8, 8:-8].max()), 0.01)

    def test_depth_near_never_breaks(self):
        """深度门：far < 1 − reach1 − soft 的近处逐像素不变（近处不碎）；远处确实在碎"""
        img = rand_img(72, 96, seed=8)
        far = np.zeros((72, 96), np.float32)
        far[:, :48] = 1.0                        # 左半远、右半近
        p = R.Params(melt0=1.0, length=0.6, depth_reach0=0.5, depth_reach1=0.5, depth_soft=0.1, sort_amt=0.0, burst_amp=0.0)
        out, inf = R.Melter(72, 96, p, 5.0).step(img, None, 2.0, far=far)
        np.testing.assert_array_equal(out[:, 48:], img[:, 48:])
        self.assertGreater(float(inf["alpha"][:, :48].mean()), 0.01)

    def test_depth_reach_advances(self):
        """前沿从远往近推进：中等深度的区域开场不碎、结尾碎（reach 0.3 → 0.9，far = 0.5）"""
        img = rand_img(72, 96, seed=9)
        far = np.full((72, 96), 0.5, np.float32)
        p = R.Params(melt0=1.0, length=0.6, depth_reach0=0.3, depth_reach1=0.9, depth_soft=0.05, zone_ema=1.0, b_ema=1.0,
                     sort_amt=0.0, burst_amp=0.0)
        m = R.Melter(72, 96, p, 4.0)
        a0 = float(m.step(img, None, 0.0, far=far)[1]["alpha"].mean())
        m2 = R.Melter(72, 96, p, 4.0)
        a1 = float(m2.step(img, None, 4.0, far=far)[1]["alpha"].mean())
        self.assertEqual(a0, 0.0)
        self.assertGreater(a1, 0.01)

    def test_temporal_flicker_lower_than_naive(self):
        """⭐时间相干：静止画面上，相邻帧输出变化 << 每帧换场的对照（naive）"""
        rng = np.random.default_rng(9)
        from scipy.ndimage import gaussian_filter
        g = gaussian_filter(rng.random((90, 120)), 2).astype(np.float32)
        img = np.repeat(((g - g.min()) / (g.max() - g.min()))[..., None], 3, -1)

        def flick(p):
            m = R.Melter(90, 120, p, 4.0)
            outs = [m.step(img, None, k / 24)[0] for k in range(12)]
            return float(np.mean([np.abs(outs[k] - outs[k - 1]).mean() for k in range(1, 12)]))
        good, bad = flick(R.Params(melt0=1.0)), flick(R.Params(melt0=1.0, naive=True))
        self.assertLess(good, 0.2 * bad)


if __name__ == "__main__":
    unittest.main()
