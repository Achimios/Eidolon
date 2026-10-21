#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""开头工具（tools/opening.py）的纯数学自检：重分配 STFT、瀑布图分 bin、弦折叠 = 延迟嵌入。"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import opening as O  # noqa: E402


class TestReassigned(unittest.TestCase):
    def test_sine_frequency_and_power(self):
        """必须相等：正弦 441.3 Hz、振幅 0.5 → 重分配频率 ±0.5 Hz，主瓣功率和 = A² = 0.25"""
        sr, f0, A = 44100, 441.3, 0.5
        x = A * np.sin(2 * np.pi * f0 * np.arange(sr) / sr)
        P = O.reassigned_power(x, sr, [sr // 2], n=4096, df=0.5, fmax=3000)[0]
        fb = (np.argmax(P) + 0.5) * 0.5
        self.assertLess(abs(fb - f0), 0.5)
        i = np.argmax(P)
        self.assertAlmostEqual(P[i - 20:i + 21].sum(), A * A, delta=0.01)

    def test_two_tones_separate(self):
        sr = 44100
        t = np.arange(sr) / sr
        x = 0.3 * np.sin(2 * np.pi * 41 * t) + 0.3 * np.sin(2 * np.pi * 82 * t)
        P = O.reassigned_power(x, sr, [sr // 2], n=4096, df=0.5, fmax=200)[0]
        for f in (41, 82):
            self.assertGreater(P[int((f - 2) / 0.5):int((f + 2) / 0.5)].sum(), 0.08)


class TestWaterfallBinning(unittest.TestCase):
    def test_energy_conserved(self):
        """必须相等：任意（非整数）列宽下，列功率之和 = 范围内细 bin 功率之和"""
        rng = np.random.default_rng(0)
        wf = O.Waterfall("song", 0.0, 0.5, 6000)
        blk = rng.random((3, 6000))
        cols = wf.bin_cols(blk, 100.0, 1300.0, 1037)
        self.assertTrue(np.allclose(cols.sum(1), blk[:, 200:2600].sum(1)))

    def test_flat_noise_no_stripes(self):
        wf = O.Waterfall("song", 0.0, 0.5, 6000)
        cols = wf.bin_cols(np.ones((1, 6000)), 13.0, 1213.0, 1043)[0]
        self.assertLess(cols.std() / cols.mean(), 1e-9)


class TestStringFold(unittest.TestCase):
    def test_fold_is_delay_embedding(self):
        """必须相等：u = 1 时节点 n 的坐标 = G·(x(s−τ), −x(s), x(s−2τ))，s = t − |pos−p0|/c"""
        st = O.String.__new__(O.String)
        st.sr = 11025
        st.x = np.sin(2 * np.pi * 41 * np.arange(st.sr * 3) / st.sr)
        st.tau = 0.006
        s = np.array([1.0, 1.2, 1.7])
        a, b, c = st.xs(s), st.xs(s - st.tau), st.xs(s - 2 * st.tau)
        self.assertTrue(np.allclose(a, np.sin(2 * np.pi * 41 * s), atol=2e-3))
        self.assertTrue(np.allclose(b, np.sin(2 * np.pi * 41 * (s - 0.006)), atol=2e-3))
        self.assertTrue(np.allclose(c, np.sin(2 * np.pi * 41 * (s - 0.012)), atol=2e-3))


class TestCam(unittest.TestCase):
    def test_never_stops(self):
        cam = O.Cam(lambda t: 0.05, [1.0, 2.0], 0.3, 0.5)
        sp = [cam.speed(i / 24) for i in range(120)]
        self.assertGreater(min(sp), 0.049)
        self.assertGreater(cam.speed(1.0 + 3 / 24), 0.25)


if __name__ == "__main__":
    unittest.main()
