#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""粒子海（tools/particle_sea.py）自检：找「必须相等」的量。CPU、小规模。
1. J0（没有 scipy，自己写的多项式）对表值。
2. FFT 海：4σ(h) == Hs（多个种子平均，±8 %）；场是实的、周期为 L。
3. 色散：单个模 == 0.7·cos(kx + √(gk)·t)，逐格点。
4. Cauchy–Poisson 环形波：t = 0 精确等于高斯初值；t < 0 为 0（因果）；t > 0 与独立方法（2D FFT 直接演化）的径向剖面一致。
5. 临界阻尼：e(0) = e0、无初速时 |e| 单调减、ωu = 5.8 时剩 < 2 %。
6. 质量守恒：海 + 飞行中 + 已到达 == 总点数，逐时刻；越强的起音放的点越多；先放的点更低（从海里往上长）。
7. 红点：到达时刻对齐最强起音之后的第一个小节首拍（剩余距离 2 %）。
8. 溅射能量守恒；深度缓冲留最近的。
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import particle_sea as P  # noqa: E402


class TestJ0(unittest.TestCase):
    def test_table(self):
        x = np.array([0.0, 1.0, 2.0, 2.404825557695773, 3.0, 5.0, 10.0, 20.0])
        ref = np.array([1.0, 0.7651976866, 0.2238907791, 0.0, -0.2600519549, -0.1775967713, -0.2459357645, 0.1670246643])
        np.testing.assert_allclose(P.j0(x), ref, atol=2e-7)


class TestOcean(unittest.TestCase):
    def test_hs(self):
        r = [4 * P.Ocean(Hs=3.0, Tp=6.0, seed=s, L=256.0, n=128).fields(0.0)["h"].std() for s in range(8)]
        self.assertAlmostEqual(np.mean(r) / 3.0, 1.0, delta=0.08)

    def test_real_periodic(self):
        o = P.Ocean(Hs=3.0, Tp=6.0, seed=1, L=256.0, n=64)
        F = o.fields(1.3)
        Hk = o.spectrum_t(1.3)
        self.assertLess(np.abs(np.imag(np.fft.ifft2(Hk))).max(), 1e-9 * max(1.0, np.abs(F["h"]).max()))
        x = np.array([3.7, 100.2]); z = np.array([-5.0, 77.7])
        a = P.sample_periodic(F["h"], o.L, x, z); b = P.sample_periodic(F["h"], o.L, x + o.L, z - 2 * o.L)
        np.testing.assert_allclose(a, b, atol=1e-9)

    def test_dispersion(self):
        o = P.Ocean(Hs=3.0, Tp=6.0, seed=2, L=256.0, n=64)
        keep = np.zeros_like(o.h0); ix = 5                     # 一个沿 +x 的模：k = 2π·5/L
        keep[0, ix] = 0.7
        o.h0 = keep; o.h0m = np.zeros_like(keep)
        k = 2 * np.pi * ix / o.L; w = np.sqrt(P.G * k)
        x = np.arange(o.n) * o.L / o.n
        for t in (0.0, 1.7, 4.2):
            np.testing.assert_allclose(o.fields(t)["h"][0], 0.7 * np.cos(k * x + w * t), atol=1e-9)


class TestRing(unittest.TestCase):
    def test_initial_and_causal(self):
        r = np.linspace(0, 30, 200)
        eta = P.cauchy_poisson(r, 0.0, 2.0, 5.0)
        np.testing.assert_allclose(eta, 2.0 * np.exp(-(r / 5.0) ** 2), atol=2e-3)
        self.assertEqual(np.abs(P.cauchy_poisson(r, -0.1, 2.0, 5.0)).max(), 0.0)

    def test_vs_fft(self):
        """独立方法对照：同一个高斯初值在大方格上 2D FFT × cos(√(gk) t) 逆变换，径向剖面应与 J0 积分一致"""
        A, b, t = 2.0, 5.0, 2.5
        n, L = 512, 512.0
        x = (np.arange(n) - n // 2) * L / n
        X, Z = np.meshgrid(x, x)
        eta0 = A * np.exp(-(X ** 2 + Z ** 2) / b ** 2)
        k1 = 2 * np.pi * np.fft.fftfreq(n, d=L / n); KX, KZ = np.meshgrid(k1, k1)
        eta_t = np.real(np.fft.ifft2(np.fft.fft2(np.fft.ifftshift(eta0)) * np.cos(np.sqrt(P.G * np.hypot(KX, KZ)) * t)))
        eta_t = np.fft.fftshift(eta_t)
        r = x[n // 2:n // 2 + 60]
        ref = eta_t[n // 2, n // 2:n // 2 + 60]
        got = P.cauchy_poisson(r, t, A, b, nk=1024)
        self.assertLess(np.abs(got - ref).max(), 0.01 * A)


    def test_impulse_is_time_derivative(self):
        """冲量版 == tau_i · ∂(隆起版)/∂t（中心差分）；t = 0 时处处为 0（水面先平）"""
        r = np.linspace(0, 60, 120); t, dt, ti = 1.3, 1e-4, 0.6
        d = (P.cauchy_poisson(r, t + dt, 2.0, 5.0) - P.cauchy_poisson(r, t - dt, 2.0, 5.0)) / (2 * dt)
        np.testing.assert_allclose(P.cauchy_poisson(r, t, 2.0, 5.0, kind="impulse", tau_i=ti), ti * d, atol=1e-6)
        self.assertLess(np.abs(P.cauchy_poisson(r, 0.0, 2.0, 5.0, kind="impulse", tau_i=ti)).max(), 1e-12)
        self.assertLess(P.cauchy_poisson(np.array([0.0]), 0.2, 2.0, 5.0, kind="impulse", tau_i=ti)[0], 0)   # 先砸出坑


class TestSpring(unittest.TestCase):
    def test_closed_form(self):
        e0 = np.array([[10.0, -4.0, 3.0]]); v0 = np.zeros_like(e0)
        u = np.linspace(0, 3, 200)
        e = np.array([P.damped_spring(e0, v0, 6.0, uu)[0] for uu in u])
        np.testing.assert_allclose(e[0], e0[0])
        n = np.linalg.norm(e, axis=1)
        self.assertTrue(np.all(np.diff(n) <= 1e-12))
        e_arr = P.damped_spring(e0, v0, 6.0, P.ARRIVE / 6.0)[0]
        self.assertLessEqual(np.linalg.norm(e_arr), 0.02 * n[0] + 1e-12)


def small_scene(t0=128.36, t1=134.78):
    p = dict(P.BASE)
    p.update(N_sea=4000, n_surf=40000, n_depth=4000, grid=64, hud=0, K=2)
    return P.Sea(p, t0, t1, P.load_sense(), W=160, H=90, log=lambda *a: None), p


class TestScene(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sea, cls.p = small_scene()

    def test_mass_conservation(self):
        s = self.sea
        for t in np.linspace(s.t0, s.t1, 9):
            a, b, c = s.counts_at(t)
            self.assertEqual(a + b + c, s.N_total)
            L, info = s.frame(t)
            self.assertEqual(sum(info["counts"]), s.N_total)
            for q in P.PALETTES:
                img = s.compose(L, info, q)
                self.assertTrue(np.isfinite(img).all())
                self.assertEqual(img.shape, (s.H, s.W, 3))

    def test_batches_follow_threat(self):
        s = self.sea
        ev = s.sched["events"]
        self.assertGreater(len(ev), 3)
        big = max(ev, key=lambda e: e[1]); small = min(ev, key=lambda e: e[1])
        self.assertGreater(big[2], small[2])
        # 释放的总数 = end_frac − pre_absorbed
        nb = len(s.rt)
        n_rel = int((np.isfinite(s.rt)).sum())
        self.assertEqual(n_rel, s.sched["n_end"] - s.sched["n_pre"])

    def test_grows_upward(self):
        s = self.sea
        fin = np.isfinite(s.rt)
        o = np.argsort(s.rt[fin]); h = s.BL[fin][o, 1]
        q = len(h) // 4
        self.assertLess(h[:q].mean(), h[-q:].mean())

    def test_red_arrives_on_downbeat(self):
        s = self.sea
        self.assertGreaterEqual(s.red, 0)
        sense = P.load_sense()
        dbs = [d for d in sense["events"]["downbeats"] if d > s.red_hit + 0.15]
        self.assertAlmostEqual(s.red_arrive, dbs[0], places=6)


class TestRender(unittest.TestCase):
    def test_splat_conserves(self):
        r = np.random.default_rng(0)
        x = r.uniform(-5, 40, 500); y = r.uniform(-5, 30, 500); w = r.random(500)
        img, tot = P.splat_bilinear(x, y, w, 32, 24)
        self.assertAlmostEqual(img.sum(), tot, places=9)

    def test_zbuffer_nearest(self):
        X = np.array([5.0, 5.2, 5.1]); Y = np.array([5.0, 5.1, 5.3]); Z = np.array([9.0, 3.0, 7.0])
        zb = P.zbuffer(X, Y, Z, 16, 16, 4)
        self.assertEqual(zb[1, 1], 3.0)


if __name__ == "__main__":
    unittest.main()
