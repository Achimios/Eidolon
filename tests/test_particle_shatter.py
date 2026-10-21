#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""粒子碎裂 v5（tools/particle_shatter.py）自检：找「必须相等」的量。不需要 torch / 源视频。"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import particle_shatter as P  # noqa: E402


def bust(H=180, W=320):
    """合成胸像：头（椭圆）+ 肩（下半宽带），亮度带扫描线"""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    head = ((xx - 200) / 40) ** 2 + ((yy - 60) / 50) ** 2 <= 1
    shoulders = (yy > 120) & (np.abs(xx - 200) < 110)
    m = (head | shoulders).astype(np.float32)
    lum = m * (0.5 + 0.4 * (np.sin(yy * 1.3) > 0))
    return np.repeat(lum[..., None], 3, 2), m


def field(seed=1, N=20000, **kw):
    img, m = bust()
    return P.Field(img, m, np.random.default_rng(seed), N=N, **kw)


class TestField(unittest.TestCase):
    def test_at_impact_everything_in_place(self):
        """τ = 0 与 τ < 0：每个点都在采样位置（接缝 = 上一镜末帧）"""
        f = field()
        for tau in (-1.0, 0.0):
            x, y, a = f.pos(tau)
            np.testing.assert_array_equal(x, f.x0); np.testing.assert_array_equal(y, f.y0)
            np.testing.assert_array_equal(a, 1.0)

    def test_one_direction(self):
        """漂移开始前（u ≤ TS），已释放的点的横向位移符号全部 = dir（单方向）"""
        for d in (-1, 1):
            f = field(direction=d)
            tau = P.TS * 0.9
            x, _, _ = f.pos(tau)
            rel = f.tr == 0
            self.assertTrue(np.all((x - f.x0)[rel] * d >= 0))
            self.assertGreater(np.mean(np.abs(x - f.x0)[rel]), 1.0)

    def test_release_is_continuous(self):
        """释放时刻前后位置连续（闭式解在 u = 0 处位移为 0，速度有限）"""
        f = field()
        face = f.tr > 0
        self.assertTrue(face.any())
        tr = f.tr[face]
        eps = 1e-4
        # 在每个点自己的释放时刻取样：逐点调用太慢 → 直接用公式里的 u
        u = np.full(f.N, eps)
        e1 = 1 - np.exp(-u / P.T1)
        dx = f.vx * P.T1 * e1
        self.assertLess(np.max(np.abs(dx[face])), np.max(f.vx) * eps * 1.01)
        self.assertTrue(np.all(tr >= 0.30 - 1e-9) and np.all(tr <= 1.25 + 1e-9))

    def test_face_goes_last(self):
        """脸以外 τ_r = 0；脸的 τ_r 与脸权重正相关（中心最后走）"""
        f = field(N=40000)
        self.assertTrue(np.all(f.tr[~f.is_face] == 0))
        r = np.corrcoef(f.face[f.is_face], f.tr[f.is_face])[0, 1]
        self.assertGreater(r, 0.5)
        hi = f.is_face & (f.face > 0.8); lo = f.is_face & (f.face < 0.4)
        self.assertGreater(f.tr[hi].mean(), f.tr[lo].mean())

    def test_slabs_move_as_rows(self):
        """条块 = 同一行带的点同速（横向位移只差漂移项）；尘埃各自不同"""
        f = field(N=40000)
        tau = P.TS * 0.9                                   # 漂移前：位移 = v·T1·e1
        x, y, _ = f.pos(tau)
        sl = f.is_slab & (f.tr == 0) & (f.face < 1e-4)        # 受力权重 hitw ≈ 1 的点（脸附近按 v4 被 hitw 减速）
        b = f.band[sl]; dx = (x - f.x0)[sl]
        for bb in np.unique(b)[:30]:
            v = dx[b == bb]
            self.assertLess(np.ptp(v), 1e-9 + 1e-3 * np.abs(v).max())

    def test_deterministic(self):
        """同种子 → 同一个场；τ 是唯一的输入（慢放 / 回看取的是同一批点）"""
        a, b = field(seed=5), field(seed=5)
        for tau in (0.1, 0.77, 2.3):
            np.testing.assert_array_equal(a.pos(tau)[0], b.pos(tau)[0])

    def test_closed_form_matches_v4_integration(self):
        """闭式急刹 vs v4 的逐帧 ×0.80 积分：总抛距严格相等；8 帧时误差 < 一帧位移的 20%"""
        v0 = 1000.0
        x = 0.0; v = v0
        for _ in range(8):
            x += v / P.FPS; v *= 0.80
        tau = 8 / P.FPS
        xc = v0 * P.T1 * (1 - np.exp(-tau / P.T1))
        self.assertLess(abs(x - xc), 0.20 * v0 / P.FPS)
        self.assertAlmostEqual(v0 * P.T1, v0 / P.FPS / (1 - 0.80))


class TestRender(unittest.TestCase):
    def test_splat_conserves_mass(self):
        """双线性 splat：图上的总和 = 画内点的权重和"""
        rng = np.random.default_rng(0)
        x = rng.uniform(-20, 340, 5000); y = rng.uniform(-20, 200, 5000); w = rng.random(5000)
        img, tot = P.splat_bilinear(x, y, w, 320, 180)
        self.assertAlmostEqual(img.sum(), tot, places=6)
        ok = (x >= 0) & (x < 319) & (y >= 0) & (y < 179)
        self.assertAlmostEqual(tot, w[ok].sum(), places=6)

    def test_camera_identity(self):
        c = P.Camera(320, 180)
        X, Y, s = c.project(np.array([3.5, 100.0]), np.array([7.0, 50.0]))
        np.testing.assert_allclose(X, [3.5, 100.0]); np.testing.assert_allclose(Y, [7.0, 50.0])

    def test_25d_is_2d_at_release(self):
        """2.5D：刚释放（u = 0）的点都在 z = 1 平面 → 和同参数的 2D 投影一样（接缝处无跳变）"""
        f = field()
        c3 = P.Camera(320, 180, 150, 90, zoom=0.8, par=1.0, zc=0.0)
        c2 = P.Camera(320, 180, 150, 90, zoom=0.8)
        X3, Y3, _ = c3.project(f.x0, f.y0, f, np.zeros(f.N))
        X2, Y2, _ = c2.project(f.x0, f.y0)
        np.testing.assert_allclose(X3, X2); np.testing.assert_allclose(Y3, Y2)


class TestTimeline(unittest.TestCase):
    def test_src_index(self):
        """上一镜末帧 = 冲击帧；起点 = 第 0 帧；单调"""
        n = 121
        self.assertEqual(P.src_index(51.52, 45.12, 51.52, n), n - 1)
        self.assertEqual(P.src_index(60.0, 45.12, 51.52, n), n - 1)
        self.assertEqual(P.src_index(45.12, 45.12, 51.52, n), 0)
        ts = np.arange(45.12, 51.52, 1 / 24)
        ii = [P.src_index(t, 45.12, 51.52, n) for t in ts]
        self.assertTrue(np.all(np.diff(ii) >= 0))

    def test_shots_tile_the_window(self):
        """镜头首尾相接、覆盖 51.52 → 57.92、切点在 0.4 s 拍格上、余波每镜 0.4–1.6 s、下拍 54.72 / 56.32 是切点"""
        sh = P.SHOTS_V5
        self.assertAlmostEqual(sh[0]["t0"], 51.52); self.assertAlmostEqual(sh[-1]["t1"], 57.92)
        for a, b in zip(sh, sh[1:]):
            self.assertAlmostEqual(a["t1"], b["t0"])
        cuts = [s["t0"] for s in sh] + [sh[-1]["t1"]]
        for c in cuts:
            q = (c - 51.52) / 0.4
            self.assertAlmostEqual(q, round(q), places=6)
        for s in sh[1:]:
            self.assertTrue(0.4 - 1e-9 <= s["t1"] - s["t0"] <= 1.6 + 1e-9)
        for db in (53.12, 54.72, 56.32):
            self.assertTrue(any(abs(c - db) < 1e-9 for c in cuts))

    def test_src_index_real(self):
        """v5c 原速 + 定格：第 i 帧 = 源第 i 帧，放完停在末帧（= face_glitch v4 的 live 121 + held 33）"""
        n = 121
        ii = [P.src_index(45.12 + i / 24, 45.12, 51.52, n, "real") for i in range(154)]
        self.assertEqual(ii[:121], list(range(121)))
        self.assertTrue(all(x == n - 1 for x in ii[121:]))
        self.assertEqual(P.src_index(51.52, 45.12, 51.52, n, "real"), n - 1)

    def test_shots_v5c_downbeats_no_overlap(self):
        """v5c：首尾相接不重叠，切点 = 下拍 53.12 / 54.72 / 56.32，收在 57.94；脸只在 冲击 + 余波_4 里"""
        sh = P.SHOTS_V5C
        self.assertAlmostEqual(sh[0]["t0"], 51.52); self.assertAlmostEqual(sh[-1]["t1"], 57.94)
        for a, b in zip(sh, sh[1:]):
            self.assertAlmostEqual(a["t1"], b["t0"])
        cuts = [s["t0"] for s in sh]
        for db in (53.12, 54.72, 56.32):
            self.assertTrue(any(abs(c - db) < 1e-9 for c in cuts))
        self.assertEqual([s["name"] for s in sh if not s.get("hide_held")], ["冲击", "余波_4"])

    def test_streak_echo_drags_left_and_decays(self):
        img = np.zeros((4, 50, 3)); img[:, 40] = 1.0
        e = P.streak_echo(img, 10, -1, np.random.default_rng(0))
        self.assertTrue((e[:, 41:] == 0).all())                         # 不往右拖
        self.assertTrue((e[:, 30] < e[:, 39]).all() and (e[:, 30] > 0).all())

    def test_hblur_keeps_rows(self):
        img = np.zeros((5, 30)); img[2, 15] = 1.0
        b = P.hblur(img, 3)
        self.assertAlmostEqual(b[2].sum(), 1.0); self.assertEqual(b[[0, 1, 3, 4]].sum(), 0)

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.mp4"; p.write_bytes(b"0")
            with self.assertRaises(SystemExit):
                P.no_overwrite(p)
            P.no_overwrite(Path(d) / "y.mp4")


if __name__ == "__main__":
    unittest.main()
