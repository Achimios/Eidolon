#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""几何符号工具（tools/geo_symbols.py）的自检：缓动、对数螺旋自相似、模块字体、阶梯频谱、倒数真值、字阵配对、筛的真值。"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import geo_symbols as G  # noqa: E402

SENSE = G.S.MV / "感官.json"


class TestEase(unittest.TestCase):
    def test_endpoints_monotone_fast_then_slow(self):
        u = np.linspace(0, 1, 101)
        e = G.ease2(u, 10.0)
        self.assertAlmostEqual(float(e[0]), 0.0, places=12)
        self.assertAlmostEqual(float(e[-1]), 1.0, places=12)
        self.assertTrue(np.all(np.diff(e) > 0))
        self.assertGreater(float(G.ease2(0.3, 10.0)), 0.5)          # 前 3 帧走掉一半以上（Modular 实测）
        v_end = (G.ease2(1.0, 10.0) - G.ease2(0.99, 10.0)) / 0.01
        self.assertGreater(float(v_end), 0.0)                         # 终点不停
        v0 = (G.ease2(0.01, 10.0) - G.ease2(0.0, 10.0)) / 0.01
        self.assertGreater(float(v0), 5 * float(v_end))               # 先快后慢


class TestSpiral(unittest.TestCase):
    def setUp(self):
        self.b = 192.36 + 0.4 * np.arange(16) + 0.01 * np.sin(np.arange(16))   # 不等间距的拍也要自相似
        self.sp = G.Spiral(self.b)

    def test_self_similar_at_birth_and_next_beat(self):
        """必须相等：每个元素出生时、下一拍时的屏幕位置 / 尺度对所有 i 完全相同"""
        p0, h0, _, r0 = self.sp.xform(0, self.b[0])
        for i in range(1, 15):
            p, h, _, r = self.sp.xform(i, self.b[i])
            np.testing.assert_allclose(p, p0, atol=1e-9); self.assertAlmostEqual(h, h0, places=9); self.assertAlmostEqual(r, 0.0)
            p1, h1, _, r1 = self.sp.xform(i, self.b[i + 1] - 1e-9)
            q1, g1, _, _ = self.sp.xform(0, self.b[1] - 1e-9)
            np.testing.assert_allclose(p1, q1, atol=1e-6); self.assertAlmostEqual(h1, g1, places=5)

    def test_layers_never_ahead_and_moving_before_first_beat(self):
        for t in np.linspace(self.b[0], self.b[-1], 200):
            for i in range(16):
                if t >= self.b[i]:
                    self.assertLessEqual(self.sp.rel(i, t), 1e-12)
        self.assertNotAlmostEqual(self.sp.u(self.b[0] - 0.1), self.sp.u(self.b[0] - 0.05))   # 剪进来时已经在动


class TestFont(unittest.TestCase):
    def raster(self, ch, wt=0.62, trim=True):
        cv = G.Canvas(60, 90, ss=2, bg=(0, 0, 0))
        ss, wid = G.text_strokes(ch)
        for st, xo in ss:
            G.draw_stroke(cv, st, xo, (8, 10, 14.0, 0.0), 1.0, (255, 255, 255), wt=wt, trim=trim)
        return np.asarray(cv.result().convert("L"), float) / 255

    def test_strokes_inside_cell_box(self):
        for ch, strokes in G.FONT_GLYPHS.items():
            for st in strokes:
                x0, y0, x1, y1 = G.stroke_bbox(st)
                self.assertGreaterEqual(x0, -1e-9, ch); self.assertLessEqual(x1, G.GLYPH_W + 1e-9, ch)
                self.assertGreaterEqual(y0, -1e-9, ch); self.assertLessEqual(y1, G.GLYPH_H + 1e-9, ch)

    def test_digits_distinct(self):
        R = {d: self.raster(d) for d in "0123456789"}
        for a in R:
            for b in R:
                if a < b:
                    self.assertGreater(np.abs(R[a] - R[b]).mean(), 0.02, (a, b))

    def test_trim_keeps_outer_edge(self):
        """wt < 1 + trim：竖笔顶端不再伸出横笔外沿（「5」左上角 v1–v4 有个小凸起）"""
        img = self.raster("5")
        img_untrim = self.raster("5", trim=False)
        top = lambda im: int(np.argmax(im.max(1) > 0.5))
        self.assertGreater(top(img), top(img_untrim))


class TestStepSpectrumLogic(unittest.TestCase):
    def test_quantize_and_stepped_release(self):
        sp = G.StepSpectrum.__new__(G.StepSpectrum)
        sp.nb, sp.steps, sp.db_lo, sp.db_hi = 1, 8, -62.0, -12.0
        sp.level = np.zeros(1, int); sp.hold = np.zeros(1, int)
        self.assertEqual(int(sp.quantize(np.array([-100.0]))[0]), 0)
        self.assertEqual(int(sp.quantize(np.array([0.0]))[0]), 8)
        seq = [-12.0] + [-100.0] * 20
        it = iter(seq)
        sp.bands_db = lambda t: np.array([next(it)])
        lv = [int(sp.step(0)[0][0]) for _ in seq]
        self.assertEqual(lv[0], 8)
        self.assertTrue(all(0 <= a - b <= 1 for a, b in zip(lv, lv[1:])))   # 一次最多掉一级
        self.assertEqual(lv[16], 0)                                           # 每 2 帧一级 → 16 帧掉完


@unittest.skipUnless(SENSE.exists(), "需要 感官.json")
class TestCountdownTruth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sn = json.loads(SENSE.read_text("utf-8"))
        cls.cd = G.Countdown(cls.sn)
        cls.cd.spec = type("S", (), {"step": lambda self, t: (np.zeros(24, int), np.zeros(24)), "steps": 8, "db_lo": -62, "db_hi": -12})()

    def test_release_is_a_beat_and_16_births(self):
        beats = np.array(self.sn["events"]["beats"])
        self.assertAlmostEqual(self.cd.release, float(beats[np.argmin(np.abs(beats - 198.76))]))
        self.assertEqual(len(self.cd.beats), 17)
        self.assertTrue(np.all(np.diff(self.cd.beats) > 0.3))

    def test_number_equals_beats_remaining(self):
        """必须相等：第 i 个出生拍之后显示的数 = 离冲顶还剩的拍数"""
        for i in range(16):
            cv = G.Canvas(64, 36, 1)
            info = self.cd.frame(self.cd.beats[i] + 0.01, cv)
            remaining = int(np.sum((np.array(self.sn["events"]["beats"]) >= self.cd.beats[i] - 1e-6)
                                   & (np.array(self.sn["events"]["beats"]) < self.cd.release - 1e-6)))
            self.assertEqual(info["beats_to_release"], remaining)
            self.assertEqual(info["beats_to_release"], 16 - i)


@unittest.skipUnless(SENSE.exists(), "需要 感官.json")
class TestCountdownV6(unittest.TestCase):
    def test_v6_truth_and_legible_in_3_frames(self):
        sn = json.loads(SENSE.read_text("utf-8"))
        cd = G.Countdown(sn, scale=0.5, v7=True)
        cd.spec = G.StepSpectrum(cd.beats[0] - 0.1, cd.release)
        for i in range(16):
            info = cd.frame(cd.beats[i] + 0.01, G.Canvas(cd.w, cd.h, 1))
            self.assertEqual(info["beats_to_release"], 16 - i)
            if i < 15:                                      # 👤主会话：新数字前 2–3 帧就要可辨认
                progs = [pr for pr, _ in cd.digit_alpha(i, cd.beats[i] + 3.5 / G.FPS)]
                self.assertGreater(min(progs), 0.7, i)
                self.assertGreater(float(np.mean(progs)), 0.85, i)


@unittest.skipUnless(SENSE.exists(), "需要 感官.json")
class TestMatrixTruth(unittest.TestCase):
    def test_voices_and_matching(self):
        sn = json.loads(SENSE.read_text("utf-8"))
        m = G.Matrix(sn, 128.36, 141.14, scale=0.25)
        on = np.array(sn["events"]["onsets"])
        self.assertEqual(m.N, int(np.sum((on >= 77.11) & (on < 115.51))))
        self.assertEqual(len(m.match), int(np.sum((on >= 128.36 - 1e-6) & (on < 141.14))))
        js = [j for _, j, _ in m.match]
        self.assertEqual(len(js), len(set(js)))                      # 一格只被认出一次
        info = m.frame(141.13, G.Canvas(m.w, m.h, 1))
        arrived = sum(1 for o, _, _ in m.match if 141.13 - o - 8.0 / G.FPS >= m.fly)
        self.assertEqual(info["assimilated"], arrived)


class TestSieveTruth(unittest.TestCase):
    def test_sieve(self):
        sv = G.Sieve(scale=0.25, v=4)
        primes = G.primes_upto(169)
        self.assertEqual(len(primes), 39)
        self.assertEqual(sv.P, [2, 3, 5, 7, 11, 13])
        struck = {n for _, n, _ in sv.events}
        self.assertEqual(struck, set(range(2, 170)) - set(primes.tolist()))     # 划掉的 = 合数，一个不多一个不少
        for n in range(4, 170):
            k = sum(1 for _, m, _ in sv.events if m == n)
            self.assertEqual(k, sum(1 for p in sv.P if n % p == 0 and p * p <= n), n)
        info = sv.frame(sv.T - 0.01, G.Canvas(sv.w, sv.h, 1))
        self.assertEqual(info["remaining"], 39)
        self.assertEqual(info["struck"] + info["remaining"] + 1, 169)
        self.assertEqual(info["primes_lit"], 39)
        sv5 = G.Sieve(scale=0.25, v=5)
        i5 = sv5.frame(sv5.T - 0.01, G.Canvas(sv5.w, sv5.h, 1))
        self.assertEqual(i5["proven"], 39)
        self.assertEqual(sum(i5["new_by_p"].values()), i5["struck"])            # 每个合数恰好被「第一次」划一次
        self.assertEqual(i5["new_by_p"][2], 83)                                   # 4..168 的偶数
        mid = sv5.frame(sv5.t_pass[3] + 0.1, G.Canvas(sv5.w, sv5.h, 1))          # p = 7 这一遍：2/3/5 已完 → 49 以下的幸存者都已证明
        self.assertEqual(mid["proven"], int(np.sum(G.primes_upto(48) < 49)))
        x = sv.audio()
        self.assertLessEqual(float(np.abs(x).max()), 0.8 + 1e-9)


class TestSieveLayoutV6(unittest.TestCase):
    def test_front_numeral_clears_previous_and_frame(self):
        """v6：每个大数字落定（下一拍前）时，和退到 −1 层的上一个数字外接框不相交；也不出画（v5 的 13 / 39 叠在一起）"""
        sv = G.Sieve(scale=1.0, v=6)
        def box(i, t):
            pos, hpx, _, r = sv.sp.xform(i, t)
            ss_, wid = G.text_strokes(sv.big[i][1])
            unit = min(hpx / G.GLYPH_H, 0.24 * sv.w * (sv.sp.R ** r) / wid)
            return pos[0] - wid / 2 * unit, pos[1] - G.GLYPH_H / 2 * unit, pos[0] + wid / 2 * unit, pos[1] + G.GLYPH_H / 2 * unit
        for i in range(1, len(sv.big)):
            t = sv.big[i][0] + 0.45
            a, b = box(i, t), box(i - 1, t)
            inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
            self.assertLess(inter, 0.06 * (a[2] - a[0]) * (a[3] - a[1]), i)
            self.assertGreater(a[0], 0.03 * sv.w); self.assertLess(a[2], sv.gx0)


class TestPalette(unittest.TestCase):
    def test_switch(self):
        try:
            G.set_palette("bwr")
            self.assertEqual(G.BG, (0, 0, 0))
            self.assertEqual(G.Canvas(4, 4, 1).im.getpixel((0, 0)), (0, 0, 0))
            G.set_palette("acid")
            self.assertEqual(G.Canvas(4, 4, 1).im.getpixel((0, 0)), G.PALETTES["acid"]["BG"])
        finally:
            G.set_palette("cold")


if __name__ == "__main__":
    unittest.main()
