#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""示波器模拟（tools/scope_sim.py）自检：找「必须相等」的量。CPU 跑、小画幅，几秒内完。"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import scope_sim as S  # noqa: E402

SR = 48000


def write_xy(path, fx, fy, ax, ay, dur=0.5, phase=np.pi / 2):
    t = np.arange(int(dur * SR)) / SR
    x = ax * np.sin(2 * np.pi * fx * t + phase)
    y = ay * np.sin(2 * np.pi * fy * t)
    sf.write(str(path), np.stack([x, y], 1).astype(np.float32), SR, subtype="FLOAT")
    return path


def small(**kw):
    base = dict(W=200, H=200, scale=1.0, beam=0.0, step=0.25, persist=1e9, fps=60, lead=0.0)
    base.update(kw)
    return S.Params(**base)


class TestScope(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        cls.circ3 = write_xy(d / "c3.wav", 440, 440, 0.3, 0.3)
        cls.circ6 = write_xy(d / "c6.wav", 440, 440, 0.6, 0.6)
        cls.ell = write_xy(d / "e.wav", 440, 440, 0.8, 0.3)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()   # 系统临时目录里自己刚建的测试文件

    def deposit(self, path, p, t=0.2):
        """3 帧的沉积之和：3/60 s × 440 Hz = 22 整圈（单帧 7.33 圈，多出的那段弧会让象限不均）"""
        sc = S.Scope(S.Audio(path, p.upsample), p, "cpu")
        return sum(sc.deposit(t - k / p.fps).sum(0).numpy() for k in range(3)) / 3

    def test_energy_conserved(self):
        """光束一整帧都在屏内、无衰减 → 本帧沉积能量总和 = 1（能量单位定义：静止一整帧 = 1）"""
        E = self.deposit(self.circ6, small())
        self.assertAlmostEqual(float(E.sum()), 1.0, places=4)

    def test_circle_centered_radius_prop_amplitude(self):
        """440 Hz 圆：亮度重心在画面中心；亮度加权半径 = 幅度 · scale · H/2（∝ 幅度）"""
        p = small()
        yy, xx = np.mgrid[0:p.H, 0:p.W] + 0.5
        radii = []
        for path, A in ((self.circ3, 0.3), (self.circ6, 0.6)):
            E = self.deposit(path, p)
            cx, cy = (E * xx).sum() / E.sum(), (E * yy).sum() / E.sum()
            self.assertAlmostEqual(cx, p.W / 2, delta=0.3)
            self.assertAlmostEqual(cy, p.H / 2, delta=0.3)
            r = (E * np.hypot(xx - p.W / 2, yy - p.H / 2)).sum() / E.sum()
            self.assertAlmostEqual(r, A * p.scale * p.H / 2, delta=0.5)
            radii.append(r)
        self.assertAlmostEqual(radii[1] / radii[0], 2.0, delta=0.02)

    def test_circle_uniform_brightness(self):
        """匀速圆：沿圆周亮度均匀（四个象限能量相等）"""
        E = self.deposit(self.circ6, small())
        h = 100
        q = [E[:h, :h].sum(), E[:h, h:].sum(), E[h:, :h].sum(), E[h:, h:].sum()]
        self.assertLess(np.std(q) / np.mean(q), 0.02)

    def test_ellipse_slow_is_bright(self):
        """椭圆 x = a cos, y = b sin（a > b）：长轴两端速度 = ωb（慢）、短轴两端 = ωa（快）
        → 单位弧长能量之比 = a / b。用两端附近的小窗口（同样的弧长）比较"""
        p = small()
        E = self.deposit(self.ell, p)
        s = p.scale * p.H / 2
        a, b = 0.8 * s, 0.3 * s
        cx, cy = p.W / 2, p.H / 2
        w = 3   # 窗口半宽（px），内含的弧长近似 2w（两端曲率不同，只验大小顺序 + 粗比例）
        major = E[int(cy) - w:int(cy) + w, int(cx + a) - 2:int(cx + a) + 3].sum()
        minor = E[int(cy - b) - 2:int(cy - b) + 3, int(cx) - w:int(cx) + w].sum()
        self.assertGreater(major, minor)
        self.assertGreater(major / minor, 0.5 * (a / b))

    def test_fast_jump_faint_line_not_dot(self):
        """两点间瞬跳（一个采样）：中间是一条连续的暗线（每个像素都有能量），不是两个点"""
        p = small(upsample=1)
        pts = torch.tensor([[50.0, 100.0], [150.0, 100.0]])
        E = S.splat_path(pts, torch.tensor([1.0]), p.H, p.W, 0.25).numpy()
        row = E[99:101].sum(0)[52:148]
        self.assertTrue((row > 0).all())
        self.assertLess(row.std() / row.mean(), 0.05)
        self.assertAlmostEqual(float(E.sum()), 1.0, places=5)

    def test_chunking_does_not_change_samples(self):
        """上采样按全局下标取，分块大小不同 → 逐样本相同（可续渲的前提）"""
        a1 = S.Audio(self.ell, 4, chunk_sec=0.05)
        a2 = S.Audio(self.ell, 4, chunk_sec=0.3)
        ja, jb = 12345, 12345 + 3200
        np.testing.assert_allclose(a1.get(ja, jb), a2.get(ja, jb), atol=1e-6)

    def test_resume_equals_continuous(self):
        """单帧模式（预热后渲第 k 帧）== 连续逐帧渲到第 k 帧（余辉预热收敛）"""
        p = small(persist=0.016, beam=1.0, W=120, H=120)
        a = S.Audio(self.ell, p.upsample)
        sc = S.Scope(a, p, "cpu")
        t0 = 0.1
        buf = sc.warm(t0)
        for i in range(8):
            buf = sc.step(buf, t0 + i / p.fps)
        img_cont = sc.display(buf)
        te = t0 + 7 / p.fps
        img_single = sc.display(sc.step(sc.warm(te), te))
        self.assertLessEqual(int(np.abs(img_cont.astype(int) - img_single).max()), 1)


if __name__ == "__main__":
    unittest.main()
