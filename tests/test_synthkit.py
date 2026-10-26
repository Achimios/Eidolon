# -*- coding: utf-8 -*-
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import synthkit as sk  # noqa: E402


class TestFeedbackSat(unittest.TestCase):
    def test_bounded_when_pole_outside_unit_circle(self):
        """g = 1.05 持续 2 秒：环里有 tanh，输出必须有界、没有 inf / NaN"""
        x = np.zeros(2 * sk.SR)
        x[:2000] = sk.osc("saw", 110.0, 2000 / sk.SR)
        y = sk.feedback_delay_sat(x, 0.05, 1.05)
        self.assertTrue(np.all(np.isfinite(y)))
        self.assertLess(float(np.abs(y).max()), 2.5)
        self.assertGreater(float(np.abs(y[-sk.SR // 4:]).max()), 0.05)   # 自激：越过 1 不会自己消失

    def test_decays_inside_unit_circle(self):
        x = np.zeros(2 * sk.SR)
        x[:2000] = 0.5
        y = sk.feedback_delay_sat(x, 0.05, 0.5)
        self.assertLess(float(np.abs(y[-sk.SR // 4:]).max()), 1e-3)


if __name__ == "__main__":
    unittest.main()


class TestFeedbackSatXY(unittest.TestCase):
    def test_bounded_and_round(self):
        """径向饱和：g = 1.05 持续 2 秒有界；自激后的点落在圆盘里，不贴着正方形的角"""
        n = 2 * sk.SR
        xy = np.zeros((n, 2))
        u = np.arange(2000) / sk.SR * 110
        xy[:2000] = np.stack([np.sin(2 * np.pi * u), np.cos(2 * np.pi * u)], 1)
        y = sk.feedback_delay_sat_xy(xy, (0.0071, 0.0113), 1.05, drive=1.0)
        self.assertTrue(np.all(np.isfinite(y)))
        tail = y[-sk.SR // 2:]
        r = np.linalg.norm(tail, axis=1)
        self.assertLessEqual(float(r.max()), 1.0 + 1e-6)          # 圆盘半径 1/drive
        self.assertGreater(float(r.max()), 0.05)                  # 越过 1：自激不消失
        corner = np.mean((np.abs(tail[:, 0]) > 0.8) & (np.abs(tail[:, 1]) > 0.8))
        self.assertLess(float(corner), 0.01)                      # 不出正方形的角
