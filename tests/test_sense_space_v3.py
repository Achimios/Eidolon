#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""感官空间-正面（tools/sense_space_v3.py）自检 —— 👤对 v5 的两条：推拉不许来回；节奏在平移和层上的事件。
- 推拉速度符号恒定（整段只推近）
- 平移起步落在小节首拍 ±2 帧
- 镜头永不静止（最低屏幕速度 > 0），也不超过看得清的上限
- 平移目标有意义：人声起所在的小节，目标 = f0 与审计 ▼ 的中点"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import sense_space_v3 as S  # noqa: E402


@unittest.skipUnless(S.SENSE.exists() and (S.MV / "分轨" / "drums.wav").exists(), "没有 感官.json / 分轨")
class TestSenseSpaceV3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.D = S.Data()
        cls.T = int(round((S.S1 - S.S0) * S.FPS))
        cls.cams, cls.moves, cls.bt = S.camera(cls.D, cls.T)
        cls.pan, cls.push, cls.tot = S.speeds(cls.cams)

    def test_push_monotonic(self):
        dz = np.diff(self.cams[:, 2])
        self.assertTrue(np.all(dz > 0), "cz 必须一直往前（推近），不许来回")
        self.assertTrue(np.all(self.push[1:] > 0))

    def test_pan_starts_on_downbeats(self):
        starts, _ = S.pan_starts(self.pan)
        db = np.array([int(round((d - S.S0) * S.FPS)) for d in self.D.downbeats if S.S0 < d < S.S1])
        self.assertGreaterEqual(len(starts), 5)
        for f in starts:
            self.assertLessEqual(int(np.min(np.abs(db - f))), 2, f"平移起步帧 {f} 不在首拍 ±2 帧（首拍帧 {db.tolist()}）")
        for m in self.moves:
            self.assertIn(m["f_beat"] - m["f_start"], (1, 2))

    def test_never_still_and_readable(self):
        self.assertGreater(float(self.tot[1:].min()), 0.0)
        self.assertGreater(float(self.pan[1:].min()), 0.0)
        self.assertLess(float(self.pan.max()), 45.0, "平移太快看不清")

    def test_voice_bar_targets_f0(self):
        b = [x for x in self.bt if x["d"] <= 47.19 < x["d"] + 1.7][0]
        self.assertEqual(b["ev"]["s"], "voice_on")
        self.assertEqual(b["z"], S.Z["voice"])
        p = min(self.D.phrases, key=lambda p: abs(p["start"] - 47.19))
        self.assertAlmostEqual(b["y"], (S.y_f0(p["f0_med"]) + S.Y_MARK) / 2)      # f0 与审计 ▼ 同框


if __name__ == "__main__":
    unittest.main()
