#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""感官层自检：合成信号，答案已知。python -X utf8 -m unittest discover -s tests -q"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import sensorium as S  # noqa: E402

SR = S.SR


def click_track(bpm=120.0, dur=12.0, quiet=None):
    """每拍一个短促的宽带噪声 + 持续的 60Hz 低音；quiet=(a,b) 秒内全静"""
    rng = np.random.default_rng(0)
    n = int(SR * dur); t = np.arange(n) / SR
    x = 0.15 * np.sin(2 * np.pi * 60 * t)
    period = 60 / bpm
    for k in range(int(dur / period)):
        i = int(k * period * SR)
        L = int(0.02 * SR)
        x[i:i + L] += rng.uniform(-0.8, 0.8, min(L, n - i)) * np.exp(-np.arange(min(L, n - i)) / (0.005 * SR))
    if quiet:
        x[int(quiet[0] * SR):int(quiet[1] * SR)] = 0
    return x


class TestSensorium(unittest.TestCase):
    def test_bpm_and_beats(self):
        for bpm in (96.0, 120.0, 140.0):
            d = S.analyze(click_track(bpm))
            self.assertAlmostEqual(d["bpm"], bpm, delta=1.5, msg=bpm)
            gaps = np.diff(d["events"]["beats"])
            self.assertLess(abs(np.median(gaps) - 60 / bpm), 0.02)

    def test_onsets_land_on_clicks(self):
        d = S.analyze(click_track(120.0))
        on = np.array(d["events"]["onsets"])
        # 每个起音离最近的拍点（0.5s 网格）不超过 12ms（分析帧 10ms；长窗版曾整体早 ~30ms）
        self.assertGreater(len(on), 15)
        self.assertLess(np.max(np.abs(on - np.round(on / 0.5) * 0.5)), 0.012)

    def test_hunger_high_in_silence_and_shapes(self):
        d = S.analyze(click_track(120.0, dur=16.0, quiet=(6, 11)))
        ch = d["channels"]; fps = d["fps"]
        n = int(16 * fps)
        for k in ("heartbeat", "threat", "hunger", "tension", "rhythm"):
            v = np.array(ch[k]); self.assertEqual(len(v), n, k)
            self.assertGreaterEqual(v.min(), 0); self.assertLessEqual(v.max(), 1)
        hunger = np.array(ch["hunger"])
        self.assertGreater(hunger[int(8.5 * fps)], hunger[int(3 * fps)] + 0.4, "静的地方应该更饿")
        hb = np.array(ch["heartbeat"])
        self.assertLess(hb[int(8.5 * fps)], 0.1, "静默里没有心跳")

    def test_rhythm_phase_wraps_once_per_beat(self):
        d = S.analyze(click_track(120.0))
        ph = np.array(d["channels"]["rhythm"])
        wraps = int(np.sum(np.diff(ph) < -0.5))
        self.assertAlmostEqual(wraps, 12 * 2 - 2, delta=2)  # 12 秒 × 2 拍/秒，首尾少一两拍

    def test_length_is_floor_of_duration_times_fps(self):
        # 规格：长度 = floor(duration × fps)。用一个乘出来不是整数的时长
        x = click_track(120.0, dur=5.083)
        d = S.analyze(x)
        import math
        n = math.floor(len(x) / S.SR * d["fps"])
        for k, v in d["channels"].items():
            self.assertEqual(len(v), n, k)

    def test_write_is_valid_json(self):
        d = S.analyze(click_track(120.0, dur=4.0))
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s.json"
            S.write(d, p, source="click")
            back = json.loads(p.read_text("utf-8"))
            self.assertEqual(back["source"], "click"); self.assertEqual(back["bpm"], d["bpm"])
            self.assertEqual(len(back["channels"]["threat"]), len(d["channels"]["threat"]))


if __name__ == "__main__":
    unittest.main()
