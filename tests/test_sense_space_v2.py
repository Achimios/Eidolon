#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""感官空间 v2（tools/sense_space_v2.py）自检：
- Kuramoto 抖动和开头（opening.py）是同一条 K 曲线：15_开头素材_wf_kura_v3 的 HUD 在 24.46 s 写 K = 2.06
- 抖动是局部的：任一帧里被踢起来（|位移| > 半个山脊间距）的山脊占比很小
- 镜头不静止、结尾不推近"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import sense_space_v2 as S  # noqa: E402


@unittest.skipUnless(S.SENSE.exists() and (S.MV / "分轨" / "drums.wav").exists(), "没有 感官.json / 分轨")
class TestSenseSpaceV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sense = json.loads(S.SENSE.read_text("utf-8"))
        cls.dbs = [S.stem_bands(n, 52.0) for n, _ in S.STEMS]
        cls.wig, cls.info, _ = S.kura_tremor(cls.sense, cls.dbs, 51.51)

    def test_K_matches_opening(self):
        f = int(round((24.46 - S.K_T0) * S.FPS))
        self.assertAlmostEqual(float(self.info[f, 0]), 2.06, delta=0.02)

    def test_tremor_is_local(self):
        a = np.abs(self.wig[:, int((38.71 - S.K_T0) * S.FPS):])
        frac = (a > S.HH / S.NR / 2).mean(axis=(0, 2))           # 每帧：被踢起来的山脊占比
        self.assertLess(float(np.median(frac)), 0.05)
        self.assertLess(float(frac.max()), 0.35)
        self.assertGreater(float(frac.max()), 0.0)

    def test_camera_moves_and_no_push_in(self):
        seg = (38.71, 51.51)
        T = int(round((seg[1] - seg[0]) * S.FPS))
        cams, ev = S.camera(seg, T, self.sense, [(47.19, S.yf(554.8)), (48.85, S.yf(369.2))])
        refs = np.array([[x, y, z] for x in np.linspace(-S.X1, 0, 7) for y in (0, S.HH) for z in (S.Z_SENSE, 0.0, 1270.0)])
        sp = S.screen_speed(cams, refs)
        self.assertGreater(float(sp[1:].min()), 0.1)              # 没有静止的帧（实测最慢 ≈ 0.2 px/帧 ≈ 5 px/s：慢，但在漂）
        last = int(T - 1.5 * S.FPS)
        self.assertGreaterEqual(float(cams[-1, 5]), float(cams[last, 5]))   # 最后 1.5 s 距离不缩短


@unittest.skipUnless(S.SENSE.exists() and (S.MV / "分轨" / "drums.wav").exists(), "没有 感官.json / 分轨")
class TestSenseSpaceV3(unittest.TestCase):
    """v3「在里面搜寻」：目标 = 真事件（人声起必在），镜头峰值在出发帧之前 0–2 帧，最慢也在动，层距压缩"""

    def setUp(self):
        self._saved = (S.STEMS, S.Z_SENSE, S.Z_EXP, S.Z_VOICE, dict(S.FOG), dict(S.VER))
        S.set_v3()
        self.sense = json.loads(S.SENSE.read_text("utf-8"))
        self.dbs = [S.stem_bands(n, 52.0) for n, _ in S.STEMS]

    def tearDown(self):
        S.STEMS, S.Z_SENSE, S.Z_EXP, S.Z_VOICE, fog, ver = self._saved
        S.FOG.clear(); S.FOG.update(fog); S.VER.clear(); S.VER.update(ver)

    def test_search(self):
        seg = (38.71, 51.51)
        T = int(round((seg[1] - seg[0]) * S.FPS))
        tgs = S.search_targets(self.sense, self.dbs, seg)
        self.assertIn(47.19, [round(t["t"], 2) for t in tgs if t["kind"] == "voice"])
        cams, ev, _ = S.camera_v3(seg, T, self.sense, tgs)
        refs = np.array([[x, y, z] for x in np.linspace(-S.X1, 0, 7) for y in (0, S.HH)
                         for z in (S.Z_SENSE, S.Z_EXP, S.Z_VOICE) + tuple(z_ for _, z_ in S.STEMS)])
        sp = S.screen_speed(cams, refs)
        self.assertGreater(float(sp[1:].min()), 0.4)        # 实测最慢 0.49 px/帧（v2 0.16）
        for t, k, d in S.alignment(sp, ev, T):
            self.assertIn(d, (-2, -1, 0), (t + seg[0], k, d))
        zs = [z for _, z in S.STEMS]
        self.assertLessEqual(max(np.diff(zs)), 150)


if __name__ == "__main__":
    unittest.main()
