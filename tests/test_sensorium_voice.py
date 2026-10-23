#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""人声通道自检（src/voice.py）：合成信号，答案已知。python -X utf8 -m unittest tests.test_sensorium_voice -q
「必须相等」的量：合成音高 = 检出 f0；合成颤音的频率 / 半幅 = 检出值；静音位置 = 乐句边界；帧数 = channels 帧数。
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import sensorium as S  # noqa: E402
import voice as V  # noqa: E402

SR = V.SR


def tone(dur, f, vib=0.0, rate=5.5, amp=0.3, kind="saw"):
    """锯齿（20 谐波，人声式的丰富谐波）或正弦；vib = 颤音半幅（音分）"""
    n = int(dur * SR); t = np.arange(n) / SR
    fi = f * 2 ** (vib / 1200 * np.sin(2 * np.pi * rate * t))
    ph = 2 * np.pi * np.cumsum(fi) / SR
    if kind == "sine":
        return amp * np.sin(ph)
    y = sum(np.sin(h * ph) / h * (h * fi < SR * 0.45) for h in range(1, 21))
    return amp * y / 1.8


def z(dur):
    return np.zeros(int(dur * SR))


def cents(a, f):
    a = a[np.isfinite(a)]
    return 1200 * np.log2(a / f)


try:
    import torchcrepe  # noqa: F401
    HAS_CREPE = True
except Exception:
    HAS_CREPE = False


class TestVoice(unittest.TestCase):
    """旧算法（numpy YIN，回退）；下面 TestVoiceCrepe 用同一组断言跑默认的 crepe + YIN 细化"""
    METHOD = "yin"

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        # 0.5 静 | 1.5 s 220 Hz 锯齿 | 0.3 s 换气 | 2.0 s 440 Hz ±50 音分 5.5 Hz 颤音 | 2.0 s 休止 | 1.5 s 660 Hz 正弦 | 0.5 静 | 1.0 s 白噪 | 0.5 静
        cls.x = np.concatenate([z(0.5), tone(1.5, 220), z(0.3), tone(2.0, 440, vib=50), z(2.0), tone(1.5, 660, kind="sine"),
                                z(0.5), 0.1 * rng.standard_normal(int(1.0 * SR)), z(0.5)]) + 1e-4 * rng.standard_normal(int(9.8 * SR))
        cls.d = V.analyze_voice(cls.x, f0_method=cls.METHOD)
        cls.F = cls.d["_frames"]

    def test_f0_matches_synth_pitch(self):
        f0 = self.F["f0"]
        for a, b, f in ((0.6, 1.9, 220), (6.4, 7.7, 660)):
            s = f0[int(a * 100):int(b * 100)]
            self.assertGreater(np.isfinite(s).mean(), 0.98, f)
            c = cents(s, f)
            self.assertLess(np.percentile(np.abs(c), 95), 5, f"{f} Hz：95% 帧误差 < 5 音分")
        # 颤音段：中心音高
        self.assertLess(abs(np.median(cents(f0[240:420], 440))), 5)

    def test_no_octave_errors_on_sawtooth(self):
        f0 = self.F["f0"]
        for a, b, f in ((0.6, 1.9, 220), (2.4, 4.2, 440)):
            c = cents(f0[int(a * 100):int(b * 100)], f)
            self.assertLess(np.max(np.abs(c)), 200, f"{f} Hz：没有八度跳变")

    def test_vibrato_rate_and_depth(self):
        hz = np.array([v for v in self.d["vibrato_hz"] if v is not None])
        ct = np.array([v for v in self.d["vibrato_cents"] if v is not None])
        self.assertGreater(len(hz), 20)
        self.assertAlmostEqual(np.median(hz), 5.5, delta=0.2)
        self.assertAlmostEqual(np.median(ct), 50, delta=5)      # 已补 YIN 40 ms 盒式平均（sinc）
        vib = np.array(self.d["vibrato"])
        self.assertGreater(vib[int(2.7 * 24):int(4.0 * 24)].mean(), 0.8)
        self.assertEqual(vib[int(0.7 * 24):int(1.8 * 24)].max(), 0.0, "稳定音没有颤音")
        self.assertEqual(vib[int(6.5 * 24):int(7.6 * 24)].max(), 0.0)

    def test_noise_and_silence_are_unvoiced(self):
        f0 = self.F["f0"]
        self.assertEqual(np.isfinite(f0[860:940]).sum(), 0, "白噪不是有声")
        self.assertEqual(np.isfinite(f0[:45]).sum(), 0)
        self.assertTrue(all(v is None for v in self.d["f0"][int(5.0 * 24):int(6.0 * 24)]))

    def test_phrases_breaths_sections(self):
        ph = self.d["phrases"]
        self.assertEqual(len(ph), 3)
        for p, (a, b) in zip(ph, ((0.5, 2.0), (2.3, 4.3), (6.3, 7.8))):
            self.assertLess(abs(p["start"] - a), 0.03); self.assertLess(abs(p["end"] - b), 0.03)
        self.assertEqual(len(self.d["breaths"]), 1, "0.3 s 是换气，2 s 是休止")
        b0, b1 = self.d["breaths"][0]
        self.assertLess(abs(b0 - 2.0), 0.03); self.assertLess(abs(b1 - 2.3), 0.03)
        self.assertAlmostEqual(ph[1]["f0_med"], 440, delta=3)
        self.assertGreater(ph[1]["tail_vibrato"], 0.8); self.assertEqual(ph[0]["tail_vibrato"], 0.0)

    def test_frame_count_and_ranges(self):
        n = 200
        d = V.analyze_voice(self.x, n_out=n, f0_method=self.METHOD)
        for k in ("loudness", "voiced", "f0", "vibrato", "vibrato_cents", "vibrato_hz"):
            self.assertEqual(len(d[k]), n, k)
        lo = np.array(d["loudness"]); self.assertGreaterEqual(lo.min(), 0); self.assertLessEqual(lo.max(), 1)

    def test_timing_is_playback_time(self):
        """音在 1.000 s 起：乐句起点 = 1.00 ± 20 ms（中心对齐的分析帧，无整体偏移）"""
        d = V.analyze_voice(np.concatenate([z(1.0), tone(1.0, 300), z(1.0)]), f0_method=self.METHOD)
        self.assertLess(abs(d["phrases"][0]["start"] - 1.0), 0.02)
        self.assertLess(abs(d["phrases"][0]["end"] - 2.0), 0.02)

    def test_write_keeps_other_fields(self):
        """sensorium.write 加 voice 节点后，其余字段逐个不变"""
        base = {"fps": 24, "duration": 1.0, "bpm": 120.0, "channels": {"heartbeat": [0.1, 0.2]}, "events": {"onsets": [0.5]}}
        d = V.analyze_voice(self.x, n_out=2, f0_method=self.METHOD)
        node = {k: v for k, v in d.items() if not k.startswith("_")}
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s.json"
            S.write({**base, "voice": node}, p, source="x")
            back = json.loads(p.read_text("utf-8"))
            vb = back.pop("voice")
            self.assertEqual(back, {"source": "x", **base})
            self.assertEqual(vb["phrases"], node["phrases"])


@unittest.skipUnless(HAS_CREPE, "torchcrepe 没装")
class TestVoiceCrepe(TestVoice):
    METHOD = "crepe"

    def test_method_label(self):
        self.assertTrue(self.d["method"].startswith("crepe-full"), self.d["method"])
        self.assertIn("yin_refine", self.d["method"])


class TestFallback(unittest.TestCase):
    def test_crepe_failure_falls_back_to_yin_and_says_so(self):
        orig = V.crepe_track

        def boom(*a, **k):
            raise RuntimeError("no gpu")
        V.crepe_track = boom
        try:
            d = V.analyze_voice(np.concatenate([z(0.5), tone(1.0, 300), z(0.5)]))
        finally:
            V.crepe_track = orig
        self.assertTrue(d["method"].startswith("yin (crepe failed"), d["method"])
        self.assertEqual(len(d["phrases"]), 1)


if __name__ == "__main__":
    unittest.main()
