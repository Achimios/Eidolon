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


SONG = Path(__file__).resolve().parents[3] / "Projekt_Visual" / "Assets" / "The Glitch Mob - Can't Kill Us [2020 Edit]_190189282.mp4"


def _lag(a, b, m=2000):
    """b 在 a 里的位置偏差（样本）；0 = 对齐"""
    return int(np.argmax(np.correlate(a[:len(b)], b[m:-m], "valid")) - m)


class TestLoadAudioPlaybackTime(unittest.TestCase):
    """样本 i ↔ 播放时间 start + i/sr，即使容器里音频比视频晚开始（原曲：0.042 s）"""
    LEAD, CLICKS = 0.25, (0.5, 1.5, 2.25)

    @classmethod
    def setUpClass(cls):
        import subprocess
        import imageio_ffmpeg
        import wave
        cls.td = tempfile.TemporaryDirectory()
        d = Path(cls.td.name); sr = 48000
        x = np.zeros(int(3.0 * sr))
        for c in cls.CLICKS:
            i = int(c * sr); x[i:i + 48] = np.hanning(48) * 0.9 * (1 if c != 1.5 else -1)
        with wave.open(str(d / "c.wav"), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
            w.writeframes((x * 32767).astype(np.int16).tobytes())
        # mov + PCM：音频起点恰好 = LEAD（无编码器预滚），点击的播放时间 = c + LEAD，答案已知
        # mp4 + AAC：起点被编码器预滚挪到 LEAD − 1024/48k 左右，只拿来验「从头解码 ≡ -ss 截取」
        cls.mov, cls.mp4 = d / "late.mov", d / "late.mp4"
        for out, ca in ((cls.mov, "pcm_s16le"), (cls.mp4, "aac")):
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:r=24:d=3.5",
                            "-itsoffset", str(cls.LEAD), "-i", str(d / "c.wav"), "-map", "0:v", "-map", "1:a",
                            "-c:v", "mpeg4", "-c:a", ca, str(out)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    def test_lead_is_probed(self):
        self.assertAlmostEqual(S.audio_lead(self.mov), self.LEAD, delta=1e-6)
        self.assertGreater(S.audio_lead(self.mp4), 0.2)

    def test_clicks_land_at_playback_time(self):
        x = S.load_audio(self.mov)
        hop = S.HOP / SR
        for c in self.CLICKS:
            t = c + self.LEAD
            a, b = int((t - 0.2) * SR), int((t + 0.2) * SR)
            got = (a + np.argmax(np.abs(x[a:b]))) / SR
            self.assertLess(abs(got - t), hop, f"click {c}: {got:.4f} vs {t:.4f}")

    def test_start_inside_lead_and_after_agree(self):
        for f in (self.mov, self.mp4):
            x = S.load_audio(f)
            for st in (0.1, 1.0):
                y = S.load_audio(f, start=st, dur=1.5)
                i = int(round(st * SR))
                self.assertLessEqual(abs(_lag(x[i:i + len(y) + 4000], y)), 2, (f.name, st))

    @unittest.skipUnless(SONG.exists(), "原曲不在")
    def test_song_start40_vs_start0_agree(self):
        x = S.load_audio(SONG, dur=52.0)
        y = S.load_audio(SONG, start=40.0, dur=10.0)
        self.assertAlmostEqual(S.audio_lead(SONG), 2016 / 48000, delta=1e-6)
        self.assertLessEqual(abs(_lag(x[40 * SR:], y)), 3)   # 修之前 −465 样本（−42 ms）


if __name__ == "__main__":
    unittest.main()
