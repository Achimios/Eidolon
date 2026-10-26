# -*- coding: utf-8 -*-
"""composer_render：用一份和页面导出格式一样的小作曲（2 小节）渲染，查有声、不削波、响度对、场景静音生效、侧链生效、确定性。"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
import composer_render as cr  # noqa: E402

BPM = 120.0
SPB = 0.5


def comp(work):
    # 一段 0.5 秒的「录音」clip（正弦），放在作品文件夹里
    (work / "作曲").mkdir(parents=True, exist_ok=True)
    t = np.arange(int(0.5 * cr.SR)) / cr.SR
    sf.write(str(work / "作曲" / "哼唱_v1_第1遍.wav"), (0.2 * np.sin(2 * np.pi * 330 * t)).astype(np.float32), cr.SR, subtype="PCM_24")
    notes = [{"t": 0.0 + i * SPB, "dur": 0.4, "pitch": [40, 43, 47, 52][i % 4], "vel": 0.8, "layer": "bass", "input": "keyboard"} for i in range(8)]
    notes += [{"t": 0.0, "dur": 1.9, "pitch": 64, "vel": 0.7, "layer": "pad", "input": "keyboard"},
              {"t": 2.0, "dur": 1.9, "pitch": 67, "vel": 0.7, "layer": "pad", "input": "keyboard"},
              {"t": 1.0, "dur": 0.8, "pitch": 71, "vel": 0.6, "layer": "voice", "input": "hum", "bend": {"dt": 0.02, "cents": [0, 20, 40, 20, 0, -20]}}]
    notes.sort(key=lambda n: n["t"])
    hits = [{"t": i * SPB, "layer": "drums", "sound": "kick", "vel": 0.9} for i in range(8)]
    hits += [{"t": i * SPB + 0.25, "layer": "drums", "sound": s, "vel": 0.6} for i, s in enumerate(["hat", "clap", "hat", "snare", "ohat", "perc", "tom", "hat"])]
    hits.append({"t": 3.5, "layer": "drums", "sound": "impact", "vel": 1.0})
    fps = 60
    v = [0.6] * int(4.2 * fps)
    return {
        "bpm": BPM, "window": [0.0, 4.0], "grid": {"anchor": 0.0, "beats_per_bar": 4, "downbeats": [0.0, 2.0, 4.0]},
        "layers": [
            {"id": "drums", "name": "鼓", "kind": "drums", "preset": "drums", "octave": 0, "gain_db": 0, "mute": False, "input": "keyboard", "params": {"drive": 0.1, "delaySend": 0, "reverbSend": 0.08, "duck": False, "pan": 0}},
            {"id": "bass", "name": "贝斯", "kind": "synth", "preset": "bass", "octave": 2, "gain_db": 0, "mute": False, "input": "keyboard", "params": dict(cr.PRESETS["bass"])},
            {"id": "pad", "name": "铺底", "kind": "synth", "preset": "pad", "octave": 4, "gain_db": -3, "mute": False, "input": "keyboard", "params": dict(cr.PRESETS["pad"])},
            {"id": "voice", "name": "哼唱", "kind": "voice", "preset": "voice", "octave": 4, "gain_db": 0, "mute": False, "input": "hum",
             "params": {"mode": "hum", "snap": "scale", "rawGainDb": -6, "synthOn": True, "synthPreset": "sing", "bendOn": True, "drive": 0, "delaySend": 0.1, "reverbSend": 0.2, "duck": False, "pan": 0}},
        ],
        "notes": notes, "hits": hits,
        "clips": [{"t": 1.0, "dur": 0.5, "layer": "voice", "input": "hum", "active": True, "take": 1, "file": "作曲/哼唱_v1_第1遍.wav", "src": "mic"}],
        # 第二小节（2.0 s 起）场景里没有 pad → pad 在后半不响
        "scenes": [{"t": 0.0, "name": "A", "layers": ["drums", "bass", "pad", "voice"]}, {"t": 2.0, "name": "B", "layers": ["drums", "bass", "voice"]}],
        "macros": {"drive": {"fps": fps, "t0": 0.0, "v": v}},
        "macro_rules": {"cutoff": {"a": 0.55, "b": 0.45}, "reso": {"a": 1, "b": 0}, "drive": {"a": 0.35, "b": 0.65}, "space": {"a": 0.5, "b": 0.5}, "pole_g": {"a": 0.75, "b": 0.35}, "sidechain": {"a": 0.35, "b": 0.65}},
        "macro_defaults": dict(cr.MACRO_DEFAULTS),
        "tension": [[0.0, 0.3], [4.0, 0.9]], "reveals": [], "mood": [],
    }


class TestComposerRender(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.work = Path(cls.tmp.name)
        cls.J = comp(cls.work)
        cls.mix, cls.stems, cls.rep = cr.render(cls.J, cls.work, master=True, log=lambda s: None)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_loudness_and_peak(self):
        self.assertAlmostEqual(self.rep["lufs"], -14.0, delta=0.3)
        self.assertLessEqual(self.rep["true_peak_dbtp"], -1.0)
        self.assertEqual(self.rep["clipped_samples"], 0)
        self.assertTrue(np.all(np.isfinite(self.mix)))

    def test_scene_mutes_pad(self):
        pad = self.stems["铺底"]
        a = float(np.sqrt(np.mean(pad[int(0.5 * cr.SR):int(1.8 * cr.SR)] ** 2)))
        b = float(np.sqrt(np.mean(pad[int(3.6 * cr.SR):int(4.0 * cr.SR)] ** 2)))   # 场景 B 里没有 pad（2.0 s 的新音不放；旧音的尾巴 1.2 s 释放后已很小）
        self.assertGreater(a, 0.01)
        self.assertLess(b, a * 0.05)

    def test_sidechain_ducks_bass(self):
        bass = np.abs(self.stems["贝斯"]).mean(1)
        at_kick = bass[int(1.005 * cr.SR):int(1.03 * cr.SR)].mean()
        later = bass[int(1.15 * cr.SR):int(1.3 * cr.SR)].mean()
        self.assertLess(at_kick, later)

    def test_voice_has_clip_and_synth(self):
        v = self.stems["哼唱"]
        self.assertGreater(float(np.abs(v[int(1.05 * cr.SR):int(1.4 * cr.SR)]).max()), 0.01)

    def test_deterministic(self):
        mix2, _, _ = cr.render(self.J, self.work, master=True, log=lambda s: None)
        self.assertTrue(np.allclose(self.mix, mix2))


if __name__ == "__main__":
    unittest.main()
