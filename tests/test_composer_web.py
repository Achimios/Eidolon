# -*- coding: utf-8 -*-
"""HTML 作曲器：本地服务（存档不覆盖、路径不越界、分轨切片）+ 页面纯函数的 node 自测。"""
import base64
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "composer_web"))
import serve  # noqa: E402


class TestServe(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory(prefix="composer_web_")   # 和其它测试一样：只放测试自己生成的临时文件
        self.tmp = Path(self._td.name)
        self.work = self.tmp / "MV_测试"
        (self.work / "分轨").mkdir(parents=True)
        sr = 8000
        x = np.stack([np.linspace(-0.5, 0.5, sr * 3), np.zeros(sr * 3)], 1).astype("float32")
        sf.write(str(self.work / "分轨" / "drums.wav"), x, sr, subtype="FLOAT")
        (self.work / "感官.json").write_text(json.dumps({"bpm": 150, "duration": 3, "events": {"downbeats": [0.1, 1.7]}}), encoding="utf-8")

    def tearDown(self):
        self._td.cleanup()

    def bundle(self, ver, audio=()):
        return {"ver": ver, "truth": {"bpm": 150, "notes": []}, "session": {"format": "composer_web/1"}, "audio": list(audio)}

    def test_work_name_cannot_escape_root(self):
        for bad in ["..", "../x", "a/b", "a\\b", "", ".git", "不存在"]:
            with self.assertRaises(ValueError, msg=bad):
                serve.safe_work(self.tmp, bad)
        self.assertEqual(serve.safe_work(self.tmp, "MV_测试"), self.work.resolve())
        for bad in ["../x.json", "a/b.wav", "..\\x", ".hidden", "C:x"]:
            with self.assertRaises(ValueError, msg=bad):
                serve.safe_name(bad)

    def test_save_versions_never_overwrite(self):
        out = self.work / serve.OUT_DIR
        self.assertEqual(serve.next_version(out), 1)
        pcm = base64.b64encode(np.full(800, 0.25, "<f4").tobytes()).decode()
        r = serve.save_bundle(self.work, self.bundle("v1", [{"name": "哼唱_v1_第1遍.wav", "sr": 8000, "b64": pcm}]))
        self.assertIn("作曲_v1.json", r["written"])
        y, sr = sf.read(str(out / "哼唱_v1_第1遍.wav"))
        self.assertEqual((len(y), sr), (800, 8000))
        self.assertAlmostEqual(float(y[0]), 0.25, places=4)
        before = (out / "作曲_v1.json").read_bytes()
        with self.assertRaises(FileExistsError):
            serve.save_bundle(self.work, self.bundle("v1"))
        self.assertEqual((out / "作曲_v1.json").read_bytes(), before)
        self.assertEqual(serve.next_version(out), 2)
        # 有一个录音文件名撞了 → 整包都不写
        with self.assertRaises(FileExistsError):
            serve.save_bundle(self.work, self.bundle("v2", [{"name": "哼唱_v1_第1遍.wav", "sr": 8000, "b64": pcm}]))
        self.assertFalse((out / "作曲_v2.json").exists())
        with self.assertRaises(ValueError):
            serve.save_bundle(self.work, self.bundle("v3/../x"))

    def test_stem_slice_pads_outside_file(self):
        b = serve.stem_slice(self.work, "drums.wav", -0.5, 1.0)
        y, sr = sf.read(__import__("io").BytesIO(b))
        self.assertEqual(sr, 8000)
        self.assertEqual(len(y), 12000)
        self.assertTrue(np.all(y[:4000] == 0))
        self.assertAlmostEqual(float(y[4000, 0]), -0.5, places=3)
        self.assertEqual(serve.list_stems(self.work), ["drums.wav"])
        self.assertEqual(serve.read_sense(self.work)["downbeats"], [0.1, 1.7])


@unittest.skipUnless(shutil.which("node"), "没装 node")
class TestPageLogic(unittest.TestCase):
    def test_node_selftest(self):
        r = subprocess.run(["node", str(ROOT / "tools" / "composer_web" / "test" / "run.mjs")], capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
