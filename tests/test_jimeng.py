# -*- coding: utf-8 -*-
"""即梦封装的命令拼装（替换 _run，不调 CLI、不扣积分）。"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import jimeng  # noqa: E402


class T(unittest.TestCase):
    def go(self, f, *a, **k):
        calls = []
        def run(args, timeout=120):
            calls.append(args)
            return 'submit_id: abc-1'
        with mock.patch.object(jimeng, "_run", run), tempfile.TemporaryDirectory() as d:
            sid = f("p", d, *a, block=False, **k)
        self.assertEqual(sid, "abc-1")
        return calls[0]

    def test_routes(self):
        self.assertEqual(self.go(jimeng.image)[0], "text2image")
        self.assertIn("--images=a.png", self.go(jimeng.image, refs=["a.png"]))
        c = self.go(jimeng.video, first="a.png", last="b.png")
        self.assertEqual(c[0], "frames2video"); self.assertIn("--last=b.png", c)
        self.assertNotIn("--ratio=None", c)
        self.assertEqual(self.go(jimeng.video, first="a.png")[0], "image2video")
        c = self.go(jimeng.video, refs=["a.png", "b.png"], ref_audios=["x.wav"])
        self.assertEqual(c[0], "multimodal2video"); self.assertEqual(c.count("--image=a.png"), 1); self.assertIn("--audio=x.wav", c)

    def test_either_or(self):
        with self.assertRaises(ValueError):
            jimeng.video("p", ".", first="a", refs=["b"], block=False)


if __name__ == "__main__":
    unittest.main()
