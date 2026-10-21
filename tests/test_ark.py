#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ark 客户端离线自检（不联网、不读 key）：参考视频只收 URL / asset://（2026-10-21 实测其余写法全被 400 拒）。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import ark as A  # noqa: E402


class TestArkOffline(unittest.TestCase):
    def test_vid_accepts_url_and_asset(self):
        for u in ("https://b.tos-cn-beijing.volces.com/a.mp4?X-Tos-Signature=x", "asset://asset-2026-abc"):
            d = A.vid(u)
            self.assertEqual(d, {"type": "video_url", "video_url": {"url": u}, "role": "reference_video"})

    def test_vid_rejects_local_and_non_urls(self):
        for bad in ("样片/09c.mp4", r"C:\x\a.mp4", "data:video/mp4;base64,AAAA", "file-20261021214722-5ndqv", "file://file-x"):
            with self.assertRaises(ValueError):
                A.vid(bad)

    def test_media_passthrough(self):
        for u in ("https://e.com/a.png", "asset://asset-x", "data:image/png;base64,AA"):
            self.assertEqual(A.media(u), u)


if __name__ == "__main__":
    unittest.main()
