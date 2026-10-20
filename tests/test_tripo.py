#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tripo 客户端离线自检（不联网、不读 key）：multipart 拼装、后缀跟着服务器格式走、输入识别、预设不越界。"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import tripo as T  # noqa: E402


class TestTripoOffline(unittest.TestCase):
    def test_multipart(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.png"; p.write_bytes(b"\x89PNG-data")
            body, ctype = T._multipart("file", p)
            b = ctype.split("boundary=")[1]
            self.assertTrue(body.startswith(f"--{b}\r\n".encode()))
            self.assertTrue(body.endswith(f"\r\n--{b}--\r\n".encode()))
            self.assertIn(b'name="file"; filename="a.png"', body)
            self.assertIn(b"Content-Type: image/png", body)
            self.assertIn(b"\x89PNG-data", body)

    def test_fix_suffix(self):
        # quad=true 时服务器给 fbx：别存成 .glb
        self.assertEqual(T._fix_suffix("x/人台.glb", "https://cdn.x/a/model.fbx?sig=1").suffix, ".fbx")
        self.assertEqual(T._fix_suffix("x/人台.glb", "https://cdn.x/a/model.glb?sig=1").name, "人台.glb")
        self.assertEqual(T._fix_suffix("x/人台.glb", "https://cdn.x/a/noext").name, "人台.glb")

    def test_input_passthrough(self):
        # 实测：V3 的 task_id 是 UUID（文档示例写 task_xxx）；两种都原样传，不触发上传
        for s in ("https://e.com/a.png", "task_abc", "file_abc", "4b07d03c-3218-44e9-9ecc-cf0d456304b9"):
            self.assertEqual(T._input(s), s)

    def test_presets_sane(self):
        self.assertEqual(T.PRESETS["draft"]["texture"], False)
        self.assertEqual(T.PRESETS["draft"]["pbr"], False)  # pbr=true 会强制 texture=true → 白模必须两个都关
        for p in T.PRESETS.values():
            self.assertNotIn("export_orientation", p)  # 设了朝向，后续 rig 会静默转错

    def test_retarget_max5(self):
        with self.assertRaises(ValueError):
            T.retarget("t", ["preset:idle"] * 6, "x.glb")


if __name__ == "__main__":
    unittest.main()
