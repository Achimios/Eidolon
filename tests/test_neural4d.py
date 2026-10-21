#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Neural4D 客户端离线自检（不联网、不读 key）：请求体拼装、面数范围、多视图剥参数、下载计划、sidecar 不带 key。"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import neural4d as N  # noqa: E402

FID = "550e8400-e29b-41d4-a716-446655440000"


def fake_upload(path):
    return "uploaded:" + Path(path).name


class TestNeural4DOffline(unittest.TestCase):
    def test_text_body_draft_is_grey(self):
        b = N.text_body("a cube")
        self.assertEqual(b["model"], "dreamtech/neural4d-2.5")
        self.assertIs(b["with_texture"], False)
        self.assertIs(b["with_pbr"], False)
        self.assertEqual(b["target_polycount"], 100000)

    def test_prompt_limit(self):
        with self.assertRaises(ValueError):
            N.text_body("")
        with self.assertRaises(ValueError):
            N.text_body("x" * 501)
        N.text_body("x" * 500)

    def test_pbr_needs_texture(self):
        with self.assertRaises(ValueError):
            N.text_body("a", preset=None, with_pbr=True)
        N.text_body("a", preset=None, with_pbr=True, with_texture=True)

    def test_polycount_ranges(self):
        N.check_polycount("triangle", "standard", 100000)
        N.check_polycount("triangle", "high", 1000000)
        N.check_polycount("quad", "standard", 1000)
        N.check_polycount("quad", "standard", 100100 - 100)
        for args in (("triangle", "standard", 99000), ("triangle", "standard", 600000),
                     ("triangle", "standard", 100500),     # 步长 1000
                     ("triangle", "high", 400000), ("quad", "standard", 150), ("quad", "standard", 1050)):
            with self.assertRaises(ValueError, msg=str(args)):
                N.check_polycount(*args)
        with self.assertRaises(ValueError):
            N.text_body("a", preset="draft", mesh_quality="high")  # draft 的 10 万面在 high 下不合法

    def test_presets_valid(self):
        for name in N.PRESETS:
            N.text_body("a", preset=name)

    def test_n_range(self):
        with self.assertRaises(ValueError):
            N.text_body("a", n=5)

    def test_ref_kinds(self):
        self.assertEqual(N._ref("https://e.com/a.png"), {"url": "https://e.com/a.png"})
        self.assertEqual(N._ref(FID), {"file_id": FID})
        self.assertEqual(N._ref("x/图.png", fake_upload), {"file_id": "uploaded:图.png"})
        with self.assertRaises(ValueError):
            N._ref("http://e.com/a.png")

    def test_image_body_single_and_batch(self):
        b = N.image_body("a.png", uploader=fake_upload, pose="T-pose", n=2)
        self.assertEqual(b["input_references"], [{"type": "image", "file_id": "uploaded:a.png"}])
        self.assertEqual(b["pose"], "T-pose")
        b2 = N.image_body(["a.png", FID], uploader=fake_upload)
        self.assertEqual(len(b2["input_references"]), 2)
        with self.assertRaises(ValueError):
            N.image_body(["a.png", "b.png"], uploader=fake_upload, n=2)
        with self.assertRaises(ValueError):
            N.image_body([f"{i}.png" for i in range(11)], uploader=fake_upload)

    def test_multiview_strips_forbidden(self):
        b = N.multiview_body({"front": FID, "back": FID}, preset="grey_hi")
        for k in ("mesh_quality", "topology", "pose"):
            self.assertNotIn(k, b)
        self.assertEqual(b["target_polycount"], 1000000)
        b = N.multiview_body({"front": FID, "back": FID}, preset="draft")  # 10 万面不合法 → 交给服务器默认
        self.assertNotIn("target_polycount", b)
        self.assertEqual([v["view_type"] for v in b["multi_view_images"]], ["front", "back"])
        with self.assertRaises(ValueError):
            N.multiview_body({"front": FID, "left": FID})
        with self.assertRaises(ValueError):
            N.multiview_body({"front": FID, "back": FID, "side": FID})

    def test_task_args(self):
        with self.assertRaises(ValueError):
            N.task()
        with self.assertRaises(ValueError):
            N.task("a", uuid="b")

    def test_out_paths(self):
        self.assertEqual(N.out_paths("x/人台.glb", 1), [Path("x/人台.glb")])
        self.assertEqual([p.name for p in N.out_paths("x/人台.glb", 3)], ["人台_1.glb", "人台_2.glb", "人台_3.glb"])

    def test_plan_files_tri_and_quad(self):
        tri = {"uuid": "u", "format": "glb", "model_urls": {"glb": "https://s3.x/r/model.glb?sig=1"},
               "thumbnail_urls": {"front": "https://s3.x/r/t1.png?sig=2"}}
        f = N.plan_files(tri, "out/人台.glb")
        self.assertEqual([p.name for _, p in f], ["人台.glb", "人台.glb.preview.png"])
        quad = {"uuid": "u", "format": "obj",
                "model_urls": {"obj": "https://s3.x/r/m.obj?s", "mtl": "https://s3.x/r/m.mtl?s"},
                "texture_urls": {"albedo": "https://s3.x/r/m_albedo.png?s"}, "thumbnail_urls": {}}
        f = N.plan_files(quad, "out/人台.glb")
        names = [p.name for _, p in f]
        self.assertEqual(names[0], "人台.obj")   # 四边面服务器给 obj：后缀跟着走
        self.assertIn("人台.mtl", names)
        self.assertIn("m_albedo.png", names)    # 贴图名保持和 mtl 引用一致
        with self.assertRaises(N.Neural4DError):
            N.plan_files({"uuid": "u", "model_urls": {}}, "o.glb")

    def test_sidecar_has_no_key(self):
        st = {"id": "normal-3d-x", "model": N.MODEL, "usage": {"credits": 30}}
        m = {"uuid": "u", "topology": "triangle", "target_polycount": 100000, "model_urls": {"glb": "https://s/x.glb"}}
        body = N.text_body("a cube")
        s = N.sidecar(st, m, "/3dmodels/generations", body, None, [("https://s/x.glb", Path("a.glb"))])
        txt = str(s)
        self.assertEqual(s["usage_credits_task"], 30)
        self.assertEqual(s["server"]["target_polycount"], 100000)
        for bad in ("Authorization", "Bearer", "api_key", "eyJ"):
            self.assertNotIn(bad, txt)

    def test_multipart(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.png"; p.write_bytes(b"\x89PNG-data")
            body, ctype = N._multipart("file", p)
            b = ctype.split("boundary=")[1]
            self.assertTrue(body.startswith(f"--{b}\r\n".encode()))
            self.assertTrue(body.endswith(f"\r\n--{b}--\r\n".encode()))
            self.assertIn(b'name="file"; filename="a.png"', body)


if __name__ == "__main__":
    unittest.main()
