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
        # 有一个录音文件名撞了 → 整包都不写，已有的那个也不动
        (out / "哼唱_v2_第1遍.wav").write_bytes(b"old")
        with self.assertRaises(FileExistsError):
            serve.save_bundle(self.work, self.bundle("v2", [{"name": "哼唱_v2_第1遍.wav", "sr": 8000, "b64": pcm}]))
        self.assertFalse((out / "作曲_v2.json").exists())
        self.assertEqual((out / "哼唱_v2_第1遍.wav").read_bytes(), b"old")
        self.assertFalse([p for p in out.iterdir() if p.name.startswith(".tmp_")])
        with self.assertRaises(ValueError):
            serve.save_bundle(self.work, self.bundle("v3/../x"))

    def test_audio_names_must_belong_to_this_version(self):
        pcm = base64.b64encode(np.zeros(80, "<f4").tobytes()).decode()
        out = self.work / serve.OUT_DIR
        for bad in ["作曲_v1.json", "哼唱_v2_第1遍.wav", "x.wav", "哼唱_v1_第1遍.wav.json"]:
            with self.assertRaises(ValueError, msg=bad):
                serve.save_bundle(self.work, self.bundle("v1", [{"name": bad, "sr": 8000, "b64": pcm}]))
        a = {"name": "哼唱_v1_第1遍.wav", "sr": 8000, "b64": pcm}
        with self.assertRaises(ValueError):
            serve.save_bundle(self.work, self.bundle("v1", [a, dict(a)]))
        with self.assertRaises(ValueError):          # 采样率不对：整包一个文件都不落
            serve.save_bundle(self.work, self.bundle("v1", [{**a, "sr": 1}]))
        self.assertEqual(sorted(p.name for p in out.iterdir()) if out.exists() else [], [])

    def test_http_rejects_other_sites(self):
        import http.client
        import threading
        from http.server import ThreadingHTTPServer
        serve.Handler.root = self.tmp
        srv = ThreadingHTTPServer(("127.0.0.1", 0), serve.Handler)
        port = srv.server_address[1]
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        try:
            def req(method, path, headers, body=None):
                c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                c.putrequest(method, path, skip_host=True)
                for k, v in headers.items():
                    c.putheader(k, v)
                data = body.encode("utf-8") if body else b""
                if body is not None:
                    c.putheader("Content-Length", str(len(data)))
                c.endheaders(data)
                r = c.getresponse()
                r.read()
                c.close()
                return r.status
            me = f"127.0.0.1:{port}"
            body = json.dumps(self.bundle("v1"))
            self.assertEqual(req("GET", "/api/works", {"Host": me}), 200)
            self.assertEqual(req("GET", "/api/works", {"Host": f"evil.com:{port}"}), 403)   # DNS 重绑定
            self.assertEqual(req("POST", "/api/save?work=MV_%E6%B5%8B%E8%AF%95", {"Host": me, "Origin": "https://evil.com", "Content-Type": "application/json"}, body), 403)
            self.assertEqual(req("POST", "/api/save?work=MV_%E6%B5%8B%E8%AF%95", {"Host": me, "Content-Type": "text/plain"}, body), 415)
            self.assertFalse((self.work / serve.OUT_DIR).exists())
            self.assertEqual(req("POST", "/api/save?work=MV_%E6%B5%8B%E8%AF%95", {"Host": me, "Origin": f"http://{me}", "Content-Type": "application/json"}, body), 200)
            self.assertTrue((self.work / serve.OUT_DIR / "作曲_v1.json").is_file())
        finally:
            srv.shutdown()
            srv.server_close()

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
