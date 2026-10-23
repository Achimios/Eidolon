# -*- coding: utf-8 -*-
"""edl.rows_from_timeline：Synapsis 时间轴导出 → 剪辑表行"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import edl  # noqa: E402


def item(id_, t0, t1, track=0, pics=(), src=None, label=None):
    d = {"id": id_, "label": label or id_, "t0": t0, "t1": t1, "track": track, "pics": list(pics), "body": "", "ref": None}
    if src:
        d["src"] = src
    return d


class RowsFromTimeline(unittest.TestCase):
    def test_order_gaps_and_sources(self):
        tl = {"page": "p", "unit": "s", "fps": 24, "items": [
            item("b", 2.0, 3.0, pics=["C:/x/b.jpg"]),
            item("a", 0.5, 1.5, src="样片:a.mp4@1.2"),
            item("c", 3.0, 4.0, label="她在下坠"),
        ]}
        rows = edl.rows_from_timeline(tl)
        self.assertEqual(rows[0], (0.0, "黑", 1.0))                 # 开头空隙补黑
        self.assertEqual(rows[1], (0.5, "样片:a.mp4@1.2", 1.0))     # src 优先
        self.assertEqual(rows[2], (1.5, "黑", 1.0))                 # 中间空隙
        self.assertEqual(rows[3], (2.0, "图:C:/x/b.jpg", 1.0))      # 有图 → 静帧
        self.assertEqual(rows[4], (3.0, "文字:她在下坠", 1.0))       # 都没有 → 文字卡
        self.assertEqual(rows[-1], (4.0, "黑", 1.0))                # 收尾

    def test_skips_bad_and_other_tracks(self):
        tl = {"items": [item("x", 2.0, 1.0), item("y", 0.0, 1.0, track=1), item("z", None, 1.0)]}
        self.assertEqual(edl.rows_from_timeline(tl), [(0.0, "黑", 1.0)])

    def test_overlap_clamped(self):
        tl = {"items": [item("a", 0.0, 2.0, label="a"), item("b", 1.0, 3.0, label="b")]}
        rows = edl.rows_from_timeline(tl)
        self.assertEqual([r[0] for r in rows], [0.0, 2.0, 3.0])     # 重叠的后一条从前一条结束处开始


class Flatten(unittest.TestCase):
    def test_sections_expand_to_shots(self):
        pages = {
            "剪辑": {"items": [item("剪辑/04", 51.5, 60.0), item("剪辑/05", 60.0, 70.0, src="原片@1:00")]},
            "剪辑/04": {"items": [item("剪辑/04/a", 51.5, 55.0, src="样片:x.mp4@0"), item("剪辑/04/b", 55.0, 60.0)]},
        }
        flat = edl.flatten_items(pages["剪辑"], pages.get)
        self.assertEqual([i["id"] for i in flat], ["剪辑/04/a", "剪辑/04/b", "剪辑/05"])   # 有 src 的不展开


if __name__ == "__main__":
    unittest.main()
