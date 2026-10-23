#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""期待 / 意外自检（src/expect.py + tools/sense_expect.py）：合成事件序列，答案已知。python -X utf8 -m unittest tests.test_expect -q
「必须相等 / 必须成立」的量：常见事件低、罕见事件高；换段后先意外、再适应；同一个事件在稀段比在密段意外；
写 expect 时其余字段（channels / events / voice …）逐字节不变。
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import expect as E  # noqa: E402
import sensorium as S  # noqa: E402
import sense_expect as X  # noqa: E402

BEAT = 0.4


class Surprisal(unittest.TestCase):
    def test_common_low_rare_high(self):
        """一段里每拍一个同样大小的起音：热身后接近 0 比特；中间插一个特别大的 → 高"""
        ts = np.arange(0, 40, BEAT)
        m = np.full(len(ts), 0.5); m[60] = 1.0
        r = E.surprisal(ts, m, [(0, 40)], beat=BEAT)
        common = np.delete(r["bits"][20:], 40)
        self.assertLess(common.max(), 0.8)
        self.assertGreater(r["bits"][60], 4.0)
        self.assertGreater(r["bits"][60], 5 * np.median(common))

    def test_rare_occurrence_high(self):
        """稀段里（每 4 s 一个）一个事件的发生意外度 > 密段里（每拍一个）"""
        sparse = np.arange(0, 40, 4.0); dense = np.arange(40, 80, BEAT)
        ts = np.r_[sparse, dense]
        r = E.surprisal(ts, np.ones(len(ts)), [(0, 40), (40, 80)], beat=BEAT, use_mag=False)
        self.assertGreater(np.median(r["bits"][3:len(sparse)]), 2.0)
        self.assertLess(np.median(r["bits"][-40:]), 0.3)

    def test_readapt_after_switch(self):
        """稀 → 密换段：换段后头几个事件仍带着稀段的期待（意外），2 小节（3.2 s）内适应到 < 0.5 比特"""
        sparse = np.arange(0, 40, 4.0); dense = np.arange(40, 60, BEAT)
        ts = np.r_[sparse, dense]
        r = E.surprisal(ts, np.ones(len(ts)), [(0, 40), (40, 60)], beat=BEAT, use_mag=False)
        b = r["bits"][len(sparse):]; t = dense - 40
        self.assertGreater(b[0], 1.5)
        self.assertLess(b[t >= 3.2].max(), 0.5)
        self.assertGreater(b[t < 1.0].min(), 1.0)              # 第一小节里还是意外的
        self.assertTrue(np.all(np.diff(b[:6]) <= 1e-9))          # 单调适应

    def test_same_event_depends_on_segment(self):
        """同一个大小 0.9 的起音：在安静段（其余 0.2）意外，在吵段（其余 0.9）日常 —— 段落上下文，不是全曲一个分布"""
        q = np.arange(0, 30, BEAT); l = np.arange(30, 60, BEAT)
        ts = np.r_[q, l]
        m = np.r_[np.full(len(q), 0.2), np.full(len(l), 0.9)]
        iq = len(q) - 5; m[iq] = 0.9
        r = E.surprisal(ts, m, [(0, 30), (30, 60)], beat=BEAT)
        il = len(ts) - 5
        self.assertGreater(r["mag"][iq], 3.0)
        self.assertLess(r["mag"][il], 0.3)
        # 全曲一个段落时，两处的区别会被抹小（这就是要按段落的原因）
        g = E.surprisal(ts, m, [(0, 60)], beat=BEAT)
        self.assertGreater(r["mag"][iq] - r["mag"][il], g["mag"][iq] - g["mag"][il] - 1e-9)

    def test_sliding_window_without_segments(self):
        """没有段落：滑动窗口也能重新适应（密 → 稀 → 密）"""
        ts = np.r_[np.arange(0, 30, BEAT), np.arange(30, 60, 4.0), np.arange(60, 100, BEAT)]
        r = E.surprisal(ts, np.ones(len(ts)), None, beat=BEAT, use_mag=False)
        late = r["bits"][ts > 85]
        self.assertLess(late.max(), 0.5)
        self.assertGreater(r["bits"][(ts > 30) & (ts < 60)].max(), 1.5)

    def test_absence(self):
        """密段里突然停 3 s：缺席比特冲过阈值，记成一个 absence 事件；之后回来就塌回去"""
        ts = np.r_[np.arange(0, 20, BEAT), np.arange(23, 40, BEAT)]
        fps = 24; ft = np.arange(int(40 * fps)) / fps
        ab = E.absence(ts, ft, [(0, 40)])
        ev = E.absence_events(ab, fps)
        self.assertEqual(len(ev), 1)
        self.assertTrue(20.0 < ev[0][0] < 21.5)
        self.assertGreater(ev[0][1], 8.0)
        self.assertLess(ab[int(30 * fps):].max(), 2.0)


class Salience(unittest.TestCase):
    def test_gate_and_attend(self):
        ex = [{"tolerance": "低", "attend": {"onset": 1.0}}, {"tolerance": "高", "attend": {"onset": 1.0}}]
        lo = E.salience(2.0, "onset", 0, ex); hi = E.salience(2.0, "onset", 1, ex)
        self.assertGreater(lo, 1.8)                 # 低容忍：2 比特就过门
        self.assertLess(hi, 0.2)                    # 高容忍：2 比特过不了门
        self.assertAlmostEqual(E.salience(6.0, "voice_on", 0, ex), E.UNATTENDED * 6.0, delta=0.01)   # 没在听的那一路：弱


def fake_sense(dur=30.0, fps=24):
    n = int(dur * fps)
    rng = np.random.default_rng(0)
    on = np.round(np.arange(0.5, dur - 0.5, BEAT), 3).tolist()
    return {"source": "x.wav", "fps": fps, "duration": dur, "bpm": 150.0, "beat_source": "t",
            "motifs": {"抽搐": {"template": [1, 2], "track": "other", "hits": [5.1, 17.3]}},
            "channels": {k: np.round(rng.random(n), 3).tolist() for k in ("heartbeat", "threat", "hunger", "tension", "rhythm")},
            "events": {"onsets": on, "beats": on, "downbeats": on[::4]},
            "voice": {"source": "分轨/vocals.wav", "loudness": [0.1] * n, "f0": [None] * n,
                      "phrases": [{"start": 3.0, "end": 5.0, "peak": 0.5, "tail_vibrato": 0.0},
                                  {"start": 12.0, "end": 16.5, "peak": 0.8, "tail_vibrato": 0.2}],
                      "breaths": [], "sections": [[3.0, 16.5]]}}


class Integration(unittest.TestCase):
    def run_tool(self, with_context):
        d = Path(tempfile.mkdtemp())
        sense = fake_sense()
        src = sense.pop("source")
        S.write(sense, d / "感官.json", source=src)
        before = (d / "感官.json").read_text("utf-8")
        if with_context:
            (d / "情绪大图").mkdir()
            (d / "情绪大图" / "草案.json").write_text(json.dumps({"段": [{"id": "00", "t": [0, 10], "名": "甲"}, {"id": "01", "t": [10, 30], "名": "乙"}]}, ensure_ascii=False), "utf-8")
            (d / "情绪大图" / "期待.json").write_text(json.dumps({"段": [{"id": "00", "expecting": "无", "tolerance": "低", "attend": {"onset": 1}},
                                                                     {"id": "01", "expecting": "冲击", "tolerance": "高", "attend": {"voice_on": 1}}]}, ensure_ascii=False), "utf-8")
        old_argv = sys.argv
        sys.argv = ["sense_expect.py", str(d)]
        try:
            X.main()
        finally:
            sys.argv = old_argv
        return d, before

    def test_old_fields_untouched(self):
        for ctx in (False, True):
            d, before = self.run_tool(ctx)
            after = json.loads((d / "感官.json").read_text("utf-8"))
            old = json.loads(before)
            ex = after.pop("expect")
            self.assertEqual(after, old)
            self.assertEqual(list(after), list(old))
            self.assertEqual(len(ex["surprise"]), len(old["channels"]["heartbeat"]))
            self.assertEqual(len(ex["absence"]), len(old["channels"]["heartbeat"]))
            self.assertEqual({e["s"] for e in ex["events"]} >= {"onset", "motif", "voice_on", "voice_off"}, True)
            if ctx:
                self.assertIsNotNone(ex["salience"]); self.assertIsNotNone(ex["segments"])
            else:
                self.assertIsNone(ex["salience"]); self.assertIsNotNone(ex["salience_note"])
            # 文本层面：expect 之前的每一行和旧文件一模一样（下游按行扫也不受影响）
            new_text = (d / "感官.json").read_text("utf-8")
            self.assertTrue(new_text.startswith(before[:before.rindex(" }\n}")]))
            # 再跑一次：幂等（expect 被替换，不叠加）
            sys.argv = ["sense_expect.py", str(d)]
            try:
                X.main()
            finally:
                sys.argv = ["x"]
            self.assertEqual((d / "感官.json").read_text("utf-8"), new_text)

    def test_write_roundtrip_without_expect(self):
        """没有 expect 的文件：write 的输出和改 TAIL 之前一样（voice 仍在最后）"""
        sense = fake_sense(); src = sense.pop("source")
        p = Path(tempfile.mkdtemp()) / "a.json"
        S.write(sense, p, source=src)
        t = p.read_text("utf-8")
        self.assertNotIn('"expect"', t)
        self.assertTrue(t.rstrip().endswith("}\n}") or t.rstrip().endswith("}"))
        self.assertLess(t.index('"events"'), t.index('"voice"'))
        S.write(json.loads(t) | {}, p.with_name("b.json"))
        self.assertEqual(json.loads(p.with_name("b.json").read_text("utf-8")), json.loads(t))


if __name__ == "__main__":
    unittest.main()
