#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""信息图（tools/info_graph.py）的自检：找「必须相等」的量 —— 符号说真话，所以每个画出来的数都要能对回去。"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import info_graph as G  # noqa: E402


def flood_partition(mask):
    """暴力 BFS 连通域（4 邻接）→ 每个像素的分组代表（最小线性下标）"""
    H, W = mask.shape
    rep = -np.ones((H, W), int)
    for y in range(H):
        for x in range(W):
            if mask[y, x] and rep[y, x] < 0:
                st, comp = [(y, x)], []
                rep[y, x] = 0
                while st:
                    a, b = st.pop()
                    comp.append((a, b))
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        u, v = a + da, b + db
                        if 0 <= u < H and 0 <= v < W and mask[u, v] and rep[u, v] < 0:
                            rep[u, v] = 0
                            st.append((u, v))
                r = min(a * W + b for a, b in comp)
                for a, b in comp:
                    rep[a, b] = r
    return rep


class TestGraphPrimitives(unittest.TestCase):
    def test_label_cc_matches_bfs(self):
        """连通域标记 == 暴力 BFS 的划分（同组同标号、异组异标号），含蛇形长连通域（考验传播收敛）"""
        rng = np.random.default_rng(0)
        masks = [rng.random((40, 50)) > 0.55 for _ in range(4)]
        snake = np.zeros((31, 31), bool)
        snake[::2, :] = True
        for r in range(1, 31, 2):
            snake[r, 30 if (r // 2) % 2 == 0 else 0] = True
        masks.append(snake)
        for m in masks:
            lab, n = G.label_cc(m)
            ref = flood_partition(m)
            self.assertEqual(n, len(np.unique(ref[m])))
            pairs = set(zip(lab[m].tolist(), ref[m].tolist()))
            self.assertEqual(len(pairs), n)                      # 双射
            self.assertTrue((lab[~m] == -1).all())
        lab, n = G.label_cc(snake)
        self.assertEqual(n, 1)

    def test_components_stats(self):
        """矩形块：面积、质心、外接框逐个对"""
        lum = np.zeros((60, 80))
        lum[10:20, 30:46] = 0.9
        lum[40:43, 5:8] = 0.7
        comps, _ = G.components(lum, 0.5, 1)
        comps = sorted(comps, key=lambda c: -c["a"])
        self.assertEqual(comps[0]["a"], 160)
        self.assertEqual(comps[0]["box"], (30, 10, 45, 19))
        np.testing.assert_allclose(comps[0]["c"], (37.5, 14.5))
        self.assertEqual(comps[1]["a"], 9)
        np.testing.assert_allclose(comps[0]["lum"], 0.9)

    def test_mst_weight_equals_kruskal(self):
        """Prim 的总权 == Kruskal 的总权；边数 n−1；连通"""
        rng = np.random.default_rng(3)
        for n in (2, 5, 17, 40):
            P = rng.random((n, 2)) * 100
            E = G.mst_prim(P)
            self.assertEqual(len(E), n - 1)
            wp = sum(np.hypot(*(P[a] - P[b])) for a, b in E)
            pairs = sorted((np.hypot(*(P[i] - P[j])), i, j) for i in range(n) for j in range(i + 1, n))
            par = list(range(n))

            def f(x):
                while par[x] != x:
                    par[x] = par[par[x]]
                    x = par[x]
                return x
            wk = 0.0
            for w, i, j in pairs:
                a, b = f(i), f(j)
                if a != b:
                    par[a] = b
                    wk += w
            self.assertAlmostEqual(wp, wk, places=6)

    def test_dijkstra_equals_bellman_ford(self):
        rng = np.random.default_rng(5)
        n = 25
        adj = [[] for _ in range(n)]
        edges = []
        for i in range(n):
            for j in rng.choice(n, 3, replace=False):
                if i != j:
                    w = float(rng.random() * 5)
                    adj[i].append((int(j), w)); adj[int(j)].append((i, w)); edges.append((i, int(j), w))
        order = G.dijkstra_order(n, adj, 0)
        d = {u: dd for u, dd, _ in order}
        bf = [np.inf] * n; bf[0] = 0.0
        for _ in range(n):
            for a, b, w in edges:
                bf[b] = min(bf[b], bf[a] + w); bf[a] = min(bf[a], bf[b] + w)
        for u, dd in d.items():
            self.assertAlmostEqual(dd, bf[u], places=9)
        ds = [dd for _, dd, _ in order]
        self.assertEqual(ds, sorted(ds))                          # 弹出顺序 = 距离单调不减

    def test_bezier(self):
        pts, s = G.bezier((0, 0), (100, 0), 0.0)
        np.testing.assert_allclose(pts[0], (0, 0)); np.testing.assert_allclose(pts[-1], (100, 0))
        self.assertAlmostEqual(s[-1], 100.0, places=6)
        pts, s = G.bezier((0, 0), (100, 0), 0.25)
        self.assertAlmostEqual(abs(pts[len(pts) // 2][1]), 12.5, delta=0.6)   # 二次贝塞尔中点 = 控制点偏移的一半

    def test_tracker_keeps_id(self):
        tr = G.Tracker(gate=10)
        ids = []
        for f in range(6):
            cs = tr.step([dict(c=(10 + 3 * f, 20.0)), dict(c=(60.0, 5 + 2 * f))], f)
            ids.append(tuple(c["id"] for c in cs))
        self.assertEqual(len(set(ids)), 1)
        cs = tr.step([dict(c=(200.0, 200.0))], 6)                # 跳太远 → 新 ID
        self.assertEqual(cs[0]["id"], 2)

    def test_component_tree_containment(self):
        """高阈值连通域的每个像素都落在它记下的父（低一层）连通域里"""
        yy, xx = np.mgrid[0:90, 0:120]
        lum = np.exp(-((xx - 40) ** 2 + (yy - 45) ** 2) / 300) + 0.9 * np.exp(-((xx - 72) ** 2 + (yy - 42) ** 2) / 160)
        frames = (np.repeat(lum[None, ..., None], 3, -1) / lum.max() * 255).astype(np.uint8)
        data, trk = G.analyse_track(frames, [0.2, 0.5, 0.8], [1, 1, 1], gate=10)
        lv = data[0]
        self.assertEqual(len(lv[0]), 1)                          # 低阈值两峰连成一块
        self.assertEqual(len(lv[2]), 2)
        top = {c["id"]: c for c in lv[1]}
        for c in lv[2]:
            p = top[c["parent"]]
            x0, y0, x1, y1 = c["box"]
            ys, xs = np.nonzero(c["mask"])
            X, Y = xs + x0, ys + y0
            px0, py0, px1, py1 = p["box"]
            self.assertTrue(((X >= px0) & (X <= px1) & (Y >= py0) & (Y <= py1)).all())
            self.assertTrue(p["mask"][Y - py0, X - px0].all())


class TestMemoryGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not G.SENSE.exists():
            raise unittest.SkipTest("没有 感官.json")
        cls.seg = (115.51, 128.31)
        cls.mem = G.build_memory(seg=cls.seg)
        cls.sch = G.schedule(cls.mem, cls.seg)

    def test_features_are_channel_means(self):
        """节点特征 == 感官.json 在该镜头区间的通道平均（直接重算一个）"""
        _, fps, ch, ev = G.load_sense()
        nd = self.mem.nodes[10]
        a, b = int(round(nd["t0"] * fps)), int(round(nd["t1"] * fps))
        self.assertAlmostEqual(nd["feat"]["tension"], float(ch["tension"][a:b].mean()), places=12)
        on = ev["onsets"]
        self.assertAlmostEqual(nd["feat"]["rho"], ((on >= nd["t0"]) & (on < nd["t1"])).sum() / (nd["t1"] - nd["t0"]), places=12)

    def test_path_cost_and_confidence(self):
        """路径边权之和 == Dijkstra 给 John 的距离；CONFIDENCE == (1 + cos) / 2"""
        m = self.mem
        w = {(min(a, b), max(a, b)): ww for a, b, _, ww in m.edges}
        cost = sum(w[(min(a, b), max(a, b))] for a, b in zip(m.path[:-1], m.path[1:]))
        dj = next(d for u, d, _ in m.order if u == m.john)
        self.assertAlmostEqual(cost, dj, places=9)
        self.assertEqual(m.path[0], m.now); self.assertEqual(m.path[-1], m.john)
        zn, zj = np.array(m.nodes[m.now]["z"]), np.array(m.nodes[m.john]["z"])
        self.assertAlmostEqual(m.conf, (1 + zn @ zj / np.linalg.norm(zn) / np.linalg.norm(zj)) / 2, places=9)
        self.assertIn("走廊背影", m.nodes[m.john]["src"])

    def test_edges(self):
        m = self.mem
        n_story = sum(e[2] == "story" for e in m.edges)
        self.assertEqual(n_story, len(m.nodes) - 2)               # 剪辑表镜头数 − 1（「此刻」不在故事线里）
        for a, b, _, ww in m.edges:
            self.assertAlmostEqual(ww, float(np.linalg.norm(np.array(m.nodes[a]["z"]) - np.array(m.nodes[b]["z"]))), places=9)

    def test_pca_projection(self):
        """PCA 坐标 == (z − 记忆均值) · 载荷；载荷正交归一"""
        m = self.mem
        V = m.pcs
        np.testing.assert_allclose(V @ V.T, np.eye(2), atol=1e-9)
        Z = np.array([nd["z"] for nd in m.nodes[:-1]])
        np.testing.assert_allclose((Z - Z.mean(0)) @ V.T, np.array([nd["pca"] for nd in m.nodes[:-1]]), atol=1e-9)

    def test_schedule_on_real_onsets(self):
        """每个弹出时刻都是 06 段里的真实起音，强度 ≥ 阈值，且按 Dijkstra 顺序单调"""
        _, fps, ch, ev = G.load_sense()
        on = set(np.round(ev["onsets"], 4))
        ts = []
        for u, d, _ in self.mem.order:
            if u in self.sch["pop"]:
                t, s = self.sch["pop"][u]
                self.assertIn(round(t + self.seg[0], 4), on)
                self.assertGreaterEqual(s, 0.3)
                ts.append(t)
        self.assertEqual(ts, sorted(ts))
        self.assertIn(self.mem.john, self.sch["pop"])


class TestMemoryV4(unittest.TestCase):
    """v4 只改节奏：顺序 / 数值和 v3 一样；锚点踩在指定的真实起音上"""
    @classmethod
    def setUpClass(cls):
        if not G.SENSE.exists():
            raise unittest.SkipTest("没有 感官.json")
        cls.seg = (115.51, G.V4["end_t"])
        cls.mem = G.build_memory(seg=cls.seg)
        cls.sch = G.schedule_v4(cls.mem, cls.seg)

    def test_order_preserved_and_stops_at_john(self):
        order = [u for u, _, _ in self.mem.order]
        pops = sorted(self.sch["pop"].items(), key=lambda kv: kv[1][0])
        self.assertEqual([u for u, _ in pops], order[:self.sch["jk"] + 1])   # 弹出顺序 == Dijkstra 顺序，到 John 停
        ts = [tt for _, (tt, _) in pops]
        self.assertTrue(all(b > a for a, b in zip(ts, ts[1:])))

    def test_anchors(self):
        s0 = self.seg[0]
        self.assertAlmostEqual(self.sch["pop"][self.sch["unstable"]][0] + s0, 118.552, places=2)
        self.assertAlmostEqual(self.sch["pop"][self.mem.john][0] + s0, 127.741, places=3)
        self.assertEqual(self.sch["jk"] + 1, 28)
        zj = np.array(self.mem.nodes[self.mem.john]["z"])
        zu = np.array(self.mem.nodes[self.sch["unstable"]]["z"])
        cu = zu @ zj / np.linalg.norm(zu) / np.linalg.norm(zj)
        for u in [u for u, _, _ in self.mem.order][1:12]:                    # 心魔 = 前 11 个里和 John 最不像的
            z = np.array(self.mem.nodes[u]["z"])
            self.assertGreaterEqual(z @ zj / np.linalg.norm(z) / np.linalg.norm(zj), cu - 1e-12)
        _, fps, ch, ev = G.load_sense()
        on = np.array(ev["onsets"])
        for h in G.V4["hops"]:
            self.assertLess(np.abs(on - h).min(), 0.002)                     # 四声都是真实起音

    def test_takens_delay(self):
        nd = self.mem.nodes[self.sch["unstable"]]
        P3, m = G.takens_points(nd["t0"], nd["t1"])
        tau = int(round(m["tau_ms"] * m["sr"] / 1000))
        np.testing.assert_allclose(P3[tau:, 1], P3[:-tau, 0], atol=1e-12)    # 第二坐标 == 第一坐标晚 τ
        self.assertLessEqual(np.abs(P3).max(), 1.0 + 1e-12)


class TestMemoryV5(unittest.TestCase):
    """v5：前馈镜头 —— 每个排程事件的镜头速度峰值离事件 ≤ 1 帧；John 在四声第一声弹出；弦两端固定"""
    @classmethod
    def setUpClass(cls):
        if not G.SENSE.exists():
            raise unittest.SkipTest("没有 感官.json")
        cls.seg = (115.51, G.V4["end_t"])
        cls.mem = G.build_memory(seg=cls.seg)
        cls.sch = G.schedule_v4(cls.mem, cls.seg, john_hop=0)

    def test_camera_peak_on_event(self):
        T = int(round(self.sch["end"] * G.FPS))
        cam, zs, ev = G.camera_ff(self.mem, self.sch, self.seg, T)
        al = G.camera_alignment(cam, zs, ev, T)
        self.assertGreater(len(al), 25)
        for t, k, d in al:
            self.assertLessEqual(abs(d), 1, (t + self.seg[0], k, d))

    def test_camera_v6_lead_and_cap(self):
        """v6：提前 2 帧起步（峰值在起音前 1 帧或当帧），第一段平均速度 ≤ 上限，第二段比第一段快"""
        T = int(round(self.sch["end"] * G.FPS))
        cam, zs, ev = G.camera_ff(self.mem, self.sch, self.seg, T, v6=True)
        al = G.camera_alignment(cam, zs, ev, T, lead=G.V6["lead_frames"])
        for t, k, d in al:
            self.assertIn(d, (-1, 0, 1), (t + self.seg[0], k, d))
        sp = np.hypot(*np.diff(cam, axis=0).T) * zs[1:] * G.FPS
        c = int(self.sch["cut"] * G.FPS)
        self.assertLessEqual(sp[:c].mean(), G.V6["cap1"])
        self.assertGreater(sp[c:].mean(), sp[:c].mean())

    def test_camera_v7(self):
        """v7：v1 式目标 + 前馈；速度峰值离事件 ≤ 1 帧；锁定后不冲进（变焦不超过锁定时的 1.05 倍）"""
        T = int(round(self.sch["end"] * G.FPS))
        cam, zs, ev = G.camera_ff_v7(self.mem, self.sch, self.seg, T)
        for t, k, d in G.camera_alignment(cam, zs, ev, T, lead=G.V7["lead_frames"]):
            self.assertIn(d, (-1, 0, 1), (t + self.seg[0], k, d))
        fl = int(self.sch["hops"][3] * G.FPS)
        self.assertLess(zs[-1], zs[fl] * 1.05)

    def test_camera_v8_never_still(self):
        """v8：镜头永不停（每帧屏幕位移 > 0.5 px），速度峰值仍贴事件"""
        T = int(round(self.sch["end"] * G.FPS))
        cam, zs, ev = G.camera_ff_v7(self.mem, self.sch, self.seg, T, v8=True)
        sp = np.hypot(*np.diff(cam, axis=0).T) * zs[1:]
        self.assertGreater(sp.min(), 0.5)
        for t, k, d in G.camera_alignment(cam, zs, ev, T, lead=G.V7["lead_frames"]):
            self.assertIn(d, (-1, 0, 1), (t + self.seg[0], k, d))

    def test_ease_shape(self):
        """快起慢收：E(0)=0、E(1)=1、速度单调减（起步那一刻最快）"""
        u = np.linspace(0, 1, 201)
        E = np.array([G.ease_ff(x) for x in u])
        self.assertAlmostEqual(E[0], 0.0); self.assertAlmostEqual(E[-1], 1.0)
        v = np.diff(E)
        self.assertTrue((np.diff(v) <= 1e-12).all())

    def test_john_on_first_sound(self):
        self.assertAlmostEqual(self.sch["pop"][self.mem.john][0] + self.seg[0], G.V4["hops"][0], places=3)
        order = [u for u, _, _ in self.mem.order]
        pops = sorted(self.sch["pop"].items(), key=lambda kv: kv[1][0])
        self.assertEqual([u for u, _ in pops], order[:self.sch["jk"] + 1])

    def test_string_fixed_ends(self):
        for dt in (0.0, 0.07, 0.3):
            y = G.string_offset(49, dt, 5.0)
            self.assertAlmostEqual(y[0], 0.0, places=9); self.assertAlmostEqual(y[-1], 0.0, places=9)
        self.assertLessEqual(np.abs(G.string_offset(49, 0.0, 5.0)).max(), 5.0 * 1.6)


if __name__ == "__main__":
    unittest.main()
