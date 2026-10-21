#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""四维生长（浮现：从手长出来）自检 —— 合成的体素形状，不读模型。python -X utf8 -m unittest discover -s tests -q"""
import importlib.util
import math
import sys
import unittest
from pathlib import Path

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import four_d as F  # noqa: E402

N = 56


def _box(x, y, z, cx, cy, cz, hx, hy, hz):
    qx, qy, qz = (x - cx).abs() - hx, (y - cy).abs() - hy, (z - cz).abs() - hz
    out = torch.sqrt(qx.clamp(min=0) ** 2 + qy.clamp(min=0) ** 2 + qz.clamp(min=0) ** 2)
    return out + torch.maximum(qx, torch.maximum(qy, qz)).clamp(max=0)


def _u_shape(x, y, z):
    """U 形：两根竖柱（x = ±0.2，缝宽 0.4 − 2·0.06 = 0.28）+ 底下一根横梁"""
    a = _box(x, y, z, -0.2, 0.1, 0, 0.06, 0.5, 0.06)
    b = _box(x, y, z, 0.2, 0.1, 0, 0.06, 0.5, 0.06)
    c = _box(x, y, z, 0.0, -0.45, 0, 0.26, 0.06, 0.06)
    return torch.minimum(a, torch.minimum(b, c))


def _fake_mesh(fn):
    m = F.MeshSDF("synthetic.glb", n=N)
    ax = torch.linspace(-F.MESH_B, F.MESH_B, N, device=F.DEV)
    Z, Y, X = torch.meshgrid(ax, ax, ax, indexing="ij")
    m.vol = fn(X, Y, Z)[None, None].contiguous()
    return m


@unittest.skipUnless(HAS_TORCH, "four_d 需要 torch（.venv）")
class TestGrow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.msh = _fake_mesh(_u_shape)
        cls.gr = F.Grow(cls.msh, seed=(-0.2, 0.58, 0.0), apex=1e-4, smooth=0)
        cls.gr.build()
        cls.obj = F.grow_obj(cls.msh)
        cls.obj.grow = cls.gr
        cls.obj.f, cls.obj.cap, cls.obj.tracer, cls.obj.lb, cls.obj.emit = cls.gr.f, cls.gr.cap, cls.gr.tracer, cls.gr.lb, cls.gr.emit

    def _at(self, G, x, y, z):
        t = lambda v: torch.tensor([v], device=F.DEV)  # noqa: E731
        return float(self.gr._samp(G, t(x), t(y), t(z)))

    def test_geodesic_goes_along_structure_not_across_gap(self):
        """必须相等的量：右柱顶的测地距离 ≈ 下去 + 横过 + 上来 的路径长；远大于穿过缝的直线距离 0.4"""
        d = self._at(self.gr.D, 0.2, 0.58, 0.0)
        path = (0.58 + 0.45) * 2 + 0.4
        self.assertGreater(d, 3 * 0.4)
        self.assertLess(abs(d - path) / path, 0.10)

    def test_straight_run_equals_euclid(self):
        """沿柱子直走：测地距离 = 欧氏距离（误差 < 1.5 体素）"""
        for y in (0.4, 0.1, -0.2):
            self.assertAlmostEqual(self._at(self.gr.D, -0.2, y, 0.0), 0.58 - y, delta=1.5 * self.gr.h)

    def test_lipschitz_hull_is_below_and_lipschitz(self):
        D, DL, ins = self.gr.D[0, 0], self.gr.DL[0, 0], self.gr.inside
        self.assertTrue(bool((DL[ins] <= D[ins] + 1e-6).all()))
        for dim in range(3):
            diff = (DL - torch.roll(DL, 1, dim)).abs()
            sl = [slice(None)] * 3; sl[dim] = slice(1, None)
            self.assertLessEqual(float(diff[tuple(sl)].max()), self.gr.h * 1.0001)

    def test_lower_bound_is_a_lower_bound(self):
        """lb(p) ≤ p 到截面（已长出的体素中心）的真实距离 + 一个体素（插值）"""
        c = 0.9
        occ = self.gr.occupancy(c)
        ax = torch.linspace(-F.MESH_B, F.MESH_B, N, device=F.DEV)
        Z, Y, X = torch.meshgrid(ax, ax, ax, indexing="ij")
        P = torch.stack([X[occ], Y[occ], Z[occ]], 1)
        g = torch.Generator(device="cpu").manual_seed(3)
        q = (torch.rand(3000, 3, generator=g) * 1.6 - 0.8).to(F.DEV)
        true = torch.cdist(q, P).min(1).values
        lb = self.gr.lb(c)(q)
        self.assertLessEqual(float((lb - true).max()), self.gr.h)

    def test_birth_only_at_the_seed(self):
        """w = w0：只剩 seed 那一个体素；w0 + ε：只在 seed 附近（欧氏距离 ≤ r + 1 体素）"""
        self.assertEqual(int(self.gr.occupancy(0.0).sum()), 1)
        ax = torch.linspace(-F.MESH_B, F.MESH_B, N, device=F.DEV)
        Z, Y, X = torch.meshgrid(ax, ax, ax, indexing="ij")
        s = self.gr.seed_xyz
        for r in (1.5 * self.gr.h, 3 * self.gr.h):     # 体素 h = 0.041：比 1 个体素小的 r 只有 seed 一个点
            occ = self.gr.occupancy(r)
            self.assertGreater(int(occ.sum()), 1)
            dist = torch.sqrt((X - s[0]) ** 2 + (Y - s[1]) ** 2 + (Z - s[2]) ** 2)
            self.assertLessEqual(float(dist[occ].max()), r + 1.01 * self.gr.h * math.sqrt(3))

    def test_volume_monotone_and_final_equals_body(self):
        vols = [int(self.gr.occupancy(c).sum()) for c in torch.linspace(0, self.gr.Dmax * 1.05, 40).tolist()]
        self.assertTrue(all(a <= b for a, b in zip(vols, vols[1:])))
        self.assertTrue(torch.equal(self.gr.occupancy(self.gr.Dmax), self.gr.inside))
        self.assertEqual(self.gr.unreached, 0)

    def test_tangency_only_at_birth(self):
        """红 = 4D 法向的 xyz 分量 → 0：锥尖（抹圆后）满足；生长中途的平切面上 = 1/√(1+K²) ≈ 0.707，不红"""
        gr = F.Grow(self.msh, seed=(-0.2, 0.58, 0.0), apex=3 * self.gr.h, smooth=0); gr.build()   # 抹圆半径要 ≥ 几个体素（模型 256³ 时 0.03 ≈ 3.4 体素）
        R = torch.eye(4, device=F.DEV)
        n = gr.n; i = gr.seed_idx
        ax = torch.linspace(-F.MESH_B, F.MESH_B, n).tolist()
        s = (ax[i % n], ax[(i // n) % n], ax[i // (n * n)])                 # seed 体素中心
        obj = F.Obj("g", "grow", "", gr.f, 1.0)
        q = torch.tensor([[s[0], s[1], s[2], 0.0005]], device=F.DEV)
        g4 = F.grad4(obj, R, q)
        self.assertLess(float(g4[0, :3].norm() / g4[0].norm()), 0.32)
        t1 = lambda v: torch.tensor([v], device=F.DEV)  # noqa: E731
        w = float(gr.dist(t1(-0.2), t1(0.08), t1(0.0)))              # 柱子中轴上、正好在前沿上的那一点
        q = torch.tensor([[-0.2, 0.08, 0.0, w]], device=F.DEV)
        g4 = F.grad4(obj, R, q)
        self.assertAlmostEqual(float(g4[0, :3].norm() / g4[0].norm()), 1 / math.sqrt(2), delta=0.08)

    def test_tracer_sees_grown_part_only(self):
        """竖直向下看柱子顶上方：长出前（r 很小时只有顶端）看得见左柱顶、看不见右柱顶；hit 处真截面 ≈ 0"""
        eye = torch.tensor([0.0, 0.0, 1.2], device=F.DEV)
        tg = torch.tensor([[-0.2, 0.4, 0.0], [0.2, 0.4, 0.0]], device=F.DEV)
        rd = tg - eye; rd = rd / rd.norm(dim=1, keepdim=True)
        hit, t = self.gr.tracer(0.4, eye, rd, 1.6)
        self.assertTrue(bool(hit[0])); self.assertFalse(bool(hit[1]))
        hit, t = self.gr.tracer(self.gr.Dmax, eye, rd, 1.6)
        self.assertTrue(bool(hit.all()))
        p = eye[None] + rd * t[:, None]
        g = F.slice_fn(self.obj, torch.eye(4, device=F.DEV), self.gr.Dmax)(p)
        self.assertLess(float(g.abs().max()), self.gr.h)

    def test_render_rejects_rotation(self):
        with self.assertRaises(ValueError):
            F.render(self.obj, F.rot4(xw=0.3), 0.5, 32, 18, "sculpt")

    def test_emerge_track(self):
        sp = F.EmergeSpec(2.0, size=(64, 36), d0=0.5, d1=8.0)
        tr = F.emerge_track(self.gr, sp)
        self.assertEqual(len(tr), 48)
        rs = [k["r"] for k in tr]
        self.assertTrue(all(a < b for a, b in zip(rs, rs[1:])))       # 前沿一直在走（不停）
        self.assertGreater(tr[0]["v"], 0)
        self.assertLessEqual(max(k["r"] for k in tr if k["t"] <= sp.full), self.gr.Dmax + 1e-6)   # 长全之前没超
        self.assertGreaterEqual(tr[-1]["r"], self.gr.Dmax)                                         # 最后一帧整个身体都在
        self.assertTrue(torch.equal(self.gr.occupancy(tr[-1]["c"]), self.gr.inside))
        ld = [math.log(k["dist"]) for k in tr]
        st = [b - a for a, b in zip(ld, ld[1:])]
        self.assertLess(max(st) - min(st), 1e-9)                        # 对数匀速拉远
        self.assertAlmostEqual(tr[0]["dist"], 0.5); self.assertAlmostEqual(tr[-1]["dist"], 8.0)
        s0 = self.gr.seed_xyz                                            # 镜头从 seed 开始，最后对准整个身体的重心
        self.assertLess(math.dist(tr[0]["target"], s0), 2 * self.gr.h)
        self.assertLess(math.dist(tr[-1]["target"], self.gr.center), 0.1)
        with self.assertRaises(ValueError):
            F.EmergeSpec(2.0, gv_ratio=1.5)
        with self.assertRaises(ValueError):
            F.EmergeSpec(2.0, d0=2.0, d1=1.0)


if __name__ == "__main__":
    unittest.main()
