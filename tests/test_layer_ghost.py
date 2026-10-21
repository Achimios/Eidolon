#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多层重影（tools/layer_ghost.py）的数值自检：找「必须相等」的量。"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import layer_ghost as G  # noqa: E402

# 什么都不加的参数：palette none、无纹理 / 散射 / bloom / 色调 / 黑位 / 色散
BARE = dict(palette="none", grain=0.0, scatter=0.0, bloom=0.0, tone=0.0, black=0.0, chan_sep=0.0, unit=1.0, tex_rgb=0)


def rand_img(H=64, W=96, seed=0):
    return np.random.default_rng(seed).random((H, W, 3)).astype(np.float32)


class TestLayerGhost(unittest.TestCase):
    def test_identity_single_layer(self):
        """N = 1、无纹理 / 散射 / bloom / 色调 → 输出 = 输入（逐像素严格相等）"""
        img = rand_img()
        out = G.run(img, G.Params(n_layers=1, n_vis=1, **BARE))
        np.testing.assert_array_equal(out, img)

    def test_layer_offsets_exact(self):
        """单个亮点 → 第 k 层的拷贝恰好落在 y0 + round(spacing·k^growth)，权重 = w_k；通道错位 R −s·k / B +s·k"""
        H, W, y0, x0 = 80, 40, 10, 20
        img = np.zeros((H, W, 3), np.float32)
        img[y0, x0] = 1.0
        p = G.Params(n_layers=4, n_vis=4, spacing=6.0, growth=1.5, falloff=0.7, edge="zero", **BARE)
        out = G.run(img, p)
        w = G.layer_weights(p)
        offs = G.layer_offsets(p, 1.0)
        self.assertEqual([o[0] for o in offs], [0, 6, int(round(6 * 2 ** 1.5)), int(round(6 * 3 ** 1.5))])
        for k, (dy, dx, _) in enumerate(offs):
            np.testing.assert_allclose(out[y0 + dy, x0 + dx], w[k], rtol=1e-6)
        self.assertAlmostEqual(float(out.sum()), 3.0, places=5)     # 3 个通道各自 Σw = 1
        # 通道错位
        p2 = G.Params(n_layers=3, n_vis=3, spacing=10.0, growth=1.0, falloff=1.0, edge="zero",
                      **{**BARE, "chan_sep": 2.0})
        out2 = G.run(img, p2)
        for k in range(3):
            self.assertAlmostEqual(float(out2[y0 + 10 * k - 2 * k, x0, 0]), 1 / 3, places=6)   # R
            self.assertAlmostEqual(float(out2[y0 + 10 * k, x0, 1]), 1 / 3, places=6)           # G
            self.assertAlmostEqual(float(out2[y0 + 10 * k + 2 * k, x0, 2]), 1 / 3, places=6)   # B

    def test_shift_is_exact(self):
        a = rand_img(20, 30)[..., 0]
        s = G.shift(a, 3, -2)
        np.testing.assert_array_equal(s[3:, :-2], a[:-3, 2:])
        self.assertTrue((s[:3] == 0).all() and (s[:, -2:] == 0).all())
        r = G.shift(a, 3, 0, "replicate")
        np.testing.assert_array_equal(r[3:], a[:-3])
        np.testing.assert_array_equal(r[:3], np.repeat(a[:1], 3, 0))

    def test_weights_and_tints_normalised(self):
        """Σ w_k = 1；每个通道 Σ_k w_k·tint_k,c = 1（层全重合处是白的，亮度守恒）"""
        for pal in ("cold", "ref", "none"):
            for nv in (1.0, 2.4, 6.0):
                p = G.Params(palette=pal, n_layers=6, n_vis=nv, tint_sat=-1)
                w = G.layer_weights(p)
                self.assertAlmostEqual(float(w.sum()), 1.0, places=9)
                self.assertTrue((w[int(np.ceil(nv)):] == 0).all())
                t = G.layer_tints(p, w)
                np.testing.assert_allclose((w[:, None] * t).sum(0), 1.0, rtol=1e-5)

    def test_texture_mask_mean_preserves_brightness(self):
        """纹理蒙版均值 = 1；乘到一张平的灰图上，整体均值 ≈ 原亮度（< 1 %）"""
        bank = G.TextureBank()
        rng = np.random.default_rng(3)
        z = G.frame_texture(bank, 180, 320, 1.0, rng)
        self.assertAlmostEqual(float(z.mean()), 0.0, places=3)
        self.assertAlmostEqual(float(z.std()), 1.0, places=3)
        T = G.texture_mask(z, 0.5, 1.3)
        self.assertAlmostEqual(float(T.mean()), 1.0, places=5)
        self.assertGreater(float(T.std()), 0.2)                     # 真有纹理
        np.testing.assert_array_equal(G.texture_mask(z, 0.0, 1.3), np.ones_like(z))
        img = np.full((180, 320, 3), 0.1, np.float32)     # 暗一点：输出最后 clip 到 1，亮的平图会被削掉一截
        for mode in ("carry", "screen"):
            out = G.run(img, G.Params(n_layers=4, n_vis=4, tex_mode=mode, **{**BARE, "grain": 0.6, "tex_rgb": 1}), bank)
            self.assertLess(abs(float(out.mean()) / 0.1 - 1), 0.01, mode)
            self.assertGreater(float(out.std()), 0.01)

    def test_texture_not_a_lattice(self):
        """纹理来自真实照片、不是规则晶格：横向自相关 2 px 外 ≈ 0（无周期）；
        竖向继承参考图自己的 ~9 px 准周期（参考图 = 带纹理的图竖向错 ~9.2 px 叠了多层），峰在 8–10 px"""
        bank = G.TextureBank()
        z = G.frame_texture(bank, 256, 256, 1.0, np.random.default_rng(1))
        F = np.fft.fft2(z - z.mean())
        ac = np.real(np.fft.ifft2(np.abs(F) ** 2))
        ac /= ac[0, 0]
        self.assertLess(float(np.abs(ac[0, 3:64]).max()), 0.2)
        self.assertIn(int(np.argmax(ac[4:30, 0])) + 4, (8, 9, 10))
        self.assertLess(float(ac[9, 0]), 0.95)                     # 准周期，不是严格重复

    def test_scatter_conserves_energy(self):
        """散射只搬光：s > 0 时（零填充、内容远离画边）总光量不变"""
        img = np.zeros((200, 120, 3), np.float32)
        img[80:110, 50:70] = rand_img(30, 20)
        out = G.run(img, G.Params(n_layers=3, n_vis=3, spacing=5, edge="zero",
                                  **{**BARE, "scatter": 0.6, "scatter_r": 6.0, "aniso": 0.3}))
        self.assertAlmostEqual(float(out.sum()), float(img.sum()), delta=1e-3 * float(img.sum()))

    def test_knobs(self):
        """倍率旋钮：sep_mul / chan_g 的通道位移逐像素精确；spacing_mul 乘层距；hl_smooth 平图均值仍 = 1；tex_scale < 1 纹粒更细"""
        H, W, y0, x0 = 120, 40, 10, 20
        img = np.zeros((H, W, 3), np.float32)
        img[y0, x0] = 1.0
        p = G.Params(n_layers=3, n_vis=3, spacing=10.0, growth=1.0, falloff=1.0, edge="zero",
                     **{**BARE, "chan_sep": 2.0, "sep_mul": 1.5, "chan_g": 0.5, "spacing_mul": 1.2})
        out = G.run(img, p)
        for k in range(3):
            dy, s = int(round(12 * k)), 3.0 * k
            self.assertAlmostEqual(float(out[y0 + dy - int(round(s)), x0, 0]), 1 / 3, places=6)
            self.assertAlmostEqual(float(out[y0 + dy + int(round(0.5 * s)), x0, 1]), 1 / 3, places=6)
            self.assertAlmostEqual(float(out[y0 + dy + int(round(s)), x0, 2]), 1 / 3, places=6)
        bank = G.TextureBank()
        flat = np.full((180, 320, 3), 0.1, np.float32)
        o = G.run(flat, G.Params(n_layers=3, n_vis=3, **{**BARE, "grain": 0.6, "hl_smooth": 0.8, "tex_scale": 0.7}), bank)
        self.assertLess(abs(float(o.mean()) / 0.1 - 1), 0.01)
        def corr1(scale):
            z = G.frame_texture(bank, 256, 256, scale, np.random.default_rng(2))
            return float((z[:, 1:] * z[:, :-1]).mean())
        self.assertLess(corr1(0.5), corr1(1.0))                    # 相邻像素更不像 = 纹粒更细
        self.assertEqual(G.parse_set("sep_mul=2,n_layers=7"), {"sep_mul": 2.0, "n_layers": 7})

    def test_versions_A_B(self):
        """A 回忆：阶梯停帧帧号；B 清晰：g = 0 → 只剩第 0 层（无纹理 / 散射 / bloom）= 干净画面；x 轴分裂的层位移逐像素精确"""
        idx = [G.hold_index(i, 10) for i in range(12)]
        self.assertEqual(idx, [0, 0, 0, 2, 2, 4, 4, 4, 7, 7, 9, 9])   # 每 2.4 帧换一张源帧
        self.assertEqual([G.hold_index(i, 0) for i in range(5)], list(range(5)))
        img = rand_img(60, 80)
        q = G.params_from_sense(0.9, 0.9, 0.9, 0.0, G.Params(style="split", **BARE))
        self.assertEqual((q.n_vis, q.scatter, q.grain, q.bloom, q.chan_sep), (1.0, 0.0, 0.0, 0.0, 0.0))
        np.testing.assert_array_equal(G.run(img, q), img)
        H, W, y0, x0 = 30, 120, 15, 10
        dot = np.zeros((H, W, 3), np.float32)
        dot[y0, x0] = 1.0
        px = G.Params(n_layers=3, n_vis=3, spacing=8.0, growth=1.0, falloff=1.0, edge="zero", axis="x",
                      **{**BARE, "chan_sep": 2.0})
        o = G.run(dot, px)
        for k in range(3):
            self.assertAlmostEqual(float(o[y0, x0 + 8 * k - 2 * k, 0]), 1 / 3, places=6)
            self.assertAlmostEqual(float(o[y0, x0 + 8 * k, 1]), 1 / 3, places=6)
            self.assertAlmostEqual(float(o[y0, x0 + 8 * k + 2 * k, 2]), 1 / 3, places=6)
        self.assertEqual([a for a, _ in G.split_schedule([(1, 1)] * 4, "alt")], ["y", "x", "y", "x"])

    def test_full_style_equals_still(self):
        """全强度：g = 1 → 参数和静帧（BEST on cold）一模一样，与威胁 / 张力 / 饥饿无关；g 只缩放层距 / 色散 / 散射 / bloom"""
        base = G.Params(**{**G.PRESETS["cold"], **G.BEST, "style": "full"})
        q = G.params_from_sense(0.1, 0.9, 0.3, 1.0, base)
        for f in ("spacing", "chan_sep", "scatter_r", "bloom", "n_layers", "n_vis", "grain", "scatter", "tex_scale"):
            self.assertEqual(getattr(q, f), getattr(base, f), f)
        r = G.params_from_sense(0.1, 0.9, 0.3, 1.15, base)
        self.assertAlmostEqual(r.spacing, base.spacing * 1.15)
        self.assertEqual(r.n_vis, base.n_vis)

    def test_tone_curve(self):
        x = np.linspace(0, 1, 11)
        np.testing.assert_array_equal(G.tone_curve(x, 0.0), x)
        y = G.tone_curve(x, 1.2)
        self.assertAlmostEqual(float(y[-1]), 1.0, places=9)
        self.assertTrue((np.diff(y) > 0).all())

    def test_pulses_threshold_and_gap(self):
        """威胁峰 < thr 不触发；两次触发间隔 ≥ min_gap（连续 0.4 s 的起音 → 隔一个）；脉冲之间 = base"""
        fps = 24
        t = np.arange(0, 4, 1 / fps)
        threat = np.full(len(t) + 10, 0.8)
        threat[int(0.5 * fps) - 2:int(0.5 * fps) + 3] = 0.3
        onsets = [0.5, 1.0, 1.4, 1.8, 2.2, 2.6]
        g, fired = G.pulses(t, onsets, threat, thr=0.6, min_gap=0.45, base=0.25)
        self.assertEqual([o for o, _ in fired], [1.0, 1.8, 2.6])
        self.assertAlmostEqual(float(g[0]), 0.25)
        self.assertAlmostEqual(float(g[int(round(1.0 * fps))]), 1.0)
        self.assertGreaterEqual(float(g.min()), 0.25)

    def test_sense_mapping_monotone(self):
        lo = G.params_from_sense(0.1, 0.1, 0.1, 1.0)
        hi = G.params_from_sense(0.9, 0.9, 0.9, 1.0)
        self.assertLess(lo.spacing, hi.spacing)
        self.assertLess(lo.n_vis, hi.n_vis)
        self.assertLess(lo.chan_sep, hi.chan_sep)
        self.assertLess(lo.scatter_r, hi.scatter_r)
        quiet = G.params_from_sense(0.9, 0.9, 0.9, 0.25)
        self.assertLess(quiet.spacing, hi.spacing)
        self.assertLess(quiet.n_vis, hi.n_vis)


if __name__ == "__main__":
    unittest.main()
