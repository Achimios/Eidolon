#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<👁视觉层-抠像-BiRefNet（ZhengPeng7/BiRefNet，HF trust_remote_code）图片 / 图片目录 / 视频 → 前景 alpha PNG。
要在本仓库的 .venv 里跑（torch + torchvision cu128 + transformers + timm + kornia）：
  .venv/Scripts/python.exe tools/matte.py 输入 [--out 路径或目录] [--rgba] [--size 1024] [--every 1]
  · 输入是图片：默认写到同目录 <名>_alpha.png
  · 输入是目录 / 视频：默认写到 <名>_alpha/000000.png ...（视频用 imageio_ffmpeg 解帧，--every N 隔帧抽）
  · --rgba：额外写一张带透明通道的抠图 <名>_rgba.png
逐帧独立推理，没有时序一致性（视频会有轻微闪烁；要跟踪得换 SAM2）。
"""
import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VID_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class Matter:
    def __init__(self, repo="ZhengPeng7/BiRefNet", size=1024):
        import torch
        from transformers import AutoModelForImageSegmentation
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.dev == "cuda" else torch.float32
        self.size = size
        m = AutoModelForImageSegmentation.from_pretrained(repo, trust_remote_code=True)
        self.model = m.to(self.dev, self.dtype).eval()

    def __call__(self, im: Image.Image) -> Image.Image:
        """RGB 图 → 同尺寸 L 模式 alpha"""
        torch = self.torch
        W, H = im.size
        x = (np.asarray(im.convert("RGB").resize((self.size, self.size), Image.BILINEAR), np.float32) / 255 - MEAN) / STD
        x = torch.from_numpy(x.transpose(2, 0, 1))[None].to(self.dev, self.dtype)
        with torch.no_grad():
            a = self.model(x)[-1].sigmoid()
        a = torch.nn.functional.interpolate(a.float(), size=(H, W), mode="bilinear", align_corners=False)[0, 0]
        return Image.fromarray((a.clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8), "L")


def video_frames(path, every=1):
    import imageio_ffmpeg
    rd = imageio_ffmpeg.read_frames(str(path))
    meta = next(rd)
    W, H = meta["size"]
    for i, buf in enumerate(rd):
        if i % every == 0:
            yield i, Image.frombytes("RGB", (W, H), buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", default=None)
    ap.add_argument("--rgba", action="store_true")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--repo", default="ZhengPeng7/BiRefNet")
    a = ap.parse_args()
    src = Path(a.src)
    t0 = time.time()
    mt = Matter(a.repo, a.size)
    t1 = time.time()
    if src.is_file() and src.suffix.lower() in IMG_EXT:
        out = Path(a.out) if a.out else src.with_name(src.stem + "_alpha.png")
        im = Image.open(src).convert("RGB")
        al = mt(im); al.save(out)
        if a.rgba:
            rgba = im.copy(); rgba.putalpha(al); rgba.save(out.with_name(src.stem + "_rgba.png"))
        n = 1
    else:
        od = Path(a.out) if a.out else src.with_name(src.stem + "_alpha")
        od.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            items = ((i, Image.open(p).convert("RGB")) for i, p in enumerate(sorted(q for q in src.iterdir() if q.suffix.lower() in IMG_EXT)))
        elif src.suffix.lower() in VID_EXT:
            items = video_frames(src, a.every)
        else:
            raise SystemExit(f"不认识的输入：{src}")
        n = 0
        for i, im in items:
            al = mt(im); al.save(od / f"{i:06d}.png")
            if a.rgba:
                rgba = im.copy(); rgba.putalpha(al); rgba.save(od / f"{i:06d}_rgba.png")
            n += 1
        out = od
    t2 = time.time()
    gpu = ""
    if mt.dev == "cuda":
        gpu = f"  峰值显存 {mt.torch.cuda.max_memory_allocated() / 2**20:.0f}MB"
    print(f"BiRefNet on {mt.dev}: 载入 {t1 - t0:.1f}s  推理 {n} 张 {t2 - t1:.2f}s（{(t2 - t1) / max(n, 1):.3f}s/张）{gpu}  -> {out}")


if __name__ == "__main__":
    main()
