# -*- coding: utf-8 -*-
""">v<☁生成-即梦-出图 / 出视频：包 `dreamina` CLI（走订阅积分，OAuth 登录态在 CLI 自己那里，这里不碰凭证）。
⚠️生成都扣积分：提交前告诉 👤。flag 以 `dreamina <子命令> -h` 为准，能力和坑见 `_设计/接口_生成.md`「即梦」节。
返回的结果下载到作品文件夹，旁边写 `_jimeng_<submit_id>.json` 记 prompt / 模型 / 参数（和 ark.py 一样可追溯）。
    credit()                                        # 查积分，不扣
    image(prompt, out_dir, refs=[...])              # refs 空 = text2image，有 = image2image
    video(prompt, out_dir, first=, last=)           # 首尾帧 / 单图；都不给 = text2video
    video(prompt, out_dir, refs=[...], ref_videos=[...], ref_audios=[...])   # 全能参考
"""
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

BIN = shutil.which("dreamina") or str(Path.home() / "bin" / "dreamina.exe")


def _run(args, timeout=120):
    r = subprocess.run([BIN, *map(str, args)], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        raise RuntimeError(f"dreamina {args[0]} -> {r.returncode}: {out[:500]}")
    return out


def _flags(**kw):
    a = []
    for k, v in kw.items():
        if v is None:
            continue
        for x in (v if isinstance(v, (list, tuple)) else [v]):
            a.append(f"--{k}={x}")
    return a


def _submit_id(out):
    m = re.search(r'submit_id["\s:=]+([\w-]+)', out)
    if not m:
        raise RuntimeError(f"没拿到 submit_id：{out[:500]}")
    return m.group(1)


def wait(submit_id, out_dir, timeout=900, every=5):
    """等任务结束并下载；返回这次下载到的文件。"""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    before, t0 = set(out_dir.iterdir()), time.time()
    while time.time() - t0 < timeout:
        out = _run(["query_result", f"--submit_id={submit_id}", f"--download_dir={out_dir}"])
        m = re.search(r'gen_status["\s:=]+(\w+)', out)
        st = (m.group(1) if m else "").lower()
        if st in ("fail", "failed", "error"):
            raise RuntimeError(f"任务失败 {submit_id}：{out[:500]}")
        new = sorted(set(out_dir.iterdir()) - before)
        if st in ("success", "succeed", "done") or new:
            return new
        time.sleep(every)
    raise TimeoutError(f"{submit_id} 等了 {timeout} s 还没好；可以稍后 wait() 接着取，别重新提交（会再扣积分）")


def _go(cmd, flags, prompt, out_dir, meta, block):
    sid = _submit_id(_run([cmd, *flags, f"--prompt={prompt}"]))
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"_jimeng_{sid}.json").write_text(json.dumps({"cmd": cmd, "prompt": prompt, **meta}, ensure_ascii=False, indent=1), "utf-8")
    return wait(sid, out_dir) if block else sid


def credit():
    return _run(["user_credit"])


def image(prompt, out_dir, refs=None, model="5.0", res="2k", ratio="16:9", n=1, block=True):
    cmd = "image2image" if refs else "text2image"
    kw = dict(images=refs or None, model_version=model, resolution_type=res, ratio=ratio, generate_num=n)
    return _go(cmd, _flags(**kw), prompt, out_dir, kw, block)


def video(prompt, out_dir, first=None, last=None, refs=None, ref_videos=None, ref_audios=None,
          model="seedance2.5", res="480p", duration=5, ratio=None, block=True):
    if first and (refs or ref_videos or ref_audios):
        raise ValueError("首尾帧和全能参考二选一")
    if refs or ref_videos or ref_audios:
        cmd, kw = "multimodal2video", dict(image=refs, video=ref_videos, audio=ref_audios)
    elif last:
        cmd, kw = "frames2video", dict(first=first, last=last)
    elif first:
        cmd, kw = "image2video", dict(image=first)
    else:
        cmd, kw = "text2video", {}
    kw |= dict(model_version=model, video_resolution=res, duration=duration, ratio=ratio)
    return _go(cmd, _flags(**kw), prompt, out_dir, kw, block)
