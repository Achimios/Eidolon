#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🎛合成器词表页面：读 synth_vocab.TERMS（文字的唯一真源）+ 某版本的 词表.json → -给人类的解释/合成器词表.html

用法：python tools/synth_vocab_page.py --ver v1
页面用相对路径引用 Projekt_Visual/样片_示波器/词表/<ver>/ 里的 wav / mp4 / jpg（本地双击打开即可）。
"""
import argparse
import html
import json
from pathlib import Path

from synth_vocab import OUT_BASE, ROOT, TERMS

PAGE = ROOT / "-给人类的解释" / "合成器词表.html"

CSS = """
:root{--bg:#070908;--panel:#0d110f;--line:#1c2520;--ink:#d6e6da;--dim:#7f978a;--g:#4dff7c;--o:#ff8a4c}
*{box-sizing:border-box}html,body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.7 "Microsoft YaHei","PingFang SC",system-ui,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:40px 20px 80px}
h1{font-size:30px;margin:0 0 4px;letter-spacing:.02em}h1 small{color:var(--dim);font-weight:400;font-size:15px;margin-left:10px}
.lead{color:var(--dim);max-width:820px;margin:0 0 26px}
.how{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:0 0 40px}
.how div{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:12px 14px;font-size:14px}
.how b{color:var(--g);font-weight:600}
nav{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 36px}
nav a{color:var(--dim);text-decoration:none;border:1px solid var(--line);border-radius:4px;padding:2px 9px;font-size:13px}
nav a:hover{color:var(--g);border-color:var(--g)}
section{border-top:1px solid var(--line);padding:30px 0 10px}
.hd{display:flex;align-items:baseline;gap:14px;margin-bottom:6px}
.num{font:600 13px Consolas,monospace;color:var(--g)}
h2{margin:0;font-size:23px}h2 span{color:var(--dim);font-weight:400;font-size:15px;margin-left:8px}
.one{font-size:17px;margin:4px 0 14px;max-width:900px}
.row{display:grid;grid-template-columns:420px 1fr;gap:18px;align-items:start}
video{width:100%;border-radius:4px;background:#000;display:block}
.cues p{margin:0 0 10px}.cues b{color:var(--g);font-weight:600;margin-right:6px}
.listen{display:flex;align-items:center;gap:10px;margin-top:10px}.listen span{font-size:13px;color:var(--g);white-space:nowrap}
.listen audio{width:100%;height:34px}
.warn{font-size:12.5px;color:var(--o);margin:6px 0 0;line-height:1.55}.warn.ok{color:var(--dim)}
.meta{font:12px Consolas,monospace;color:var(--dim);margin-top:8px}
.meta a{color:var(--dim)}
img{width:100%;display:block;border-radius:4px;margin-top:12px}
.foot{margin-top:50px;color:var(--dim);font-size:14px;border-top:1px solid var(--line);padding-top:20px}
.foot b{color:var(--ink)}
@media (max-width:820px){.row{grid-template-columns:1fr}.how{grid-template-columns:1fr}}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ver", required=True)
    a = ap.parse_args()
    meta = {m["num"]: m for m in json.loads((OUT_BASE / a.ver / "词表.json").read_text(encoding="utf-8"))}
    rel = f"../../../Projekt_Visual/样片_示波器/词表/{a.ver}/"
    e = html.escape
    parts = [f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>合成器词表</title><style>{CSS}</style></head><body><div class="wrap">
<h1>合成器词表<small>15 个词 · 听 + 看 · {e(a.ver)}</small></h1>
<p class="lead">够用来指挥的 15 个词。每个词一句人话、一段几秒的声音、它在示波器上的样子。
示波器画的是一个立体声 wav：左声道推光点左右走，右声道推光点上下走。真立体声的例子，听到的就是画出来的；延迟嵌入的例子，另有一个「听音色」的版本（只有真实的声音）。所有声音都是 numpy 现算的，没有采样，A 小调，120 BPM。</p>
<div class="how">
<div><b>左 = x，右 = y</b><br>两个声道一样 → 一条 45° 斜线；只有左 → 横线；只有右 → 竖线；左右各走各的 → 一团。</div>
<div><b>延迟嵌入</b><br>单声道的声音，把右声道设成左声道晚 τ（四分之一个周期，几毫秒）。正弦 → 圆；任何稳定的音 → 一个闭环；乱的声音 → 乱结构。⚠️这个副本只为画图：外放时第 2、6、10… 次泛音会被抵消，所以听音色用每格的「听音色」播放器。</div>
<div><b>亮 = 慢</b><br>光束走得慢的地方亮，快跳的地方暗。所以方波只剩四个亮点，失真的方块四个角最亮。每个例子的放大倍数单独调过（= 示波器的 V/div 旋钮）。</div>
</div><nav>"""]
    for t in TERMS:
        parts.append(f'<a href="#t{t[0]}">{t[0]} {e(t[2])}</a>')
    parts.append("</nav>")
    for num, slug, cn, en, _fn, one, hear, see in TERMS:
        m = meta[num]
        f = rel + f"{num}_{slug}"
        parts.append(f"""<section id="t{num}"><div class="hd"><span class="num">{num}</span><h2>{e(cn)}<span>{e(en)}</span></h2></div>
<p class="one">{e(one)}</p>
<div class="row"><div><video controls preload="metadata" src="{f}_scope.mp4"></video>
<div class="listen"><span>听音色</span><audio controls preload="none" src="{f}_listen.wav"></audio></div>
{'<p class="warn">视频里的声音是给示波器用的：右声道是左声道晚几毫秒的副本，外放会抵消一部分泛音。听音色请点上面这个。</p>' if m.get('embedded') else '<p class="warn ok">这一格是真立体声：视频里的声音就是要听的声音。</p>'}
<div class="meta">{e(m['mode'])} · 放大 ×{m['scale']} · {m['dur']} s · <a href="{f}.wav">示波器用 wav</a> · <a href="{f}_listen.wav">听的 wav</a></div></div>
<div class="cues"><p><b>听</b>{e(hear)}</p><p><b>看</b>{e(see)}</p></div></div>
<img src="{f}_scope.jpg" alt="{e(cn)} 示波器四格"><img src="{f}_spec.jpg" alt="{e(cn)} 波形和频谱"></section>""")
    parts.append("""<div class="foot">
<p><b>和我们的理念接上的三处</b>（<code>_设计/理念_示波器与处理器.md</code>）</p>
<p>· 延迟的反馈 g 就是 IIR 的极点：g &lt; 1 回声会散，g → 1 永远不停，g &gt; 1 炸开。和「FIR / IIR 生长」是同一个数学。<br>
· 琶音器那一格：两个音的频率比 = 音程 = 李萨如图的瓣数。越协和的音程图越简单（八度 1:2 是 ∞，五度 2:3 是三瓣）；平均律不是整数比，所以图形在慢慢转。<br>
· 调制那一格就是调制矩阵的一条线：源（包络）× 目标（截止频率）× 强度。作曲器里每条线都会写进事件轨，画面直接读。</p>
<p>生成：<code>tools/synth_vocab.py</code>（声音 + 图）、<code>tools/synth_vocab_page.py</code>（这页）、积木在 <code>src/synthkit.py</code>，示波器是 <code>tools/scope_sim.py</code>。</p></div>
</div></body></html>""")
    PAGE.write_text("\n".join(parts), encoding="utf-8")
    print(PAGE)


if __name__ == "__main__":
    main()
