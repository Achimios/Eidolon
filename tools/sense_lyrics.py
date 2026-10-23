# -*- coding: utf-8 -*-
""">v<👂感官层-歌词-词级时间戳：whisperX（Whisper 转写 + wav2vec2 强制对齐）→ <作品>/歌词.json。
  .venv_whisperx/Scripts/python.exe tools/sense_lyrics.py <作品目录> [--lang en] [--text 歌词.txt]
- 输入用 <作品>/分轨/vocals.wav（separate.py 已对齐到播放时间，所以词时间 = 作品时间）。
- 单独的 venv：whisperX 要 torch 2.8，主 .venv 是 2.11，装一起会把主环境降级（2026-10-24）。
- 有 --text（👤给的歌词原文）就跳过转写、只做强制对齐；没有就用 Whisper 转写，⚠️歌声转写会错词，低置信的词标出来给人听。
- 不写 感官.json（那份由 sense_song 整份重写），单独一份 歌词.json。
"""
import argparse
import json
import sys
from pathlib import Path

LOW = 0.5          # 对齐分数低于这个 → 标 low，给人听
ROOT = Path(__file__).resolve().parents[1]


def load16k(p):
    """whisperX 自带的 load_audio 要 PATH 上的 ffmpeg（本机没有）→ 借主 .venv 里 imageio-ffmpeg 的 exe。"""
    import subprocess
    import numpy as np
    exe = next((ROOT / ".venv/Lib/site-packages/imageio_ffmpeg/binaries").glob("ffmpeg*.exe"))
    raw = subprocess.run([str(exe), "-nostdin", "-loglevel", "error", "-i", str(p), "-f", "s16le", "-ac", "1", "-ar", "16000", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


PAD = 0.5          # 人声段前后放宽（秒）


def gate_by_voice(out, mv):
    """Whisper 在没人声的地方会编词（间奏里的「Thank you.」）→ 用 感官.json 的 voice.sections 判：
    词的中点不在任何人声段（±PAD）里 → outside_voice；整行都在外面 → 这一行标 hallucination。只标不删。"""
    sp = mv / "感官.json"
    secs = json.loads(sp.read_text("utf-8")).get("voice", {}).get("sections") if sp.exists() else None
    if not secs:
        return 0
    inside = lambda t: any(a - PAD <= t <= b + PAD for a, b in secs)
    for w in out["words"]:
        w["outside_voice"] = w["t0"] is not None and not inside((w["t0"] + w["t1"]) / 2)
    n = 0
    for l in out["lines"]:
        ws = [w for w in out["words"] if w["t0"] is not None and l["t0"] - 1e-3 <= w["t0"] <= l["t1"] + 1e-3]
        l["hallucination"] = bool(ws) and all(w["outside_voice"] for w in ws)
        n += l["hallucination"]
    out["gate"] = f"voice.sections ±{PAD:g} s（感官.json）：outside_voice = 词中点不在人声段；hallucination = 整行都在外面"
    return n


def save(out, mv):
    p = mv / "歌词.json"
    k = 2
    while p.exists():                                  # 不覆盖用户可能看过的版本
        p = mv / f"歌词_v{k}.json"; k += 1
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
    return p


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("mv")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--text", default=None, help="歌词原文（一行一句），有就只做强制对齐")
    ap.add_argument("--regate", default=None, help="不重跑：读这份已有的 歌词*.json，只加人声门控，另存新版本")
    a = ap.parse_args()
    if a.regate:
        mv = Path(a.mv)
        out = json.loads(Path(a.regate).read_text("utf-8"))
        n = gate_by_voice(out, mv)
        p = save(out, mv)
        for l in out["lines"]:
            if l.get("hallucination"):
                print(f"  幻觉行 {l['t0']:.1f}-{l['t1']:.1f}: {l['text'][:60]!r}")
        print(f"{n} hallucinated lines, {sum(w.get('outside_voice', False) for w in out['words'])} words outside voice  -> {p}")
        return
    import torch
    import whisperx
    mv = Path(a.mv)
    voc = mv / "分轨" / "vocals.wav"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    audio = load16k(voc)
    dur = len(audio) / 16000
    if a.text:
        lines = [s.strip() for s in Path(a.text).read_text("utf-8").splitlines() if s.strip()]
        segs = [{"text": " ".join(lines), "start": 0.0, "end": dur}]
        src = f"强制对齐（歌词原文 {Path(a.text).name}）"
    else:
        m = whisperx.load_model(a.model, dev, compute_type="float16" if dev == "cuda" else "int8", language=a.lang)
        segs = m.transcribe(audio, batch_size=8, language=a.lang)["segments"]
        del m
        src = f"Whisper {a.model} 转写 + 对齐（⚠️歌声转写会错词）"
    am, meta = whisperx.load_align_model(language_code=a.lang, device=dev)
    al = whisperx.align(segs, am, meta, audio, dev, return_char_alignments=False)
    words, lines_out = [], []
    for s in al["segments"]:
        ws = []
        for w in s.get("words", []):
            if "start" not in w:                       # 对不上的词（数字 / 符号）没有时间
                ws.append({"w": w["word"], "t0": None, "t1": None, "score": None, "low": True}); continue
            sc = round(float(w.get("score", 0.0)), 3)
            ws.append({"w": w["word"], "t0": round(w["start"], 3), "t1": round(w["end"], 3), "score": sc, "low": sc < LOW})
        words += ws
        lines_out.append({"text": s["text"].strip(), "t0": round(s["start"], 3), "t1": round(s["end"], 3)})
    out = {"source": src, "audio": "分轨/vocals.wav", "unit": "s（作品时间 = 播放时间）", "low_threshold": LOW,
           "lines": lines_out, "words": words}
    if a.text and sum(len(l.split()) for l in lines) == len(words):   # 按原文分行（NLTK 只按句号分，没标点的歌词会粘成几大段）
        lines_out, i = [], 0
        for l in lines:
            ws = words[i:i + len(l.split())]; i += len(ws)
            ts = [w for w in ws if w["t0"] is not None]
            lines_out.append({"text": l, "t0": ts[0]["t0"] if ts else None, "t1": ts[-1]["t1"] if ts else None})
        out["lines"] = lines_out
    gate_by_voice(out, mv)
    p = save(out, mv)
    n_low = sum(w["low"] for w in words)
    print(f"{len(lines_out)} lines, {len(words)} words, low {n_low}  -> {p}")
    for w in [w for w in words if w["low"]][:15]:
        print(f"  low: {w['w']!r} @ {w['t0']}  score {w['score']}")


if __name__ == "__main__":
    main()
