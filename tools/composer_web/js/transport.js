// >v<🎛作曲器-走带-提前量调度（每 25 ms 排未来 120 ms 的事件），循环 = 一段一段地接。
// 段（seg）= {ctx0, b0, b1}：音频时钟 ctx0 时刻在第 b0 拍，一直走到 b1。录音时用段把「按键的时刻」换算回拍。
import { audibleAt, effectiveMacros, curveAt, loopRange, songBeats, spb, layerById, MACROS } from './state.js';

// 键盘事件的时刻 → 「听到的」音频时钟。getOutputTimestamp().contextTime 是扬声器此刻正在放的那个样本，
// 所以换算出来的已经是听到的时钟（排在 c 的音恰好在 c 被听到），不能再减输出延迟。
// 拿不到时间戳时退回 now − 输出延迟。
export function heardCtxTime(ts, evStamp, now, outLat) {
  if (ts && ts.performanceTime && Number.isFinite(evStamp)) return ts.contextTime + (evStamp - ts.performanceTime) / 1000;
  return now - outLat;
}

export class Transport {
  constructor(engine, getState, hooks = {}) {
    this.e = engine; this.S = getState; this.h = hooks;
    this.playing = false; this.segs = []; this.nextB = 0; this.timer = null; this.sources = [];
  }
  _segEnd(b) {
    const s = this.S(), [ls, le] = loopRange(s);
    if (s.loop.on && b < ls - 1e-9) return ls;           // 预备拍 / 循环前面那段
    if (s.loop.on && b < le - 1e-9) return le;
    return songBeats(s);
  }
  start(fromBeat) {
    this.stop(true);
    this.playing = true;
    const ctx0 = this.e.now + 0.08;
    this.segs = [{ ctx0, b0: fromBeat, b1: this._segEnd(fromBeat) }];
    this.nextB = fromBeat;
    this._clips(this.segs[0]);
    this.h.onSegment?.(this.segs[0], null);
    this.timer = setInterval(() => this.tick(), 25);
    this.tick();
  }
  stop(silent = false) {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
    for (const src of this.sources) { try { src.stop(); } catch (e) { /* 已停 */ } }
    this.sources = [];
    const was = this.playing;
    this.playing = false;
    if (was && !silent) this.h.onStop?.();
  }
  // 音频时钟 → 拍（找包含它的那一段；在两段之间的就算前一段的末尾）
  beatAt(t) {
    const s = this.S();
    for (let i = this.segs.length - 1; i >= 0; i--) {
      const g = this.segs[i];
      if (t >= g.ctx0 - 1e-6 || i === 0) return Math.min(g.b1, g.b0 + (t - g.ctx0) / spb(s));
    }
    return 0;
  }
  // 现在听到的是第几拍（排程比听到早 outputLatency）
  get beatNow() {
    if (!this.playing) return null;
    const lat = (this.e.ctx.outputLatency || 0) + (this.e.ctx.baseLatency || 0);
    return this.beatAt(this.e.now - lat);
  }
  segAt(t) { for (let i = this.segs.length - 1; i >= 0; i--) if (t >= this.segs[i].ctx0) return this.segs[i]; return this.segs[0]; }

  tick() {
    if (!this.playing) return;
    const s = this.S(), sp = spb(s);
    const horizon = this.e.now + 0.12;
    for (let guard = 0; guard < 8; guard++) {
      const g = this.segs[this.segs.length - 1];
      const gEnd = g.ctx0 + (g.b1 - g.b0) * sp;
      const tEnd = Math.min(horizon, gEnd);
      const bB = g.b0 + (tEnd - g.ctx0) / sp;
      if (bB > this.nextB) this._events(g, this.nextB, bB);
      this.nextB = Math.max(this.nextB, bB);
      if (tEnd < gEnd - 1e-6) break;
      // 到段尾：循环 → 回到循环开头；不循环 → 歌尾停
      const [ls, le] = loopRange(s);
      if (s.loop.on && g.b1 >= le - 1e-9) {
        const ng = { ctx0: gEnd, b0: ls, b1: le };
        this.segs.push(ng); if (this.segs.length > 6) this.segs.shift();
        this.nextB = ls;
        this._clips(ng);
        this.h.onSegment?.(ng, g);
      } else if (g.b1 < songBeats(s) - 1e-9) {
        const ng = { ctx0: gEnd, b0: g.b1, b1: this._segEnd(g.b1) };
        this.segs.push(ng); if (this.segs.length > 6) this.segs.shift();
        this._clips(ng);
        this.h.onSegment?.(ng, g);
      } else {
        setTimeout(() => this.stop(), Math.max(0, (gEnd - this.e.now) * 1000 + 50));
        this.h.onSegment?.(null, g);
        break;
      }
    }
    this._macros();
  }
  _when(g, b) { return g.ctx0 + (b - g.b0) * spb(this.S()); }

  _events(g, bA, bB) {
    const s = this.S(), sp = spb(s);
    // 节拍器
    if (s.settings.metro || (this.h.countingIn && this.h.countingIn())) {
      for (let k = Math.ceil(bA - 1e-9); k < bB; k++) this.e.click(this._when(g, k), ((k % s.beatsPerBar) + s.beatsPerBar) % s.beatsPerBar === 0);
    }
    const events = this.h.events ? this.h.events() : s.notes;
    for (const n of events) {
      if (n.b < bA - 1e-9 || n.b >= bB - 1e-9) continue;
      const L = layerById(s, n.layer);
      if (!L || !audibleAt(s, L, n.b)) continue;
      const when = this._when(g, n.b);
      if (n.sound) {
        this.e.hit(L, n.sound, n.vel, when);
        if (n.sound === 'kick') this.e.duck(when, this._macroEff(n.b).sidechain);
      } else if (L.kind === 'synth' || (L.kind === 'voice' && L.params.synthOn)) {
        this.e.note(L, n.pitch, n.vel, when, Math.max(0.03, n.d * sp), L.kind === 'voice' && L.params.bendOn ? n.bend : null);
      }
    }
  }
  // 录音 / 分轨：按段排；场景边界处切开（场景里不响的那段不放）
  _clips(g) {
    const s = this.S(), sp = spb(s);
    const clips = this.h.clips ? this.h.clips() : s.clips;
    for (const c of clips) {
      if (!c.active) continue;
      const L = layerById(s, c.layer); if (!L) continue;
      const buf = this.h.buffer?.(c.audio); if (!buf) continue;
      const cEnd = c.b + c.dur / sp;
      const a = Math.max(c.b, g.b0), z = Math.min(cEnd, g.b1);
      if (z <= a) continue;
      const cuts = [a, ...s.scenes.map(sc => sc.bar * s.beatsPerBar).filter(x => x > a && x < z), z];
      for (let i = 0; i < cuts.length - 1; i++) {
        const p0 = cuts[i], p1 = cuts[i + 1];
        if (!audibleAt(s, L, p0 + 1e-6)) continue;
        const gain = L.kind === 'voice' ? Math.pow(10, (L.params.rawGainDb ?? 0) / 20) : 1;
        if (gain < 0.01) continue;
        this.sources.push(this.e.clip(L, buf, this._when(g, p0), (p0 - c.b) * sp + (c.offset || 0), (p1 - p0) * sp, gain));
      }
    }
    if (this.sources.length > 200) this.sources = this.sources.slice(-100);
  }
  _macroEff(b) {
    const s = this.S();
    const vals = {};
    for (const m of MACROS) vals[m.id] = this.h.liveMacro?.(m.id) ?? curveAt(s.curves[m.id], b, m.def);
    return effectiveMacros(vals, s.settings.tensionParent);
  }
  _macros() {
    const b = this.beatNow;
    if (b === null) return;
    const eff = this._macroEff(b);
    this.e.setMacros(eff);
    this.lastEff = eff;
  }
}
