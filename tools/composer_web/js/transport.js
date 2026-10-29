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
  // hooks.countingIn(k)：第 k 拍（排到的拍，不是听到的拍）要不要打预备拍的咔哒
  constructor(engine, getState, hooks = {}) {
    this.e = engine; this.S = getState; this.h = hooks;
    this.playing = false; this.segs = []; this.nextB = 0; this.timer = null;
    this.live = [];            // 排好的录音 / 分轨片段 { src, layer, p0, p1（拍）, when, end（音频时钟）, v }：每 tick 按静音 / 独奏 / 场景调它的增益
    this.endTimer = null; this.ending = false;      // 歌尾的那一次 stop（只排一次）
    this.countInEnd = -Infinity;
    this._warned = false;
  }
  _segEnd(b) {
    const s = this.S(), [ls, le] = loopRange(s);
    if (s.loop.on && b < ls - 1e-9) return ls;           // 预备拍 / 循环前面那段
    if (s.loop.on && b < le - 1e-9) return le;
    return songBeats(s);
  }
  // opts.countInEnd = 录音从第几拍开始（预备拍只打它之前的拍）。没给：开循环 = 循环起点；不循环 = 起点 + 一小节（有预备拍时）
  start(fromBeat, opts = {}) {
    this.stop(true);
    this.playing = true;
    const s = this.S(), [ls] = loopRange(s);
    this.countInEnd = opts.countInEnd ?? (s.loop.on ? ls : fromBeat + (s.settings.countIn ? s.beatsPerBar : 0));
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
    if (this.endTimer) clearTimeout(this.endTimer);
    this.endTimer = null; this.ending = false;
    for (const c of this.live) { try { c.src.stop(); } catch (e) { /* 已停 */ } }
    this.live = [];
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
    let skipped = 0;
    for (let guard = 0; guard < 8; guard++) {
      const g = this.segs[this.segs.length - 1];
      const gEnd = g.ctx0 + (g.b1 - g.b0) * sp;
      const tEnd = Math.min(horizon, gEnd);
      // 弹窗（改名 / 定稿确认）挡住了页面、音频时钟照走：醒来时排程落在过去 → 错过的那段不补排（不然那段的音一起涌出来），从现在接着排
      const late = g.b0 + (this.e.now - 0.02 - g.ctx0) / sp;
      if (late > this.nextB + 1e-9) { const to = Math.min(late, g.b1); skipped += (to - this.nextB) * sp; this.nextB = to; }
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
        // 歌尾：只进来一次（之后每 tick 还会走到这里，不能再排 stop、再报 onSegment(null)）
        if (!this.ending) {
          this.ending = true;
          this.endTimer = setTimeout(() => { this.endTimer = null; this.stop(); }, Math.max(0, (gEnd - this.e.now) * 1000 + 50));
          this.h.onSegment?.(null, g);
        }
        break;
      }
    }
    if (skipped > 0.05) this.h.onSkip?.(skipped);
    this._clipGates();
    this._macros();
  }
  _when(g, b) { return g.ctx0 + (b - g.b0) * spb(this.S()); }

  _events(g, bA, bB) {
    const s = this.S(), sp = spb(s);
    // 节拍器；预备拍按「排到的拍」k 判断（排程比听到早 120 ms + 输出延迟，按听到的拍判会多打循环起点那一下）
    const ci = this.h.countingIn;
    for (let k = Math.ceil(bA - 1e-9); k < bB; k++) {
      if (s.settings.metro || (ci && k < this.countInEnd - 1e-9 && ci(k))) this.e.click(this._when(g, k), ((k % s.beatsPerBar) + s.beatsPerBar) % s.beatsPerBar === 0);
    }
    const events = this.h.events ? this.h.events() : s.notes;
    for (const n of events) {
      if (!(n.b >= bA - 1e-9 && n.b < bB - 1e-9)) continue;         // 写反过来：b 是 NaN 时也跳过
      const L = layerById(s, n.layer);
      if (!L || !audibleAt(s, L, n.b)) continue;
      const when = this._when(g, n.b);
      // 一个坏音（比如力度 NaN）不能让整个走带卡住：报一次，接着排后面的
      try {
        if (n.sound) {
          this.e.hit(L, n.sound, n.vel, when);
          if (n.sound === 'kick') this.e.duck(when, this._macroEff(n.b).sidechain);
        } else if (L.kind === 'synth' || (L.kind === 'voice' && L.params.synthOn)) {
          // 和 Python 渲染一样：只有跟唱带滑音时音高可以是小数（小数并进音分曲线）；别的一律放整数音高
          const bend = L.kind === 'voice' && L.params.bendOn && n.bend?.cents?.length > 1 ? n.bend : null;
          this.e.note(L, bend ? n.pitch : Math.round(n.pitch), n.vel, when, Math.max(0.03, n.d * sp), bend);
        }
      } catch (err) {
        if (!this._warned) { this._warned = true; console.warn('有一个音排不进去，跳过', n, err); }
      }
    }
  }
  // 片段在第 b 拍的增益：场景里不响 / 静音 / 被别人独奏 → 0；哼唱层乘原声音量
  _clipGain(s, L, b) {
    if (!L || !audibleAt(s, L, b)) return 0;
    const g = L.kind === 'voice' ? Math.pow(10, (L.params.rawGainDb ?? 0) / 20) : 1;
    return g < 0.01 ? 0 : g;
  }
  // 录音 / 分轨：按段排；场景边界处切开（每片单独一个增益，场景切点是准的）。
  // 不响的片也排上（增益 0）：静音 / 独奏 / 场景勾选 / 原声音量在播放中改了，_clipGates 下一 tick 就生效，不用等下一遍
  // 段的开头已经过去了（弹窗挡住以后补排、reclip 中途重排）：只排从现在起的那部分（片内起点跟着往后挪，声音不错位）
  _clips(g) {
    const s = this.S(), sp = spb(s);
    const clips = this.h.clips ? this.h.clips() : s.clips;
    const nowB = g.b0 + (this.e.now + 0.01 - g.ctx0) / sp;
    for (const c of clips) {
      if (!c.active) continue;
      const L = layerById(s, c.layer); if (!L) continue;
      const buf = this.h.buffer?.(c.audio); if (!buf) continue;
      const cEnd = c.b + c.dur / sp;
      const a = Math.max(c.b, g.b0, nowB), z = Math.min(cEnd, g.b1);
      if (!(z > a)) continue;
      const cuts = [a, ...s.scenes.map(sc => sc.bar * s.beatsPerBar).filter(x => x > a && x < z), z];
      for (let i = 0; i < cuts.length - 1; i++) {
        const p0 = cuts[i], p1 = cuts[i + 1];
        const v = this._clipGain(s, L, p0 + 1e-6), when = this._when(g, p0), dur = (p1 - p0) * sp;
        const src = this.e.clip(L, buf, when, (p0 - c.b) * sp + (c.offset || 0), dur, v);
        if (src) this.live.push({ src, layer: L.id, p0, p1, when, end: when + dur, v });
      }
    }
  }
  // 片段换了（挑哪一遍试听 / 挑定了 / 剪了录音）：正在放的录音 / 分轨从现在起按新的重排，不等下一段。旧的淡出 40 ms 再停（不咔一下）
  reclip() {
    if (!this.playing) return;
    const now = this.e.now, sp = spb(this.S());
    for (const c of this.live) { this.e.clipGain?.(c.src, 0); try { c.src.stop(now + 0.04); } catch (e) { /* 已停 */ } }
    this.live = [];
    for (const g of this.segs) if (g.ctx0 + (g.b1 - g.b0) * sp > now) this._clips(g);
  }
  // 每 tick：按片子此刻放到的那一拍重算增益（还没开始的片按它的起点）。场景没改时片内不跨场景，等于按起点算
  _clipGates() {
    if (!this.live.length) return;
    const s = this.S(), now = this.e.now, sp = spb(s);
    this.live = this.live.filter(c => c.end > now - 0.1);               // 放完的丢掉
    for (const c of this.live) {
      const b = Math.min(c.p1 - 1e-6, Math.max(c.p0 + 1e-6, c.p0 + (now - c.when) / sp));
      const v = this._clipGain(s, layerById(s, c.layer), b);
      if (Math.abs(v - c.v) > 1e-4) { c.v = v; this.e.clipGain?.(c.src, v); }
    }
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
