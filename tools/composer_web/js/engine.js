// >v<🎛作曲器-预览引擎-WebAudio。⚠️只是预览：最终音质由 Python（synthkit）读同一份 作曲.json 渲染。
// 结构：每层一条总线（失真 → 音量 → 侧链闪避 → 声像，另有延迟 / 混响送出）→ 母线（截止 / 共振 → 失真 → 限幅）→ 输出 + 示波器。
// 旋钮的名字用合成器词表的 15 个词。
import { midiHz } from './theory.js';

export const PRESETS = {
  bass:  { label: '贝斯', octave: 2, p: { wave: 'sawtooth', voices: 1, detune: 0, fmRatio: 0, fmIndex: 0, cutoff: 520, reso: 0.25, envAmt: 0.5, attack: 0.003, decay: 0.25, sustain: 0.5, release: 0.08, lfoRate: 5, lfoDepth: 0, lfoTarget: 'cutoff', drive: 0.25, delaySend: 0, reverbSend: 0.03, duck: true, pan: 0 } },
  pad:   { label: '铺底', octave: 4, p: { wave: 'sawtooth', voices: 5, detune: 18, fmRatio: 0, fmIndex: 0, cutoff: 1800, reso: 0.1, envAmt: 0.1, attack: 0.6, decay: 0.5, sustain: 0.8, release: 1.2, lfoRate: 0.25, lfoDepth: 0.15, lfoTarget: 'cutoff', drive: 0, delaySend: 0.1, reverbSend: 0.45, duck: true, pan: 0 } },
  lead:  { label: '主旋律', octave: 4, p: { wave: 'square', voices: 2, detune: 8, fmRatio: 0, fmIndex: 0, cutoff: 3000, reso: 0.25, envAmt: 0.35, attack: 0.005, decay: 0.2, sustain: 0.7, release: 0.15, lfoRate: 5.5, lfoDepth: 0, lfoTarget: 'pitch', drive: 0.1, delaySend: 0.25, reverbSend: 0.2, duck: false, pan: 0 } },
  pluck: { label: '拨弦', octave: 4, p: { wave: 'triangle', voices: 1, detune: 0, fmRatio: 0, fmIndex: 0, cutoff: 2500, reso: 0.15, envAmt: 0.6, attack: 0.002, decay: 0.35, sustain: 0, release: 0.2, lfoRate: 5, lfoDepth: 0, lfoTarget: 'pitch', drive: 0, delaySend: 0.3, reverbSend: 0.25, duck: false, pan: 0 } },
  bell:  { label: 'FM 钟', octave: 5, p: { wave: 'sine', voices: 1, detune: 0, fmRatio: 3.5, fmIndex: 2.5, cutoff: 9000, reso: 0, envAmt: 0, attack: 0.002, decay: 1.2, sustain: 0, release: 1.0, lfoRate: 5, lfoDepth: 0, lfoTarget: 'pitch', drive: 0, delaySend: 0.15, reverbSend: 0.35, duck: false, pan: 0 } },
  arp:   { label: '琶音', octave: 4, p: { wave: 'sawtooth', voices: 1, detune: 0, fmRatio: 0, fmIndex: 0, cutoff: 1800, reso: 0.3, envAmt: 0.5, attack: 0.002, decay: 0.12, sustain: 0.2, release: 0.05, lfoRate: 5, lfoDepth: 0, lfoTarget: 'pitch', drive: 0.1, delaySend: 0.25, reverbSend: 0.15, duck: true, pan: 0 } },
  sing:  { label: '跟唱合成', octave: 4, p: { wave: 'triangle', voices: 2, detune: 6, fmRatio: 1, fmIndex: 0.4, cutoff: 2600, reso: 0.1, envAmt: 0.2, attack: 0.02, decay: 0.3, sustain: 0.8, release: 0.2, lfoRate: 5.5, lfoDepth: 0, lfoTarget: 'pitch', drive: 0, delaySend: 0.2, reverbSend: 0.3, duck: false, pan: 0 } },
};
export const DRUM_SOUNDS = ['kick', 'snare', 'clap', 'hat', 'ohat', 'perc', 'tom', 'impact'];
export const DRUM_LABEL = { kick: '底鼓', snare: '军鼓', clap: '拍手', hat: '闭镲', ohat: '开镲', perc: '边击', tom: '通鼓', impact: '冲击' };
export const DRUM_PARAMS = { drive: 0.1, delaySend: 0, reverbSend: 0.08, duck: false, pan: 0 };

const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const tanhCurve = (k, n = 1024) => {
  const c = new Float32Array(n), t = Math.tanh(k);
  for (let i = 0; i < n; i++) { const x = i / (n - 1) * 2 - 1; c[i] = Math.tanh(k * x) / t; }
  return c;
};

function noiseBuffer(ctx, sec = 2) {
  const b = ctx.createBuffer(1, Math.floor(ctx.sampleRate * sec), ctx.sampleRate), d = b.getChannelData(0);
  let s = 12345;
  for (let i = 0; i < d.length; i++) { s = (s * 1103515245 + 12345) & 0x7fffffff; d[i] = s / 0x3fffffff - 1; }
  return b;
}

// ---- 一个合成音（离线测响度也走这一个函数）----
// 返回 { release(when) }；dur 给了就自动松开
export function synthVoice(ctx, dest, p, pitch, vel, when, dur = null, bend = null) {
  const f = midiHz(pitch);
  // 力度同时推音量和亮度（真乐器越响越亮）
  const amp = 0.18 + 0.82 * Math.pow(vel, 1.5);
  const bright = Math.pow(2, (vel - 0.7) * 1.5);
  const nodes = [];
  const out = ctx.createGain(); out.gain.value = 0;
  const f1 = ctx.createBiquadFilter(), f2 = ctx.createBiquadFilter();
  f1.type = f2.type = 'lowpass';
  f1.Q.value = 0.5 + p.reso * 14; f2.Q.value = 0.707;
  f1.connect(f2); f2.connect(out); out.connect(dest);
  const fMax = Math.min(18000, ctx.sampleRate * 0.45);
  const base = clamp(p.cutoff * bright, 40, fMax), peak = clamp(base * Math.pow(2, p.envAmt * 4), 40, fMax);
  for (const fl of [f1, f2]) {
    fl.frequency.setValueAtTime(base, when);
    fl.frequency.linearRampToValueAtTime(peak, when + Math.max(0.003, p.attack));
    fl.frequency.setTargetAtTime(base + (peak - base) * p.sustain, when + Math.max(0.003, p.attack), Math.max(0.01, p.decay / 3));
  }
  const n = Math.max(1, p.voices | 0);
  let bendSrc = null;
  if (bend && bend.cents && bend.cents.length > 1) {
    bendSrc = ctx.createConstantSource(); bendSrc.offset.value = bend.cents[0];
    const curve = Float32Array.from(bend.cents);
    try { bendSrc.offset.setValueCurveAtTime(curve, when, Math.max(0.02, bend.dt * curve.length)); } catch (e) { /* 曲线和别的自动化冲突时就不弯 */ }
    bendSrc.start(when); nodes.push(bendSrc);
  }
  let lfo = null, lfoG = null;
  if (p.lfoDepth > 0) {
    lfo = ctx.createOscillator(); lfo.frequency.value = p.lfoRate;
    lfoG = ctx.createGain();
    lfoG.gain.value = p.lfoTarget === 'pitch' ? p.lfoDepth * 60 : p.lfoTarget === 'cutoff' ? p.lfoDepth * 1800 : p.lfoDepth * 0.5;
    lfo.connect(lfoG); lfo.start(when); nodes.push(lfo);
    if (p.lfoTarget === 'cutoff') { lfoG.connect(f1.detune); lfoG.connect(f2.detune); }
  }
  let trem = null;
  if (lfoG && p.lfoTarget === 'amp') { trem = ctx.createGain(); trem.gain.value = 1; lfoG.connect(trem.gain); trem.connect(f1); }
  const into = trem || f1;
  for (let v = 0; v < n; v++) {
    const o = ctx.createOscillator();
    o.type = p.wave; o.frequency.value = f;
    o.detune.value = n > 1 ? (v / (n - 1) - 0.5) * 2 * p.detune : 0;
    if (bendSrc) bendSrc.connect(o.detune);
    if (lfoG && p.lfoTarget === 'pitch') lfoG.connect(o.detune);
    if (p.fmIndex > 0 && p.fmRatio > 0) {
      const m = ctx.createOscillator(), mg = ctx.createGain();
      m.frequency.value = f * p.fmRatio; mg.gain.value = p.fmIndex * f * p.fmRatio;
      m.connect(mg); mg.connect(o.frequency); m.start(when); nodes.push(m);
    }
    let tail = o;
    if (n > 1) { const pn = ctx.createStereoPanner(); pn.pan.value = (v / (n - 1) - 0.5) * 1.2; o.connect(pn); tail = pn; }
    const g = ctx.createGain(); g.gain.value = 1 / Math.sqrt(n);
    tail.connect(g); g.connect(into);
    o.start(when); nodes.push(o);
  }
  out.gain.setValueAtTime(0, when);
  out.gain.linearRampToValueAtTime(amp, when + Math.max(0.002, p.attack));
  out.gain.setTargetAtTime(amp * p.sustain, when + Math.max(0.002, p.attack), Math.max(0.01, p.decay / 3));
  let released = false;
  const release = t => {
    if (released) return; released = true;
    t = Math.max(t, when + 0.005);
    const r = Math.max(0.01, p.release);
    if (out.gain.cancelAndHoldAtTime) out.gain.cancelAndHoldAtTime(t); else out.gain.cancelScheduledValues(t);
    out.gain.setTargetAtTime(0, t, r / 4);
    const end = t + r * 1.6 + 0.05;
    for (const nd of nodes) { try { nd.stop(end); } catch (e) { /* 已停 */ } }
    nodes[nodes.length - 1]?.addEventListener?.('ended', () => { try { out.disconnect(); } catch (e) { /* */ } });
  };
  if (dur !== null) release(when + dur);
  return { release };
}

// ---- 鼓（全是现算的，没有采样）----
export function drumHit(ctx, dest, noise, sound, vel, when) {
  const g = ctx.createGain(); g.connect(dest);
  const a = 0.25 + 0.75 * Math.pow(vel, 1.3);
  const env = (node, peak, dec, t0 = when) => { node.gain.setValueAtTime(0, t0); node.gain.linearRampToValueAtTime(peak, t0 + 0.002); node.gain.setTargetAtTime(0, t0 + 0.002, dec / 4); };
  const nz = (t0, dur) => { const s = ctx.createBufferSource(); s.buffer = noise; s.start(t0, Math.random() * 1.0, dur); return s; };
  const osc = (type, f0, f1, tf, t0 = when, dur = 0.6) => { const o = ctx.createOscillator(); o.type = type; o.frequency.setValueAtTime(f0, t0); if (f1) o.frequency.exponentialRampToValueAtTime(f1, t0 + tf); o.start(t0); o.stop(t0 + dur); return o; };
  g.gain.value = a;
  if (sound === 'kick') {
    const o = osc('sine', 150, 45, 0.11, when, 0.6), e = ctx.createGain(); o.connect(e); e.connect(g); env(e, 1, 0.45);
    const c = nz(when, 0.01), hp = ctx.createBiquadFilter(), ce = ctx.createGain(); hp.type = 'highpass'; hp.frequency.value = 2000; c.connect(hp); hp.connect(ce); ce.connect(g); env(ce, 0.3, 0.01);
  } else if (sound === 'snare') {
    const s = nz(when, 0.3), bp = ctx.createBiquadFilter(), e = ctx.createGain(); bp.type = 'bandpass'; bp.frequency.value = 1800; bp.Q.value = 0.7; s.connect(bp); bp.connect(e); e.connect(g); env(e, 0.8, 0.18);
    const o = osc('triangle', 185, 160, 0.05, when, 0.2), e2 = ctx.createGain(); o.connect(e2); e2.connect(g); env(e2, 0.5, 0.1);
  } else if (sound === 'clap') {
    const s = nz(when, 0.4), bp = ctx.createBiquadFilter(), e = ctx.createGain(); bp.type = 'bandpass'; bp.frequency.value = 1100; bp.Q.value = 1.2; s.connect(bp); bp.connect(e); e.connect(g);
    e.gain.setValueAtTime(0, when);
    for (const dt of [0, 0.012, 0.024]) { e.gain.setValueAtTime(0.9, when + dt); e.gain.setTargetAtTime(0.1, when + dt + 0.001, 0.004); }
    e.gain.setValueAtTime(0.7, when + 0.034); e.gain.setTargetAtTime(0, when + 0.035, 0.05);
  } else if (sound === 'hat' || sound === 'ohat') {
    const dec = sound === 'hat' ? 0.045 : 0.32;
    const s = nz(when, dec * 2 + 0.05), hp = ctx.createBiquadFilter(), e = ctx.createGain(); hp.type = 'highpass'; hp.frequency.value = 7500; s.connect(hp); hp.connect(e); e.connect(g); env(e, 0.45, dec);
  } else if (sound === 'perc') {
    const o = osc('square', 820, 0, 0, when, 0.06), bp = ctx.createBiquadFilter(), e = ctx.createGain(); bp.type = 'bandpass'; bp.frequency.value = 1700; o.connect(bp); bp.connect(e); e.connect(g); env(e, 0.5, 0.03);
  } else if (sound === 'tom') {
    const o = osc('sine', 120, 78, 0.25, when, 0.6), e = ctx.createGain(); o.connect(e); e.connect(g); env(e, 0.9, 0.35);
  } else if (sound === 'impact') {
    const o = osc('sine', 60, 28, 1.4, when, 2.2), e = ctx.createGain(); o.connect(e); e.connect(g); env(e, 1, 1.6);
    const s = nz(when, 2), lp = ctx.createBiquadFilter(), e2 = ctx.createGain(); lp.type = 'lowpass'; lp.frequency.value = 500; s.connect(lp); lp.connect(e2); e2.connect(g); env(e2, 0.6, 1.4);
  }
}

export class Engine {
  constructor() {
    this.ctx = new (window.AudioContext || window.webkitAudioContext)({ latencyHint: 'interactive' });
    const c = this.ctx;
    this.noise = noiseBuffer(c);
    this.sum = c.createGain();
    this.mf1 = c.createBiquadFilter(); this.mf2 = c.createBiquadFilter();
    this.mf1.type = this.mf2.type = 'lowpass'; this.mf1.frequency.value = this.mf2.frequency.value = 20000;
    this.mshape = c.createWaveShaper(); this.mshape.curve = tanhCurve(1); this.mshape.oversample = '2x';
    this.makeup = c.createGain();
    this.limiter = c.createDynamicsCompressor();
    this.limiter.threshold.value = -8; this.limiter.knee.value = 4; this.limiter.ratio.value = 14; this.limiter.attack.value = 0.002; this.limiter.release.value = 0.12;
    this.out = c.createGain(); this.out.gain.value = 0.9;
    this.sum.connect(this.mf1); this.mf1.connect(this.mf2); this.mf2.connect(this.mshape); this.mshape.connect(this.makeup);
    this.makeup.connect(this.limiter); this.limiter.connect(this.out); this.out.connect(c.destination);
    // 示波器：左 = x，右 = y（同合成器词表）
    const sp = c.createChannelSplitter(2);
    this.scopeL = c.createAnalyser(); this.scopeR = c.createAnalyser();
    this.scopeL.fftSize = this.scopeR.fftSize = 2048;
    this.out.connect(sp); sp.connect(this.scopeL, 0); sp.connect(this.scopeR, 1);
    // 混响：生成的冲激响应（立体声噪声 × 指数衰减）
    this.revIn = c.createGain(); this.conv = c.createConvolver(); this.revRet = c.createGain(); this.revRet.gain.value = 0.3;
    this.conv.buffer = this._ir(2.8);
    this.revIn.connect(this.conv); this.conv.connect(this.revRet); this.revRet.connect(this.sum);
    // 延迟：环里有 tanh，g 推过 1 也是有界的（= 极点越过单位圆的那一下，限幅接住）
    this.dlyIn = c.createGain(); this.dly = c.createDelay(4); this.dlyDamp = c.createBiquadFilter(); this.dlySat = c.createWaveShaper(); this.dlyFb = c.createGain(); this.dlyRet = c.createGain();
    this.dlyDamp.type = 'lowpass'; this.dlyDamp.frequency.value = 4500; this.dlySat.curve = tanhCurve(1.5); this.dlyFb.gain.value = 0.35; this.dlyRet.gain.value = 0.6;
    this.dlyIn.connect(this.dly); this.dly.connect(this.dlyDamp); this.dlyDamp.connect(this.dlySat); this.dlySat.connect(this.dlyFb); this.dlyFb.connect(this.dly);
    this.dly.connect(this.dlyRet); this.dlyRet.connect(this.sum);
    this.layers = new Map();
    this.trims = {};           // 每种音色的响度修正（dB），开机离线测
    this._lastDrive = -1;
  }
  _ir(sec) {
    const c = this.ctx, n = Math.floor(c.sampleRate * sec), b = c.createBuffer(2, n, c.sampleRate);
    for (let ch = 0; ch < 2; ch++) {
      const d = b.getChannelData(ch); let s = 777 + ch * 31, lp = 0;
      for (let i = 0; i < n; i++) {
        s = (s * 1103515245 + 12345) & 0x7fffffff;
        const w = s / 0x3fffffff - 1; lp += 0.35 * (w - lp);
        d[i] = lp * Math.exp(-i / n * 6.5) * (i < c.sampleRate * 0.01 ? i / (c.sampleRate * 0.01) : 1);
      }
    }
    return b;
  }
  get now() { return this.ctx.currentTime; }
  setDelayTime(sec) { this.dly.delayTime.setTargetAtTime(clamp(sec, 0.01, 3.9), this.now, 0.05); }

  // 每层总线
  ensureLayer(layer) {
    let L = this.layers.get(layer.id);
    if (!L) {
      const c = this.ctx;
      L = { in: c.createGain(), shape: c.createWaveShaper(), vol: c.createGain(), duck: c.createGain(), pan: c.createStereoPanner(), rev: c.createGain(), dly: c.createGain(), meter: c.createAnalyser() };
      L.meter.fftSize = 512;
      L.in.connect(L.shape); L.shape.connect(L.vol); L.vol.connect(L.duck); L.duck.connect(L.pan); L.pan.connect(this.sum);
      L.duck.connect(L.rev); L.rev.connect(this.revIn); L.duck.connect(L.dly); L.dly.connect(this.dlyIn); L.pan.connect(L.meter);
      this.layers.set(layer.id, L);
    }
    this.updateLayer(layer);
    return L;
  }
  updateLayer(layer, audible = true) {
    const L = this.layers.get(layer.id); if (!L) return;
    const p = layer.params || {};
    const drv = p.drive || 0;
    if (L._drv !== drv) { L.shape.curve = tanhCurve(1 + drv * 18); L._drv = drv; }
    const trim = this.trims[layer.preset] ?? 0;
    const g = audible ? Math.pow(10, ((layer.gainDb || 0) + trim) / 20) / (1 + drv * 1.2) : 0;
    L.vol.gain.setTargetAtTime(g, this.now, 0.015);
    L.pan.pan.setTargetAtTime(clamp(p.pan || 0, -1, 1), this.now, 0.02);
    L.rev.gain.setTargetAtTime(p.reverbSend || 0, this.now, 0.02);
    L.dly.gain.setTargetAtTime(p.delaySend || 0, this.now, 0.02);
    L.ducked = !!p.duck;
  }
  removeLayer(id) { const L = this.layers.get(id); if (L) { try { L.pan.disconnect(); L.duck.disconnect(); } catch (e) { /* */ } this.layers.delete(id); } }

  note(layer, pitch, vel, when, dur = null, bend = null) {
    const L = this.ensureLayer(layer);
    const p = layer.kind === 'voice' ? PRESETS[layer.params.synthPreset || 'sing'].p : layer.params;
    return synthVoice(this.ctx, L.in, p, pitch, vel, when, dur, bend);
  }
  hit(layer, sound, vel, when) { const L = this.ensureLayer(layer); drumHit(this.ctx, L.in, this.noise, sound, vel, when); }
  clip(layer, buffer, when, offset, dur, gain = 1) {
    const L = this.ensureLayer(layer);
    const s = this.ctx.createBufferSource(); s.buffer = buffer;
    const g = this.ctx.createGain(); g.gain.value = gain;
    s.connect(g); g.connect(L.in);
    s.start(Math.max(when, this.now), Math.max(0, offset), Math.max(0.001, dur));
    return s;
  }
  click(when, accent) {
    const c = this.ctx, o = c.createOscillator(), g = c.createGain();
    o.type = 'square'; o.frequency.value = accent ? 1760 : 1180;
    g.gain.setValueAtTime(0, when); g.gain.linearRampToValueAtTime(accent ? 0.16 : 0.09, when + 0.001); g.gain.setTargetAtTime(0, when + 0.002, 0.012);
    o.connect(g); g.connect(this.out); o.start(when); o.stop(when + 0.08);
  }
  // 侧链：底鼓一响，被闪避的层压下去再弹回来
  duck(when, depth) {
    if (depth <= 0.001) return;
    for (const L of this.layers.values()) {
      if (!L.ducked) continue;
      L.duck.gain.cancelScheduledValues(when);
      L.duck.gain.setValueAtTime(1 - depth, when);
      L.duck.gain.setTargetAtTime(1, when + 0.02, 0.09);
    }
  }
  // 旋钮（已经过「张力当父级」换算的有效值，0–1；pole_g 可到 1.1）
  setMacros(m) {
    const t = this.now;
    if (m.cutoff !== undefined) { const f = 60 * Math.pow(2, m.cutoff * 8.4); this.mf1.frequency.setTargetAtTime(f, t, 0.02); this.mf2.frequency.setTargetAtTime(f, t, 0.02); }
    if (m.reso !== undefined) this.mf1.Q.setTargetAtTime(0.7 + m.reso * 12, t, 0.02);
    if (m.drive !== undefined && Math.abs(m.drive - this._lastDrive) > 0.01) {
      this.mshape.curve = tanhCurve(1 + m.drive * 14); this.makeup.gain.setTargetAtTime(1 / (1 + m.drive * 1.6), t, 0.02); this._lastDrive = m.drive;
    }
    if (m.space !== undefined) this.revRet.gain.setTargetAtTime(m.space * 1.2, t, 0.03);
    if (m.pole_g !== undefined) this.dlyFb.gain.setTargetAtTime(m.pole_g, t, 0.03);
  }
  meterDb(id) {
    const L = this.layers.get(id); if (!L) return -90;
    const a = new Float32Array(L.meter.fftSize); L.meter.getFloatTimeDomainData(a);
    let s = 0; for (const v of a) s += v * v;
    return 10 * Math.log10(s / a.length + 1e-12);
  }

  // 每种音色离线渲一个音，测 RMS，记修正量：换音色时音量平衡不乱
  async measureTrims(target = -20) {
    // 和 Python 渲染（composer_render.py 的 trim_for）同法同采样率。两边的修正量会差约 1.45 dB：WebAudio 自带的锯齿 / 方波
    // 按吉布斯过冲归一过（峰值 1.18 → 1），页面的单个音本来就轻一点；修正以后两边都落在 −20 dB，所以成片的平衡一致
    const sr = 48000;
    for (const [k, pr] of Object.entries(PRESETS)) {
      const oc = new OfflineAudioContext(2, sr * 2, sr);
      const pitch = 12 * (pr.octave + 1) + 4;
      synthVoice(oc, oc.destination, pr.p, pitch, 0.8, 0.0, 1.2);
      const buf = await oc.startRendering();
      let s = 0, n = 0;
      for (let ch = 0; ch < 2; ch++) { const d = buf.getChannelData(ch); for (let i = 0; i < sr * 1.2; i++) { s += d[i] * d[i]; n++; } }
      const db = 10 * Math.log10(s / n + 1e-12);
      this.trims[k] = clamp(target - db, -12, 18);
    }
    this.trims.drums = 0; this.trims.voice = 0; this.trims.stem = 0;
    return this.trims;
  }
}
