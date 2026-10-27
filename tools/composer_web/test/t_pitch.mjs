// >v<🎛作曲器-自测-哼唱识别（真人声回归）：帧时刻对齐、录音开头的音、跑调门限、元音里 YIN 断续不切、短辅音照切。
// 用法：node tools/composer_web/test/t_pitch.mjs   （纯计算，不出声）
import assert from 'node:assert/strict';
import { voiceLike } from './synth_signals.mjs';
import * as Pi from '../js/pitch.js';

let n = 0, failed = 0;
const t = (name, f) => { n++; try { f(); } catch (e) { failed++; console.error(`✗ ${name}\n  ${e.message}`); } };
const pitches = segs => segs.map(s => Math.round(s.pitchRaw));
const ms = v => `${(v * 1000).toFixed(1)} ms`;
const lcg = seed => () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff - 0.5; };

// 一串谐波音，起音 5 ms；on 可以 < 0（录音开始前就在唱，前面那段被切掉）
function tones(notes, total, sr = 48000) {
  const x = new Float32Array(Math.round(total * sr)), rand = lcg(3);
  for (const { on, dur, midi } of notes) {
    const f = 440 * Math.pow(2, (midi - 69) / 12);
    const a = Math.round(on * sr), z = Math.min(x.length, Math.round((on + dur) * sr));
    for (let i = Math.max(0, a); i < z; i++) {
      const tt = (i - a) / sr, w = 2 * Math.PI * f * tt;
      const env = Math.min(1, tt / 0.005, (z - i) / sr / 0.01) * 0.3;
      x[i] += env * (Math.sin(w) + 0.4 * Math.sin(2 * w) + 0.2 * Math.sin(3 * w));
    }
  }
  for (let i = 0; i < x.length; i++) x[i] += 0.002 * rand();
  return x;
}
// 相位连续的分段音（f = 0 → 静音，里面有 1e-4 的微弱周期信号）；硬起音
function steps(segs, sr = 48000) {
  const x = new Float32Array(Math.round(segs.reduce((a, s) => a + s.dur, 0) * sr));
  let i = 0, ph = 0;
  for (const s of segs) {
    for (let k = 0, m = Math.round(s.dur * sr); k < m && i < x.length; k++, i++) {
      if (s.f > 0) { ph += 2 * Math.PI * s.f / sr; x[i] = 0.3 * (Math.sin(ph) + 0.5 * Math.sin(2 * ph) + 0.25 * Math.sin(3 * ph)); }
      else x[i] = 0.5e-4 * Math.sin(i * 0.37);
    }
  }
  return x;
}
// 一个长元音，每 every 秒相位跳一下（声门不规则）：没有噪声，只让 YIN 的谷变浅几帧（真人声里元音常这样断续）
function brokenVowel(midi, { jump = 2.2, every = 0.06, dur = 1.2, sr = 48000 } = {}) {
  const x = new Float32Array(Math.ceil((dur + 0.45) * sr)), rand = lcg(7);
  const f = 440 * Math.pow(2, (midi - 69) / 12), N = Math.floor(dur * sr), o = Math.floor(0.15 * sr);
  let ph = 0, next = every;
  for (let i = 0; i < N; i++) {
    if (i / sr >= next) { ph += jump; next += every; }
    ph += 2 * Math.PI * f / sr;
    const env = Math.min(1, i / (0.01 * sr), (N - i) / (0.01 * sr)) * 0.3;
    let s = 0;
    for (let h = 1; h <= 8; h++) s += Math.sin(h * ph) / (h * h * 0.6 + 0.4);
    x[o + i] += env * s;
  }
  for (let i = 0; i < x.length; i++) x[i] += 0.002 * rand();
  return { x, on: 0.15, off: 0.15 + dur };
}

// ---- 帧时刻 ----
t('帧时刻 = YIN 积分那一半的中心：第一帧在 0、帧距均匀（48k / 44.1k）', () => {
  for (const sr of [48000, 44100]) {
    const fr = Pi.track(tones([{ on: 0.1, dur: 0.3, midi: 60 }], 0.6, sr), sr);
    assert.ok(Math.abs(fr[0].t) < 1e-9, `${sr}: 第一帧 ${fr[0].t}`);
    const hop = fr[1].t - fr[0].t;
    for (let i = 2; i < fr.length; i++) assert.ok(Math.abs(fr[i].t - fr[i - 1].t - hop) < 1e-9, `${sr}: 第 ${i} 帧不等距`);
  }
});
t('起点对齐：录音一开头的音和中间的音都在 ±6 ms 内（MIDI 50 / 66，三种模式）', () => {
  for (const sr of [48000, 44100]) for (const base of [50, 66]) {
    const ons = [0, 0.4, 0.8, 1.2];
    const x = tones(ons.map((on, k) => ({ on, dur: 0.3, midi: base + 2 * k })), 1.6, sr);
    for (const mode of ['hum', 'voice', 'melody']) {
      const r = Pi.analyzeHum(x, sr, { mode });
      assert.equal(r.segs.length, 4, `${sr} midi ${base} ${mode}: ${r.segs.length} 个音`);
      r.segs.forEach((s, i) => assert.ok(Math.abs(s.t - ons[i]) <= 0.006, `${sr} midi ${base} ${mode} 第 ${i + 1} 个音起点差 ${ms(s.t - ons[i])}`));
    }
  }
});
t('legato 换音（大三度，切点已知）：第二个音的起点在 ±7 ms 内', () => {
  for (const f1 of [110, 200, 330]) {
    const r = Pi.analyzeHum(steps([{ f: 0, dur: 0.2 }, { f: f1, dur: 0.3 }, { f: f1 * Math.pow(2, 4 / 12), dur: 0.3 }, { f: 0, dur: 0.2 }]), 48000, { mode: 'hum' });
    assert.equal(r.segs.length, 2, `${f1} Hz: ${r.segs.length} 个音`);
    assert.ok(Math.abs(r.segs[1].t - 0.5) <= 0.007, `${f1} Hz: 第二个音起点差 ${ms(r.segs[1].t - 0.5)}`);
  }
});
t('静音里有微弱的周期信号、然后硬起音：起点不提前（YIN 那一半是静音的帧不测音高）', () => {
  for (const f of [110, 200, 330]) {
    const r = Pi.analyzeHum(steps([{ f: 0, dur: 0.3 }, { f, dur: 0.5 }, { f: 0, dur: 0.3 }]), 48000, { mode: 'hum' });
    assert.equal(r.segs.length, 1, `${f} Hz: ${r.segs.length} 个音`);
    assert.ok(Math.abs(r.segs[0].t - 0.3) <= 0.006, `${f} Hz: 起点差 ${ms(r.segs[0].t - 0.3)}`);
    const m = 69 + 12 * Math.log2(f / 440);   // pitchRaw 已扣掉跑调 → 加回来再比
    assert.ok(Math.abs(r.segs[0].pitchRaw + r.tuneCents / 100 - m) < 0.3, `${f} Hz: 音高 ${r.segs[0].pitchRaw}`);
  }
});
t('录音开始前就唱了（强拍提前 10 / 30 ms）：第一个音从 0 起，没有负的时刻', () => {
  for (const early of [0.01, 0.03]) {
    const ons = [-early, 0.4 - early, 0.8 - early];
    const r = Pi.analyzeHum(tones(ons.map((on, k) => ({ on, dur: 0.3, midi: 60 + 3 * k })), 1.3), 48000, { mode: 'hum' });
    assert.equal(r.segs.length, 3);
    assert.ok(r.segs[0].t >= 0 && r.segs[0].t <= 0.006, `提前 ${early}: 第一个音起点 ${ms(r.segs[0].t)}`);
    assert.ok(Math.abs(r.segs[0].t + r.segs[0].dur - (0.3 - early)) <= 0.016, `提前 ${early}: 第一个音终点 ${ms(r.segs[0].t + r.segs[0].dur)}`);
    for (const s of r.segs) assert.ok(s.t >= 0 && s.curve.every(c => c.t >= 0));
  }
});

// ---- 整体跑调 ----
const mel = [{ midi: 64, dur: 0.4 }, { midi: 67, dur: 0.4 }, { midi: 71, dur: 0.6 }, { midi: 69, dur: 0.3 }, { midi: 67, dur: 0.5 }];
t('tuningOffset：相位集中才算跑调；均匀散开（R < 0.5）→ 0', () => {
  const fr = ms_ => ms_.map(m => ({ voiced: true, midi: m }));
  assert.ok(Math.abs(Pi.tuningOffset(fr([60.3, 62.3, 64.28, 65.32])) - 30) < 2);
  assert.equal(Pi.tuningOffset(fr([60, 60.25, 60.5, 60.75])), 0);
  assert.equal(Pi.tuningOffset(fr([])), 0);
});
t('只有颤音（±50 音分）、没跑调：不扣跑调，音高照对', () => {
  for (const scoop of [0, 1.5]) {
    const r = Pi.analyzeHum(voiceLike(mel, 48000, { vib: 0.5, scoop }), 48000);
    assert.ok(Math.abs(r.tuneCents) < 6, `scoop ${scoop}: 测到 ${r.tuneCents}`);
    assert.deepEqual(pitches(r.segs), [64, 67, 71, 69, 67]);
  }
});
t('唱高 30 音分 + 颤音 ±40：不会被假的「−38 音分」推高一个半音', () => {
  const r = Pi.analyzeHum(voiceLike(mel.map(m => ({ ...m, cents: 30 })), 48000, { vib: 0.4 }), 48000);
  assert.deepEqual(pitches(r.segs), [64, 67, 71, 69, 67], `tune ${r.tuneCents}`);
});
t('颤音小（±20 音分）+ 唱低 40 音分：照样测出跑调并扣掉', () => {
  const r = Pi.analyzeHum(voiceLike(mel.map(m => ({ ...m, cents: -40 })), 48000, { vib: 0.2 }), 48000);
  assert.ok(Math.abs(r.tuneCents + 40) < 6, `测到 ${r.tuneCents}`);
  assert.deepEqual(pitches(r.segs), [64, 67, 71, 69, 67]);
});

// ---- 人声模式：辅音才切，元音里的断续不切 ----
t('元音里 YIN 断续（没噪声，谷 0.15–0.4）：人声模式还是一个音、不比哼唱模式短', () => {
  for (const [midi, o] of [[57, { jump: 2.2, every: 0.06 }], [64, { jump: 2.2, every: 0.06 }], [64, { jump: Math.PI / 2, every: 0.07 }], [72, { jump: Math.PI, every: 0.05 }]]) {
    const { x, on, off } = brokenVowel(midi, o);
    const v = Pi.analyzeHum(x, 48000, { mode: 'voice' }), h = Pi.analyzeHum(x, 48000, { mode: 'hum' });
    // 信号本身得真有断续（响、没音高、谷不深），不然这条测不到东西
    const drop = v.frames.filter(f => f.t > on + 0.05 && f.t < off - 0.05 && f.loud && !f.voiced);
    assert.ok(drop.length >= 10 && drop.every(f => f.ap >= 0.15 && f.ap < 0.4), `midi ${midi}: 断续帧 ${drop.length}，ap ${drop.map(f => f.ap.toFixed(2))}`);
    const len = r => r.segs.reduce((s, q) => s + q.dur, 0);
    assert.equal(v.segs.length, 1, `midi ${midi} ${JSON.stringify(o)}: 人声模式 ${v.segs.length} 个音`);
    assert.equal(pitches(v.segs)[0], midi);
    assert.ok(len(v) >= len(h) - 0.02 && len(v) > 0.9 * (off - on), `midi ${midi}: 人声 ${len(v).toFixed(2)} s，哼唱 ${len(h).toFixed(2)} s`);
  }
});
t('同音音节 + 短辅音（20 / 30 / 60 ms 噪声）：人声模式照样一个音节一个音', () => {
  const syl = Array.from({ length: 6 }, () => ({ midi: 62, dur: 0.18 }));
  for (const c of [0.02, 0.03, 0.06]) {
    const r = Pi.analyzeHum(voiceLike(syl, 48000, { consonant: c, gapSec: c }), 48000, { mode: 'voice' });
    assert.equal(r.segs.length, 6, `辅音 ${c * 1000} ms: ${r.segs.length} 个音`);
  }
  const r3 = Pi.analyzeHum(voiceLike(mel, 48000, { consonant: 0.06, gapSec: 0.06 }), 48000);
  assert.equal(r3.mode, 'voice');
  assert.deepEqual(pitches(r3.segs), [64, 67, 71, 69, 67]);
});

t('预读：早唱 40 ms 的强拍放在这一遍开头、整音不缺头，early ≈ 40 ms；上一遍的尾巴丢掉；拖过来的长音不算抢拍', () => {
  // voiceLike 前面有 0.15 s 静音：把「这一遍」的开头放在 0.19 s，强拍就早了 40 ms
  const x = voiceLike([{ midi: 64, dur: 0.5 }, { midi: 67, dur: 0.4 }], 48000);
  const an = Pi.analyzeHum(x, 48000, { mode: 'hum' });
  const segs = Pi.trimPreroll(an.segs, 0.19);
  assert.deepEqual(segs.map(s => Math.round(s.pitchRaw)), [64, 67]);
  assert.equal(segs[0].t, 0);
  assert.ok(Math.abs(segs[0].early - 0.04) <= 0.012, `early ${segs[0].early}`);
  assert.ok(Math.abs(segs[0].dur - 0.46) <= 0.02, `dur ${segs[0].dur}`);
  assert.equal(segs[1].early, undefined);
  for (const s of segs) for (const c of s.curve) assert.ok(c.t >= 0);
  // 构造：上一遍的尾巴（预读里就结束了）、拖过来的长音（从预读开头就在响）
  const mk = (t, dur) => ({ t, dur, pitchRaw: 60, db: -20, zcr: 100, curve: [{ t, m: 60 }, { t: t + dur - 0.01, m: 60 }] });
  const out = Pi.trimPreroll([mk(0.0, 0.1), mk(0.0, 0.5)], 0.15);
  assert.equal(out.length, 1);
  assert.equal(out[0].early, undefined);
  assert.ok(Math.abs(out[0].dur - 0.35) < 1e-9);
});

console.log(`${n - failed}/${n} 通过`);
if (failed) process.exit(1);
