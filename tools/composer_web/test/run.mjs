// >v<🎛作曲器-自测-纯函数（node 跑，不需要浏览器）：乐理、哼唱识别、解释层、真值导出。
// 用法：node tools/composer_web/test/run.mjs   （tests/test_composer_web.py 也会调它）
import assert from 'node:assert/strict';
import { voiceLike } from './synth_signals.mjs';
import * as Th from '../js/theory.js';
import * as Pi from '../js/pitch.js';
import * as In from '../js/interp.js';
import * as St from '../js/state.js';

let n = 0, failed = 0;
const t = (name, f) => { n++; try { f(); } catch (e) { failed++; console.error(`✗ ${name}\n  ${e.message}`); } };
const Em = { root: 4, mode: 'aeolian' };
const pitches = segs => segs.map(s => Math.round(s.pitchRaw));

// ---- 乐理 ----
t('锁调键盘：A 排 = E 小调音阶，Q 排同列高八度，Z 排低八度', () => {
  assert.deepEqual(['KeyA', 'KeyS', 'KeyD', 'KeyF', 'KeyG', 'KeyH', 'KeyJ', 'KeyK'].map(c => Th.degreeToMidi(Em, 4, Th.codeToDegree(Em, c))), [64, 66, 67, 69, 71, 72, 74, 76]);
  assert.equal(Th.degreeToMidi(Em, 4, Th.codeToDegree(Em, 'KeyQ')), 76);
  assert.equal(Th.degreeToMidi(Em, 4, Th.codeToDegree(Em, 'KeyZ')), 52);
  for (const c of [...Th.ROW_A, ...Th.ROW_Q, ...Th.ROW_Z]) assert.ok(Th.inScale(Em, Th.degreeToMidi(Em, 3, Th.codeToDegree(Em, c))), c);
});
t('和弦：i VI III VII = Em C G D', () => {
  assert.deepEqual(Th.parseProgression('i VI III VII').map(d => Th.chordName(Em, d)), ['Em', 'C', 'G', 'D']);
  assert.deepEqual(Th.parseProgression('1 6 3 7'), [0, 5, 2, 6]);
});
t('吸附：调内 + 两边差不多时跟上一个音', () => {
  assert.equal(Th.snapToScale(Em, 64.9), 64);            // F 不在 E 小调里 → 离得近的 E
  assert.equal(Th.snapToScale(Em, 65.2), 66);            //                  → 离得近的 F#
  assert.equal(Th.snapMelodic(Em, 65.5, { prev: 66 }), 66);
});

// ---- 哼唱识别 ----
const mel = [{ midi: 64, dur: 0.4 }, { midi: 67, dur: 0.4 }, { midi: 71, dur: 0.6 }, { midi: 69, dur: 0.3 }, { midi: 67, dur: 0.5 }];
t('干净的哼唱：5 个音、音高对、起点误差 < 30 ms', () => {
  const r = Pi.analyzeHum(voiceLike(mel), 48000);
  assert.deepEqual(pitches(r.segs), [64, 67, 71, 69, 67]);
  let t0 = 0.15;
  r.segs.forEach((s, i) => { assert.ok(Math.abs(s.t - t0) < 0.03, `第 ${i + 1} 个音起点 ${s.t} vs ${t0}`); t0 += mel[i].dur; });
});
t('进音滑音 + 颤音不会被切成多个音', () => {
  const r = Pi.analyzeHum(voiceLike(mel, 48000, { vib: 0.4, scoop: 1.5 }), 48000);
  assert.deepEqual(pitches(r.segs), [64, 67, 71, 69, 67]);
});
t('整体唱低 40 音分：测出跑调并在切音时扣掉', () => {
  const r = Pi.analyzeHum(voiceLike(mel.map(m => ({ ...m, cents: -40 }))), 48000);
  assert.ok(Math.abs(r.tuneCents + 40) < 6, `测到 ${r.tuneCents}`);
  assert.deepEqual(pitches(r.segs), [64, 67, 71, 69, 67]);
});
t('同音重唱（中间有一点点停）切成三个音', () => {
  const r = Pi.analyzeHum(voiceLike([{ midi: 64, dur: 0.3 }, { midi: 64, dur: 0.3 }, { midi: 64, dur: 0.3 }], 48000, { gapSec: 0.05 }), 48000);
  assert.equal(r.segs.length, 3);
});
t('三种类型的猜测：哼唱 / 人声（辅音）/ 口哨', () => {
  assert.equal(Pi.analyzeHum(voiceLike(mel), 48000).guess.mode, 'hum');
  assert.equal(Pi.analyzeHum(voiceLike(mel, 48000, { consonant: 0.06, gapSec: 0.06 }), 48000).guess.mode, 'voice');
  assert.equal(Pi.analyzeHum(voiceLike([{ midi: 84, dur: 0.4 }, { midi: 86, dur: 0.4 }]), 48000).guess.mode, 'melody');
});
t('秒 → 拍：吸附到调、力度在 0–1、音分曲线留着', () => {
  const r = Pi.analyzeHum(voiceLike(mel.map(m => ({ ...m, cents: 30 }))), 48000);
  const ns = Pi.notesToBeats(r.segs, { t0Beat: 4, spb: 0.4, key: Em, prog: [0, 5, 2, 6], snap: 'scale' });
  assert.deepEqual(ns.map(x => x.pitch), [64, 67, 71, 69, 67]);
  for (const x of ns) { assert.ok(x.vel > 0 && x.vel <= 1); assert.ok(x.bend.cents.length > 3); assert.ok(x.b >= 4); }
});

// ---- 解释层 ----
const played = [0.06, 0.55, 1.07, 1.56, 2.05, 2.57].map((b, i) => ({ b, d: 0.4, pitch: 64 + i, vel: 0.8 }));
t('稳定拖后 ~0.06 拍 → 认成延迟', () => {
  const o = In.meanOffset(played);
  assert.ok(o.consistent && Math.abs(o.offset - 0.06) < 0.02, JSON.stringify(o));
});
t('吸附 100% 落在 1/8 格上；60% 介于原样和格子之间；原始值不动', () => {
  const q = In.quantize(played, { grid: '1/8', strength: 1 });
  for (const x of q) assert.ok(Math.abs(x.b * 2 - Math.round(x.b * 2)) < 1e-9);
  const h = In.quantize(played, { grid: '1/8', strength: 0.6 });
  h.forEach((x, i) => assert.ok(Math.abs(x.b - Math.round(played[i].b * 2) / 2) < Math.abs(played[i].b - Math.round(played[i].b * 2) / 2) + 1e-9));
  assert.equal(played[0].b, 0.06);
});
t('律动模板：从「每个 16 分都晚 20 ms」的起音里学出偏移，并能吸上去', () => {
  const spb = 0.4, late = 0.02 / spb;
  const ons = []; for (let k = 0; k < 64; k++) ons.push(k * 0.25 + (k % 2 ? late : 0));
  const g = In.grooveFromOnsets(ons);
  assert.ok(Math.abs(g.offsets[1] - late) < 1e-6 && Math.abs(g.offsets[0]) < 1e-6);
  const q = In.quantize([{ b: 0.24, d: 0.2 }], { grid: '1/16', strength: 1, groove: g });
  assert.ok(Math.abs(q[0].b - (0.25 + late)) < 1e-6);
});
t('变体：第一个永远是原样；再来一组换别的', () => {
  const ctx = { key: Em, prog: [0], beatsPerBar: 4, tensionAt: () => 0.5, groove: null };
  const v = In.makeVariants(played, ctx);
  assert.equal(v[0].kind, 'raw'); assert.equal(v.length, 4);
  assert.deepEqual(v[0].notes.map(x => x.b), played.map(x => x.b));
  const v2 = In.makeVariants(played, ctx, { reroll: 2 });
  assert.equal(v2[0].kind, 'raw'); assert.equal(new Set(v2.map(x => x.kind)).size, 4);
});
t('三度和声 / 八度叠的新音都在调内', () => {
  const ctx = { key: Em, prog: [0], beatsPerBar: 4, tensionAt: () => 0.5 };
  for (const k of [1, 2, 3, 4, 5, 6, 7]) for (const o of In.makeVariants(played.map(x => ({ ...x, pitch: Th.snapToScale(Em, x.pitch) })), ctx, { reroll: k }))
    for (const x of o.notes) if (x.pitch !== undefined) assert.ok(Th.inScale(Em, x.pitch), `${o.label} ${x.pitch}`);
});
t('起音检测：等间隔的咔哒都找到', () => {
  const sr = 22050, x = new Float32Array(sr * 3);
  for (let k = 0; k < 10; k++) for (let i = 0; i < 200; i++) x[Math.round((0.2 + k * 0.25) * sr) + i] = Math.exp(-i / 40) * (i % 2 ? 1 : -1);
  const ons = In.detectOnsets(x, sr);
  assert.equal(ons.length, 10, ons.join(','));
});

// ---- 真值导出 ----
t('导出：秒 = anchor + 拍 × 60/bpm；鼓进 hits；按时间排；格式自检通过', () => {
  const s = St.makeState(); s.anchor = 25.95; s.bpm = 150;
  s.notes.push({ id: 'a', layer: 'lead', b: 4, d: 1, pitch: 64.2, vel: 0.8, input: 'keyboard', raw: { b: 4.05, d: 1 } });
  s.notes.push({ id: 'b', layer: 'drums', b: 0, d: 0.25, sound: 'kick', vel: 1, input: 'keyboard' });
  s.notes.push({ id: 'c', layer: 'lead', b: 2, d: 0.5, pitch: 67, vel: 0.5, input: 'tap' });
  s.curves.cutoff = [[0, 0], [16, 1]]; s.curves.tension = [[0, 0.2], [8, 0.9]];
  s.scenes.push({ id: 's', bar: 0, name: '回想', layers: ['lead'], text: '她开始想起自己是谁' });
  const j = St.exportTruth(s);
  assert.deepEqual(St.validateTruth(j), []);
  assert.equal(j.notes[0].t, 26.75); assert.equal(j.notes[1].t, 27.55); assert.equal(j.notes[1].pitch, 64);
  assert.equal(j.notes[1].raw.t, 27.57);
  assert.equal(j.hits[0].sound, 'kick'); assert.equal(j.hits[0].t, 25.95);
  assert.equal(j.macros.cutoff.fps, 60); assert.equal(j.macros.cutoff.t0, 25.95);
  assert.equal(j.macros.cutoff.v[0], 0); assert.ok(Math.abs(j.macros.cutoff.v[j.macros.cutoff.v.length - 1] - 1) < 0.01);
  assert.deepEqual(j.tension[1], [29.15, 0.9]);
  assert.deepEqual(j.mood, [{ t: 25.95, text: '她开始想起自己是谁' }]);
  assert.equal(j.grid.downbeats[1], 27.55);
});
t('张力当父级：低张力把失真压住，回声 g 只有高张力才能过 1', () => {
  const lo = St.effectiveMacros({ tension: 0, drive: 1, pole_g: 1 }), hi = St.effectiveMacros({ tension: 1, drive: 1, pole_g: 1 });
  assert.ok(lo.drive < 0.4 && hi.drive === 1);
  assert.ok(lo.pole_g < 1 && hi.pole_g > 1);
});
t('曲线：写一段会替换旧点、插值正确', () => {
  let p = [[0, 0], [2, 1], [4, 0]];
  p = St.curveWrite(p, 1, 3, 0.5);
  assert.deepEqual(p, [[0, 0], [3, 0.5], [4, 0]]);
  assert.equal(St.curveAt(p, 1.5, 9), 0.25);
  assert.equal(St.curveAt([], 1, 0.7), 0.7);
});

console.log(`${n - failed}/${n} 通过`);
if (failed) process.exit(1);
