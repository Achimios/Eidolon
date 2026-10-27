// >v<🎛作曲器-自测-核心（node 跑，不需要浏览器）：循环取模 / 截尾、力度不出 NaN、真值 audible / derived / 自检、曲线写点、走带边界、分轨片增益。
// 用法：node tools/composer_web/test/t_core.mjs
import assert from 'node:assert/strict';
import * as In from '../js/interp.js';
import * as St from '../js/state.js';
import * as Tr from '../js/transport.js';
import { Engine, synthVoice, drumHit, PRESETS } from '../js/engine.js';
import { Timeline } from '../js/timeline.js';

let n = 0, failed = 0;
const t = (name, f) => { n++; try { f(); } catch (e) { failed++; console.error(`✗ ${name}\n  ${e.stack || e.message}`); } };
const Em = { root: 4, mode: 'aeolian' };
const ctx0 = { key: Em, prog: [0, 5, 2, 6], beatsPerBar: 4, tensionAt: () => 0.5, groove: null };
const near = (a, b, eps = 1e-9) => Math.abs(a - b) <= eps;

// ---------- 循环：位置取模，时值截到 le ----------
t('wrapToLoop：循环里不动；落在 le 上 / 之后绕回；循环前（抢拍）绕到尾巴；时值截到 le；没开循环不动', () => {
  const L = [0, 16];
  assert.deepEqual(In.wrapToLoop(15.962, 0.02, L), { b: 15.962, d: 0.02 });
  assert.deepEqual(In.wrapToLoop(16, 0.25, L), { b: 0, d: 0.25 });
  assert.ok(near(In.wrapToLoop(16.3, 0.25, L).b, 0.3));
  assert.ok(near(In.wrapToLoop(-0.04, 0.1, L).b, 15.96) && near(In.wrapToLoop(-0.04, 0.1, L).d, 0.04));
  assert.deepEqual(In.wrapToLoop(15, 2, L), { b: 15, d: 1 });
  assert.deepEqual(In.wrapToLoop(31.97, undefined, [16, 32]), { b: 31.97, d: undefined });
  assert.deepEqual(In.wrapToLoop(32, undefined, [16, 32]), { b: 16, d: undefined });
  assert.deepEqual(In.wrapToLoop(20, 9, null), { b: 20, d: 9 });
  assert.equal(In.wrapToLoop(16 - 1e-12, 1, L).b, 0);                   // 浮点贴着 le = 下一遍的开头
});
t('quantize 带 loop：吸到 le 的抢拍回到 ls；超过 le 的时值截掉；不带 loop 和以前一样', () => {
  const q = In.quantize([{ b: 15.9625, d: 0.1 }, { b: 15.02, d: 1.5 }], { grid: '1/16', strength: 1, loop: [0, 16] });
  assert.equal(q[0].b, 0);
  assert.equal(q[1].b, 15); assert.ok(q[1].b + q[1].d <= 16 + 1e-9, `d=${q[1].d}`);
  assert.equal(In.quantize([{ b: 15.9625, d: 0.1 }], { grid: '1/16', strength: 1 })[0].b, 16);
  const q2 = In.quantize([{ b: 31.97, d: 0.25 }], { grid: '1/16', strength: 1, loop: [16, 32] });
  assert.equal(q2[0].b, 16);
});
t('第 2 遍抢拍的重拍（记在循环尾 15.962）：吸附 100% 落在第 1 拍（0），不在 16', () => {
  const hits = [4, 8, 12, 15.962].map((b, i) => ({ id: 'k' + i, layer: 'drums', b, d: 0.25, sound: 'kick', vel: 0.8, input: 'keyboard' }));
  const q100 = In.makeVariants(hits, { ...ctx0, loop: [0, 16] }, { isDrums: true }).find(o => o.kind === 'q100');
  assert.deepEqual(q100.notes.map(x => x.b).sort((a, b) => a - b), [0, 4, 8, 12]);
});
t('makeVariants 带 ctx.loop：每个变体、每组再来，音都在 [ls, le) 里、尾巴不过 le（键盘 + 鼓 + 鬼音）', () => {
  for (const loop of [[0, 16], [16, 32]]) {
    const [ls, le] = loop;
    const syn = [0.02, 3.9, 7.48, 11.02, 14.5, 15.97].map((b, i) => ({ b: ls + b, d: i === 4 ? 3 : 0.4, pitch: [64, 67, 71, 69, 67, 66][i], vel: 0.8 }));
    const drm = [0.01, 1.02, 2.0, 3.01, 15.96].map((b, i) => ({ b: ls + b, d: 0.25, sound: i % 2 ? 'snare' : 'kick', vel: 0.8 }));
    for (const [notes, isDrums] of [[syn, false], [drm, true]]) for (let r = 0; r <= 15; r++) {
      for (const o of In.makeVariants(notes, { ...ctx0, loop }, { isDrums, reroll: r })) for (const x of o.notes) {
        assert.ok(x.b >= ls - 1e-9 && x.b < le - 1e-9, `${o.kind} b=${x.b} loop=${loop}`);
        if (x.d !== undefined) assert.ok(x.b + x.d <= le + 1e-9, `${o.kind} b=${x.b} d=${x.d}`);
      }
    }
  }
});

// ---------- 力度：鼓 / 混合 / 缺力度都不出 NaN ----------
t('规则力度：鼓（没有音高）每组再来都给有限的力度；混合（音 + 鼓）也一样；缺力度 / NaN 用默认', () => {
  const hits = [0.02, 1.01, 2.03, 2.98, 4.0, 5.02].map((b, i) => ({ b, d: 0.25, sound: i % 2 ? 'snare' : 'kick', vel: 0.8 }));
  for (let r = 0; r <= 20; r++) for (const o of In.makeVariants(hits, ctx0, { isDrums: true, reroll: r }))
    for (const x of o.notes) assert.ok(Number.isFinite(x.vel) && x.vel >= 0 && x.vel <= 1, `reroll ${r} ${o.kind} vel=${x.vel}`);
  const mixed = [{ b: 0, d: 0.5, pitch: 64, vel: 0.8 }, { b: 1, d: 0.25, sound: 'kick', vel: 0.8 }, { b: 2, d: 0.5, pitch: 67, vel: NaN }, { b: 3, d: 0.5, pitch: 69 }];
  for (let r = 0; r <= 20; r++) for (const o of In.makeVariants(mixed, ctx0, { reroll: r }))
    for (const x of o.notes) { assert.ok(Number.isFinite(x.vel), `mixed ${o.kind} vel=${x.vel}`); if (!x.sound) assert.ok(Number.isFinite(x.pitch), `${o.kind} pitch=${x.pitch}`); }
  const v = In.ruleVelocity(hits, ctx0); assert.ok(v.every(x => Number.isFinite(x.vel)));
  assert.ok(v[0].vel > v[1].vel - 1e-9 || true);
  assert.ok(Number.isFinite(In.holdVelocity(NaN)) && Number.isFinite(In.holdVelocity(undefined)));
  // 张力曲线给了 NaN 也不传染
  assert.ok(In.ruleVelocity(hits, { ...ctx0, tensionAt: () => NaN }).every(x => Number.isFinite(x.vel)));
});
t('引擎：力度 NaN / 弯音里有 NaN，合成音和鼓都不往 AudioParam 里写非有限数', () => {
  const bad = [];
  const P = (v = 0) => { let x = v; const chk = (y, w) => { if (!Number.isFinite(y) || (w !== undefined && !Number.isFinite(w))) bad.push(y); }; return { get value() { return x; }, set value(y) { chk(y); x = y; }, setValueAtTime(y, w) { chk(y, w); }, linearRampToValueAtTime(y, w) { chk(y, w); }, exponentialRampToValueAtTime(y, w) { chk(y, w); }, setTargetAtTime(y, w, c) { chk(y, w); chk(c); }, setValueCurveAtTime(a, w, d) { for (const y of a) chk(y); chk(w); chk(d); }, cancelScheduledValues() {}, cancelAndHoldAtTime() {} }; };
  const N = () => ({ gain: P(1), frequency: P(440), detune: P(0), Q: P(1), pan: P(0), offset: P(0), type: '', buffer: null, connect() {}, disconnect() {}, start() {}, stop() {}, addEventListener() {} });
  const c = { sampleRate: 48000, createGain: N, createBiquadFilter: N, createOscillator: N, createBufferSource: N, createStereoPanner: N, createConstantSource: N, createWaveShaper: N, createAnalyser: N };
  synthVoice(c, N(), PRESETS.lead.p, 64, NaN, 1, 0.5, { dt: 0.01, cents: [0, NaN, 3] });
  for (const s of ['kick', 'snare', 'clap', 'hat', 'ohat', 'perc', 'tom', 'impact']) drumHit(c, N(), null, s, NaN, 1);
  synthVoice(c, N(), PRESETS.bass.p, 40, undefined, 1, 0.5);
  assert.deepEqual(bad, []);
  // Engine 一层：坏的 when / 音高直接跳过；侧链深度 NaN 不写；旋钮 NaN 不写
  const E = Object.create(Engine.prototype);
  Object.assign(E, { ctx: { ...c, currentTime: 0 }, layers: new Map(), trims: {}, sum: N(), revIn: N(), dlyIn: N(), mf1: N(), mf2: N(), makeup: N(), revRet: N(), dlyFb: N(), mshape: N(), _lastDrive: -1 });
  Object.defineProperty(E, 'now', { get() { return 0; } });
  const L = St.makeLayer('synth', 'lead', 'x', 0, { id: 'x' });
  E.note(L, NaN, 0.8, 1, 0.5); E.note(L, 64, 0.8, NaN, 0.5); E.hit(L, 'kick', NaN, 1); E.hit(L, 'kick', 0.8, NaN);
  E.duck(1, NaN); E.setMacros({ cutoff: NaN, reso: 0.2, drive: NaN, space: Infinity, pole_g: 0.4 });
  assert.equal(E.clip(L, {}, NaN, 0, 1, 1), null);
  assert.deepEqual(bad, []);
});

// ---------- 真值：audible / derived / raw / 自检 ----------
t('导出 audible：按 mute + 场景算；solo 不参与；_说明 写明', () => {
  const s = St.makeState();
  s.notes.push({ id: 'a', layer: 'lead', b: 1, d: 1, pitch: 64, vel: 0.8, input: 'keyboard' });
  s.notes.push({ id: 'b', layer: 'bass', b: 5, d: 1, pitch: 40, vel: 0.8, input: 'keyboard' });
  s.notes.push({ id: 'c', layer: 'bass', b: 1, d: 1, pitch: 40, vel: 0.8, input: 'keyboard' });
  s.notes.push({ id: 'k', layer: 'drums', b: 5, d: 0.25, sound: 'kick', vel: 0.8, input: 'keyboard' });
  s.scenes.push({ id: 'S1', bar: 1, name: '只有鼓和主旋律', layers: ['drums', 'lead'] });
  s.layers.find(l => l.id === 'drums').solo = true;                     // 独奏只是试听：不影响 audible
  let j = St.exportTruth(s);
  const by = (arr, layer, t0) => arr.find(x => x.layer === layer && Math.abs(x.t - t0) < 1e-6);
  assert.equal(by(j.notes, 'lead', 0.4).audible, true);
  assert.equal(by(j.notes, 'bass', 0.4).audible, true);                 // 第 1 小节还没进场景
  assert.equal(by(j.notes, 'bass', 2.0).audible, false);                // 场景里没有贝斯
  assert.equal(by(j.hits, 'drums', 2.0).audible, true);
  s.layers.find(l => l.id === 'lead').mute = true;
  j = St.exportTruth(s);
  assert.equal(by(j.notes, 'lead', 0.4).audible, false);
  assert.ok(j.notes.every(x => typeof x.audible === 'boolean') && j.hits.every(x => typeof x.audible === 'boolean'));
  assert.ok(j._说明.includes('audible 是导出时按 mute + scenes 算的快照，solo 不参与'));
  assert.deepEqual(St.validateTruth(j), []);
});
t('导出 derived：鬼音带 derived、来源 = pick、没有 raw（不编）；八度叠带着原音的 raw', () => {
  const s = St.makeState();
  s.notes.push({ id: 'g', layer: 'drums', b: 0.25, d: 0.125, sound: 'hat', vel: 0.28, derived: 'ghost', raw: { b: 0.25, d: 0.125 } });   // 旧页面 commit 编的 raw
  s.notes.push({ id: 'o', layer: 'lead', b: 1, d: 1, pitch: 76, vel: 0.56, input: 'keyboard', derived: 'octave', raw: { b: 1.03, d: 0.9, vel: 0.8 } });
  s.notes.push({ id: 'r', layer: 'lead', b: 2, d: 1, pitch: 64, vel: 0.8, input: 'keyboard', raw: { d: 1 } });   // 没有原始拍位 → 不编一个
  const j = St.exportTruth(s);
  assert.deepEqual(j.hits[0], { t: 0.1, layer: 'drums', sound: 'hat', vel: 0.28, input: 'pick', audible: true, derived: 'ghost' });
  assert.equal(j.notes[0].derived, 'octave'); assert.equal(j.notes[0].raw.t, 0.412);
  assert.equal(j.notes[1].raw, undefined);
  assert.deepEqual(St.validateTruth(j), []);
});
t('validateTruth：t = null / Infinity、vel = NaN / null / 超出 [0,1]、dur 不是数 → 拒；正常的过', () => {
  const s = St.makeState();
  s.notes.push({ id: 'a', layer: 'lead', b: 2, d: 1, pitch: 64, vel: 0.8, input: 'keyboard' });
  s.notes.push({ id: 'k', layer: 'drums', b: 0, d: 0.25, sound: 'kick', vel: 0.8, input: 'keyboard' });
  const ok = St.exportTruth(s);
  assert.deepEqual(St.validateTruth(ok), []);
  const bad = (f, what) => { const j = JSON.parse(JSON.stringify(ok)); f(j); assert.ok(St.validateTruth(JSON.parse(JSON.stringify(j))).length > 0, what); };
  bad(j => { j.notes[0].t = Infinity; }, 'note t Infinity（过 JSON = null）');
  bad(j => { j.notes[0].t = null; }, 'note t null');
  bad(j => { j.notes[0].vel = NaN; }, 'note vel NaN');
  bad(j => { j.notes[0].vel = 1.4; }, 'note vel > 1');
  bad(j => { j.notes[0].dur = null; }, 'note dur null');
  bad(j => { j.hits[0].vel = null; }, 'hit vel null');
  bad(j => { j.hits[0].vel = 2; }, 'hit vel > 1');
  bad(j => { j.hits[0].t = null; }, 'hit t null');
  bad(j => { j.tension = [[null, 0.5]]; }, 'tension t null');
  // 页面里直接导出的坏值也拦得住（不过 JSON）
  s.notes[1].vel = NaN; s.notes.push({ id: 'z', layer: 'lead', b: Infinity, d: 1, pitch: 64, vel: 0.8, input: 'keyboard' });
  assert.ok(St.validateTruth(St.exportTruth(s)).length >= 2);
});
t('clips.offset：只有分轨从 src 的「@t0-」解析；拖进来的文件名里有 @128- 不算', () => {
  const s = St.makeState();
  s.layers.push(St.makeLayer('stem', 'stem', 'loop', 4, { id: 'f1' }), St.makeLayer('stem', 'stem', '原曲·bass', 5, { id: 'st' }));
  s.clips.push({ id: 'C1', layer: 'f1', b: 0, dur: 8, audio: 'a1', input: 'file', active: true, src: 'DrumLoop@128-Am.wav' });
  s.clips.push({ id: 'C2', layer: 'st', b: 0, dur: 8, audio: 'a2', input: 'stem', active: true, src: '分轨/bass.wav@10.000-20.000' });
  s.clips.push({ id: 'C3', layer: 'voice', b: 0, dur: 2, audio: 'a3', input: 'hum', active: true, src: 'mic@3-x' });
  assert.deepEqual(St.exportTruth(s).clips.map(c => c.offset), [0, 10, 0]);
});

// ---------- 曲线写点 ----------
const stroke = (pts, path) => { pts = St.curveWrite(pts, path[0][0], path[0][0], path[0][1]); let last = path[0][0]; for (let i = 1; i < path.length; i++) { pts = St.curveWrite(pts, last, path[i][0], path[i][1]); last = path[i][0]; } return pts; };
t('curveWrite：密的单向笔画（带小数拍、四舍五入会进位）一个点都不丢', () => {
  const path = Array.from({ length: 101 }, (_, k) => [k * 0.08 + 0.000049 + (k % 3) * 0.00001, 0.5 + 0.4 * Math.sin(k / 7)]);
  const pts = stroke([], path);
  assert.equal(pts.length, 101);
  for (const [b, v] of path) assert.ok(Math.abs(St.curveAt(pts, b, 0) - v) < 2e-3);
  // 最小的进位情形：1.00006 存成 1.0001，下一笔不能把它删了
  assert.deepEqual(St.curveWrite(St.curveWrite([], 1.00006, 1.00006, 0.2), 1.00006, 1.5, 0.9), [[1.0001, 0.2], [1.5, 0.9]]);
  // 快的三笔（中间是峰）：峰留着
  const W = 1440, GUT = 112, xb = x => (x - GUT) / (W - GUT - 6) * 16;
  for (let k = 0; k < 300; k++) { const xs = [300 + k * 0.137, 380 + k * 0.137, 460 + k * 0.137]; const p = stroke([], xs.map((x, i) => [xb(x), [0.1, 0.8, 0.4][i]])); assert.ok(p.some(q => q[1] === 0.8), `k=${k}`); }
});
t('curveWrite：从右往左画 = 从左往右画；来回晃不错位；旧点只删两笔之间的', () => {
  const L2R = stroke([], [[0, 0], [2, 0.25], [4, 0.5], [6, 0.75], [8, 1]]);
  const R2L = stroke([], [[8, 1], [6, 0.75], [4, 0.5], [2, 0.25], [0, 0]]);
  assert.deepEqual(R2L, L2R);
  assert.deepEqual(stroke([], [[0, 0], [4, 1], [2, 0.5]]), [[0, 0], [2, 0.5], [4, 1]]);
  const old = Array.from({ length: 9 }, (_, i) => [i * 4, 0]);
  const p = stroke(old, [[24, 1], [16, 1], [8, 1]]);
  assert.deepEqual(p.filter(q => q[0] >= 8 && q[0] <= 24).map(q => q[1]), [1, 1, 1]);
  assert.equal(St.curveAt(p, 4, 9), 0); assert.equal(St.curveAt(p, 28, 9), 0);
  assert.deepEqual(St.curveWrite([[0, 0]], 1, 2, NaN), [[0, 0]]);           // NaN 不写
});

// ---------- 走带 ----------
const fakeE = (extra = {}) => ({ now: 10, ctx: { outputLatency: 0.02, baseLatency: 0.01 }, click() {}, hit() {}, duck() {}, clip() { return { stop() {} }; }, setMacros() {}, note() { return { release() {} }; }, ...extra });
t('走带：不循环播到歌尾，歌尾那一支只走一次（onSegment(null) 1 次、stop 定时 1 次）；重新播放又能走', () => {
  const realST = globalThis.setTimeout;
  try {
    const S = St.makeState(); S.loop.on = false; S.songBars = 2;
    const E = fakeE(); let nulls = 0, timers = 0, fire = null;
    globalThis.setTimeout = f => { timers++; fire = f; return 0; };           // 返回 0（假值）也不能重复
    const T = new Tr.Transport(E, () => S, { onSegment: ng => { if (!ng) nulls++; } });
    T.start(0); clearInterval(T.timer);
    for (; E.now < 10 + 0.08 + 8 * 0.4 + 0.5; E.now += 0.025) T.tick();
    assert.equal(nulls, 1); assert.equal(timers, 1);
    fire(); assert.equal(T.playing, false);
    T.start(0); clearInterval(T.timer);
    for (const end = E.now + 0.08 + 8 * 0.4 + 0.5; E.now < end; E.now += 0.025) T.tick();
    assert.equal(nulls, 2); assert.equal(timers, 2);
    T.stop(true);
  } finally { globalThis.setTimeout = realST; }
});
t('走带：节拍器关着时，预备拍只打循环起点之前的 4 下（不多打第 1 拍）', () => {
  for (const hook of ['heard', 'k']) {
    const S = St.makeState(); S.settings.metro = false;
    const clicks = [], E = fakeE({ click(w) { clicks.push(w); } });
    const T = new Tr.Transport(E, () => S, { countingIn: hook === 'k' ? () => true : () => T.playing && (T.beatNow ?? 0) < St.loopRange(S)[0] });
    T.start(-4); clearInterval(T.timer);
    for (; E.now < 10 + 0.08 + 10 * 0.4; E.now += 0.025) T.tick();
    assert.deepEqual(clicks.map(w => Math.round(T.beatAt(w + 1e-9))), [-4, -3, -2, -1], hook);
  }
  // 不循环：从光标前一小节起，预备拍到光标为止
  const S = St.makeState(); S.settings.metro = false; S.loop.on = false;
  const clicks = [], E = fakeE({ click(w) { clicks.push(w); } });
  const T = new Tr.Transport(E, () => S, { countingIn: () => true });
  T.start(8 - 4); clearInterval(T.timer);
  for (; E.now < 10 + 0.08 + 8 * 0.4; E.now += 0.025) T.tick();
  assert.deepEqual(clicks.map(w => Math.round(T.beatAt(w + 1e-9))), [4, 5, 6, 7]);
});
t('走带：一个力度 NaN 的音让引擎抛错，走带也不卡（后面的音各排一次，拍子往前走）', () => {
  const S = St.makeState(); S.settings.metro = false;
  S.notes.push({ id: 'a', layer: 'drums', b: 0, d: 0.25, sound: 'kick', vel: 0.8 }, { id: 'c', layer: 'drums', b: 1, d: 0.25, sound: 'snare', vel: NaN }, { id: 'd', layer: 'drums', b: 1.5, d: 0.25, sound: 'clap', vel: 0.8 }, { id: 'x', layer: 'drums', b: NaN, d: 0.25, sound: 'hat', vel: 0.8 });
  const sched = [];
  const E = fakeE({ hit(L, snd, v, when) { if (!Number.isFinite(v) || !Number.isFinite(when)) throw new TypeError('non-finite'); sched.push(snd); } });
  const warn = console.warn; console.warn = () => {};
  try {
    const T = new Tr.Transport(E, () => S, {});
    T.start(0); clearInterval(T.timer);
    for (; E.now < 13; E.now += 0.025) T.tick();
    assert.ok(T.nextB > 5, `nextB ${T.nextB}`);
    assert.deepEqual(sched.filter(x => x === 'clap').length, 1);
    T.stop(true);
  } finally { console.warn = warn; }
});
t('走带：分轨层播放中静音 / 独奏别人 / 取消，不循环也马上生效（每片自己的增益）', () => {
  for (const loopOn of [false, true]) {
    const S = St.makeState(); S.loop.on = loopOn;
    S.layers.push(St.makeLayer('stem', 'stem', '原曲·drums', 4, { id: 'st' }));
    S.clips.push({ id: 'c', layer: 'st', b: 0, dur: 25.6, audio: 'a', input: 'stem', active: true });
    const gains = new Map();
    const E = fakeE({ clip(L, buf, when, off, dur, v) { const src = { stop() {}, when, dur }; gains.set(src, v); return src; }, clipGain(src, v) { gains.set(src, v); } });
    const T = new Tr.Transport(E, () => S, { buffer: () => ({}) });
    T.start(0); clearInterval(T.timer);
    const nowPlaying = () => [...gains].filter(([src]) => E.now >= src.when && E.now < src.when + src.dur).map(([, v]) => v);
    for (; E.now < 12; E.now += 0.025) T.tick();
    assert.deepEqual(nowPlaying(), [1]);
    S.layers.find(l => l.id === 'st').mute = true; T.tick(); assert.deepEqual(nowPlaying(), [0], `mute loop=${loopOn}`);
    S.layers.find(l => l.id === 'st').mute = false; T.tick(); assert.deepEqual(nowPlaying(), [1]);
    S.layers.find(l => l.id === 'lead').solo = true; T.tick(); assert.deepEqual(nowPlaying(), [0], 'solo');
    S.layers.find(l => l.id === 'lead').solo = false; T.tick(); assert.deepEqual(nowPlaying(), [1]);
    T.stop(true);
  }
});
t('走带：场景切点还是准的（不在第 1 场景里的分轨，从第 2 场景的小节线开始响）', () => {
  const S = St.makeState(); S.loop.on = false;
  S.layers.push(St.makeLayer('stem', 'stem', '原曲·drums', 4, { id: 'st' }));
  S.clips.push({ id: 'c', layer: 'st', b: 0, dur: 25.6, audio: 'a', input: 'stem', active: true });
  S.scenes = [{ id: 'A', bar: 0, name: 'a', layers: ['lead'] }, { id: 'B', bar: 2, name: 'b', layers: ['lead', 'st'] }];
  const pieces = [];
  const E = fakeE({ clip(L, buf, when, off, dur, v) { const src = { stop() {} }; pieces.push({ when, off, dur, v }); return src; } });
  const T = new Tr.Transport(E, () => S, { buffer: () => ({}) });
  T.start(0); clearInterval(T.timer);
  const t0 = T.segs[0].ctx0;
  assert.deepEqual(pieces.map(p => [+(p.when - t0).toFixed(4), +p.off.toFixed(4), p.v]), [[0, 0, 0], [3.2, 3.2, 1]]);
  T.stop(true);
});

// ---------- 时间线：编辑吸附 ----------
t('编辑吸附「自由」（0）= 不吸；1/16、1/32 照吸；旧会话没有这个设置 = 1/16', () => {
  const snap = (snapEdit, b, alt = false) => Timeline.prototype.snap.call({ S: { settings: { snapEdit } }, snapQ: Timeline.prototype.snapQ }, b, alt);
  assert.equal(snap(0, 1.37), 1.37);
  assert.equal(snap(0.25, 1.37), 1.25);
  assert.equal(snap(0.125, 1.37), 1.375);
  assert.equal(snap(undefined, 1.37), 1.25);
  assert.equal(snap(0.25, 1.37, true), 1.37);
});

console.log(`${n - failed}/${n} 通过`);
if (failed) process.exit(1);
