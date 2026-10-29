// >v<🎛作曲器-自测-页面主逻辑（node 跑，不需要浏览器）：把整份 main.js 放进假的 DOM / 假时钟 / 假引擎里跑，真的走带、真的解释层。
// 测：录音落点（循环取模 / 截尾 / 预备拍）、录音锁层、琶音跨循环、关循环哼唱、等分析完再挑、挑哪一遍的试听、复制粘贴、
// 撤销 / 变体、音色变体撤销、新层进场景、改速度、XY 按住、保存 / 打开 / 换作品 / 离线、分轨重切、后台线程分析。
// 用法：node tools/composer_web/test/t_main.mjs
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import { voiceLike } from './synth_signals.mjs';
import * as Th from '../js/theory.js';
import * as En from '../js/engine.js';
import * as Tr from '../js/transport.js';
import * as St from '../js/state.js';
import * as Pi from '../js/pitch.js';
import * as In from '../js/interp.js';

const MAIN = readFileSync(new URL('../js/main.js', import.meta.url), 'utf8');

// ---------- hum-worker.js：用假的 self 载入，拿到它的 onmessage ----------
// （worker 里的 self 在调用时才查，所以这个假的 self 一直留着）
globalThis.self = { postMessage() {} };
await import('../js/hum-worker.js');
const workerOnMessage = globalThis.self.onmessage;
const workerSelf = globalThis.self;

// ---------- 假时钟 + 定时器（走带 / 琶音 / 自动存档 / 哼唱等待都走它）----------
const REAL_TIMERS = { setTimeout: globalThis.setTimeout, setInterval: globalThis.setInterval, clearTimeout: globalThis.clearTimeout, clearInterval: globalThis.clearInterval };
const sched = { now: 10, timers: [], id: 0 };
const fakeSetTimeout = (f, ms = 0) => { const id = ++sched.id; sched.timers.push({ id, at: sched.now + Math.max(0, +ms || 0) / 1000, f, every: 0 }); return id; };
const fakeSetInterval = (f, ms = 0) => { const id = ++sched.id; sched.timers.push({ id, at: sched.now + ms / 1000, f, every: Math.max(0.001, ms / 1000) }); return id; };
const fakeClear = id => { sched.timers = sched.timers.filter(t => t.id !== id); };
const flush = async (k = 6) => { for (let i = 0; i < k; i++) await new Promise(r => setImmediate(r)); };

// ---------- 假 DOM ----------
const ctx2d = new Proxy({}, { get: (t, p) => (p in t ? t[p] : () => ({ width: 0 })), set: (t, p, v) => { t[p] = v; return true; } });
function fakeEl(tag = 'div', id = '', doc) {
  const el = {
    tagName: tag.toUpperCase(), id, style: {}, dataset: {}, value: '', textContent: '', checked: false, disabled: false, type: '', width: 300, height: 150, files: [],
    _kids: [], _listeners: {}, _html: '',
    classList: { _s: new Set(), add(c) { this._s.add(c); }, remove(c) { this._s.delete(c); }, toggle(c, on) { if (on === undefined) on = !this._s.has(c); if (on) this._s.add(c); else this._s.delete(c); return on; }, contains(c) { return this._s.has(c); } },
    get innerHTML() { return this._html; }, set innerHTML(v) { this._html = v; this._kids = []; },
    get children() { const k = this._kids; return new Proxy(k, { get: (a, p) => (typeof p === 'string' && /^\d+$/.test(p) ? (a[p] || (a[p] = fakeEl('div', '', doc))) : Reflect.get(a, p)) }); },
    get lastChild() { return this._kids[this._kids.length - 1] || fakeEl('span', '', doc); },
    get offsetParent() { return null; },
    appendChild(c) { this._kids.push(c); return c; },
    querySelector() { return fakeEl('div', '', doc); }, querySelectorAll() { return []; },
    addEventListener(k, f) { (this._listeners[k] ||= []).push(f); }, removeEventListener() {},
    getContext() { return ctx2d; }, getBoundingClientRect() { return { left: 0, top: 0, width: 100, height: 100 }; },
    setPointerCapture() {}, focus() { if (doc) doc.activeElement = el; }, blur() { el._blurred = (el._blurred || 0) + 1; if (doc && doc.activeElement === el) doc.activeElement = null; },
    click() { el._clicked = (el._clicked || 0) + 1; }, closest() { return null; },
  };
  return el;
}
function fakeDocument() {
  const els = new Map(), listeners = {};
  const doc = {
    activeElement: null, title: '', body: null,
    getElementById(id) { if (!els.has(id)) els.set(id, fakeEl(/Sel$|^work$|^root$|^mode$|^addLayer$|^velMode$|^tapPat$|^snapEdit$|^loopBars$|^anchorSel$/.test(id) ? 'select' : /^(bpm|songBars|anchor|prog|fileIn|tParent)$/.test(id) ? 'input' : 'div', id, doc)); return els.get(id); },
    createElement(tag) { return fakeEl(tag, '', doc); },
    addEventListener(k, f) { (listeners[k] ||= []).push(f); },
    fire(k, ev) { for (const f of listeners[k] || []) f(ev); },
  };
  doc.body = fakeEl('body', 'body', doc);
  for (const id of ['bpm', 'songBars', 'anchor']) doc.getElementById(id).type = 'number';
  return doc;
}

// ---------- 假音频 ----------
function fakeBuffer(nch, n, sr) {
  const ch = Array.from({ length: nch }, () => new Float32Array(Math.max(1, n)));
  return { sampleRate: sr, length: n, duration: n / sr, numberOfChannels: nch, getChannelData: i => ch[i], copyToChannel(src, i) { ch[i].set(src.subarray(0, ch[i].length)); } };
}
class FakeEngine {
  constructor({ silent = false } = {}) {
    this.silent = silent; this.notes = []; this.hits = []; this.clicks = []; this.clipsStarted = []; this.layers = new Map(); this.trims = {};
    const w = FakeEngine.world;
    this.ctx = {
      sampleRate: 48000, outputLatency: w.outLat, baseLatency: 0, destination: {},
      resume: async () => {},
      getOutputTimestamp: () => ({ contextTime: sched.now - w.outLat, performanceTime: sched.now * 1000 }),
      createBuffer: (c, n, sr) => fakeBuffer(c, n, sr),
      decodeAudioData: async ab => { const b = fakeBuffer(ab.ch || 1, Math.round((ab.dur || 1) * 48000), 48000); for (let k = 0; k < b.numberOfChannels; k++) b.getChannelData(k).fill(ab.fill?.[k] ?? 0.1); return b; },
      audioWorklet: { addModule: async () => {} },
      createMediaStreamSource: () => ({ connect() {} }),
      createGain: () => ({ gain: { value: 1 }, connect() {} }),
    };
    this.speaker = {}; this.scopeL = { getFloatTimeDomainData() {} }; this.scopeR = { getFloatTimeDomainData() {} };
  }
  get now() { return sched.now; }
  note(L, pitch, vel, when, dur = null) { const v = { layer: L.id, pitch, vel, when, dur, at: sched.now, canceled: false, release(t) { v.rel = t; }, cancel() { v.canceled = true; } }; this.notes.push(v); return v; }
  hit(L, sound, vel, when) { this.hits.push({ layer: L.id, sound, vel, when, at: sched.now }); }
  duck() {} click(when) { this.clicks.push(when); }
  clip(L, buf, when, off, dur, gain) { const s = { layer: L.id, buf, when, off, dur, gain, stop() {} }; this.clipsStarted.push(s); return s; }
  clipGain() {} setMacros() {} updateLayer(L, a) { this.updated = (this.updated || 0) + 1; this.lastUpdate = { id: L.id, params: { ...L.params } }; } ensureLayer(L) { this.layers.set(L.id, L); return {}; } removeLayer() {}
  meterDb() { return -60; } measureTrims() { return Promise.resolve({}); } setDelayTime() {}
}
class FakeTimeline { constructor(cv, api) { this.api = api; this.sel = new Set(); this.range = null; this.cursor = 0; } resize() {} draw() {} fit() {} }
class FakeWorkletNode { constructor() { this.port = {}; } connect() {} disconnect() {} }
// 后台线程：消息协议和 hum-worker.js 一样；mode = 'real'（真跑 hum-worker.js）| 'empty'（秒回空结果）
function makeWorkerClass(w) {
  return class FakeWorker {
    constructor(url, opts) { w.workerUrls.push([url, opts]); if (w.worker === 'throws') throw new Error('no worker'); }
    postMessage(msg) {
      w.workerJobs++;
      setImmediate(() => {
        if (w.worker === 'empty') return this.onmessage({ data: { id: msg.id, an: { mode: 'hum', gate: -60, guess: { mode: 'hum', why: '测试', stats: {} }, tuneCents: 0, segs: [] } } });
        if (w.worker === 'crash') return this.onerror({ preventDefault() {} });
        workerSelf.postMessage = m => this.onmessage({ data: m });
        workerOnMessage({ data: msg });
      });
    }
    terminate() {}
  };
}

// ---------- 一个世界 = 一份全新的 main.js ----------
const MODS = { theory: Th, engine: { ...En, Engine: FakeEngine }, transport: Tr, timeline: { Timeline: FakeTimeline }, state: St, pitch: Pi, interp: In };
// 没有的名字给 undefined（拿旧版 main.js 跑同一个场景做对照时用：opts.src = 旧源码）
const NAMES = ['mic', 'hum', 'rec', 'held', 'xy', 'audio',
  'boot', 'frame', 'onKeyDown', 'noteDown', 'noteUp', 'arpTick', 'toggleRec', 'stopRec', 'processHumTake', 'analyzeAsync', 'openTakeVariants', 'openNoteVariants', 'openHumInterp',
  'openTimbreVariants', 'previewVariant', 'commitVariant', 'closeVariants', 'undo', 'redo', 'snapshot', 'copySel', 'pasteAt', 'duplicate', 'deleteSel', 'reanalyze', 'copyHumToSynth',
  'save', 'loadSession', 'xyWrite', 'xyPos', 'xyApply', 'injectHum', 'events', 'clipsNow', 'clipsForPlay', 'placeRecBeat', 'selectLayer', 'reloadStems', 'retimeHum', 'hasUnsavedAudio', 'micOn',
  'addFileLayer', 'loadStems', 'joinScenes', 'micExtract', 'humPassEnd'];
const EXPOSE = `return {
  get S() { return S; }, set S(v) { S = v; }, get E() { return E; }, get T() { return T; }, get TL() { return TL; },
  get variants() { return variants; }, set variants(v) { variants = v; }, get clipboard() { return clipboard; }, get undoStack() { return undoStack; }, get redoStack() { return redoStack; },
  get groove() { return groove; }, get server() { return server; }, set server(v) { server = v; },
  ${NAMES.map(k => `${k}: typeof ${k} === 'undefined' ? undefined : ${k}`).join(',\n  ')}
};`;
const PARAMS = ['document', 'window', 'location', 'navigator', 'localStorage', 'setInterval', 'setTimeout', 'clearInterval', 'clearTimeout', 'requestAnimationFrame', 'confirm', 'prompt', 'fetch', 'Worker', 'AudioWorkletNode', 'console'];
const factories = new Map();
export const SRC = { override: null };                     // 对照用：换一份 main.js 源码跑同样的测试
function factoryOf(src) {
  if (factories.has(src)) return factories.get(src);
  const imports = [...src.matchAll(/^import \{([^}]+)\} from '\.\/([\w-]+)\.js';\s*$/gm)];
  const importNames = imports.flatMap(m => m[1].split(',').map(s => s.trim()).filter(Boolean).map(name => [name, m[2]]));
  const body = src.replace(/^import \{[^}]+\} from '\.\/[\w-]+\.js';\s*$/gm, '');
  const f = { importNames, make: new Function(...importNames.map(([nm]) => nm), ...PARAMS, body + '\n' + EXPOSE) };
  factories.set(src, f);
  return f;
}

export async function world(opts = {}) {
  const w = { outLat: 0.03, worker: 'real', server: true, confirmAnswer: true, confirms: [], saves: [], fetches: [], workerUrls: [], workerJobs: 0, downloads: 0, sessions: {}, files: {}, mic: null, ...opts };
  FakeEngine.world = w;
  sched.now = 10; sched.timers = [];
  const doc = fakeDocument();
  w.statusLog = [];
  { const st = doc.getElementById('status'); let txt = ''; Object.defineProperty(st, 'textContent', { get: () => txt, set: v => { txt = v; if (v) w.statusLog.push(v); } }); }
  const winListeners = {};
  const window = { addEventListener(k, f) { (winListeners[k] ||= []).push(f); }, fire(k, ev) { for (const f of winListeners[k] || []) f(ev); } };
  const store = new Map();
  const localStorage = { getItem: k => store.get(k) ?? null, setItem: (k, v) => store.set(k, v) };
  const track = { _l: [], addEventListener(k, f) { this._l.push(f); }, stop() { this.stopped = true; } };
  const navigator = { mediaDevices: { getUserMedia: async () => { if (w.micDenied) throw new Error('NotAllowedError'); return { getAudioTracks: () => [track], getTracks: () => [track] }; }, addEventListener() {} } };
  const jsonRes = (j, status = 200) => ({ ok: status < 400, status, json: async () => j, text: async () => JSON.stringify(j), arrayBuffer: async () => j });
  const fetch = async (url, opt = {}) => {
    w.fetches.push(url);
    if (!w.server) throw new TypeError('Failed to fetch');
    const u = new URL(url, 'http://127.0.0.1');
    const p = u.pathname, q = Object.fromEntries(u.searchParams);
    if (p === '/api/works') return jsonRes([{ name: 'MV_T' }, { name: 'MV_B' }]);
    if (p === '/api/sense') return jsonRes(w.sense || { bpm: 150, downbeats: [0, 1.6] });
    if (p === '/api/final') return jsonRes({ final: null });
    if (p === '/api/stems') return jsonRes(w.stems || []);
    if (p === '/api/stem') { w.stemReq = (w.stemReq || []).concat([q]); return jsonRes({ dur: +q.t1 - +q.t0, ch: 2 }); }
    if (p === '/api/next') return jsonRes({ ver: 'v' + (w.saves.length + 1) });
    if (p === '/api/save') { const b = JSON.parse(opt.body); w.saves.push(b); return jsonRes({ dir: 'x', written: [`作曲_${b.ver}.json`], truth: `作曲_${b.ver}.json` }); }
    if (p === '/api/sessions') return jsonRes(Object.keys(w.sessions).map(name => ({ name, mtime: '' })));
    if (p === '/api/file') { if (w.sessions[q.name]) return jsonRes(w.sessions[q.name]); return jsonRes({ dur: 2, ch: 1 }); }
    return jsonRes({ err: p }, 404);
  };
  const rafs = [];
  const quiet = { log() {}, warn() {}, error: (...a) => console.error(...a) };
  const fac = factoryOf(w.src || SRC.override || MAIN);
  const M = fac.make(...fac.importNames.map(([nm, mod]) => MODS[mod][nm]), doc, window, { search: '?silent=1' }, navigator, localStorage,
    fakeSetInterval, fakeSetTimeout, fakeClear, fakeClear, f => rafs.push(f), q => { w.confirms.push(q); return w.confirmAnswer; }, () => null, fetch,
    w.worker === 'none' ? undefined : makeWorkerClass(w), FakeWorkletNode, quiet);
  // 走带 transport.js 用全局的 setTimeout / setInterval：换成假时钟（每个测试完换回来）
  globalThis.setTimeout = fakeSetTimeout; globalThis.setInterval = fakeSetInterval; globalThis.clearTimeout = fakeClear; globalThis.clearInterval = fakeClear;
  w.restoreTimers = () => { Object.assign(globalThis, REAL_TIMERS); };
  Object.assign(w, { M, doc, window, track, rafs, store, $: id => doc.getElementById(id) });
  await M.boot(); await flush();
  w.status = () => doc.getElementById('status').textContent;
  w.sel = id => (M.selectLayer ? M.selectLayer(id) : (M.S.sel = id));     // 旧版 main.js 没有 selectLayer
  w.xyDown = ev => doc.getElementById('xy')._listeners.pointerdown[0](ev);
  // 麦克风：worklet 每 1024 个样本发一块；w.voice = { t0, x } 是「唱进麦克风」的信号（音频时钟 t0 起）
  let micT = null;
  const feedMic = () => {
    const m = M.mic; if (!m.on || !m.node || !m.node.port.onmessage) { micT = null; return; }
    if (micT === null) micT = sched.now;
    while (micT + 1024 / 48000 <= sched.now) {
      let x;
      if (w.voice) { const i0 = Math.round((micT - w.voice.t0) * 48000); x = new Float32Array(1024); for (let i = 0; i < 1024; i++) { const k = i0 + i; x[i] = k >= 0 && k < w.voice.x.length ? w.voice.x[k] : 0; } }
      else x = w.voiceConst !== undefined ? new Float32Array(1024).fill(w.voiceConst) : ZEROS;
      m.node.port.onmessage({ data: { t: micT, x } });
      micT += 1024 / 48000;
    }
  };
  w.advance = async (dt, step = 0.004) => {
    const end = sched.now + dt;
    while (sched.now < end - 1e-12) {
      sched.now = Math.min(end, sched.now + step);
      feedMic();
      let oneShot = false;
      for (let g = 0; g < 1000; g++) {
        const due = sched.timers.filter(t => t.at <= sched.now + 1e-12).sort((a, b) => a.at - b.at)[0];
        if (!due) break;
        if (due.every) due.at += due.every; else { sched.timers = sched.timers.filter(t => t !== due); oneShot = true; }
        due.f();
      }
      if (oneShot) await flush();
    }
    await flush();
  };
  w.until = async (cond, max = 30, step = 0.004) => { const end = sched.now + max; while (!cond() && sched.now < end) await w.advance(0.02, step); return cond(); };
  // 键：现在按下 / 松开（听到的时刻 = 现在 − 输出延迟）
  const kev = (code, extra = {}) => ({ code, key: code, target: doc.body, timeStamp: sched.now * 1000, shiftKey: false, repeat: false, preventDefault() {}, ...extra });
  w.down = (code, extra) => M.onKeyDown(kev(code, extra));
  w.up = code => M.noteUp(kev(code));
  w.tap = async (code, hold = 0.05) => { w.down(code); await w.advance(hold); w.up(code); };
  w.heardBeat = () => M.T.beatNow;
  // 等到听到第 b 拍（在当前段里）
  w.toBeat = async (b, step = 0.002) => { await w.until(() => M.T.playing && M.T.beatNow !== null && M.T.beatNow >= b - 1e-9, 120, step); };
  return w;
}
const ZEROS = new Float32Array(1024);

let n = 0, failed = 0;
const tests = [];
const t = (name, f) => tests.push([name, f]);
const near = (a, b, eps = 1e-6) => Math.abs(a - b) <= eps;
const truthOf = M => St.exportTruth(M.S);

// ====================== C5 / C16 / C37：复制 / 粘贴 ======================
t('C5 Ctrl+D 没选东西、剪贴板里有旧的：什么都不贴（不出 Infinity），只提示；Ctrl+V 没有光标也不贴', async () => {
  const w = await world(), M = w.M;
  M.S.notes.push({ id: 'n1', layer: 'lead', b: 2, d: 1, pitch: 64, vel: 0.8, input: 'keyboard', raw: { b: 2, d: 1 } });
  M.TL.sel = new Set(['n1']); assert.equal(M.copySel(), true);
  M.TL.sel.clear(); M.TL.range = null;
  M.duplicate();
  assert.equal(M.S.notes.length, 1);
  assert.match(w.status(), /先选/);
  M.TL.sel = new Set(['gone']);                            // 选中的音已经不在了
  M.duplicate(); assert.equal(M.S.notes.length, 1);
  M.TL.cursor = null; M.pasteAt(M.TL.cursor);
  assert.equal(M.S.notes.length, 1); assert.match(w.status(), /贴到哪/);
  assert.ok(M.S.notes.every(x => Number.isFinite(x.b)));
  assert.deepEqual(St.validateTruth(truthOf(M)), []);
  w.restoreTimers();
});
t('C16/C37 贴出来的哼唱音不再挂在那一遍上：录新的一遍 / 重新识别都不动它；raw 跟着挪', async () => {
  const w = await world(), M = w.M;
  M.audio.set('a1', { buffer: fakeBuffer(1, 4800, 48000), pcm: new Float32Array(4800), sr: 48000, file: null });
  M.S.clips.push({ id: 'C1', layer: 'voice', b: 0, dur: 6.4, audio: 'a1', input: 'hum', take: 1, active: true, src: 'mic', analysis: { mode: 'hum' } });
  M.S.notes.push({ id: 'h1', layer: 'voice', b: 1, d: 1, pitch: 64, vel: 0.8, input: 'hum', take: 'C1', raw: { b: 1.02, d: 1 } });
  M.TL.range = [0, 4]; M.copySel(); M.pasteAt(16);
  const pasted = M.S.notes.find(x => x.b === 17);
  assert.ok(pasted && pasted.take === undefined, '贴出来的没有 take');
  assert.ok(near(pasted.raw.b, 17.02), 'raw.b 跟着挪');
  for (const c of M.S.clips) c.active = false;            // 录了新的一遍：旧的那遍不用了
  assert.deepEqual(truthOf(M).notes.map(x => x.t), [St.exportTruth(M.S).window[0] + 17 * 0.4].map(x => +x.toFixed(4)));
  M.S.clips[0].active = true;
  await M.reanalyze(M.S.layers.find(l => l.id === 'voice')); await w.advance(0.05);
  assert.ok(M.S.notes.some(x => x.id === pasted.id), '重新识别不删贴出来的');
  w.restoreTimers();
});

// ====================== C7：只取半音的音分 ======================
t('C7 原样（半音）：换了音高，音分曲线跟着挪（音高 + 音分 = 唱的音）', async () => {
  const w = await world(), M = w.M;
  const mel = [{ midi: 64, dur: 0.5 }, { midi: 65, dur: 0.5 }, { midi: 70, dur: 0.5 }, { midi: 67, dur: 0.5 }];
  const r = Pi.analyzeHum(voiceLike(mel), 48000);
  const ns = Pi.notesToBeats(r.segs, { t0Beat: 0, spb: 0.4, key: M.S.key, prog: M.S.prog, snap: 'scale' }).map((x, i) => ({ id: 'h' + i, layer: 'voice', input: 'hum', take: 'C1', ...x, raw: { b: x.b, d: x.d, ...x.raw } }));
  M.S.notes = ns; M.S.clips = [{ id: 'C1', layer: 'voice', b: 0, dur: 2, audio: 'a', input: 'hum', take: 1, active: true }];
  M.openHumInterp('voice', M.S.clips[0]);
  const med = a => [...a].sort((p, q) => p - q)[a.length >> 1];
  const un = M.variants.options[0];
  assert.equal(un.label, '原样（半音）');
  un.notes.forEach((x, i) => { const eff = x.pitch + med(x.bend.cents) / 100; assert.ok(Math.abs(eff - mel[i].midi) < 0.15, `唱 ${mel[i].midi} 放 ${eff.toFixed(2)}`); });
  assert.ok(ns.every(x => x.bend.cents !== un.notes[0].bend.cents), '不改原来的音');
  w.restoreTimers();
});

// ====================== D1：录音落点 ======================
t('D1/C8/C33 开循环：第一遍抢拍的强拍放到循环尾巴（在 [ls, le) 里），raw 是真按的拍；吸附 100% 绕回 0；能存', async () => {
  const w = await world(), M = w.M;
  w.sel('drums');
  await M.toggleRec();
  const sp = 0.4, g0 = M.T.segs[0];
  await w.until(() => sched.now - w.outLat >= g0.ctx0 + 4 * sp - 0.015);    // 听到的是第 0 拍前 15 ms
  await w.tap('KeyA', 0.05);
  const nt = M.S.notes[0];
  assert.ok(nt.b >= 0 && nt.b < 16 && nt.b > 15.9, `落在循环尾巴 ${nt.b}`);
  assert.ok(nt.raw.b <= 0, 'raw = 真按下的拍（循环头之前，不早于歌的开头）');
  assert.ok(nt.b + nt.d <= 16 + 1e-9, '时值截在循环尾巴');
  M.toggleRec();
  const q100 = M.variants.options.find(o => o.kind === 'q100');
  assert.ok(q100.notes.every(x => x.b >= 0 && x.b < 16), 'q100 不出循环：' + q100.notes.map(x => x.b));
  assert.ok(q100.notes.some(x => near(x.b, 0)), '吸到循环头');
  for (const o of M.variants.options) { const s = St.makeState(); s.notes = o.notes.map(x => ({ ...x, id: St.uid() })); assert.deepEqual(St.validateTruth(St.exportTruth(s)), [], o.kind); }
  w.restoreTimers();
});
t('D1/C36 第二遍以后抢拍的强拍也在尾巴上，吸附 100% 绕回循环头；关循环：预备拍里乱按的不记，晚半拍以内的记且不早于 0', async () => {
  const w = await world(), M = w.M;
  w.sel('drums');
  await M.toggleRec();
  await w.toBeat(4);                                      // 进了循环
  await w.until(() => M.T.segs.length >= 3);            // 第二遍的段排出来了
  const g2 = M.T.segs[M.T.segs.length - 1];
  await w.until(() => sched.now - w.outLat >= g2.ctx0 - 0.015);
  await w.tap('KeyA');
  M.toggleRec();
  const last = M.S.notes[M.S.notes.length - 1];
  assert.ok(last.b > 15.9 && last.b < 16, `第二遍抢拍 ${last.b}`);
  const q100 = M.variants.options.find(o => o.kind === 'q100');
  assert.ok(q100.notes.every(x => x.b < 16) && q100.notes.some(x => near(x.b, 0)));
  M.closeVariants(); M.T.stop(true);
  // 关循环
  M.S.loop.on = false; M.S.notes = []; M.TL.cursor = 0;
  w.sel('lead');
  await M.toggleRec();
  const g = M.T.segs[0];
  await w.until(() => sched.now - w.outLat >= g.ctx0 + 2.5 * 0.4);      // 第 -1.5 拍（预备拍里）
  await w.tap('KeyA');
  await w.until(() => sched.now - w.outLat >= g.ctx0 + 4 * 0.4 - 0.012);  // 第 0 拍前 12 ms
  await w.tap('KeyS');
  M.toggleRec();
  assert.equal(M.S.notes.length, 1, '预备拍里的那下不记');
  assert.equal(M.S.notes[0].b, 0);
  assert.deepEqual(St.validateTruth(truthOf(M)), []);
  w.restoreTimers();
});
t('D1 按住拖过循环尾巴：时值截在循环尾巴（不拆），raw 留真按住的时长', async () => {
  const w = await world(), M = w.M;
  w.sel('lead');
  await M.toggleRec();
  await w.toBeat(15);
  const nSeg = M.T.segs.length;
  w.down('KeyA');
  await w.until(() => M.T.segs.length > nSeg && M.T.beatNow >= 1.5 && M.T.beatNow < 4);
  w.up('KeyA');                                           // 从 15 按到下一遍的 1.5
  M.toggleRec();
  const x = M.S.notes[0];
  assert.ok(near(x.b + x.d, 16, 0.02) && x.b + x.d <= 16 + 1e-9, `截在 le：${x.b}+${x.d}`);
  assert.ok(x.raw.d > 2, 'raw 是真按住的');
  w.restoreTimers();
});

// ====================== D3 / C10 / C21：录音锁层 ======================
t('D3/C21 录音中换层：键盘弹的、记的都还是开始录的那层；变体不把音挪层', async () => {
  const w = await world(), M = w.M;
  w.sel('lead');
  await M.toggleRec();
  await w.toBeat(1);
  w.down('Digit1');                                       // 换到鼓
  assert.equal(M.S.sel, 'drums'); assert.match(w.status(), /还在录「主旋律」/);
  await w.tap('KeyA');
  M.toggleRec();
  assert.equal(M.S.notes.length, 1);
  assert.equal(M.S.notes[0].layer, 'lead'); assert.ok(M.S.notes[0].pitch !== undefined && !M.S.notes[0].sound);
  assert.ok(w.M.E.notes.some(v => v.layer === 'lead') && !w.M.E.hits.length, '听到的也是主旋律');
  M.commitVariant('replace');
  assert.ok(M.S.notes.every(x => x.layer === 'lead'));
  w.restoreTimers();
});
t('C10 哼唱录音中换层：每一遍还是存到开始录的哼唱层', async () => {
  const w = await world({ worker: 'empty' }), M = w.M;
  M.S.loop.bars = 1; w.sel('voice');
  await M.toggleRec(); await flush();
  await w.toBeat(1); w.sel('drums');
  await w.until(() => M.hum.sessionTakes.length >= 2, 20);
  M.toggleRec(); await w.advance(1.5);
  assert.ok(M.S.clips.length >= 2 && M.S.clips.every(c => c.layer === 'voice'));
  w.restoreTimers();
});

// ====================== C11 / C35：琶音 ======================
t('C11/C35 琶音：第一格就是按下的那一格，跨过循环接着弹（每遍都有第 0 拍），松开以后的格子不记、不响', async () => {
  const w = await world(), M = w.M;
  // 静音这一层：走带不回放录下来的琶音，引擎里收到的 arp 音就全是手上按出来的
  M.S.layers.push(St.makeLayer('synth', 'arp', '琶音', 4, { id: 'arp', mute: true })); M.S.loop.bars = 1;
  w.sel('arp');
  await M.toggleRec();
  await w.toBeat(1.0);                                    // 预备拍完、循环第 1 拍
  w.down('KeyA');
  await w.until(() => M.T.segs.length >= 4, 20);         // 跨过两次循环
  const segEnd = M.T.segs[M.T.segs.length - 1];
  await w.until(() => sched.now - w.outLat >= segEnd.ctx0 + 2 * 0.4, 10);   // 第三遍的第 2 拍
  const relHeard = sched.now - w.outLat;
  w.up('KeyA');
  const before = M.S.notes.length;
  await w.advance(0.5);
  M.toggleRec();
  const bs = M.S.notes.filter(x => x.derived === 'arp').map(x => x.b);
  assert.equal(M.S.notes.length, before, '松开以后不再记');
  assert.ok(near(bs[0], 1.0), `第一格 = 按下那格 ${bs[0]}`);
  const zeros = bs.filter(b => near(b, 0)).length;
  assert.ok(zeros >= 2, `每遍循环头都有 ${zeros}`);
  assert.ok(bs.every(b => b >= 0 && b < 4));
  const last = bs[bs.length - 1];
  assert.ok(near(last, 1.75), `最后一格在松开（第 2 拍）之前 ${last}`);
  const late = w.M.E.notes.filter(v => v.layer === 'arp' && !v.canceled && v.when >= relHeard - 1e-6);
  assert.equal(late.length, 0, `松开以后的格子停掉了：${late.map(v => (v.when - relHeard).toFixed(3))}`);
  w.restoreTimers();
});

// ====================== D2 / C12 / C14：关循环哼唱、缓冲 ======================
t('D2/C12 关循环录哼唱：从开始录到停 = 一条；10 分钟自动停', async () => {
  const w = await world({ worker: 'empty' }), M = w.M;
  M.S.loop.on = false; M.S.songBars = 256; M.TL.cursor = 8; w.sel('voice');
  await M.toggleRec(); await flush();
  assert.ok(M.hum.pass && M.hum.pass.b0 === 8, '从光标开始收');
  await w.advance(12, 0.01);
  M.toggleRec(); await w.advance(1);
  assert.equal(M.S.clips.length, 1);
  assert.equal(M.S.clips[0].b, 8);
  assert.ok(Math.abs(M.S.clips[0].dur - (12 - 0.08 - 4 * 0.4)) < 0.2, `时长 ${M.S.clips[0].dur}`);
  assert.equal(M.variants?.kind ?? 'none', 'none', '一条没认出音符：不开挑选（只提示）');
  // 10 分钟自动停
  M.T.stop(); M.S.bpm = 60; M.TL.cursor = 0;               // 256 小节 × 4 拍 × 1 s：歌比 10 分钟长
  await M.toggleRec(); await flush();
  await w.advance(605, 0.05);                              // 预备拍 4 s + 600 s
  assert.equal(M.rec.on, false, '自动停了');
  await w.advance(2, 0.05);
  assert.match(w.status(), /10 分钟/, '分析完以后提示还在：' + w.statusLog.slice(-4).join(" / "));
  assert.ok(Math.abs(M.S.clips[M.S.clips.length - 1].dur - 600) < 0.5, `最长 600 s：${M.S.clips[M.S.clips.length - 1].dur}`);
  w.restoreTimers();
});
t('C14 停在长循环（16 小节 / 120 BPM）快完的时候：最后一条从头到尾都有声音（缓冲没被提前裁掉）', async () => {
  const w = await world({ worker: 'empty', voiceConst: 0.5 }), M = w.M;
  M.S.bpm = 120; M.S.loop.bars = 16; M.S.songBars = 16; M.S.settings.latencyMs = 30; w.sel('voice');
  await M.toggleRec(); await flush();
  await w.toBeat(64 * 0.97, 0.01);
  M.toggleRec();
  await w.advance(2, 0.01);
  const c = M.S.clips[0], pcm = M.audio.get(c.audio).pcm;
  let z = 0; while (z < pcm.length && pcm[z] === 0) z++;
  assert.ok(c.dur > 30, `一条 ${c.dur} s`);
  assert.equal(z, 0, `开头丢了 ${(z / 48000).toFixed(2)} s`);
  w.restoreTimers();
});

// ====================== C13 / C31：等每一遍都分析完再挑 ======================
t('C13/C31 延迟 300 ms：停了以后等最后一遍也分析完才开挑选；挑了以后同一起点只剩一遍在用', async () => {
  const w = await world({ worker: 'empty' }), M = w.M;
  M.S.loop.bars = 1; M.S.settings.latencyMs = 300; w.sel('voice');
  await M.toggleRec(); await flush();
  await w.toBeat(1);
  await w.until(() => M.T.segs.length >= 4, 20);
  await w.advance(1.1);                                   // 第三遍走了一半多
  M.toggleRec();
  assert.equal(M.variants, null, '还没分析完不开');
  assert.ok(M.hum.pending >= 1, '最后一遍还在等');
  await w.advance(2);
  assert.equal(M.variants?.kind, 'takes');
  assert.equal(M.variants.options.length, M.S.clips.length, '每一遍都在面板里');
  assert.ok(M.S.clips.length >= 3);
  M.variants.idx = 0; M.commitVariant('replace');
  assert.equal(M.S.clips.filter(c => c.active).length, 1);
  w.restoreTimers();
});

// ====================== C15：挑哪一遍的试听 ======================
t('C15 挑哪一遍：音符跟着正在放的那一遍录音；放着的时候马上换（录音从这一拍起重排、片内起点跟着挪），挑定 / 放弃也马上换回', async () => {
  const w = await world(), M = w.M;
  for (const k of [1, 2, 3]) {
    M.audio.set('a' + k, { buffer: fakeBuffer(1, 48000, 48000), pcm: null, sr: 48000, file: null });
    M.S.clips.push({ id: 'C' + k, layer: 'voice', b: 0, dur: 6.4, audio: 'a' + k, input: 'hum', take: k, active: k === 3, analysis: { mode: 'hum' } });
    M.S.notes.push({ id: 'n' + k, layer: 'voice', b: k, d: 1, pitch: 60 + k, vel: 0.8, input: 'hum', take: 'C' + k });
  }
  M.hum.sessionTakes = ['C1', 'C2', 'C3']; M.openTakeVariants('voice');
  M.variants.idx = 0; M.previewVariant();                 // 没在放：马上换
  assert.deepEqual(M.clipsNow().filter(c => c.active).map(c => c.id), ['C1']);
  assert.deepEqual(M.events().filter(x => x.take).map(x => x.id), ['n1']);
  M.T.start(0); await w.advance(1.0);
  const n0 = M.E.clipsStarted.length;
  M.variants.idx = 1; M.previewVariant();                 // 放着：马上换成第 2 遍
  assert.deepEqual(M.events().filter(x => x.take).map(x => x.id), ['n2']);
  const st = M.E.clipsStarted.slice(n0);
  assert.equal(st.length, 1, '只重排了一条');
  assert.equal(st[0].buf, M.audio.get('a2').buffer);
  assert.ok(st[0].off > 0.8 && st[0].off < 1.1, `片内起点跟着现在这一拍 ${st[0].off}`);
  assert.ok(st[0].when >= sched.now - 1e-9, '不排在过去');
  const n1 = M.E.clipsStarted.length;
  M.variants.idx = 2; M.commitVariant('discard');         // 放弃 = 留原样（第 3 遍）：马上换回
  assert.equal(M.E.clipsStarted.slice(n1)[0].buf, M.audio.get('a3').buffer);
  M.T.stop(true);
  w.restoreTimers();
});
t('弹窗挡住播放 2 s：醒来时那段的音不一起涌出来（不排在过去），提示跳过了；录音片段从现在这一拍起接着放', async () => {
  const w = await world(), M = w.M;
  M.S.notes = Array.from({ length: 16 }, (_, k) => ({ id: 'm' + k, layer: 'lead', b: k, d: 0.5, pitch: 64, vel: 0.8, input: 'keyboard' }));
  M.audio.set('a1', { buffer: fakeBuffer(1, 48000 * 7, 48000), pcm: null, sr: 48000, file: null });
  M.S.clips.push({ id: 'C1', layer: 'voice', b: 0, dur: 6.4, audio: 'a1', input: 'hum', take: 1, active: true });
  M.T.start(0); await w.advance(0.5);
  const k0 = M.E.notes.length, c0 = M.E.clipsStarted.length;
  // 弹窗：定时器都不跑，音频时钟照走 2 s（这一段 6.4 s 里）
  sched.now += 2;
  for (const tm of sched.timers) if (tm.at < sched.now) tm.at = sched.now;
  await w.advance(0.05);
  const fresh = M.E.notes.slice(k0);
  assert.ok(fresh.every(v => v.when >= sched.now - 0.06), '没有排在过去的音：' + fresh.map(v => (v.when - sched.now).toFixed(3)));
  assert.ok(fresh.length <= 2, `只排了马上要响的 ${fresh.length}`);
  assert.ok(w.statusLog.some(s => /弹窗挡住/.test(s)));
  // 走到下一遍循环：录音从段头起排（不受影响）
  await w.until(() => M.T.segs.length >= 2, 10);
  const cs = M.E.clipsStarted.slice(c0);
  assert.ok(cs.length >= 1 && near(cs[cs.length - 1].off, 0, 1e-6));
  M.T.stop(true);
  w.restoreTimers();
});

// ====================== C17 / G10：改速度 ======================
t('C17 播放中改 BPM：不回跳、不补排过去的音；撤销回原速也一样', async () => {
  const w = await world(), M = w.M;
  M.S.loop.on = false; M.S.songBars = 64;
  for (let b = 0; b < 256; b += 0.5) M.S.notes.push({ id: 'n' + b, layer: 'lead', b, d: 0.25, pitch: 64, vel: 0.8, input: 'keyboard' });
  M.T.start(0); await w.advance(15);
  const before = M.T.beatNow;
  w.$('bpm').value = '120'; w.$('bpm').onchange();
  const after = M.T.beatNow;
  assert.ok(after >= before - 0.01 && after < before + 0.5, `${before.toFixed(2)} → ${after.toFixed(2)}`);
  M.E.notes.length = 0; await w.advance(0.3);
  assert.ok(M.E.notes.every(v => v.when >= v.at - 1e-6), '没有排到过去的音');
  M.E.notes.length = 0; M.undo(); await w.advance(0.3);
  assert.equal(M.S.bpm, 150);
  assert.ok(M.E.notes.every(v => v.when >= v.at - 1e-6), '撤销以后也没有一堆过去的音');
  w.restoreTimers();
});
t('G10 改速度：哼唱的音跟着它那一遍录音（秒）走', async () => {
  const w = await world(), M = w.M;
  M.S.clips.push({ id: 'C1', layer: 'voice', b: 0, dur: 6.4, audio: 'a', input: 'hum', take: 1, active: true });
  M.S.notes.push({ id: 'h', layer: 'voice', b: 8, d: 1, pitch: 64, vel: 0.8, input: 'hum', take: 'C1', raw: { b: 8, d: 1, t: 3.2 } }, { id: 'k', layer: 'lead', b: 8, d: 1, pitch: 64, vel: 0.8, input: 'keyboard' });
  w.$('bpm').value = '120'; w.$('bpm').onchange();
  const tr = truthOf(M);
  assert.ok(near(tr.notes.find(x => x.layer === 'voice').t, 3.2, 1e-3), '哼唱的音还在录音的第 3.2 s');
  assert.ok(near(tr.notes.find(x => x.layer === 'lead').t, 4.0, 1e-3), '键盘的音跟格子走');
  w.restoreTimers();
});

// ====================== C19：XY 按住不动 ======================
t('C19 录音中按住 XY 不动：值一直写进曲线（真值 = 听到的），点不会每帧一个', async () => {
  const w = await world(), M = w.M;
  M.S.curves.cutoff = [[0, 0], [4, 1], [8, 0], [12, 1]];
  await M.toggleRec();
  await w.toBeat(2);
  const cv = w.$('xy');
  cv._listeners.pointerdown[0]({ clientX: 50, clientY: 50, pointerId: 1 });
  for (let i = 0; i < 500 && M.T.beatNow < 10; i++) { await w.advance(1 / 60); M.frame(); }
  cv._listeners.pointerup[0]({});
  const pts = M.S.curves.cutoff;
  for (const b of [4, 6, 8, 9.9]) assert.ok(near(St.curveAt(pts, b, 0), 0.5, 1e-3), `第 ${b} 拍 = ${St.curveAt(pts, b, 0)}`);
  assert.ok(pts.filter(p => p[0] > 2 && p[0] < 10).length <= 3, `按住不动只留两头：${pts.length} 个点`);
  w.restoreTimers();
});

// ====================== C20：哼唱层用键盘 ======================
t('C20 哼唱层切到键盘：不开麦、按合成层那样记音，停了给音符变体', async () => {
  const w = await world({ micDenied: true }), M = w.M;
  w.sel('voice'); M.S.layers.find(l => l.id === 'voice').input = 'keyboard';
  await M.toggleRec();
  assert.equal(M.rec.on, true, '不要麦也能录');
  assert.equal(M.hum.active, false);
  await w.toBeat(1); await w.tap('KeyA');
  M.toggleRec();
  assert.equal(M.S.clips.length, 0);
  assert.equal(M.variants?.kind, 'notes');
  assert.equal(M.S.notes[0].input, 'keyboard');
  w.restoreTimers();
});

// ====================== C23 / C43：打开会话 ======================
t('C23/C43 有没存的录音时打开会话先问；打开后撤销不会翻出没有声音的旧录音；片段记着文件名', async () => {
  const w = await world(), M = w.M;
  M.audio.set('a1', { buffer: fakeBuffer(1, 48000, 48000), pcm: new Float32Array(48000), sr: 48000, file: null });
  M.S.clips.push({ id: 'C1', layer: 'voice', b: 0, dur: 1, audio: 'a1', input: 'hum', take: 1, active: true });
  const sess = { ...St.makeState(), clips: [{ id: 'C9', layer: 'voice', b: 0, dur: 2, audio: 'a9', input: 'hum', take: 4, active: true }], audioFiles: { a9: '作曲/哼唱_v1_第4遍.wav' } };
  w.sessions['作曲_v1_会话.json'] = sess;
  w.confirmAnswer = false;
  await M.loadSession('作曲_v1_会话.json');
  assert.equal(w.confirms.length, 1); assert.equal(M.S.clips[0].id, 'C1', '取消 = 不打开');
  w.confirmAnswer = true; M.snapshot();
  await M.loadSession('作曲_v1_会话.json');
  assert.equal(M.S.clips[0].id, 'C9');
  assert.equal(M.undoStack.length, 0);
  M.undo(); assert.equal(M.S.clips[0].id, 'C9');
  assert.equal(truthOf(M).clips[0].file, '作曲/哼唱_v1_第4遍.wav');
  // 存的会话里片段带 file
  await M.save();
  const saved = w.saves[w.saves.length - 1];
  assert.equal(saved.session.clips[0].file, '作曲/哼唱_v1_第4遍.wav');
  w.restoreTimers();
});

// ====================== C24：生成的音 ======================
t('C24 加鬼音：生成的镲来源是 pick、不编 raw', async () => {
  const w = await world(), M = w.M;
  for (const b of [0, 1, 2, 3]) M.S.notes.push({ id: 'k' + b, layer: 'drums', b, d: 0.25, sound: 'kick', vel: 0.8, input: 'keyboard', raw: { b, d: 0.25 } });
  M.openNoteVariants('drums', ['k0', 'k1', 'k2', 'k3']);
  M.variants.idx = M.variants.options.findIndex(o => o.kind === 'ghost');
  M.commitVariant('replace');
  const g = M.S.notes.filter(x => x.derived === 'ghost');
  assert.ok(g.length && g.every(x => x.input === 'pick' && !x.raw));
  assert.ok(M.S.notes.filter(x => !x.derived).every(x => x.raw));
  w.restoreTimers();
});

// ====================== C25 / C38 / C32 / D4：变体和撤销 ======================
t('C25/C38 音色变体：用了能撤销回原来的音色；放弃 + 撤销不会把试听的音色拿回来；开始录音时试听的音色还原', async () => {
  const w = await world(), M = w.M;
  const lead = () => M.S.layers.find(l => l.id === 'lead');
  w.sel('lead'); const orig = lead().params.cutoff;
  M.openTimbreVariants(); M.variants.idx = 1; M.previewVariant();
  assert.notEqual(lead().params.cutoff, orig);
  M.commitVariant('replace'); const bright = lead().params.cutoff;
  M.undo(); assert.equal(lead().params.cutoff, orig, '撤销回原音色');
  M.redo(); assert.equal(lead().params.cutoff, bright);
  M.undo();
  M.openTimbreVariants(); M.variants.idx = 2; M.previewVariant(); M.commitVariant('discard');
  M.undo(); assert.equal(lead().params.cutoff, orig, '放弃 + 撤销 = 还是原音色');
  M.openTimbreVariants(); M.variants.idx = 1; M.previewVariant();
  await M.toggleRec();
  assert.equal(M.variants, null); assert.equal(lead().params.cutoff, orig, '录音前还原');
  M.toggleRec();
  w.restoreTimers();
});
t('D4/C32 撤销 / 重做一律关掉变体面板；过期的面板 Enter 不动东西', async () => {
  const w = await world(), M = w.M;
  w.sel('lead');
  await M.toggleRec(); await w.toBeat(1); await w.tap('KeyA'); await w.toBeat(2); await w.tap('KeyS');
  M.toggleRec();
  assert.equal(M.variants?.kind, 'notes');
  M.undo();
  assert.equal(M.variants, null, '撤销关掉面板');
  M.redo(); assert.equal(M.variants, null);
  // 过期的面板（比如别的路子留下的）：Enter 不把撤销掉的音加回来
  M.openNoteVariants('lead', M.S.notes.map(x => x.id)); const v = M.variants;
  M.undo(); M.variants = v;
  const nNotes = M.S.notes.length;
  M.commitVariant('replace');
  assert.equal(M.S.notes.length, nNotes); assert.match(w.status(), /撤销/);
  w.restoreTimers();
});
t('C40 哼唱录了几遍：每按一次 Ctrl+Z 撤掉一遍（不会隔一次没反应）', async () => {
  const w = await world({ worker: 'empty' }), M = w.M;
  M.S.loop.bars = 1; w.sel('voice');
  await M.toggleRec(); await flush();
  await w.until(() => M.hum.sessionTakes.length >= 3, 20);
  M.toggleRec(); await w.advance(1.5);
  M.closeVariants();
  const k = M.S.clips.length;
  for (let i = 1; i <= k; i++) { M.undo(); assert.equal(M.S.clips.length, k - i, `第 ${i} 下`); }
  w.restoreTimers();
});

// ====================== 2026-10-29 第一批：低优先的尾巴 ======================
t('#9 不吸附的哼唱音：复制成合成层取整；合成层 / 关了滑音的跟唱一律放整数音高（和渲染一样）', async () => {
  const w = await world(), M = w.M;
  M.S.clips.push({ id: 'C1', layer: 'voice', b: 0, dur: 6.4, audio: 'x', input: 'hum', take: 1, active: true });
  M.S.notes.push({ id: 'h1', layer: 'voice', b: 0.5, d: 1, pitch: 64.37, vel: 0.8, input: 'hum', take: 'C1', raw: { b: 0.5, d: 1, pitch_raw: 64.37 }, bend: { dt: 0.02, cents: [0, 5, 0] } });
  M.copyHumToSynth(M.S.layers.find(l => l.id === 'voice'));
  const c = M.S.notes.find(x => x.layer !== 'voice');
  assert.equal(c.pitch, 64); assert.equal(c.raw.pitch_raw, 64.37); assert.equal(c.bend, undefined);
  c.pitch = 64.37;                                           // 万一别的路子塞进来小数
  const V = M.S.layers.find(l => l.id === 'voice');
  M.T.start(0); await w.advance(0.6);
  const played = M.E.notes.filter(v => v.when > 0);
  assert.ok(played.filter(v => v.layer === c.layer).every(v => v.pitch === 64), '合成层放整数');
  assert.ok(played.some(v => v.layer === 'voice' && near(v.pitch, 64.37)), '带滑音的跟唱放小数（小数并进音分曲线导出）');
  M.T.stop(true); V.params.bendOn = false; M.E.notes.length = 0;
  M.T.start(0); await w.advance(0.6);
  assert.ok(M.E.notes.filter(v => v.layer === 'voice').every(v => v.pitch === 64), '关了滑音：整数');
  M.T.stop(true); w.restoreTimers();
});
t('#10 变体面板开着（刚录完）：范围复制 / Ctrl+D 也带上刚录的音', async () => {
  const w = await world(), M = w.M;
  w.sel('lead');
  await M.toggleRec(); await w.toBeat(1); await w.tap('KeyA'); await w.toBeat(2); await w.tap('KeyS');
  M.toggleRec();
  assert.equal(M.variants?.kind, 'notes');
  M.TL.range = [0, 4]; assert.equal(M.copySel(), true);
  assert.equal(M.clipboard.notes.length, 2);
  w.restoreTimers();
});
t('#11 键盘弹的音：raw.pitch_raw = 真弹的音（吸和弦音 / 改音高以后还在）；琶音的 held_pitch 导出', async () => {
  const w = await world(), M = w.M;
  w.sel('lead');
  await M.toggleRec(); await w.toBeat(1); await w.tap('KeyA');
  M.toggleRec(); M.closeVariants();
  const nt = M.S.notes[0], p0 = nt.pitch;
  assert.equal(nt.raw.pitch_raw, p0);
  nt.pitch = p0 + 3;
  const ex = truthOf(M).notes[0];
  assert.equal(ex.raw.pitch_raw, p0); assert.equal(ex.pitch, p0 + 3);
  M.S.notes.push({ id: 'a1', layer: 'lead', b: 2, d: 0.2, pitch: 67, vel: 0.8, input: 'keyboard', derived: 'arp', raw: { b: 2, d: 0.2, held_pitch: 64 } });
  assert.equal(truthOf(M).notes.find(x => x.derived === 'arp').raw.held_pitch, 64);
  assert.deepEqual(St.validateTruth(truthOf(M)), []);
  w.restoreTimers();
});
t('#28 Esc 放弃变体以后：第一下 Ctrl+Z 就撤掉这次录音（看得见）', async () => {
  const w = await world(), M = w.M;
  w.sel('lead');
  await M.toggleRec(); await w.toBeat(1); await w.tap('KeyA');
  M.toggleRec();
  w.down('Escape');
  assert.equal(M.variants, null); assert.equal(M.S.notes.length, 1);
  M.undo();
  assert.equal(M.S.notes.length, 0, '第一下就撤掉了');
  w.restoreTimers();
});
t('C8 关循环从第 0 拍录、预备拍里就按住琶音：歌开头之前的格子不记（不挤到第 0 拍叠成一团）', async () => {
  const w = await world(), M = w.M;
  M.S.layers.push(St.makeLayer('synth', 'arp', '琶音', 4, { id: 'arp', mute: true }));
  M.S.loop.on = false; M.TL.cursor = 0; w.sel('arp');
  await M.toggleRec();
  const g = M.T.segs[0];
  await w.until(() => sched.now - w.outLat >= g.ctx0 + (4 - 0.45) * 0.4);    // 第 0 拍前 0.45 拍按下
  w.down('KeyA');
  await w.toBeat(1);
  w.up('KeyA'); M.toggleRec();
  const bs = M.S.notes.filter(x => x.derived === 'arp').map(x => x.b);
  assert.equal(bs.filter(b => near(b, 0)).length, 1, `第 0 拍只有一个 ${bs}`);
  assert.ok(bs.every(b => b >= 0));
  assert.ok(truthOf(M).notes.every(x => !x.raw || x.raw.t >= 0), 'raw.t 不是负的');
  w.restoreTimers();
});
t('#25 一遍循环的撤销点等这一遍听完再记：最后一拍的音和这一遍一起撤', async () => {
  const w = await world(), M = w.M;
  M.S.loop.bars = 1; w.sel('drums');
  await M.toggleRec();
  await w.toBeat(0.5); await w.tap('KeyA');                // 第 1 遍
  const g2 = await (async () => { await w.until(() => M.T.segs.length >= 3); return M.T.segs[M.T.segs.length - 1]; })();
  // 排程已经进了第 2 遍、但还没听到：这时按的还是第 1 遍的最后一拍
  await w.until(() => sched.now - w.outLat >= g2.ctx0 - 0.4 * 0.3);
  await w.tap('KeyS', 0.02);
  await w.toBeat(0.5); await w.advance(0.3);               // 进了第 2 遍
  await w.tap('KeyD');
  M.T.stop();                                              // 停 = 停录
  M.closeVariants();
  M.undo();
  const left = M.S.notes.map(x => x.sound).sort();
  assert.deepEqual(left, ['kick', 'snare'], `撤掉第 2 遍以后第 1 遍的两下都在：${left}`);
  w.restoreTimers();
});

// ====================== G14：范围删除 / 复制也剪录音 ======================
t('G14 范围删除剪掉录音的那一段：播放跳过、audible_spans 不含它；原曲分轨不剪；撤销回来', async () => {
  const w = await world(), M = w.M;
  M.audio.set('a1', { buffer: fakeBuffer(1, 48000 * 7, 48000), pcm: null, sr: 48000, file: null });
  M.audio.set('s1', { buffer: fakeBuffer(2, 48000 * 7, 48000), pcm: null, sr: 48000, file: '分轨/drums.wav' });
  M.S.clips.push({ id: 'C1', layer: 'voice', b: 0, dur: 6.4, audio: 'a1', input: 'hum', take: 1, active: true });
  M.S.layers.push(St.makeLayer('stem', 'stem', '鼓轨', 4, { id: 'st' }));
  M.S.clips.push({ id: 'C2', layer: 'st', b: 0, dur: 6.4, audio: 's1', input: 'stem', active: true, fileOffset: 0 });
  M.TL.range = [4, 8]; M.deleteSel();
  assert.deepEqual(M.S.clips[0].gaps, [[4, 8]]);
  assert.equal(M.S.clips[1].gaps, undefined, '分轨不剪');
  const tr = truthOf(M);
  assert.deepEqual(tr.clips[0].audible_spans, [[0, 1.6], [3.2, 6.4]]);
  assert.deepEqual(tr.clips[1].audible_spans, [[0, 6.4]]);
  M.T.start(0); await w.advance(0.2);
  const vs = M.E.clipsStarted.filter(c => c.layer === 'voice');
  assert.deepEqual(vs.map(c => [+c.off.toFixed(3), +c.dur.toFixed(3)]), [[0, 1.6], [3.2, 3.2]], '跳过剪掉的那段');
  M.T.stop(true);
  M.undo(); assert.equal(M.S.clips[0].gaps, undefined);
  assert.deepEqual(St.validateTruth(truthOf(M)), []);
  w.restoreTimers();
});
t('G14 范围复制 / 粘贴带上录音：贴出来的片段指着同一段声音（offset 对），剪掉的不复制；保存只存一份 wav', async () => {
  const w = await world(), M = w.M;
  M.audio.set('a1', { buffer: fakeBuffer(1, 48000 * 7, 48000), pcm: new Float32Array(48000 * 7), sr: 48000, file: null });
  M.S.clips.push({ id: 'C1', layer: 'voice', b: 0, dur: 6.4, audio: 'a1', input: 'hum', take: 1, active: true, gaps: [[5, 6]] });
  M.TL.range = [4, 8]; assert.equal(M.copySel(), true);
  assert.equal(M.clipboard.clips.length, 2, '剪掉的那拍把它分成两段');
  M.pasteAt(16);
  const ps = M.S.clips.filter(c => c.input === 'paste');
  assert.deepEqual(ps.map(c => [c.b, +c.dur.toFixed(3), +c.offset.toFixed(3)]), [[16, 0.4, 1.6], [18, 0.8, 2.4]]);
  const tr = truthOf(M);
  const pc = tr.clips.filter(c => c.input === 'paste');
  assert.deepEqual(pc.map(c => [c.t, c.offset, c.audible_spans]), [[6.4, 1.6, [[6.4, 6.8]]], [7.2, 2.4, [[7.2, 8]]]]);
  await M.save();
  const sv = w.saves[w.saves.length - 1];
  assert.equal(sv.audio.length, 1, '共用一段声音只存一次');
  assert.ok(sv.truth.clips.every(c => c.file === '作曲/' + sv.audio[0].name));
  assert.deepEqual(St.validateTruth(sv.truth), []);
  w.restoreTimers();
});

// ====================== C26 / D8：新层进场景 ======================
t('D8/C26 场景建好以后才加的层：加一层 / 叠成新层 / 复制成合成层 / 拖进来的文件都进每个场景', async () => {
  const w = await world(), M = w.M;
  M.S.scenes.push({ id: 'S1', bar: 0, name: '一', layers: ['drums', 'lead'], text: '' }, { id: 'S2', bar: 2, name: '二', layers: ['bass'], text: '' });
  w.$('addLayer').value = 'synth:pad'; w.$('addLayer').onchange();
  const pad = M.S.layers[M.S.layers.length - 1];
  assert.ok(M.S.scenes.every(sc => sc.layers.includes(pad.id)));
  M.S.notes.push({ id: 'x', layer: 'lead', b: 0, d: 1, pitch: 64, vel: 0.8, input: 'keyboard', raw: { b: 0, d: 1 } });
  M.openNoteVariants('lead', ['x']); M.commitVariant('overlay');
  const ov = M.S.layers[M.S.layers.length - 1];
  assert.ok(M.S.scenes.every(sc => sc.layers.includes(ov.id)));
  M.S.notes.push({ id: 'h', layer: 'voice', b: 0, d: 1, pitch: 64, vel: 0.8, input: 'hum' });
  M.copyHumToSynth(M.S.layers.find(l => l.id === 'voice'));
  assert.ok(M.S.scenes.every(sc => sc.layers.includes(M.S.sel)));
  await M.addFileLayer({ name: 'loop.wav', arrayBuffer: async () => ({ dur: 4, ch: 2 }) });
  const fl = M.S.layers[M.S.layers.length - 1];
  assert.ok(M.S.scenes.every(sc => sc.layers.includes(fl.id)));
  assert.equal(M.S.clips[M.S.clips.length - 1].fileOffset, 0);
  w.restoreTimers();
});

// ====================== C28 / G5：循环区 ======================
t('C28 全曲改短：循环区挪回歌里；G5 录音中不让挪循环区', async () => {
  const w = await world(), M = w.M;
  M.S.loop.start = 8; M.S.loop.bars = 4;
  w.$('songBars').value = '8'; w.$('songBars').onchange();
  assert.ok(M.S.loop.start + M.S.loop.bars <= 8, JSON.stringify(M.S.loop));
  w.$('songBars').value = '2'; w.$('songBars').onchange();
  assert.deepEqual([M.S.loop.start, M.S.loop.bars], [0, 2]);
  w.$('songBars').value = '16'; w.$('songBars').onchange();
  await M.toggleRec();
  const loop0 = JSON.stringify(M.S.loop);
  w.down('Equal'); assert.equal(JSON.stringify(M.S.loop), loop0);
  w.$('loopBars').value = '8'; w.$('loopBars').onchange(); assert.equal(JSON.stringify(M.S.loop), loop0);
  M.S.loop.start = 3; M.TL.api.changed('loop'); assert.equal(JSON.stringify(M.S.loop), loop0, '标尺上拖也还原');
  M.toggleRec();
  w.restoreTimers();
});

// ====================== C29：预备拍 ======================
t('C29 预备拍：节拍器关着时只打开始录那一拍之前的；关循环从光标开始录时按光标算', async () => {
  const w = await world(), M = w.M;
  M.S.settings.metro = false; M.S.loop.on = false; M.TL.cursor = 8;
  await M.toggleRec();
  await w.advance(4 * 0.4 + 1);
  const beats = M.E.clicks.map(c => M.T.beatAt(c + 1e-9));
  assert.deepEqual(beats.map(b => Math.round(b)), [4, 5, 6, 7]);
  M.toggleRec();
  w.restoreTimers();
});

// ====================== G1 / G2 / G3 / G4 / G8 / G18 ======================
t('G1 载入分轨以后改起点：分轨按新起点重新切（片段 offset = 新起点）；撤销回去还是旧的那段', async () => {
  const w = await world({ stems: ['drums.wav', 'bass.wav'] }), M = w.M;
  await M.loadStems();
  const oldAudio = M.S.clips.map(c => c.audio);
  w.$('anchor').value = '0.36'; w.$('anchor').onchange(); await flush(10);
  const tr = truthOf(M);
  assert.ok(tr.clips.every(c => near(c.offset, 0.36, 1e-6) && near(c.t, 0.36, 1e-6)), JSON.stringify(tr.clips.map(c => [c.t, c.offset])));
  assert.ok(w.stemReq.slice(-2).every(q => near(+q.t0, 0.36)));
  M.undo();
  assert.deepEqual(M.S.clips.map(c => c.audio), oldAudio);
  assert.ok(M.audio.get(oldAudio[0]), '旧的那段声音还在');
  w.restoreTimers();
});
t('G2 拖进来的立体声文件：两个声道都存（extra_channels），offset 0；G4 有没存的录音时关页面先拦', async () => {
  const w = await world(), M = w.M;
  await M.addFileLayer({ name: 'pad@128-Am.wav', arrayBuffer: async () => ({ dur: 2, ch: 2, fill: [0.25, -0.5] }) });
  let ev = { prevented: 0, preventDefault() { this.prevented++; } };
  w.window.fire('beforeunload', ev); assert.equal(ev.prevented, 1, '有没存的录音：拦');
  await M.save();
  const a = w.saves[0].audio[0];
  assert.ok(a.extra_channels && a.extra_channels.length === 1);
  const R = new Float32Array(Uint8Array.from(Buffer.from(a.extra_channels[0], 'base64')).buffer);
  assert.ok(near(R[0], -0.5) && near(new Float32Array(Uint8Array.from(Buffer.from(a.b64, 'base64')).buffer)[0], 0.25));
  assert.equal(w.saves[0].truth.clips[0].offset, 0);
  ev = { prevented: 0, preventDefault() { this.prevented++; } };
  w.window.fire('beforeunload', ev); assert.equal(ev.prevented, 0, '存过了：不拦');
  w.restoreTimers();
});
t('G3 换作品：先问；带过去的录音下次存进新作品（不沿用旧作品的文件名），分轨变成普通素材', async () => {
  const w = await world({ stems: ['drums.wav'] }), M = w.M;
  await M.loadStems();
  M.audio.set('a1', { buffer: fakeBuffer(1, 48000, 48000), pcm: new Float32Array(48000), sr: 48000, file: null });
  M.S.clips.push({ id: 'C1', layer: 'voice', b: 0, dur: 1, audio: 'a1', input: 'hum', take: 1, active: true });
  await M.save();
  w.$('work').value = 'MV_B'; await w.$('work').onchange();
  assert.equal(w.confirms.length, 1); assert.equal(M.S.work, 'MV_B');
  await M.save();
  const b = w.saves[w.saves.length - 1];
  assert.ok(b.audio.length === 2, '两段都重新存进新作品');
  assert.ok(b.truth.clips.every(c => c.file && c.file.startsWith('作曲/') && c.file.includes(b.ver)), JSON.stringify(b.truth.clips.map(c => c.file)));
  assert.ok(b.truth.clips.every(c => c.offset === 0));
  w.restoreTimers();
});
t('G8 serve.py 中途关了：保存改成下载，并说清楚', async () => {
  const w = await world(), M = w.M;
  w.server = false;
  let dl = 0; const mk = w.doc.createElement; w.doc.createElement = tag => { const e = mk(tag); if (tag === 'a') e.click = () => dl++; return e; };
  globalThis.URL.createObjectURL ||= () => 'blob:x';
  const r0 = URL.createObjectURL; URL.createObjectURL = () => 'blob:x'; const r1 = URL.revokeObjectURL; URL.revokeObjectURL = () => {};
  await M.save();
  URL.createObjectURL = r0; URL.revokeObjectURL = r1;
  assert.ok(dl >= 2, '下载了真值和会话'); assert.match(w.status(), /连不上 serve\.py/);
  w.restoreTimers();
});
t('G8 离线下载：24 bit、文件名带版本（再存一次不撞名），真值里的文件名就是下载的名字', async () => {
  const w = await world({ server: false }), M = w.M;
  M.audio.set('a1', { buffer: fakeBuffer(1, 4800, 48000), pcm: new Float32Array(4800).fill(0.5), sr: 48000, file: null });
  M.S.clips.push({ id: 'C1', layer: 'voice', b: 0, dur: 0.1, audio: 'a1', input: 'hum', take: 1, active: true });
  const got = []; const mk = w.doc.createElement;
  w.doc.createElement = tag => { const e = mk(tag); if (tag === 'a') e.click = () => got.push({ name: e.download, blob: blobs.get(e.href) }); return e; };
  const blobs = new Map(); let k = 0;
  const r0 = URL.createObjectURL, r1 = URL.revokeObjectURL; URL.createObjectURL = b => { const u = 'blob:' + (++k); blobs.set(u, b); return u; }; URL.revokeObjectURL = () => {};
  await M.save();
  URL.createObjectURL = r0; URL.revokeObjectURL = r1;
  const names = got.map(g => g.name);
  const wav = got.find(g => /\.wav$/.test(g.name)), js = got.find(g => /^作曲_离线_\d{8}-\d{6}\.json$/.test(g.name));
  assert.ok(wav && /^哼唱_离线_\d{8}-\d{6}_第1遍\.wav$/.test(wav.name), names.join(' '));
  assert.ok(js, names.join(' '));
  assert.ok(names.some(n => /_会话\.json$/.test(n)));
  const ab = new DataView(await wav.blob.arrayBuffer());
  assert.equal(ab.getUint16(34, true), 24); assert.equal(ab.byteLength, 44 + 4800 * 3);
  const s0 = ab.getUint8(44) | (ab.getUint8(45) << 8) | (ab.getUint8(46) << 16);
  assert.equal(s0, Math.round(0.5 * 8388607));
  const truth = JSON.parse(await js.blob.text());
  assert.equal(truth.clips[0].file, wav.name);
  assert.match(w.status(), /允许下载多个文件/);
  w.restoreTimers();
});
t('G11 没校准的麦克风延迟估计算上麦克风输入延迟（浏览器给的）', async () => {
  const w = await world(), M = w.M;
  w.track.getSettings = () => ({ latency: 0.02 });
  await M.micOn();
  assert.equal(M.S.settings.latencyMs, Math.round((0.03 + 0.02) * 1000 + 12));
  w.restoreTimers();
});
t('G13 起点没对上原曲重拍：不做律动模板，说清楚；loadStems 一条都没载入不留撤销点', async () => {
  const w = await world({ sense: { bpm: 150, downbeats: [0.36, 1.96] }, stems: ['drums.wav'] }), M = w.M;
  await M.loadStems();
  assert.equal(M.groove, null);
  assert.match(w.statusLog.join('\n'), /律动模板没做/);
  const w2 = await world({ stems: ['drums.wav'] }); const M2 = w2.M;
  const u0 = M2.undoStack.length;
  M2.E.ctx.decodeAudioData = async () => { throw new Error('坏文件'); };   // 分轨全读不了
  await M2.loadStems();
  assert.equal(M2.undoStack.length, u0, '没载入就没有撤销点');
  w2.restoreTimers();
});
t('G19 停了以后最后一遍还在分析时按 Ctrl+Z：只撤掉那一遍（它不再塞回来），不再往回撤一步', async () => {
  const w = await world({ outLat: 0.03 }), M = w.M;
  M.S.loop.bars = 1; w.sel('voice');
  await M.toggleRec(); await flush();
  await w.until(() => M.hum.sessionTakes.length >= 2, 20);
  await w.until(() => M.T.beatNow > 2.6, 5);                 // 这一遍唱过一半多
  M.toggleRec();                                             // 停：这一遍（残段）在等
  const ids = M.S.clips.map(c => c.id);
  assert.ok(M.hum.pending > 0, '还有一遍在等');
  M.undo();
  await w.advance(2);
  assert.deepEqual(M.S.clips.map(c => c.id), ids, '还在分析的那一遍没进来，之前的都还在');
  w.restoreTimers();
});
t('G6 自动存档不带挂在哼唱上的音；撤销栈总量有上限', async () => {
  const w = await world(), M = w.M;
  M.S.notes.push({ id: 'h1', layer: 'voice', b: 0, d: 1, pitch: 64, vel: 0.8, input: 'hum', take: 'C1', bend: { dt: 0.02, cents: new Array(5000).fill(1) } });
  M.S.notes.push({ id: 'k1', layer: 'lead', b: 0, d: 1, pitch: 64, vel: 0.8, input: 'keyboard' });
  await w.advance(4.5);
  const j = JSON.parse(w.store.get('composer_web_autosave_v1'));
  assert.deepEqual(j.notes.map(x => x.id), ['k1']);
  const big = M.S.notes[0].bend.cents; big.length = 2e6; big.fill(1);   // 一步 ~4 MB
  for (let i = 0; i < 40; i++) M.snapshot();
  const total = M.undoStack.reduce((n, x) => n + x.length, 0);
  assert.ok(total <= 40e6 + M.undoStack[0].length, `撤销栈 ${total} 字符`);
  assert.ok(M.undoStack.length >= 5);
  w.restoreTimers();
});
t('G18 改完下拉框 / 数字框：焦点还给页面；数字框里 Enter 也一样', async () => {
  const w = await world(), M = w.M;
  const s = w.$('loopBars'); w.doc.fire('change', { target: s }); assert.ok(s._blurred >= 1);
  const b = w.$('bpm'); M.onKeyDown({ code: 'Enter', key: 'Enter', target: b, preventDefault() {} }); assert.ok(b._blurred >= 1);
  w.restoreTimers();
});

// ====================== C4：后台线程 ======================
t('C4 哼唱分析走后台线程（hum-worker.js，消息协议真跑）；开不了 worker 就同步算；结果一样', async () => {
  const sig = voiceLike([{ midi: 64, dur: 0.4 }, { midi: 67, dur: 0.4 }, { midi: 71, dur: 0.6 }]);
  const notesOf = async mode => {
    const w = await world({ worker: mode }), M = w.M;
    await M.injectHum(sig, 48000, 0);
    w.restoreTimers();
    return { w, ps: M.S.notes.filter(x => x.take).map(x => x.pitch) };
  };
  const a = await notesOf('real'), b = await notesOf('throws'), c = await notesOf('none');
  assert.ok(a.w.workerJobs === 1 && a.w.workerUrls[0][0] === 'js/hum-worker.js' && a.w.workerUrls[0][1].type === 'module');
  assert.equal(b.w.workerJobs, 0);
  assert.deepEqual(a.ps, [64, 67, 71]); assert.deepEqual(b.ps, a.ps); assert.deepEqual(c.ps, a.ps);
  const d = await notesOf('crash');                      // worker 跑一半崩了：手上的活改成同步算
  assert.deepEqual(d.ps, a.ps);
});
t('C4 等分析的时候打开了别的会话：晚到的那一遍不塞进新会话', async () => {
  const w = await world({ worker: 'real' }), M = w.M;
  M.mic.sr = 48000; M.mic.chunks = [{ t: 100, x: voiceLike([{ midi: 64, dur: 0.5 }]) }];
  const p = M.processHumTake('voice', 0, 100, 100.8);
  w.sessions['s.json'] = { ...St.makeState(), audioFiles: {} };
  await M.loadSession('s.json');
  await p; await flush();
  assert.equal(M.S.clips.length, 0);
  w.restoreTimers();
});

// 直接跑这个文件 = 跑全部测试；被别的脚本 import（比如拿 world() 复现问题）就不跑
export { sched, fakeBuffer, flush, truthOf, tests, REAL_TIMERS };
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  for (const [name, f] of tests) {
    n++;
    try { await f(); } catch (e) { failed++; console.error(`✗ ${name}\n  ${e.stack || e.message}`); }
    Object.assign(globalThis, REAL_TIMERS);                // 失败的测试没来得及还原定时器
  }
  console.log(`${n - failed}/${n} 通过`);
  if (failed) process.exit(1);
}
