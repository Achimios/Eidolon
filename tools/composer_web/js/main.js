// >v<🎛作曲器-页面-锁调键盘 / 拍 / 哼唱 / XY 推子 → 循环叠录 → 变体里挑 → 时间线剪 → 保存 作曲.json + 哼唱 wav。
import { MODES, NOTE_NAMES, ROW_A, ROW_Q, ROW_Z, KEY_LABEL, codeToDegree, degreeToMidi, noteName, parseProgression, chordAtBar, chordPcs, chordRoman, chordTones, hzMidi } from './theory.js';
import { Engine, PRESETS, DRUM_SOUNDS, DRUM_LABEL } from './engine.js';
import { Transport, heardCtxTime } from './transport.js';
import { Timeline } from './timeline.js';
import { MACROS, MACRO_BY_ID, effectiveMacros, curveAt, curveWrite, makeState, makeLayer, uid, spb, loopRange, songBeats, layerById, exportTruth, exportSession, validateTruth } from './state.js';
import { analyzeHum, notesToBeats, yin, decimate, MODE_PARAMS } from './pitch.js';
import { makeVariants, TIMBRE_VARIANTS, holdVelocity, tapPitch, detectOnsets, grooveFromOnsets, meanOffset, quantize, cloneNotes } from './interp.js';

const $ = id => document.getElementById(id);
const AUTOSAVE = 'composer_web_autosave_v1';
let S = makeState();
// 浏览器里的自动存档只有音符和设置：录音 / 分轨不在里面（要从「打开…」读回存过的会话）
let restoredNote = '';
try {
  const a = localStorage.getItem(AUTOSAVE);
  if (a) {
    const j = JSON.parse(a);
    if (j && j.format === S.format) {
      const layers = (j.layers || []).filter(l => l.kind !== 'stem');
      const ids = new Set(layers.map(l => l.id));
      S = { ...S, ...j, layers, clips: [], notes: (j.notes || []).filter(n => !n.take && ids.has(n.layer)), scenes: (j.scenes || []).map(sc => ({ ...sc, layers: sc.layers.filter(id => ids.has(id)) })) };
      if (!ids.has(S.sel)) S.sel = layers[0]?.id;
      restoredNote = '接着上次浏览器里的进度（录音和分轨没存在浏览器里：要的话点「打开…」读存过的会话）';
    }
  }
} catch (e) { /* 浏览器存储不可用就算了 */ }

let E = null, T = null, TL = null;
const audio = new Map();          // audioId → { buffer, pcm, sr, file }
const peaksCache = new Map();
let undoStack = [], redoStack = [];
let server = false;
let groove = null;                // 律动模板（载入原曲鼓轨后有）
let variants = null;              // { layer, removeIds, options, idx, kind, range, reroll }
let shownLanes = new Set(['cutoff', 'drive']);
let clipboard = null;

// ---------- 小工具 ----------
const status = (msg, keep = false) => { $('status').textContent = msg; if (!keep) { clearTimeout(status._t); status._t = setTimeout(() => { $('status').textContent = ''; }, 9000); } };
const sel = () => layerById(S, S.sel) || S.layers[0];
const ctxObj = () => ({ key: S.key, prog: S.prog, beatsPerBar: S.beatsPerBar, tensionAt: b => curveAt(S.curves.tension, b, 0.5), groove });
function snapshot() { undoStack.push(JSON.stringify(S)); if (undoStack.length > 150) undoStack.shift(); redoStack = []; }
function restore(str) { const cur = S; S = JSON.parse(str); S.sel = layerById(S, cur.sel) ? cur.sel : S.layers[0]?.id; syncAll(); }
function undo() { if (!undoStack.length) return status('没有可撤销的'); redoStack.push(JSON.stringify(S)); restore(undoStack.pop()); status('撤销'); }
function redo() { if (!redoStack.length) return; undoStack.push(JSON.stringify(S)); restore(redoStack.pop()); status('重做'); }
function autosave() { try { localStorage.setItem(AUTOSAVE, JSON.stringify({ ...S, clips: [] })); } catch (e) { /* 满了 / 不可用 */ } }
setInterval(autosave, 4000);

// 当前要播的事件（变体试听时替换掉被试的那几个音）
function events() {
  const activeTakes = new Set(S.clips.filter(c => c.active).map(c => c.id));
  let ev = S.notes.filter(n => !n.take || activeTakes.has(n.take));
  if (variants) {
    const o = variants.options[variants.idx];
    if (o.notes) {
      ev = ev.filter(n => !variants.removeIds.has(n.id));
      ev = ev.concat(o.notes.map(n => ({ ...n, layer: n.layer || variants.layer, _preview: true })));
    }
  }
  return ev;
}
function clipsForPlay() {
  if (variants && variants.kind === 'takes') {
    const o = variants.options[variants.idx];
    return S.clips.map(c => variants.takeIds.includes(c.id) ? { ...c, active: c.id === o.clipId } : c);
  }
  return S.clips;
}

// ---------- 音频 ----------
function peaksOf(id) {
  if (peaksCache.has(id)) return peaksCache.get(id);
  const a = audio.get(id); if (!a) return null;
  const d = a.buffer.getChannelData(0), n = 800, step = Math.max(1, Math.floor(d.length / n)), pk = new Float32Array(n);
  let mx = 1e-6;
  for (let i = 0; i < n; i++) { let m = 0; for (let j = i * step; j < Math.min(d.length, (i + 1) * step); j += 4) m = Math.max(m, Math.abs(d[j])); pk[i] = m; mx = Math.max(mx, m); }
  for (let i = 0; i < n; i++) pk[i] /= mx;
  peaksCache.set(id, pk);
  return pk;
}
function addAudio(buffer, extra = {}) {
  const id = extra.id || uid('A');
  audio.set(id, { buffer, pcm: extra.pcm || null, sr: buffer.sampleRate, file: extra.file || null });
  return id;
}

// ---------- 麦克风 ----------
const mic = { on: false, node: null, chunks: [], live: new Float32Array(48000 * 2), liveN: 0, sr: 48000, strip: [], keepSec: 30 };
async function micOn() {
  if (mic.on) return true;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 } });
    await E.ctx.audioWorklet.addModule('js/mic-worklet.js');
    const src = E.ctx.createMediaStreamSource(stream);
    mic.node = new AudioWorkletNode(E.ctx, 'mic-rec');
    const sink = E.ctx.createGain(); sink.gain.value = 0;
    src.connect(mic.node); mic.node.connect(sink); sink.connect(E.ctx.destination);
    mic.sr = E.ctx.sampleRate;
    mic.node.port.onmessage = ev => {
      const { t, x } = ev.data;
      mic.chunks.push({ t, x });
      // 只留最近 keepSec 秒，除非正在录
      const keepFrom = E.now - (hum.active ? 600 : mic.keepSec);
      while (mic.chunks.length && mic.chunks[0].t < keepFrom) mic.chunks.shift();
      // 实时音高用的滑动缓冲
      const L = mic.live;
      if (mic.liveN + x.length > L.length) { L.copyWithin(0, x.length); mic.liveN -= x.length; }
      L.set(x, mic.liveN); mic.liveN += x.length;
    };
    mic.on = true;
    $('micBtn').classList.add('on'); $('micBtn').textContent = '麦已开';
    if (S.settings.latencyMs === null) S.settings.latencyMs = Math.round(((E.ctx.baseLatency || 0) + (E.ctx.outputLatency || 0)) * 1000 + 12);
    showLatency();
    return true;
  } catch (err) { status('开不了麦克风：' + err.message); return false; }
}
// 从缓冲里取 [t0, t1)（音频时钟）的样本
function micExtract(t0, t1) {
  const sr = mic.sr, n = Math.max(0, Math.round((t1 - t0) * sr)), out = new Float32Array(n);
  for (const c of mic.chunks) {
    const off = Math.round((c.t - t0) * sr);
    if (off + c.x.length <= 0 || off >= n) continue;
    const a = Math.max(0, -off), z = Math.min(c.x.length, n - off);
    out.set(c.x.subarray(a, z), off + a);
  }
  return out;
}
function showLatency() { $('latInfo').textContent = S.settings.latencyMs !== null ? `麦克风延迟补偿 ${S.settings.latencyMs} ms${S.settings.latencyCal ? '（实测）' : '（估计，建议校准）'}` : ''; }

// 校准：扬声器放 6 下咔哒，麦克风收，测往返延迟
async function calibrate() {
  if (!(await micOn())) return;
  status('校准中：保持安静。用哪副耳机哼唱，就把那副耳机贴着麦克风来校准（蓝牙耳机比扬声器多 150–250 ms）…', true);
  const t0 = E.now + 0.3, clicks = [];
  for (let k = 0; k < 6; k++) { const t = t0 + k * 0.5; clicks.push(t); const o = E.ctx.createOscillator(), g = E.ctx.createGain(); o.frequency.value = 2000; g.gain.setValueAtTime(0.6, t); g.gain.setTargetAtTime(0, t + 0.004, 0.004); o.connect(g); g.connect(E.speaker); o.start(t); o.stop(t + 0.05); }
  await new Promise(r => setTimeout(r, 3800));
  const x = micExtract(t0 - 0.05, t0 + 3.2), sr = mic.sr;
  let mx = 0; for (const v of x) mx = Math.max(mx, Math.abs(v));
  const found = [];
  for (const c of clicks) {
    const a = Math.round((c - t0 + 0.05) * sr), z = Math.min(x.length, a + Math.round(0.4 * sr));
    for (let i = a; i < z; i++) if (Math.abs(x[i]) > mx * 0.35) { found.push((i - a) / sr * 1000); break; }
  }
  if (found.length < 4 || mx < 0.01) { status(`校准失败：只听到 ${found.length} / 6 下（音量 ${mx.toFixed(3)}）。音量开大、拿掉耳机再试。`); return; }
  found.sort((p, q) => p - q);
  S.settings.latencyMs = Math.round(found[found.length >> 1]); S.settings.latencyCal = true;
  showLatency(); status(`往返延迟 ${S.settings.latencyMs} ms（6 下里的 ${found.length} 下），之后的哼唱都按这个对齐`);
}

// ---------- 哼唱录音（每一遍循环 = 一条 take）----------
const hum = { active: false, pass: null, takeNo: 0, sessionTakes: [] };
function humPassStart(seg) { hum.pass = { ctx0: seg.ctx0, b0: seg.b0, b1: seg.b1 }; }
function humPassEnd(ctxEnd, partial = false) {
  const p = hum.pass; hum.pass = null;
  if (!p) return;
  const L = sel(); if (!L || L.kind !== 'voice') return;
  const lat = (S.settings.latencyMs || 0) / 1000;
  const bEnd = partial ? Math.min(p.b1, p.b0 + (ctxEnd - p.ctx0) / spb(S)) : p.b1;
  if (bEnd - p.b0 < S.beatsPerBar * 0.5) return;
  const t0 = p.ctx0 + lat, t1 = p.ctx0 + (bEnd - p.b0) * spb(S) + lat;
  // 等麦克风的数据到齐（再晚 lat + 0.15 s）
  setTimeout(() => processHumTake(L.id, p.b0, t0, t1), (lat + 0.2) * 1000 + Math.max(0, (t1 - E.now) * 1000));
}
function processHumTake(layerId, b0, t0, t1) {
  const pcm = micExtract(t0, t1), sr = mic.sr;
  let peak = 0; for (const v of pcm) peak = Math.max(peak, Math.abs(v));
  const L = layerById(S, layerId); if (!L) return;
  snapshot();
  const buf = E.ctx.createBuffer(1, pcm.length || 1, sr); buf.copyToChannel(pcm, 0);
  const aid = addAudio(buf, { pcm });
  hum.takeNo++;
  const clip = { id: uid('C'), layer: layerId, b: b0, dur: pcm.length / sr, audio: aid, input: 'hum', take: hum.takeNo, active: true, src: 'mic' };
  for (const c of S.clips) if (c.layer === layerId && c.input === 'hum' && Math.abs(c.b - b0) < 1e-6) c.active = false;
  const an = analyzeHum(pcm, sr, { mode: L.params.mode, snap: L.params.snap });
  clip.analysis = { mode: an.mode, guess: an.guess.mode, why: an.guess.why, tune_cents: +an.tuneCents.toFixed(1), stats: an.guess.stats, peak: +peak.toFixed(3) };
  S.clips.push(clip);
  const notes = notesToBeats(an.segs, { t0Beat: b0, spb: spb(S), key: S.key, prog: S.prog, beatsPerBar: S.beatsPerBar, snap: L.params.snap });
  for (const n of notes) S.notes.push({ id: uid(), layer: layerId, input: 'hum', take: clip.id, ...n, raw: { b: n.b, d: n.d, ...n.raw } });
  hum.sessionTakes.push(clip.id);
  const tn = Math.abs(an.tuneCents) >= 8 ? `，整体${an.tuneCents > 0 ? '偏高' : '偏低'} ${Math.abs(an.tuneCents).toFixed(0)} 音分（吸附前已扣掉）` : '';
  status(`第 ${hum.takeNo} 遍：${MODE_PARAMS[an.mode].label}（${an.guess.why}）· ${notes.length} 个音${tn}${peak < 0.02 ? ' · ⚠️几乎没声音，麦克风选对了吗？' : ''}${peak > 0.98 ? ' · ⚠️削波了，离麦远一点' : ''}`, true);
  syncAll();
}

// ---------- 键盘 / 拍 演奏 ----------
const held = new Map();   // code → { voice, b, t, pitch, sound, layer, vel, col }
let tapCounter = 0;
function ctxTimeOfEvent(ev) {
  let ts = null;
  try { ts = E.ctx.getOutputTimestamp(); } catch (e) { /* 老浏览器 */ }
  return heardCtxTime(ts, ev.timeStamp, E.now, (E.ctx.outputLatency || 0) + (E.ctx.baseLatency || 0));
}
// 演奏到的拍：按键时刻（听到的时钟）− 你的固定偏移
function playedBeat(ev) {
  if (!T.playing) return null;
  return T.beatAt(ctxTimeOfEvent(ev) - (S.settings.kbOffsetMs || 0) / 1000);
}
function inLoop(b) { const [a, z] = loopRange(S); return !S.loop.on || (b >= a - 0.05 && b < z); }
function wrapIntoLoop(b) { const [a, z] = loopRange(S); if (!S.loop.on) return b; if (b < a) return b + (z - a); if (b >= z) return b - (z - a); return b; }

function noteDown(ev) {
  const L = sel(); if (!L) return false;
  const code = ev.code;
  const inRow = ROW_A.includes(code) || ROW_Q.includes(code) || ROW_Z.includes(code);
  if (!inRow) return false;
  if (held.has(code)) return true;
  const vel = S.settings.velMode === 'shift' ? (ev.shiftKey ? 1 : 0.72) : 0.8;
  const when = E.now, heard = ctxTimeOfEvent(ev);   // when = 现在出声；heard = 记录用（起点和时值同源）
  let pitch = null, sound = null, col = 0;
  if (L.kind === 'drums') {
    const i = [ROW_A, ROW_Q, ROW_Z].map(r => r.indexOf(code)).find(i => i >= 0);
    if (i === undefined || i < 0 || i >= DRUM_SOUNDS.length) return true;
    sound = DRUM_SOUNDS[i];
    const v = ROW_Q.includes(code) ? 1 : vel;
    E.hit(L, sound, v, when);
    if (sound === 'kick') E.duck(when, (T.lastEff || effectiveMacros({}, true)).sidechain);
    held.set(code, { b: playedBeat(ev), t: heard, sound, layer: L.id, vel: v });
  } else if (L.kind === 'synth' || L.kind === 'voice') {
    if (L.input === 'tap') {
      col = Math.max(ROW_A.indexOf(code), ROW_Q.indexOf(code), ROW_Z.indexOf(code));
      const oct = L.octave + (ROW_Q.includes(code) ? 1 : ROW_Z.includes(code) ? -1 : 0);
      const b = T.playing ? (playedBeat(ev) ?? 0) : 0;
      pitch = tapPitch(ctxObj(), b, Math.floor(col / 2), oct, S.settings.tapPattern, tapCounter++);
    } else {
      const d = codeToDegree(S.key, code); if (d === null) return true;
      pitch = degreeToMidi(S.key, L.octave, d);
    }
    if (L.kind === 'voice' && L.input === 'hum') return true;
    // 琶音器：播放中按住 = 每个 1/16 自动弹一个和弦音（arpTick 负责），这里不出长音
    if (L.preset === 'arp' && T.playing) { held.set(code, { arp: true, pitch, layer: L.id, vel, t: when, step: null, n: 0 }); litKey(code, true); return true; }
    const voice = E.note(L, pitch, vel, when);
    held.set(code, { voice, b: playedBeat(ev), t: heard, pitch, layer: L.id, vel });
  } else return true;
  litKey(code, true);
  return true;
}
function noteUp(ev) {
  const h = held.get(ev.code); if (!h) return;
  held.delete(ev.code); litKey(ev.code, false);
  const when = E.now;
  h.voice?.release(when);
  if (h.arp || !rec.on || h.b === null || !T.playing) return;
  const L = layerById(S, h.layer); if (!L) return;
  let b = h.b;
  // 预备拍里随手按的不算；循环起点前一点点（抢拍）= 循环尾巴上同一个位置
  if (S.loop.on && b < loopRange(S)[0] - 0.5) return;
  if (!inLoop(b)) b = wrapIntoLoop(b);
  if (!inLoop(b)) return;
  const holdSec = Math.max(0, ctxTimeOfEvent(ev) - h.t);
  let d = holdSec / spb(S);
  const [ls, le] = loopRange(S);
  if (S.loop.on) d = Math.min(d, le - ls);
  const vel = S.settings.velMode === 'hold' ? holdVelocity(holdSec) : h.vel;
  const input = L.input === 'tap' ? 'tap' : 'keyboard';
  const n = { id: uid(), layer: L.id, b, d: Math.max(0.05, d), vel, input, pass: rec.pass, raw: { b, d: Math.max(0.05, d), vel, hold_s: +holdSec.toFixed(3) } };
  if (h.sound) { n.sound = h.sound; n.d = 0.25; } else n.pitch = h.pitch;
  S.notes.push(n);
  rec.added.push(n.id);
}

// 琶音器：按住的键 → 从这个音往上的和弦音，每 1/16 一个；生成的每个音都是真值（录音时照记）
function arpTick() {
  if (!T || !T.playing) return;
  const sp = spb(S), look = E.now + 0.04;
  for (const h of held.values()) {
    if (!h.arp) continue;
    const L = layerById(S, h.layer); if (!L) continue;
    const seg = T.segAt(look), b = seg.b0 + (look - seg.ctx0) / sp;
    const k = Math.ceil(b * 4 - 1e-6);
    if (h.step !== null && k <= h.step) continue;
    h.step = k;
    const nb = k / 4;
    if (nb >= seg.b1) continue;                       // 段尾那一格交给下一段
    const when = seg.ctx0 + (nb - seg.b0) * sp;
    const deg = chordAtBar(S.prog, Math.floor(nb / S.beatsPerBar));
    const tones = chordTones(S.key, deg, h.pitch - 0.5, 4);
    const p = tones[h.n % 3] + (h.n % 6 >= 3 ? 12 : 0);
    h.n++;
    E.note(L, p, h.vel, Math.max(when, E.now), sp * 0.2);
    if (rec.on && inLoop(nb)) {
      const n = { id: uid(), layer: L.id, b: nb, d: 0.2, pitch: p, vel: h.vel, input: 'keyboard', derived: 'arp', pass: rec.pass, raw: { b: nb, d: 0.2, held_pitch: h.pitch } };
      S.notes.push(n); rec.added.push(n.id);
    }
  }
}
setInterval(arpTick, 8);

// ---------- 录音开关 ----------
const rec = { on: false, pass: null, added: [], layer: null, startedAt: 0 };
async function toggleRec() {
  if (rec.on) return stopRec();
  const L = sel();
  if (L.kind === 'voice' && !(await micOn())) return;
  snapshot();
  rec.on = true; rec.pass = uid('P'); rec.added = []; rec.layer = L.id;
  hum.active = L.kind === 'voice'; hum.sessionTakes = [];
  closeVariants(false);
  $('rec').classList.add('rec-on');
  if (!T.playing) {
    const [ls] = loopRange(S);
    const pre = S.settings.countIn ? S.beatsPerBar : 0;
    T.start(S.loop.on ? ls - pre : (TL.cursor || 0) - pre);
  } else if (hum.active) {
    status('从下一遍循环开头开始收音…', true);
  }
}
function stopRec() {
  rec.on = false;
  $('rec').classList.remove('rec-on');
  if (hum.active && hum.pass) humPassEnd(E.now, true);
  const wasHum = hum.active; hum.active = false;
  // 变体：键盘 / 拍 = 这次录进来的音；哼唱 = 这次的几遍（等最后一遍分析完）
  if (wasHum) setTimeout(() => openTakeVariants(rec.layer), ((S.settings.latencyMs || 0) + 450));
  else if (rec.added.length) openNoteVariants(rec.layer, rec.added);
}

// ---------- 变体 ----------
function openNoteVariants(layerId, ids, reroll = 0) {
  const L = layerById(S, layerId); if (!L) return;
  const idSet = new Set(ids);
  const src = S.notes.filter(n => idSet.has(n.id));
  if (!src.length) return;
  const isDrums = L.kind === 'drums';
  const opts = makeVariants(src, ctxObj(), { isDrums, reroll });
  const off = meanOffset(src);
  variants = { kind: 'notes', layer: layerId, removeIds: idSet, options: opts.map(o => ({ ...o, notes: o.notes.map(n => ({ ...n, layer: layerId })) })), idx: 0, reroll, ids };
  renderVariants();
  if (off.tooLarge) status('你这一段整体偏了将近半格：多半是延迟太大（比如蓝牙耳机）。吸附没扣偏移；先点「校准延迟」，或换有线耳机', true);
  else if (off.consistent) status(`你这一段平均${off.offset < 0 ? '提前' : '拖后'} ${Math.abs(off.offset * spb(S) * 1000).toFixed(0)} ms（很稳，多半是延迟）：吸附的变体里已经扣掉`, true);
}
function openTakeVariants(layerId) {
  const takes = hum.sessionTakes.map(id => S.clips.find(c => c.id === id)).filter(Boolean);
  if (!takes.length) return;
  if (takes.length === 1) return openHumInterp(layerId, takes[0]);
  const last = takes.slice(-4);
  variants = {
    kind: 'takes', layer: layerId, takeIds: last.map(c => c.id), removeIds: new Set(), idx: last.length - 1,
    options: last.map(c => ({ label: `第 ${c.take} 遍`, desc: `${MODE_PARAMS[c.analysis.mode].label} · ${S.notes.filter(n => n.take === c.id).length} 个音`, clipId: c.id, notes: null })),
  };
  renderVariants();
}
// 一条哼唱的几种解释：不吸附 / 吸到调 / 吸到调 + 节拍 60% / 100%
function openHumInterp(layerId, clip) {
  const src = S.notes.filter(n => n.take === clip.id);
  if (!src.length) { status('这一遍没认出音符（太轻？太短？）'); return; }
  const mk = (label, desc, f) => ({ label, desc, notes: f(cloneNotes(src)) });
  const unsnap = ns => ns.map(n => ({ ...n, pitch: Math.round(n.raw.pitch_raw) }));
  const off = meanOffset(src), ro = off.consistent ? off.offset : 0;
  variants = {
    kind: 'notes', layer: layerId, removeIds: new Set(src.map(n => n.id)), idx: 1,
    options: [
      mk('原样（半音）', '音高只取最近的半音，不管调', unsnap),
      mk('吸到调', '调内 + 强拍偏和弦音（默认）', ns => ns),
      mk('吸到调 + 节拍 60%', '时刻靠向格子六成', ns => quantize(ns, { strength: 0.6, removeOffset: ro })),
      mk('吸到调 + 节拍 100%', '完全对齐 1/16', ns => quantize(ns, { strength: 1, removeOffset: ro })),
    ].map(o => ({ ...o, notes: o.notes.map(n => ({ ...n, layer: layerId, take: clip.id })) })),
  };
  renderVariants();
}
function openTimbreVariants() {
  const L = sel(); if (!L || L.kind !== 'synth') return status('音色变体只给合成层');
  variants = { kind: 'timbre', layer: L.id, removeIds: new Set(), idx: 0, orig: { ...L.params }, options: [{ label: '原样', desc: '现在的音色', params: { ...L.params } }, ...TIMBRE_VARIANTS.slice(0, 4).map(v => ({ label: v.label, desc: v.desc, params: v.patch({ ...L.params }) }))] };
  renderVariants();
}
function previewVariant() {
  if (!variants) return;
  if (variants.kind === 'timbre') { const L = layerById(S, variants.layer); L.params = { ...variants.options[variants.idx].params }; E.updateLayer(L, true); renderParams(); }
  if (variants.kind === 'takes' && T.playing) { /* 下一遍循环生效 */ }
  renderVariants(); TL && TL.draw(T.beatNow);
}
function commitVariant(action) {
  if (!variants) return;
  const v = variants, o = v.options[v.idx], L = layerById(S, v.layer);
  snapshot();
  const [ls] = loopRange(S);
  S.picks.push({ at: new Date().toISOString(), t: +(S.anchor + ls * spb(S)).toFixed(3), layer: v.layer, kind: v.kind, options: v.options.map(x => x.label), chose: o.label, action, input: 'pick' });
  if (v.kind === 'timbre') {
    if (action === 'discard') L.params = { ...v.orig };
    else L.params = { ...o.params };
    E.updateLayer(L, true);
  } else if (v.kind === 'takes') {
    if (action !== 'discard') for (const c of S.clips) if (v.takeIds.includes(c.id)) c.active = c.id === o.clipId;
  } else if (action === 'replace') {
    S.notes = S.notes.filter(n => !v.removeIds.has(n.id));
    for (const n of o.notes) { const c = { ...n, id: uid(), layer: v.layer }; delete c._preview; if (!c.raw) c.raw = { b: n.b, d: n.d }; S.notes.push(c); }
  } else if (action === 'overlay') {
    const nl = makeLayer(L.kind === 'voice' ? 'synth' : L.kind, L.kind === 'voice' ? 'lead' : L.preset, `${L.name}·${o.label}`, S.layers.length);
    if (L.kind !== 'voice') nl.params = { ...L.params }; nl.octave = L.octave;
    S.layers.push(nl);
    for (const n of o.notes) { const c = { ...n, id: uid(), layer: nl.id, input: n.input || 'pick' }; delete c._preview; delete c.take; S.notes.push(c); }
  }
  const verb = { replace: '用了', overlay: '叠成新层', discard: '放弃了变体，留原样' }[action];
  status(`${verb}「${o.label}」`);
  variants = null; renderVariants(); syncAll();
}
function closeVariants(revert = true) {
  if (!variants) return;
  if (revert && variants.kind === 'timbre') { const L = layerById(S, variants.layer); L.params = { ...variants.orig }; E.updateLayer(L, true); }
  variants = null; renderVariants();
}
function renderVariants() {
  const box = $('variants');
  if (!variants) { box.classList.add('hidden'); return; }
  box.classList.remove('hidden');
  const c = $('varCards'); c.innerHTML = '';
  variants.options.forEach((o, i) => {
    const d = document.createElement('div'); d.className = 'vcard' + (i === variants.idx ? ' cur' : '');
    d.innerHTML = `<b>${i + 1}. ${o.label}</b><small>${o.desc || ''}</small>`;
    if (o.notes && o.notes.length) {
      const cv = document.createElement('canvas'); cv.width = 300; cv.height = 68; d.appendChild(cv);
      const g = cv.getContext('2d'), ns = o.notes, b0 = Math.min(...ns.map(n => n.b)), b1 = Math.max(...ns.map(n => n.b + (n.d || 0.25)));
      const ps = ns.filter(n => !n.sound).map(n => n.pitch), lo = Math.min(...ps, 60) - 1, hi = Math.max(...ps, 61) + 1;
      g.fillStyle = i === variants.idx ? '#ff8a4c' : '#4dff7c';
      for (const n of ns) {
        const x = (n.b - b0) / Math.max(0.5, b1 - b0) * 296 + 2, w = Math.max(2, (n.d || 0.2) / Math.max(0.5, b1 - b0) * 296);
        const y = n.sound ? DRUM_SOUNDS.indexOf(n.sound) * 8 + 2 : (hi - n.pitch) / (hi - lo) * 60 + 2;
        g.globalAlpha = 0.4 + 0.6 * (n.vel ?? 0.8); g.fillRect(x, y, w, 5);
      }
    }
    d.onclick = () => { variants.idx = i; previewVariant(); };
    d.ondblclick = () => { variants.idx = i; commitVariant('replace'); };
    c.appendChild(d);
  });
  $('reroll').classList.toggle('hidden', variants.kind !== 'notes' || !variants.ids);
}

// ---------- XY 推子 ----------
const xy = { down: false, x: 0.5, y: 0.5, lastB: null, live: {} };
function xyTargets() { return [$('xSel').value, $('ySel').value]; }
function xyApply(e) {
  const r = $('xy').getBoundingClientRect();
  xy.x = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
  xy.y = Math.max(0, Math.min(1, 1 - (e.clientY - r.top) / r.height));
  const [mx, my] = xyTargets();
  xy.live = { [mx]: xy.x, [my]: xy.y };
  if (rec.on && T.playing) {
    const b = T.beatNow; if (b === null) return;
    const [ls, le] = loopRange(S);
    for (const [id, v] of Object.entries(xy.live)) {
      let pts = S.curves[id] || [];
      const from = xy.lastB ?? b;
      if (S.loop.on && b < from - 0.5) { pts = curveWrite(pts, from, le - 1e-3, v); pts = curveWrite(pts, ls, b, v); }
      else pts = curveWrite(pts, from, b, v);
      S.curves[id] = pts;
      (S.curvesInput ||= {})[id] = 'xy';
      shownLanes.add(id);
    }
    xy.lastB = b;
  }
}
function setupXY() {
  const cv = $('xy');
  cv.addEventListener('pointerdown', e => { xy.down = true; xy.lastB = null; try { cv.setPointerCapture(e.pointerId); } catch (err) { /* 合成事件没有指针 */ } if (rec.on) snapshot(); xyApply(e); });
  cv.addEventListener('pointermove', e => { if (xy.down) xyApply(e); });
  const up = () => { xy.down = false; xy.live = {}; xy.lastB = null; renderLaneChecks(); };
  cv.addEventListener('pointerup', up); cv.addEventListener('pointercancel', up);
  for (const s of [$('xSel'), $('ySel')]) s.innerHTML = MACROS.map(m => `<option value="${m.id}">${m.label}</option>`).join('');
  $('xSel').value = 'cutoff'; $('ySel').value = 'drive';
}
function drawXY() {
  const cv = $('xy'), g = cv.getContext('2d'), W = cv.width, H = cv.height;
  g.fillStyle = '#0d110f'; g.fillRect(0, 0, W, H);
  g.strokeStyle = '#1c2520';
  for (let i = 1; i < 4; i++) { g.beginPath(); g.moveTo(i * W / 4, 0); g.lineTo(i * W / 4, H); g.moveTo(0, i * H / 4); g.lineTo(W, i * H / 4); g.stroke(); }
  const [mx, my] = xyTargets();
  const b = T && T.playing ? T.beatNow : TL?.cursor ?? 0;
  const vx = xy.live[mx] ?? curveAt(S.curves[mx], b, MACRO_BY_ID[mx].def), vy = xy.live[my] ?? curveAt(S.curves[my], b, MACRO_BY_ID[my].def);
  g.fillStyle = xy.down ? (rec.on ? '#ff4c5e' : '#4dff7c') : '#7f978a';
  g.beginPath(); g.arc(vx * W, (1 - vy) * H, xy.down ? 9 : 6, 0, 7); g.fill();
  g.fillStyle = '#7f978a'; g.font = '12px "Microsoft YaHei"';
  g.fillText(`→ ${MACRO_BY_ID[mx].label} ${vx.toFixed(2)}`, 8, H - 8); g.save(); g.translate(14, H - 60); g.rotate(-Math.PI / 2); g.fillText(`↑ ${MACRO_BY_ID[my].label} ${vy.toFixed(2)}`, 0, 0); g.restore();
}
function macroVals(b) { const v = {}; for (const m of MACROS) v[m.id] = xy.live[m.id] ?? curveAt(S.curves[m.id], b, m.def); return v; }
function renderMacroBars(b) {
  const raw = macroVals(b), eff = effectiveMacros(raw, S.settings.tensionParent);
  const box = $('macroBars');
  if (!box.children.length) box.innerHTML = MACROS.map(m => `<div class="mb${m.id === 'tension' ? ' t' : ''}" title="${m.word}"><span>${m.label}</span><div class="bar"><i></i><u></u></div><span></span></div>`).join('');
  MACROS.forEach((m, i) => {
    const row = box.children[i];
    row.querySelector('i').style.width = (Math.min(1, raw[m.id]) * 100) + '%';
    row.querySelector('u').style.width = (Math.min(1.1, eff[m.id]) / 1.1 * 100) + '%';
    row.lastChild.textContent = eff[m.id].toFixed(2);
  });
  return eff;
}

// ---------- 示波器 / 音高条 ----------
const scopeBufL = new Float32Array(2048), scopeBufR = new Float32Array(2048);
function drawScope() {
  const cv = $('scope'), g = cv.getContext('2d'), W = cv.width, H = cv.height;
  g.fillStyle = 'rgba(0,0,0,0.35)'; g.fillRect(0, 0, W, H);
  if (!E) return;
  E.scopeL.getFloatTimeDomainData(scopeBufL); E.scopeR.getFloatTimeDomainData(scopeBufR);
  g.strokeStyle = 'rgba(77,255,124,0.55)'; g.lineWidth = 1; g.beginPath();
  for (let i = 0; i < 2048; i += 2) { const x = W / 2 + scopeBufL[i] * W * 0.42, y = H / 2 - scopeBufR[i] * H * 0.42; if (i) g.lineTo(x, y); else g.moveTo(x, y); }
  g.stroke();
}
function livePitch() {
  if (!mic.on || mic.liveN < 2048) return null;
  const x = mic.live.subarray(mic.liveN - 2048, mic.liveN);
  const { x: d, sr } = decimate(x, mic.sr);
  let e = 0; for (const v of d) e += v * v;
  const db = 10 * Math.log10(e / d.length + 1e-12);
  if (db < -50) return { db, midi: 0 };
  const y = yin(d, sr, { fmin: 65, fmax: 3000 });
  return { db, midi: y.f0 > 0 && y.ap < 0.25 ? hzMidi(y.f0) : 0 };
}
function drawPitchStrip() {
  const cv = $('pitchStrip'); if (!cv.offsetParent) return;
  const p = livePitch(); if (p) { mic.strip.push(p); if (mic.strip.length > 240) mic.strip.shift(); }
  const g = cv.getContext('2d'), W = cv.width, H = cv.height;
  g.fillStyle = '#070908'; g.fillRect(0, 0, W, H);
  const L = sel(), c = 12 * (L.octave + 1) + S.key.root, lo = c - 7, hi = c + 17;
  // 调内的音画横线，根音橙色
  for (let m = Math.ceil(lo); m <= hi; m++) {
    const pc = ((m - S.key.root) % 12 + 12) % 12; if (!MODES[S.key.mode].steps.includes(pc)) continue;
    const y = H - (m - lo) / (hi - lo) * H; g.fillStyle = pc === 0 ? 'rgba(255,138,76,0.35)' : 'rgba(127,151,138,0.18)'; g.fillRect(0, y, W, 1);
  }
  g.fillStyle = '#4dff7c';
  mic.strip.forEach((q, i) => { if (!q.midi) return; const x = i / 240 * W, y = H - (q.midi - lo) / (hi - lo) * H; g.fillRect(x, y - 1.5, W / 240 + 1, 3); });
  const last = mic.strip[mic.strip.length - 1];
  if (last && last.midi) {
    const r = Math.round(last.midi), cents = Math.round((last.midi - r) * 100);
    $('pnName').textContent = noteName(r); $('pnCents').textContent = `${cents >= 0 ? '+' : ''}${cents} 音分`;
  } else { $('pnName').textContent = '—'; $('pnCents').textContent = last ? `${last.db.toFixed(0)} dB` : ''; }
}

// ---------- 页面：层列表 / 旋钮 / 键位 ----------
function renderLayers() {
  const box = $('layers'); box.innerHTML = '';
  S.layers.forEach((L, i) => {
    const d = document.createElement('div'); d.className = 'layer' + (L.id === S.sel ? ' sel' : '');
    const tag = L.kind === 'synth' ? PRESETS[L.preset]?.label : { drums: '鼓', voice: '哼唱', stem: '分轨' }[L.kind];
    d.innerHTML = `<div class="sw" style="background:${L.color}"></div><b class="dim">${i + 1}</b><div class="nm">${L.name}<small>${tag}</small></div>
      <div class="btns"><button data-a="m" class="${L.mute ? 'on' : ''}" title="静音">M</button><button data-a="s" class="${L.solo ? 'on' : ''}" title="独奏">S</button><button data-a="x" title="删层（可撤销）">✕</button></div><div class="meter"><i id="mt_${L.id}"></i></div>`;
    d.onclick = ev => {
      const a = ev.target.dataset.a;
      if (a === 'm') { snapshot(); L.mute = !L.mute; }
      else if (a === 's') { L.solo = !L.solo; }
      else if (a === 'x') { snapshot(); S.layers = S.layers.filter(q => q.id !== L.id); S.notes = S.notes.filter(n => n.layer !== L.id); S.clips = S.clips.filter(c => c.layer !== L.id); E.removeLayer(L.id); for (const sc of S.scenes) sc.layers = sc.layers.filter(id => id !== L.id); if (S.sel === L.id) S.sel = S.layers[0]?.id; status(`删了层「${L.name}」（Ctrl+Z 能找回来）`); }
      else S.sel = L.id;
      syncAll();
    };
    d.ondblclick = ev => { if (ev.target.dataset.a) return; const nm = prompt('层的名字', L.name); if (nm) { snapshot(); L.name = nm; syncAll(); } };
    box.appendChild(d);
  });
}
const P = (label, key, min, max, step, fmt = v => v) => ({ label, key, min, max, step, fmt });
const SYNTH_GROUPS = [
  ['振荡器 / 波形', [{ sel: 'wave', label: '波形', opts: [['sine', '正弦'], ['triangle', '三角'], ['square', '方波'], ['sawtooth', '锯齿']] }]],
  ['齐奏 / 失谐', [P('声部', 'voices', 1, 7, 1), P('失谐', 'detune', 0, 40, 1, v => v + '¢')]],
  ['FM（调频）', [P('频率比', 'fmRatio', 0, 8, 0.5), P('深度', 'fmIndex', 0, 6, 0.1)]],
  ['滤波器（截止 / 共振）', [P('截止', 'cutoff', 80, 16000, 1, v => v >= 1000 ? (v / 1000).toFixed(1) + 'k' : Math.round(v)), P('共振', 'reso', 0, 1, 0.01)]],
  ['调制（包络 → 截止）', [P('包络量', 'envAmt', 0, 1, 0.01)]],
  ['ADSR 包络', [P('A 起', 'attack', 0.001, 2, 0.001, v => (+v).toFixed(3)), P('D 落', 'decay', 0.01, 2, 0.01), P('S 停', 'sustain', 0, 1, 0.01), P('R 收', 'release', 0.01, 3, 0.01)]],
  ['LFO', [P('速度', 'lfoRate', 0.1, 12, 0.1, v => v + ' Hz'), P('深度', 'lfoDepth', 0, 1, 0.01), { sel: 'lfoTarget', label: '推谁', opts: [['pitch', '音高'], ['cutoff', '截止'], ['amp', '音量']] }]],
  ['失真 / 饱和', [P('失真', 'drive', 0, 1, 0.01)]],
  ['延迟 / 混响（送出）', [P('延迟', 'delaySend', 0, 1, 0.01), P('混响', 'reverbSend', 0, 1, 0.01)]],
  ['侧链 / 声像', [{ chk: 'duck', label: '被底鼓压' }, P('声像', 'pan', -1, 1, 0.05)]],
];
function paramRow(L, it, onChange) {
  const row = document.createElement('div'); row.className = 'prow';
  if (it.sel) {
    row.innerHTML = `<span>${it.label}</span><select>${it.opts.map(([v, t]) => `<option value="${v}">${t}</option>`).join('')}</select><span></span>`;
    const s = row.querySelector('select'); s.value = L.params[it.sel]; s.onchange = () => { snapshot(); L.params[it.sel] = s.value; onChange(); };
  } else if (it.chk) {
    row.innerHTML = `<span>${it.label}</span><input type="checkbox"><span></span>`;
    const c = row.querySelector('input'); c.checked = !!L.params[it.chk]; c.onchange = () => { snapshot(); L.params[it.chk] = c.checked; onChange(); };
  } else {
    const cur = it.root ? L[it.key] : L.params[it.key];
    row.innerHTML = `<span>${it.label}</span><input type="range" min="${it.min}" max="${it.max}" step="${it.step}"><span>${it.fmt(cur)}</span>`;
    const r = row.querySelector('input'); r.value = cur;
    r.onpointerdown = () => snapshot();
    r.oninput = () => { const v = +r.value; if (it.root) L[it.key] = v; else L.params[it.key] = v; row.lastChild.textContent = it.fmt(v); onChange(); };
  }
  return row;
}
function renderParams() {
  const L = sel(), box = $('params'); box.innerHTML = '';
  if (!L) return;
  $('paramHd').textContent = `旋钮 · ${L.name}`;
  const upd = () => { E && E.updateLayer(L, true); };
  const group = (title, items) => { const w = document.createElement('div'); w.className = 'word'; w.innerHTML = `<b>${title}</b>`; box.appendChild(w); for (const it of items) box.appendChild(paramRow(L, it, upd)); };
  group('音量 / 八度', [{ ...P('音量', 'gainDb', -30, 12, 0.5, v => v + ' dB'), root: true }, ...(L.kind === 'synth' || L.kind === 'voice' ? [{ ...P('八度', 'octave', 0, 7, 1), root: true }] : [])]);
  if (L.kind === 'synth') {
    const w = document.createElement('div'); w.className = 'prow';
    w.innerHTML = `<span>音色</span><select>${Object.entries(PRESETS).filter(([k]) => k !== 'sing').map(([k, p]) => `<option value="${k}">${p.label}</option>`).join('')}</select><span></span>`;
    const s = w.querySelector('select'); s.value = L.preset;
    s.onchange = () => { snapshot(); L.preset = s.value; L.params = { ...PRESETS[s.value].p }; L.octave = PRESETS[s.value].octave; upd(); renderParams(); renderKeys(); };
    box.appendChild(w);
    for (const [t, items] of SYNTH_GROUPS) group(t, items);
  } else if (L.kind === 'voice') {
    group('哼唱识别', [
      { sel: 'mode', label: '类型', opts: [['auto', '自动猜'], ['hum', '哼唱（要音符）'], ['voice', '人声（要原声）'], ['melody', '旋律 / 口哨']] },
      { sel: 'snap', label: '吸附', opts: [['scale', '到调式'], ['semitone', '到半音'], ['off', '不吸附']] },
    ]);
    const last = S.clips.filter(c => c.layer === L.id && c.active).pop();
    if (last && last.analysis) { const w = document.createElement('div'); w.className = 'dim'; w.style.fontSize = '12px'; w.textContent = `最近一遍：猜 ${MODE_PARAMS[last.analysis.guess].label}（${last.analysis.why}），按 ${MODE_PARAMS[last.analysis.mode].label} 切音；跑调 ${last.analysis.tune_cents} 音分`; box.appendChild(w); }
    const re = document.createElement('button'); re.textContent = '按现在的类型 / 吸附重新识别'; re.onclick = () => reanalyze(L); box.appendChild(re);
    group('放什么', [P('原声音量', 'rawGainDb', -40, 6, 0.5, v => v + ' dB'), { chk: 'synthOn', label: '合成器跟唱' }, { sel: 'synthPreset', label: '跟唱音色', opts: Object.entries(PRESETS).map(([k, p]) => [k, p.label]) }, { chk: 'bendOn', label: '保留滑音颤音' }]);
    const cp = document.createElement('button'); cp.textContent = '把音符复制成一条合成层'; cp.onclick = () => copyHumToSynth(L); box.appendChild(cp);
    group('送出', [P('延迟', 'delaySend', 0, 1, 0.01), P('混响', 'reverbSend', 0, 1, 0.01), P('失真', 'drive', 0, 1, 0.01)]);
  } else {
    group('失真 / 送出', [P('失真', 'drive', 0, 1, 0.01), P('延迟', 'delaySend', 0, 1, 0.01), P('混响', 'reverbSend', 0, 1, 0.01), { chk: 'duck', label: '被底鼓压' }, P('声像', 'pan', -1, 1, 0.05)]);
  }
  for (const s of box.querySelectorAll('select')) s.addEventListener('change', () => { if (L.kind === 'voice') renderParams(); });
}
function reanalyze(L) {
  const clips = S.clips.filter(c => c.layer === L.id && c.input === 'hum');
  if (!clips.length) return status('这一层还没有录音');
  snapshot();
  for (const c of clips) {
    const a = audio.get(c.audio); if (!a) continue;
    const pcm = a.pcm || a.buffer.getChannelData(0);
    const an = analyzeHum(pcm, a.sr, { mode: L.params.mode, snap: L.params.snap });
    S.notes = S.notes.filter(n => n.take !== c.id);
    const notes = notesToBeats(an.segs, { t0Beat: c.b, spb: spb(S), key: S.key, prog: S.prog, beatsPerBar: S.beatsPerBar, snap: L.params.snap });
    for (const n of notes) S.notes.push({ id: uid(), layer: L.id, input: 'hum', take: c.id, ...n, raw: { b: n.b, d: n.d, ...n.raw } });
    c.analysis = { ...c.analysis, mode: an.mode, guess: an.guess.mode, why: an.guess.why, tune_cents: +an.tuneCents.toFixed(1) };
  }
  status(`重新识别了 ${clips.length} 条`); syncAll();
}
function copyHumToSynth(L) {
  const act = new Set(S.clips.filter(c => c.active && c.layer === L.id).map(c => c.id));
  const src = S.notes.filter(n => n.layer === L.id && (!n.take || act.has(n.take)));
  if (!src.length) return status('没有音符可复制');
  snapshot();
  const nl = makeLayer('synth', 'lead', `${L.name}→合成`, S.layers.length);
  S.layers.push(nl);
  for (const n of src) { const c = { ...n, id: uid(), layer: nl.id }; delete c.take; delete c.bend; S.notes.push(c); }
  S.sel = nl.id; syncAll(); status('复制成了一条合成层（原哼唱层不动）');
}
function renderKeys() {
  const L = sel(), box = $('keys'); box.innerHTML = '';
  $('selName').textContent = L ? `${S.layers.indexOf(L) + 1} · ${L.name}` : '';
  for (const b of $('inputSeg').children) { b.classList.toggle('on', L && L.input === b.dataset.v); b.disabled = !L || (L.kind === 'drums' && b.dataset.v !== 'keyboard') || (L.kind === 'stem') || (L.kind === 'synth' && b.dataset.v === 'hum'); }
  $('hum').classList.toggle('hidden', !L || L.kind !== 'voice');
  $('tapPatWrap').classList.toggle('hidden', !L || L.input !== 'tap');
  const bar = T && T.playing ? Math.floor((T.beatNow ?? 0) / S.beatsPerBar) : S.loop.start;
  const deg = chordAtBar(S.prog, bar), pcs = chordPcs(S.key, deg);
  const hint = { keyboard: '按哪个都不会错音：你只管节奏和走向', tap: `只拍节奏；音高自动填当前和弦（${chordRoman(S.key, deg)}）的音`, hum: '开麦、按 ` 录：每一遍循环存一条', none: '分轨：只能静音 / 送效果 / 进场景' }[L?.input || 'none'];
  $('modeHint').textContent = L?.kind === 'drums' ? '鼓：一个键一种鼓（Q 排更重）' : hint;
  for (const row of [ROW_Q, ROW_A, ROW_Z]) {
    const r = document.createElement('div'); r.className = 'krow';
    row.forEach((code, i) => {
      const k = document.createElement('div'); k.className = 'key'; k.id = 'k_' + code;
      let txt = '';
      if (!L) txt = '';
      else if (L.kind === 'drums') { txt = DRUM_SOUNDS[i] ? DRUM_LABEL[DRUM_SOUNDS[i]] : ''; if (!txt) k.classList.add('off'); }
      else if (L.kind === 'stem' || L.input === 'hum') k.classList.add('off');
      else if (L.input === 'tap') { txt = '拍'; }
      else {
        const m = degreeToMidi(S.key, L.octave, codeToDegree(S.key, code));
        txt = noteName(m);
        if (((m - S.key.root) % 12 + 12) % 12 === 0) k.classList.add('root');
        if (pcs.includes(((m % 12) + 12) % 12)) k.classList.add('chord');
      }
      k.innerHTML = `<b>${KEY_LABEL(code)}</b><span>${txt}</span>`;
      r.appendChild(k);
    });
    box.appendChild(r);
  }
}
function litKey(code, on) { const k = $('k_' + code); if (k) k.classList.toggle('down', on); }

// ---------- 场景 ----------
function renderScenes() {
  const box = $('scenes');
  if (!S.scenes.length) { box.innerHTML = '<p class="dim">还没有场景：整首所有层都按自己的静音走。加一个场景 = 从那个小节开始，只有勾上的层在响。</p>'; return; }
  let h = '<table><tr><th>层 \\ 场景</th>';
  S.scenes.forEach((sc, i) => { h += `<th><div>Shift+${i + 1} · 第 ${sc.bar + 1} 小节 <button data-del="${sc.id}" title="删场景">✕</button></div><input data-nm="${sc.id}" value="${sc.name}"><br><input class="txt" data-tx="${sc.id}" value="${sc.text || ''}" placeholder="她此刻（一句）"></th>`; });
  h += '</tr>';
  for (const L of S.layers) { h += `<tr><td style="color:${L.color}">${L.name}</td>`; for (const sc of S.scenes) h += `<td class="cell${sc.layers.includes(L.id) ? ' on' : ''}" data-sc="${sc.id}" data-l="${L.id}">${sc.layers.includes(L.id) ? '●' : '·'}</td>`; h += '</tr>'; }
  box.innerHTML = h + '</table>';
  box.querySelectorAll('.cell').forEach(td => td.onclick = () => { snapshot(); const sc = S.scenes.find(s => s.id === td.dataset.sc); const id = td.dataset.l; sc.layers = sc.layers.includes(id) ? sc.layers.filter(x => x !== id) : [...sc.layers, id]; renderScenes(); });
  box.querySelectorAll('[data-nm]').forEach(inp => inp.onchange = () => { snapshot(); S.scenes.find(s => s.id === inp.dataset.nm).name = inp.value; TL.draw(null); });
  box.querySelectorAll('[data-tx]').forEach(inp => inp.onchange = () => { snapshot(); S.scenes.find(s => s.id === inp.dataset.tx).text = inp.value; });
  box.querySelectorAll('[data-del]').forEach(b => b.onclick = () => { snapshot(); S.scenes = S.scenes.filter(s => s.id !== b.dataset.del); renderScenes(); });
}
function addScene() {
  snapshot();
  const bar = S.loop.start;
  if (S.scenes.some(s => s.bar === bar)) return status('这个小节已经有场景了');
  S.scenes.push({ id: uid('S'), bar, name: `场景 ${S.scenes.length + 1}`, layers: S.layers.filter(l => !l.mute).map(l => l.id), text: '' });
  S.scenes.sort((a, b) => a.bar - b.bar);
  renderScenes();
}
function jumpScene(i) {
  const sc = S.scenes[i]; if (!sc) return;
  const next = S.scenes[i + 1];
  S.loop.start = sc.bar; S.loop.bars = Math.max(1, Math.min(16, (next ? next.bar : S.songBars) - sc.bar));
  syncHeader(); status(`循环区 → ${sc.name}（第 ${sc.bar + 1}–${sc.bar + S.loop.bars} 小节）`);
}

// ---------- 时间线编辑：删 / 复制 / 粘贴 ----------
function deleteSel() {
  if (TL.sel.size) { snapshot(); S.notes = S.notes.filter(n => !TL.sel.has(n.id)); TL.sel.clear(); status('删了'); return syncAll(); }
  if (TL.range) {
    snapshot(); const [a, z] = TL.range;
    S.notes = S.notes.filter(n => n.b < a || n.b >= z);
    for (const k of Object.keys(S.curves)) S.curves[k] = S.curves[k].filter(p => p[0] < a || p[0] > z);
    status(`删了第 ${(a / S.beatsPerBar + 1).toFixed(2)}–${(z / S.beatsPerBar + 1).toFixed(2)} 小节里的所有东西`); syncAll();
  }
}
function copySel() {
  if (TL.range) {
    const [a, z] = TL.range;
    clipboard = { span: z - a, notes: events().filter(n => !n._preview && n.b >= a && n.b < z).map(n => ({ ...n, b: n.b - a })), curves: Object.fromEntries(Object.entries(S.curves).map(([k, p]) => [k, p.filter(q => q[0] >= a && q[0] <= z).map(q => [q[0] - a, q[1]])])) };
  } else if (TL.sel.size) {
    const ns = S.notes.filter(n => TL.sel.has(n.id)), a = Math.min(...ns.map(n => n.b));
    clipboard = { span: Math.max(...ns.map(n => n.b + (n.d || 0.25))) - a, notes: ns.map(n => ({ ...n, b: n.b - a })), curves: {} };
  } else return status('先选：拖标尺下条选一段，或框选音符');
  status(`复制了 ${clipboard.notes.length} 个音${Object.values(clipboard.curves).some(p => p.length) ? ' + 曲线' : ''}`);
}
function pasteAt(b) {
  if (!clipboard) return status('剪贴板是空的');
  snapshot();
  const ids = [];
  for (const n of clipboard.notes) { const id = uid(); ids.push(id); S.notes.push({ ...n, id, b: n.b + b }); }
  for (const [k, pts] of Object.entries(clipboard.curves)) {
    if (!pts.length) continue;
    let cur = (S.curves[k] || []).filter(p => p[0] < b || p[0] > b + clipboard.span);
    cur = cur.concat(pts.map(([pb, v]) => [pb + b, v])).sort((p, q) => p[0] - q[0]);
    S.curves[k] = cur;
  }
  TL.sel = new Set(ids);
  status(`贴到第 ${(b / S.beatsPerBar + 1).toFixed(2)} 小节`); syncAll();
}
function duplicate() {
  copySel(); if (!clipboard) return;
  const start = TL.range ? TL.range[0] : Math.min(...S.notes.filter(n => TL.sel.has(n.id)).map(n => n.b));
  const span = TL.range ? clipboard.span : Math.ceil(clipboard.span / S.beatsPerBar) * S.beatsPerBar;
  pasteAt(start + span);
  if (TL.range) TL.range = [TL.range[0] + span, TL.range[1] + span];
}

// ---------- 原曲分轨 / 拖进来的文件 ----------
async function api(path, opt) { const r = await fetch(path, opt); if (!r.ok) throw new Error(`${r.status} ${await r.text()}`); return r; }
async function loadSense() {
  try {
    const j = await (await api(`/api/sense?work=${encodeURIComponent(S.work)}`)).json();
    const sel = $('anchorSel');
    sel.innerHTML = '<option value="">原曲从第几小节开始…</option>' + (j.downbeats || []).map((t, i) => `<option value="${t}">第 ${i + 1} 个重拍 · ${t.toFixed(2)} s</option>`).join('');
    S._sense = { bpm: j.bpm, downbeats: j.downbeats || [] };
    return j;
  } catch (e) { $('anchorSel').innerHTML = '<option value="">（这个作品没有 感官.json）</option>'; S._sense = null; return null; }
}
async function loadStems() {
  if (!server) return status('要用 serve.py 打开页面才能读原曲分轨（或把 wav 拖进来）');
  const t0 = S.anchor, t1 = S.anchor + songBeats(S) * spb(S);
  let names;
  try { names = await (await api(`/api/stems?work=${encodeURIComponent(S.work)}`)).json(); } catch (e) { return status('读不到分轨：' + e.message); }
  if (!names.length) return status('这个作品的 分轨/ 里没有 wav');
  status(`载入 ${names.length} 条分轨（${t0.toFixed(2)}–${t1.toFixed(2)} s）…`, true);
  snapshot();
  const msgs = [];
  for (const nm of names) {
    const r = await api(`/api/stem?work=${encodeURIComponent(S.work)}&name=${encodeURIComponent(nm)}&t0=${t0}&t1=${t1}`);
    const buf = await E.ctx.decodeAudioData(await r.arrayBuffer());
    const aid = addAudio(buf, { file: `分轨/${nm}` });
    const L = makeLayer('stem', 'stem', '原曲·' + nm.replace(/\.wav$/i, ''), S.layers.length);
    S.layers.push(L);
    S.clips.push({ id: uid('C'), layer: L.id, b: 0, dur: buf.duration, audio: aid, input: 'stem', active: true, fileOffset: t0, src: `分轨/${nm}@${t0.toFixed(3)}-${t1.toFixed(3)}` });
    if (/drum/i.test(nm)) msgs.push(buildGroove(buf));
  }
  msgs.push(checkGridDrift());
  status([`载入了 ${names.length} 条分轨`, ...msgs.filter(Boolean)].join('\n'), true);
  syncAll();
}
function buildGroove(buf) {
  const ch = buf.getChannelData(0);
  const ons = detectOnsets(ch, buf.sampleRate);
  groove = grooveFromOnsets(ons.map(t => t / spb(S)), { beatsPerBar: S.beatsPerBar });
  const ms = groove.offsets.map(o => Math.round(o * spb(S) * 1000));
  return `律动模板（鼓轨 ${ons.length} 个起音）：16 格偏移 ${ms.join(' ')} ms。变体里的「吸到原曲律动」用它`;
}
// 我们的格子是匀速的；原曲的重拍不一定：差太多就提醒
function checkGridDrift() {
  if (!S._sense || !S._sense.downbeats.length) return;
  const db = S._sense.downbeats; let worst = 0, at = 0;
  for (let k = 0; k <= S.songBars; k++) {
    const t = S.anchor + k * S.beatsPerBar * spb(S);
    const near = db.reduce((p, q) => Math.abs(q - t) < Math.abs(p - t) ? q : p, db[0]);
    if (Math.abs(near - t) > Math.abs(worst)) { worst = near - t; at = k; }
  }
  return Math.abs(worst) > 0.05 ? `⚠️格子是匀速 ${S.bpm} BPM，原曲第 ${at + 1} 小节的重拍差 ${(worst * 1000).toFixed(0)} ms（先在「原曲从第几小节开始」里选起点；感官.json 里有的小节不是整 4 拍）` : '';
}
async function addFileLayer(file) {
  try {
    const buf = await E.ctx.decodeAudioData(await file.arrayBuffer());
    snapshot();
    const aid = addAudio(buf, { file: null });
    const L = makeLayer('stem', 'stem', file.name.replace(/\.[^.]+$/, ''), S.layers.length);
    S.layers.push(L);
    const [ls] = loopRange(S);
    S.clips.push({ id: uid('C'), layer: L.id, b: ls, dur: buf.duration, audio: aid, input: 'file', active: true, src: file.name });
    const gm = /drum|鼓/i.test(file.name) ? '\n' + buildGroove(buf) : '';
    status(`拖进来：${file.name}（${buf.duration.toFixed(1)} s，从循环起点开始放）${gm}`, !!gm);
    syncAll();
  } catch (e) { status('读不了这个文件：' + e.message); }
}

// ---------- 保存 / 打开 ----------
function f32ToB64(f) { const u = new Uint8Array(f.buffer, f.byteOffset, f.byteLength); let s = ''; for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode.apply(null, u.subarray(i, i + 0x8000)); return btoa(s); }
async function save() {
  const errs = validateTruth(exportTruth(S));
  if (errs.length) { status('自检没过，没存：' + errs.join('；')); return; }
  if (!server) return downloadAll();
  try {
    // 版本号由服务器给；文件已存在服务器会拒（409），再要下一个号 —— 绝不覆盖
    for (let attempt = 0; attempt < 3; attempt++) {
      const { ver } = await (await api(`/api/next?work=${encodeURIComponent(S.work)}`)).json();
      // 要存的音频：麦克风录的 + 拖进来的短文件（原曲分轨只记来源，不重复存；已经存过的沿用旧文件）
      const files = {}, toSave = [];
      for (const c of S.clips) {
        const a = audio.get(c.audio); if (!a) continue;
        if (a.file) { files[c.audio] = a.file; continue; }
        if (c.input === 'file' && a.buffer.duration > 90) continue;
        const name = c.input === 'hum' ? `哼唱_${ver}_第${c.take}遍.wav` : `素材_${ver}_${c.id}.wav`;
        files[c.audio] = `作曲/${name}`;          // 路径一律相对作品文件夹（分轨是 分轨/xxx.wav）
        toSave.push({ name, sr: a.sr, b64: f32ToB64(a.pcm || a.buffer.getChannelData(0)) });
      }
      const truth = exportTruth(S, { audioFiles: files }), session = exportSession(S, files);
      const r = await fetch(`/api/save?work=${encodeURIComponent(S.work)}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ver, truth, session, audio: toSave }) });
      if (r.status === 409) continue;
      if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
      const j = await r.json();
      for (const [id, fn] of Object.entries(files)) { const a = audio.get(id); if (a) a.file = fn; }
      for (const c of S.clips) if (files[c.audio]) c.file = files[c.audio];
      status(`存好了（${ver}）：${j.dir}\n${j.written.join(' · ')}`, true);
      lastSaved = { truth: j.truth, snap: JSON.stringify(S) };
      return j.truth;
    }
    status('保存失败：连续三次撞上已有的版本号');
  } catch (e) { status('保存失败：' + e.message); }
}
// 服务器那边用同一个 exportTruth 的结果（页面先算好），这里给离线用
function downloadAll() {
  const dl = (name, blob) => { const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 3000); };
  const files = {};
  for (const c of S.clips) { const a = audio.get(c.audio); if (!a || a.file) continue; const nm = `哼唱_${c.take ?? c.id}.wav`; files[c.audio] = nm; dl(nm, wavBlob(a.pcm || a.buffer.getChannelData(0), a.sr)); }
  dl('作曲.json', new Blob([JSON.stringify(exportTruth(S, { audioFiles: files }), null, 1)], { type: 'application/json' }));
  dl('作曲_会话.json', new Blob([JSON.stringify(exportSession(S, files))], { type: 'application/json' }));
  status('没连上 serve.py：下载到浏览器的下载文件夹了');
}
function wavBlob(f, sr) {
  const n = f.length, b = new ArrayBuffer(44 + n * 2), v = new DataView(b);
  const w = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  w(0, 'RIFF'); v.setUint32(4, 36 + n * 2, true); w(8, 'WAVEfmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true); w(36, 'data'); v.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, f[i])) * 32767, true);
  return new Blob([b], { type: 'audio/wav' });
}
// 定稿：写 作曲/当前.txt（下游只读它）。改过还没存 → 先存一版再定
let lastSaved = null;
async function refreshFinal() {
  if (!server) { $('finalInfo').textContent = ''; return; }
  try { const j = await (await api(`/api/final?work=${encodeURIComponent(S.work)}`)).json(); $('finalInfo').textContent = j.final ? `定稿：${j.final.replace('.json', '')}` : '还没有定稿'; } catch (e) { $('finalInfo').textContent = ''; }
}
async function finalize() {
  if (!server) return status('要用 serve.py 打开页面才能定稿');
  let name = lastSaved && lastSaved.snap === JSON.stringify(S) ? lastSaved.truth : null;
  if (!name) { name = await save(); if (!name) return; }
  if (!confirm(`把 ${name} 定为定稿？\n感受器、剪辑、画面以后都读这一版（随时可以再定别的版本）。`)) return;
  try {
    const r = await api(`/api/final?work=${encodeURIComponent(S.work)}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name }) });
    const j = await r.json();
    status(`定稿：${j.final}（之前：${j.prev || '无'}）。写在 作曲/当前.txt`, true);
    refreshFinal();
  } catch (e) { status('定稿失败：' + e.message); }
}
async function openSessions() {
  if (!server) return status('要用 serve.py 打开页面才能读会话');
  const list = await (await api(`/api/sessions?work=${encodeURIComponent(S.work)}`)).json();
  const box = $('sessList'); box.innerHTML = list.length ? '' : '<p class="dim">这个作品还没有存过会话</p>';
  for (const s of list) { const b = document.createElement('button'); b.textContent = `${s.name}  ·  ${s.mtime}`; b.onclick = () => loadSession(s.name); box.appendChild(b); }
  $('sessBox').classList.remove('hidden');
}
async function loadSession(name) {
  $('sessBox').classList.add('hidden');
  try {
    const j = await (await api(`/api/file?work=${encodeURIComponent(S.work)}&name=${encodeURIComponent(name)}`)).json();
    T.stop(true); closeVariants(false);
    const { audioFiles = {}, ...st } = j;
    snapshot();
    S = { ...makeState(), ...st };
    audio.clear(); peaksCache.clear();
    for (const c of S.clips) {
      const fn = audioFiles[c.audio] || c.file;
      try {
        let r;
        if (c.input === 'stem' && c.src && c.src.includes('@')) {
          const [p, rng] = c.src.split('@'), [a, z] = rng.split('-');
          r = await api(`/api/stem?work=${encodeURIComponent(S.work)}&name=${encodeURIComponent(p.replace('分轨/', ''))}&t0=${a}&t1=${z}`);
        } else if (fn) r = await api(`/api/file?work=${encodeURIComponent(S.work)}&name=${encodeURIComponent(fn.replace(/^作曲\//, ''))}`);
        else continue;
        const buf = await E.ctx.decodeAudioData(await r.arrayBuffer());
        audio.set(c.audio, { buffer: buf, pcm: c.input === 'hum' ? buf.getChannelData(0) : null, sr: buf.sampleRate, file: fn || null });
      } catch (e) { status(`有一条音频没读到：${fn || c.src}`); }
    }
    hum.takeNo = Math.max(0, ...S.clips.map(c => c.take || 0));
    status(`打开了 ${name}`); syncAll();
  } catch (e) { status('打开失败：' + e.message); }
}

// ---------- 顶栏 ----------
function syncHeader() {
  $('bpm').value = S.bpm; $('root').value = S.key.root; $('mode').value = S.key.mode;
  $('prog').value = S.prog.map(d => chordRoman(S.key, d)).join(' ');
  $('loopBars').value = String(S.loop.bars); $('songBars').value = S.songBars; $('anchor').value = S.anchor;
  $('loopOn').classList.toggle('on', S.loop.on); $('metro').classList.toggle('on', S.settings.metro); $('countIn').classList.toggle('on', S.settings.countIn);
  $('velMode').value = S.settings.velMode; $('tapPat').value = S.settings.tapPattern; $('tParent').checked = S.settings.tensionParent;
  $('snapEdit').value = String(S.settings.snapEdit);
  if (!$('loopBars').value) { const o = document.createElement('option'); o.textContent = S.loop.bars; $('loopBars').appendChild(o); $('loopBars').value = String(S.loop.bars); }
  if (E) E.setDelayTime(spb(S) * 0.75);
}
function renderLaneChecks() {
  const box = $('laneChecks');
  box.innerHTML = '曲线道：' + MACROS.filter(m => m.id !== 'tension').map(m => `<label><input type="checkbox" data-l="${m.id}" ${shownLanes.has(m.id) || (S.curves[m.id] || []).length ? 'checked' : ''}>${m.label}</label>`).join('');
  box.querySelectorAll('input').forEach(c => c.onchange = () => { if (c.checked) shownLanes.add(c.dataset.l); else shownLanes.delete(c.dataset.l); TL.resize(); });
}
function syncAll() {
  syncHeader(); renderLayers(); renderParams(); renderKeys(); renderScenes(); renderLaneChecks();
  if (E) for (const L of S.layers) E.ensureLayer(L);
  if (TL) { TL.resize(); TL.draw(T?.beatNow ?? null); }
}
function bindHeader() {
  $('root').innerHTML = NOTE_NAMES.map((n, i) => `<option value="${i}">${n}</option>`).join('');
  $('mode').innerHTML = Object.entries(MODES).map(([k, m]) => `<option value="${k}">${m.label}</option>`).join('');
  const restartIfPlaying = () => { if (T.playing) { const b = T.beatNow ?? 0; T.start(b); } };
  $('bpm').onchange = () => { snapshot(); S.bpm = Math.max(40, Math.min(240, +$('bpm').value || 120)); restartIfPlaying(); syncAll(); };
  $('root').onchange = () => { snapshot(); S.key.root = +$('root').value; syncAll(); };
  $('mode').onchange = () => { snapshot(); S.key.mode = $('mode').value; syncAll(); };
  $('prog').onchange = () => { snapshot(); S.prog = parseProgression($('prog').value); syncAll(); };
  $('loopBars').onchange = () => { S.loop.bars = +$('loopBars').value; S.loop.start = Math.min(S.loop.start, Math.max(0, S.songBars - S.loop.bars)); syncAll(); };
  $('songBars').onchange = () => { snapshot(); S.songBars = Math.max(1, Math.min(256, +$('songBars').value || 16)); TL.fit('song'); syncAll(); };
  $('anchor').onchange = () => { snapshot(); S.anchor = +$('anchor').value || 0; const w = checkGridDrift(); if (w) status(w, true); syncAll(); };
  $('anchorSel').onchange = () => { const v = $('anchorSel').value; if (v === '') return; snapshot(); S.anchor = +v; if (S._sense?.bpm) S.bpm = S._sense.bpm; syncAll(); status([`起点 = ${(+v).toFixed(2)} s（第 1 小节对齐原曲这个重拍）`, checkGridDrift()].filter(Boolean).join('\n'), true); };
  $('work').onchange = async () => { S.work = $('work').value; lastSaved = null; await loadSense(); refreshFinal(); status(`作品：${S.work}`); };
  $('play').onclick = togglePlay; $('rec').onclick = toggleRec;
  $('loopOn').onclick = () => { S.loop.on = !S.loop.on; syncHeader(); };
  $('metro').onclick = () => { S.settings.metro = !S.settings.metro; syncHeader(); };
  $('countIn').onclick = () => { S.settings.countIn = !S.settings.countIn; syncHeader(); };
  $('undo').onclick = undo; $('redo').onclick = redo; $('save').onclick = save; $('load').onclick = openSessions; $('final').onclick = finalize;
  $('help').onclick = () => $('helpBox').classList.remove('hidden'); $('helpClose').onclick = () => $('helpBox').classList.add('hidden');
  $('sessClose').onclick = () => $('sessBox').classList.add('hidden');
  $('velMode').onchange = () => { S.settings.velMode = $('velMode').value; };
  $('tapPat').onchange = () => { S.settings.tapPattern = $('tapPat').value; };
  $('tParent').onchange = () => { S.settings.tensionParent = $('tParent').checked; };
  $('snapEdit').onchange = () => { S.settings.snapEdit = +$('snapEdit').value; };
  $('inputSeg').onclick = ev => { const v = ev.target.dataset.v; if (!v || ev.target.disabled) return; sel().input = v; renderKeys(); };
  $('addLayer').onchange = () => {
    const [kind, preset] = $('addLayer').value.split(':'); $('addLayer').value = '';
    if (!kind) return;
    if (kind === 'stem') { $('fileIn').click(); return; }
    snapshot();
    const name = kind === 'synth' ? PRESETS[preset].label : kind === 'drums' ? '鼓' : '哼唱';
    const L = makeLayer(kind, preset, name, S.layers.length);
    S.layers.push(L); S.sel = L.id; syncAll();
  };
  $('fileIn').onchange = () => { const f = $('fileIn').files[0]; if (f) addFileLayer(f); $('fileIn').value = ''; };
  $('micBtn').onclick = micOn; $('calib').onclick = calibrate;
  $('reroll').onclick = () => { if (variants?.ids) openNoteVariants(variants.layer, variants.ids, (variants.reroll || 0) + 1); };
  $('timbreVar').onclick = openTimbreVariants;
  $('fitSong').onclick = () => TL.fit('song'); $('fitLoop').onclick = () => TL.fit('loop');
  $('copy').onclick = copySel; $('paste').onclick = () => pasteAt(TL.cursor); $('dup').onclick = duplicate; $('del').onclick = deleteSel;
  $('addScene').onclick = addScene; $('loadStems').onclick = loadStems;
  // 点完按钮把焦点还给页面：不然空格 / 字母键会去「按」那个按钮
  document.addEventListener('click', e => { const b = e.target.closest('button'); if (b) b.blur(); });
  document.addEventListener('dragover', e => e.preventDefault());
  document.addEventListener('drop', e => { e.preventDefault(); for (const f of e.dataTransfer.files) if (/^audio\//.test(f.type) || /\.(wav|mp3|flac|ogg|m4a)$/i.test(f.name)) addFileLayer(f); });
}
function togglePlay() {
  if (T.playing) { T.stop(); return; }
  const [ls] = loopRange(S);
  T.start(S.loop.on ? ls : TL.cursor || 0);
}

// ---------- 全局键盘 ----------
function onKeyDown(ev) {
  const tag = ev.target.tagName;
  if ((tag === 'INPUT' && ev.target.type !== 'range' && ev.target.type !== 'checkbox') || tag === 'TEXTAREA' || tag === 'SELECT') { if (ev.code === 'Escape') ev.target.blur(); return; }
  if (!E) return;
  const ctrl = ev.ctrlKey || ev.metaKey;
  if (ctrl) {
    if (ev.code === 'KeyZ') { ev.preventDefault(); return ev.shiftKey ? redo() : undo(); }
    if (ev.code === 'KeyY') { ev.preventDefault(); return redo(); }
    if (ev.code === 'KeyC') { ev.preventDefault(); return copySel(); }
    if (ev.code === 'KeyV') { ev.preventDefault(); return pasteAt(TL.cursor); }
    if (ev.code === 'KeyD') { ev.preventDefault(); return duplicate(); }
    if (ev.code === 'KeyS') { ev.preventDefault(); return save(); }
    return;
  }
  if (ev.altKey) return;
  if (ev.code === 'Space') { ev.preventDefault(); if (!ev.repeat) togglePlay(); return; }
  if (ev.code === 'Backquote' || ev.key === '`' || ev.key === '·') { ev.preventDefault(); if (!ev.repeat) toggleRec(); return; }
  if (ev.code === 'Tab') {
    ev.preventDefault(); const L = sel();
    const order = L.kind === 'voice' ? ['hum', 'keyboard', 'tap'] : L.kind === 'synth' ? ['keyboard', 'tap'] : [L.input];
    L.input = order[(order.indexOf(L.input) + 1) % order.length]; renderKeys(); status(`输入方式：${{ keyboard: '键盘', tap: '拍', hum: '哼唱' }[L.input]}`); return;
  }
  if (/^Digit[1-9]$/.test(ev.code)) {
    const i = +ev.code.slice(5) - 1;
    if (ev.shiftKey) return jumpScene(i);
    if (S.layers[i]) { S.sel = S.layers[i].id; syncAll(); }
    return;
  }
  if (ev.code === 'Digit0') { S.settings.metro = !S.settings.metro; syncHeader(); return; }
  if (ev.code === 'ArrowUp' || ev.code === 'ArrowDown') { ev.preventDefault(); const L = sel(); if (L.kind === 'synth' || L.kind === 'voice') { L.octave = Math.max(0, Math.min(7, L.octave + (ev.code === 'ArrowUp' ? 1 : -1))); renderKeys(); renderParams(); status(`八度 ${L.octave}`); } return; }
  if (variants && (ev.code === 'ArrowLeft' || ev.code === 'ArrowRight')) { ev.preventDefault(); variants.idx = (variants.idx + (ev.code === 'ArrowRight' ? 1 : -1) + variants.options.length) % variants.options.length; previewVariant(); return; }
  if (variants && ev.code === 'Enter') { ev.preventDefault(); return commitVariant(ev.shiftKey ? 'overlay' : 'replace'); }
  if (ev.code === 'Escape') { if (variants) { commitVariant('discard'); } else { TL.sel.clear(); TL.range = null; } return; }
  if (ev.code === 'Delete' || ev.code === 'Backspace') { ev.preventDefault(); return deleteSel(); }
  if (ev.code === 'Minus' || ev.code === 'Equal') { const d = ev.code === 'Minus' ? -S.loop.bars : S.loop.bars; S.loop.start = Math.max(0, Math.min(S.songBars - S.loop.bars, S.loop.start + d)); syncHeader(); status(`循环区：第 ${S.loop.start + 1}–${S.loop.start + S.loop.bars} 小节`); return; }
  if (ev.repeat) { ev.preventDefault(); return; }
  if (noteDown(ev)) ev.preventDefault();
}

// ---------- 主循环 ----------
let lastKeysBar = -1;
function frame() {
  const b = T.playing ? T.beatNow : null;
  const eff = renderMacroBars(b ?? TL.cursor ?? 0);
  if (!T.playing) E.setMacros(eff);
  TL.draw(b);
  drawXY(); drawScope(); drawPitchStrip();
  for (const L of S.layers) { const m = $('mt_' + L.id); if (m) m.style.width = Math.max(0, Math.min(100, (E.meterDb(L.id) + 60) / 60 * 100)) + '%'; }
  const bar = b === null ? -1 : Math.floor(b / S.beatsPerBar);
  if (bar !== lastKeysBar && sel()?.input !== 'hum') { lastKeysBar = bar; renderKeys(); for (const code of held.keys()) litKey(code, true); }
  $('play').textContent = T.playing ? '■ 停' : '▶ 播放';
  requestAnimationFrame(frame);
}

async function boot() {
  E = new Engine({ silent: new URLSearchParams(location.search).has('silent') });
  await E.ctx.resume();
  T = new Transport(E, () => S, {
    events, clips: clipsForPlay, buffer: id => audio.get(id)?.buffer,
    liveMacro: id => xy.live[id],
    countingIn: () => rec.on && T.playing && (T.beatNow ?? 0) < loopRange(S)[0],
    onSegment: (ng, prev) => {
      if (!rec.on) return;
      const [ls, le] = loopRange(S);
      if (prev && prev.b0 <= ls + 1e-9 && prev.b1 >= le - 1e-9) {
        // 一遍循环结束：键盘 / XY 的这一遍可单独撤销；哼唱这一遍收起来
        if (hum.active) humPassEnd(ng ? ng.ctx0 : E.now);
        undoStack.push(JSON.stringify(S));
      }
      if (ng && hum.active && Math.abs(ng.b0 - ls) < 1e-9) humPassStart(ng);
    },
    onStop: () => { if (rec.on) stopRec(); },
  });
  TL = new Timeline($('tl'), {
    getState: () => S, events, peaks: peaksOf, shownLanes: () => shownLanes,
    snapshot, changed: k => { if (k === 'loop') syncHeader(); if (k === 'notes-done' || k === 'curve') renderLaneChecks(); },
    selectLayer: (id, quiet) => { if (S.sel !== id) { S.sel = id; if (!quiet) syncAll(); else { renderLayers(); renderParams(); renderKeys(); } } },
    addNote: n => { S.notes.push({ id: uid(), ...n, raw: { b: n.b, d: n.d } }); },
  });
  bindHeader(); setupXY();
  E.measureTrims().then(t => { for (const L of S.layers) E.updateLayer(L, true); console.log('音色响度修正 dB', t); });
  try {
    const works = await (await api('/api/works')).json();
    server = true;
    $('work').innerHTML = works.map(w => `<option ${w.name === S.work ? 'selected' : ''}>${w.name}</option>`).join('');
    if (!works.some(w => w.name === S.work) && works.length) S.work = works[0].name;
    await loadSense();
    refreshFinal();
  } catch (e) {
    $('work').innerHTML = `<option>${S.work}（离线）</option>`;
    status('没连上 serve.py：能玩，保存会下载到浏览器的下载文件夹');
  }
  document.addEventListener('keydown', onKeyDown);
  document.addEventListener('keyup', ev => { if (E) noteUp(ev); });
  window.addEventListener('blur', () => { for (const code of [...held.keys()]) noteUp({ code }); });
  syncAll();
  if (restoredNote && server) status(restoredNote);
  if (E.silent) { status('静音模式（网址带 ?silent）：一切照常运行，只是不从扬声器出声', true); document.title = '作曲器（静音）'; }
  requestAnimationFrame(frame);
  window.__composer = { get S() { return S; }, E, T, TL, audio, exportTruth: () => exportTruth(S), validate: () => validateTruth(exportTruth(S)), processHumBuffer: (pcm, sr, b0) => injectHum(pcm, sr, b0) };
}
// 测试 / 调试：直接塞一段音频当哼唱（不经过麦克风）
function injectHum(pcm, sr, b0 = 0) {
  mic.sr = sr; mic.chunks = [{ t: 100, x: pcm }];
  const L = sel(); if (L.kind !== 'voice') { S.sel = 'voice'; }
  hum.sessionTakes = [];
  processHumTake(sel().id, b0, 100, 100 + pcm.length / sr);
  openTakeVariants(sel().id);
}

$('gateBtn').onclick = async () => { $('gate').classList.add('hidden'); await boot(); };
