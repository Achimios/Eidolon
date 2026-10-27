// >v<🎛作曲器-页面-锁调键盘 / 拍 / 哼唱 / XY 推子 → 循环叠录 → 变体里挑 → 时间线剪 → 保存 作曲.json + 哼唱 wav。
import { MODES, NOTE_NAMES, ROW_A, ROW_Q, ROW_Z, KEY_LABEL, codeToDegree, degreeToMidi, noteName, parseProgression, chordAtBar, chordTonePcs, chordRoman, chordTones, hzMidi } from './theory.js';
import { Engine, PRESETS, DRUM_SOUNDS, DRUM_LABEL } from './engine.js';
import { Transport, heardCtxTime } from './transport.js';
import { Timeline } from './timeline.js';
import { MACROS, MACRO_BY_ID, effectiveMacros, curveAt, curveWrite, makeState, makeLayer, uid, spb, loopRange, songBeats, layerById, exportTruth, exportSession, validateTruth } from './state.js';
import { analyzeHum, notesToBeats, trimPreroll, yin, decimate, MODE_PARAMS } from './pitch.js';
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
let variants = null;              // { layer, removeIds, options, idx, kind, range, reroll, loop, playIdx }
let shownLanes = new Set(['cutoff', 'drive']);
let clipboard = null;
let sessionGen = 0;               // 打开别的会话 +1：还在分析的哼唱不往新会话里塞
const FILE_KEEP_SEC = 90;         // 拖进来的文件这么长以内随会话存成 wav；再长只记来源（重开会话时没声音）

// ---------- 小工具 ----------
// keep = 一直留着（不然 9 s 后清掉）；之前那条的清除定时器一律取消（不然会把后来要留着的这条清掉）
const status = (msg, keep = false) => { $('status').textContent = msg; clearTimeout(status._t); if (!keep) status._t = setTimeout(() => { $('status').textContent = ''; }, 9000); };
const sel = () => layerById(S, S.sel) || S.layers[0];
const mod = (a, n) => ((a % n) + n) % n;
// 解释用的上下文；loop = 录音时的循环区（吸附落到循环尾巴上 / 之后的绕回循环头，见 interp.js）
const curLoop = () => (S.loop.on ? loopRange(S) : null);
const ctxObj = (loop = curLoop()) => ({ key: S.key, prog: S.prog, beatsPerBar: S.beatsPerBar, tensionAt: b => curveAt(S.curves.tension, b, 0.5), groove, loop });
// 状态写成字符串（撤销点 / 自动存档）：音色变体还在试听时写试听之前的音色（没挑的不算数）
function serial(obj = S) {
  const L = variants?.kind === 'timbre' ? layerById(S, variants.layer) : null;
  if (!L) return JSON.stringify(obj);
  const p = L.params; L.params = variants.orig;
  try { return JSON.stringify(obj); } finally { L.params = p; }
}
function snapshot() { undoStack.push(serial()); if (undoStack.length > 150) undoStack.shift(); redoStack = []; }
// 一遍循环结束的撤销点：和栈顶一样就不推（不然 Ctrl+Z 要按两下才动一次）
function pushUndo() { const s = serial(); if (undoStack[undoStack.length - 1] === s) return; undoStack.push(s); if (undoStack.length > 150) undoStack.shift(); }
// 撤销 / 重做：变体面板一律关掉（试听中的音色先还原）；播放中速度变了 → 从排到的那一拍按新速度接着走。
// 不随撤销走的：作品（存到哪个文件夹）、麦克风延迟（量的是设备）；录音中的循环区（录音锁着它）
function restore(str) {
  closeVariants();
  const cur = S, front = playFront();
  S = JSON.parse(str); S.sel = layerById(S, cur.sel) ? cur.sel : S.layers[0]?.id;
  S.work = cur.work; S._sense = cur._sense;
  S.settings.latencyMs = cur.settings.latencyMs; S.settings.latencyCal = cur.settings.latencyCal;
  if (rec.on) S.loop = { ...cur.loop };
  tempoChanged(cur.bpm, front);
  syncAll();
}
// 和现在一模一样的撤销点跳过（空的一遍、重复推的）：每按一次都真的变
function undo() { const cur = serial(); while (undoStack.length && undoStack[undoStack.length - 1] === cur) undoStack.pop(); if (!undoStack.length) return status('没有可撤销的'); redoStack.push(cur); restore(undoStack.pop()); status('撤销'); }
function redo() { const cur = serial(); while (redoStack.length && redoStack[redoStack.length - 1] === cur) redoStack.pop(); if (!redoStack.length) return; undoStack.push(cur); restore(redoStack.pop()); status('重做'); }
// 自动存档只有一个格子：两个标签页都开着作曲器会互相盖掉 → 后开的那个停写并提示；存不进去（满了 / 被禁用）提示一次
const tabLock = { off: false, warned: false, id: Math.random().toString(36).slice(2) };
function autosave() {
  if (tabLock.off) return;
  try { localStorage.setItem(AUTOSAVE, serial({ ...S, clips: [] })); }
  catch (e) { if (!tabLock.warned) { tabLock.warned = true; status('⚠️浏览器存不了自动存档（满了 / 被禁用）：记得常按保存（Ctrl+S）', true); } }
}
function watchOtherTabs() {
  const BC = window.BroadcastChannel; if (!BC) return;
  const ch = new BC('composer_web');
  ch.onmessage = ev => {
    const m = ev.data || {}; if (m.id === tabLock.id) return;
    if (m.type === 'hello') { ch.postMessage({ type: 'here', id: tabLock.id }); status('另一个窗口也打开了作曲器：那边的自动存档停了，这边照常', true); }
    if (m.type === 'here' && !tabLock.off) { tabLock.off = true; status('⚠️另一个窗口在用作曲器：这个窗口的自动存档停了（免得互相盖掉）。这边的东西记得手动保存', true); }
  };
  ch.postMessage({ type: 'hello', id: tabLock.id });
}
setInterval(autosave, 4000);
// 播放中改了速度：从已经排到的那一拍按新速度重新起（不回跳；已经排进去的音不再排一遍）
const playFront = () => (T && T.playing ? T.nextB : null);
function tempoChanged(oldBpm, front) {
  if (front === null || S.bpm === oldBpm) return;
  if (rec.on && hum.pass) { hum.pass = null; status('录音中速度变了：正在收的这一遍哼唱不要了', true); }
  const [ls, le] = loopRange(S);
  T.start(S.loop.on && front >= le - 1e-9 ? ls : front, { countInEnd: rec.on ? rec.countEnd : undefined });
}
// 新加的层（加一层 / 叠成新层 / 复制成合成层 / 分轨 / 拖进来的文件）：进每一个已有的场景，不然在场景管着的地方一声不响
function joinScenes(id) { for (const sc of S.scenes) if (!sc.layers.includes(id)) sc.layers.push(id); }
// 还没存进硬盘的录音（拖进来的长文件本来就不随会话存，不算）
function hasUnsavedAudio() { return S.clips.some(c => { const a = audio.get(c.audio); return a && !a.file && !(c.input !== 'hum' && a.buffer.duration > FILE_KEEP_SEC); }); }

// 当前要播的事件（变体试听时替换掉被试的那几个音；挑哪一遍时，音符跟着正在放的那一遍录音）
function events() {
  const activeTakes = new Set(clipsNow().filter(c => c.active).map(c => c.id));
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
// 同一层、同一起点的几遍哼唱 = 互相替换（只能有一遍在用）
function sameTakeSlot(c, ch) { return c.layer === ch.layer && c.input === 'hum' && Math.abs(c.b - ch.b) < 1e-6; }
// 挑哪一遍：录音在下一段开头才换（走带每段开头调 clipsForPlay 一次），音符跟着录音一起换（不会第 1 遍的人声配第 3 遍的旋律）
function clipsNow() {
  if (variants && variants.kind === 'takes') {
    const o = variants.options[variants.playIdx ?? variants.idx], ch = o && S.clips.find(c => c.id === o.clipId);
    if (ch) return S.clips.map(c => sameTakeSlot(c, ch) ? { ...c, active: c.id === ch.id } : c);
  }
  return S.clips;
}
function clipsForPlay() {
  if (variants && variants.kind === 'takes') variants.playIdx = variants.idx;
  return clipsNow();
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
const mic = { on: false, node: null, stream: null, mod: false, chunks: [], live: new Float32Array(48000 * 2), liveN: 0, sr: 48000, strip: [], keepSec: 30 };
async function micOn() {
  if (mic.on) return true;
  let stream = null;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 } });
    if (!mic.mod) { await E.ctx.audioWorklet.addModule('js/mic-worklet.js'); mic.mod = true; }
    const src = E.ctx.createMediaStreamSource(stream);
    mic.node = new AudioWorkletNode(E.ctx, 'mic-rec');
    const sink = E.ctx.createGain(); sink.gain.value = 0;
    src.connect(mic.node); mic.node.connect(sink); sink.connect(E.ctx.destination);
    mic.sr = E.ctx.sampleRate;
    mic.node.port.onmessage = ev => {
      const { t, x } = ev.data;
      mic.chunks.push({ t, x });
      // 只留最近 keepSec 秒；正在收的这一遍、还没取走样本的那几遍，从它们的起点留起（一条最长 10 分钟）
      const keepFrom = Math.min(E.now - mic.keepSec, hum.pass ? hum.pass.ctx0 : Infinity, ...hum.waiting.map(w => w.t0)) - 0.5;
      while (mic.chunks.length && mic.chunks[0].t < keepFrom) mic.chunks.shift();
      // 实时音高用的滑动缓冲
      const L = mic.live;
      if (mic.liveN + x.length > L.length) { L.copyWithin(0, x.length); mic.liveN -= x.length; }
      L.set(x, mic.liveN); mic.liveN += x.length;
    };
    for (const tr of stream.getAudioTracks()) tr.addEventListener('ended', micLost);
    mic.stream = stream; mic.on = true;
    $('micBtn').classList.add('on'); $('micBtn').textContent = '麦已开';
    if (S.settings.latencyMs === null) S.settings.latencyMs = Math.round(((E.ctx.baseLatency || 0) + (E.ctx.outputLatency || 0)) * 1000 + 12);
    showLatency();
    return true;
  } catch (err) {
    if (stream) for (const tr of stream.getTracks()) tr.stop();       // 拿到了麦但后面失败：别让麦一直开着
    status('开不了麦克风：' + err.message); return false;
  }
}
// 麦克风拔了 / 被别的程序抢了：说一声（不然之后的哼唱全是空的）
function micLost() {
  if (!mic.on) return;
  mic.on = false;
  try { mic.node.disconnect(); } catch (e) { /* 已断 */ }
  mic.node = null; mic.stream = null;
  $('micBtn').classList.remove('on'); $('micBtn').textContent = '开麦';
  status('⚠️麦克风断开了：之后收到的哼唱是空的。插好以后再点「开麦」', true);
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
    // 每下之后 0.45 s 里找（咔哒间隔 0.5 s；蓝牙耳机往返可到 400 ms 以上）
    const a = Math.round((c - t0 + 0.05) * sr), z = Math.min(x.length, a + Math.round(0.45 * sr));
    for (let i = a; i < z; i++) if (Math.abs(x[i]) > mx * 0.35) { found.push((i - a) / sr * 1000); break; }
  }
  if (found.length < 4 || mx < 0.01) { status(`校准失败：只听到 ${found.length} / 6 下（音量 ${mx.toFixed(3)}）。音量开大，把哼唱时戴的那副耳机贴紧麦克风再试。`); return; }
  found.sort((p, q) => p - q);
  S.settings.latencyMs = Math.round(found[found.length >> 1]); S.settings.latencyCal = true;
  showLatency(); status(`往返延迟 ${S.settings.latencyMs} ms（6 下里的 ${found.length} 下），之后的哼唱都按这个对齐`);
}

// ---------- 哼唱分析放到后台线程（长循环的分析会卡住走带的 120 ms 提前量）；开不了 worker 就在这里同步算 ----------
let humWorker = null;             // null = 还没开；false = 开不了，一律同步
const humJobs = new Map();
let humJobId = 0;
function analyzeAsync(pcm, sr, opts) {
  const sync = () => analyzeHum(pcm, sr, opts);
  if (humWorker === null) {
    try {
      humWorker = new Worker('js/hum-worker.js', { type: 'module' });
      humWorker.onmessage = ev => { const j = humJobs.get(ev.data.id); if (!j) return; humJobs.delete(ev.data.id); if (ev.data.err) j.fallback(); else j.resolve(ev.data.an); };
      // worker 起不来 / 崩了：手上的活改成同步算，之后也不再用它
      humWorker.onerror = ev => { ev?.preventDefault?.(); try { humWorker.terminate(); } catch (e) { /* */ } humWorker = false; const js = [...humJobs.values()]; humJobs.clear(); for (const j of js) j.fallback(); };
    } catch (e) { humWorker = false; }
  }
  if (!humWorker) return Promise.resolve().then(sync);
  return new Promise((resolve, reject) => {
    const id = ++humJobId;
    humJobs.set(id, { resolve, fallback: () => { try { resolve(sync()); } catch (e) { reject(e); } } });
    const cp = pcm.slice();                            // 原样本还要存成录音：发一份拷贝过去
    humWorker.postMessage({ id, pcm: cp, sr, opts }, [cp.buffer]);
  });
}

// ---------- 哼唱录音（开循环：每一遍循环 = 一条 take；关循环：从开始录到停 = 一条，最长 10 分钟）----------
const HUM_MAX_SEC = 600;
const HUM_PREROLL = 0.15;  // 秒：每一遍往前多读这么多，只给分析看第一个音的起音（早唱的强拍不再少了头）
// pending = 排了还没分析完的遍数；waiting = 还没从麦克风缓冲里取样本的那几遍；pickFor = 停了以后等它们都完了再开「挑哪一遍」
const hum = { active: false, pass: null, takeNo: 0, sessionTakes: [], pending: 0, waiting: [], pickFor: null, afterPick: null };
// 录音锁在开始录的那一层（中途选别的层不影响这一遍存到哪）
function humPassStart(seg) { hum.pass = { ctx0: seg.ctx0, b0: seg.b0, b1: seg.b1, layer: rec.layer }; }
// 关着循环录哼唱：从 b0 到停（或歌尾）是一条；10 分钟自动停
function humStartAt(b0) {
  const g = [...T.segs].reverse().find(s => b0 >= s.b0 - 1e-9 && b0 < s.b1 - 1e-9);
  if (!g) return false;
  const ctx0 = g.ctx0 + (b0 - g.b0) * spb(S);
  humPassStart({ ctx0, b0, b1: Math.min(g.b1, b0 + HUM_MAX_SEC / spb(S)) });
  clearTimeout(rec.autoStop);
  rec.autoStop = setTimeout(() => {
    if (!(rec.on && hum.active)) return;
    hum.afterPick = '一条哼唱最长 10 分钟：自动停了录音（这一条存下了）';       // 分析完、挑选面板出来以后再说一遍（不被「第 N 遍」盖掉）
    stopRec(); status(hum.afterPick, true);
  }, Math.max(0, ctx0 - E.now) * 1000 + HUM_MAX_SEC * 1000);
  return true;
}
function humPassEnd(ctxEnd, partial = false) {
  const p = hum.pass; hum.pass = null;
  if (!p) return;
  const L = layerById(S, p.layer); if (!L || L.kind !== 'voice') { status('这一遍没存：录音的哼唱层不见了'); return; }
  const lat = (S.settings.latencyMs || 0) / 1000;
  const bEnd = partial ? Math.min(p.b1, p.b0 + (ctxEnd - p.ctx0) / spb(S)) : p.b1;
  if (bEnd - p.b0 < S.beatsPerBar * 0.5) return;
  const t0 = p.ctx0 + lat, t1 = p.ctx0 + (bEnd - p.b0) * spb(S) + lat;
  // 等麦克风的数据到齐（再晚 lat + 0.15 s）；取样本之前缓冲不裁（见 micOn），分析完之前不开挑选面板（见 maybePick）
  const job = { t0 }, sess = hum.sessionTakes, gen = sessionGen;   // 会话号在这一遍结束时就定下（之后再「打开…」，这一遍不进新会话）
  const isPartial = partial && bEnd < p.b1 - 1e-6;
  hum.waiting.push(job); hum.pending++;
  setTimeout(() => {
    const pr = processHumTake(L.id, p.b0, t0, t1, sess, gen, isPartial);      // 取样本是同步的；分析在后台线程
    hum.waiting = hum.waiting.filter(w => w !== job);
    pr.catch(e => status('这一遍分析失败：' + e.message)).finally(() => { hum.pending--; maybePick(); });
  }, (lat + 0.2) * 1000 + Math.max(0, (t1 - E.now) * 1000));
}
// 停了以后：等每一遍都分析完再开「挑哪一遍」（固定等多久不行：延迟大时最后一遍会来不及）
function maybePick() {
  if (!hum.pickFor || hum.pending > 0 || rec.on) return;
  const id = hum.pickFor; hum.pickFor = null;
  openTakeVariants(id);
  if (hum.afterPick) { status(hum.afterPick, true); hum.afterPick = null; }
}
// sess = 这次录音的那几遍（toggleRec 换新的；晚到的旧遍不会混进新一次录音）
// partial = 停在一遍中间的残段：同一位置已经有整遍时它不当默认（不顶掉整遍，挑的时候也不默认选它）
async function processHumTake(layerId, b0, t0, t1, sess = hum.sessionTakes, gen = sessionGen, partial = false) {
  // 往前多读 HUM_PREROLL：只给分析看起音（存的 wav 和片段照旧从这一遍开头起）
  const sr = mic.sr, preN = Math.round(HUM_PREROLL * sr);
  const full = micExtract(t0 - preN / sr, t1), pcm = full.slice(preN);
  if (gen !== sessionGen) return;
  let peak = 0; for (const v of pcm) peak = Math.max(peak, Math.abs(v));
  const L0 = layerById(S, layerId); if (!L0) return;
  const opts = { mode: L0.params.mode, snap: L0.params.snap }, takeNo = ++hum.takeNo;
  const an = await analyzeAsync(full, sr, opts);       // full 会被转给后台线程（之后不能再读）；pcm 是另一份
  if (gen !== sessionGen) return;                       // 等分析的时候打开了别的会话
  const segs = trimPreroll(an.segs, preN / sr);
  const L = layerById(S, layerId); if (!L) { status(`第 ${takeNo} 遍没存：哼唱层不见了（删了 / 撤销了？）`); return; }
  snapshot();
  const buf = E.ctx.createBuffer(1, pcm.length || 1, sr); buf.copyToChannel(pcm, 0);
  const aid = addAudio(buf, { pcm });
  const clip = { id: uid('C'), layer: layerId, b: b0, dur: pcm.length / sr, audio: aid, input: 'hum', take: takeNo, active: true, src: 'mic', ...(partial ? { partial: true } : {}) };
  const keepFull = partial && S.clips.some(c => sameTakeSlot(c, clip) && c.active && !c.partial);
  if (keepFull) clip.active = false;
  else for (const c of S.clips) if (sameTakeSlot(c, clip)) c.active = false;
  clip.analysis = { mode: an.mode, guess: an.guess.mode, why: an.guess.why, tune_cents: +an.tuneCents.toFixed(1), stats: an.guess.stats, peak: +peak.toFixed(3) };
  S.clips.push(clip);
  const notes = notesToBeats(segs, { t0Beat: b0, spb: spb(S), key: S.key, prog: S.prog, beatsPerBar: S.beatsPerBar, snap: opts.snap });
  notes.forEach((n, i) => {
    const raw = { b: n.b, d: n.d, ...n.raw };
    // 抢拍唱的强拍：放在这一遍开头，raw 记真起音（早了多少）
    if (segs[i].early) { raw.t = -segs[i].early; raw.b = b0 - segs[i].early / spb(S); raw.early_ms = Math.round(segs[i].early * 1000); }
    S.notes.push({ id: uid(), layer: layerId, input: 'hum', take: clip.id, ...n, raw });
  });
  sess.push(clip.id);
  const tn = Math.abs(an.tuneCents) >= 8 ? `，整体${an.tuneCents > 0 ? '偏高' : '偏低'} ${Math.abs(an.tuneCents).toFixed(0)} 音分（吸附前已扣掉）` : '';
  status(`第 ${takeNo} 遍：${MODE_PARAMS[an.mode]?.label ?? an.mode}（${an.guess.why}）· ${notes.length} 个音${tn}${peak < 0.02 ? ' · ⚠️几乎没声音，麦克风选对了吗？' : ''}${peak > 0.98 ? ' · ⚠️削波了，离麦远一点' : ''}`, true);
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
// 录到的拍放到哪（真按下的那一拍留在 raw）：
// 开循环 = 对循环长度取模放进 [循环头, 循环尾)（抢拍落在循环尾巴上，听起来就在下一遍的头前面；吸附时再绕回循环头），
// 预备拍里早于循环头半拍以上的不算；关循环 = 早于开始录的那一拍半拍以上的不算（预备拍里随手按的），不早于第 0 拍
const ANTICIPATE = 0.2;    // 拍：循环头前这么近（150 BPM 约 80 ms，比最后一个 1/16 近）按下并按住过循环头的，算循环头那一拍（见 noteUp）
function placeRecBeat(b) {
  if (S.loop.on) {
    const [ls, le] = loopRange(S);
    if (b < ls - 0.5) return null;
    return ls + mod(b - ls, le - ls);
  }
  if (b < rec.startB - 0.5) return null;
  return Math.max(0, b);
}
// 拍的音高按这个音最后落在哪个小节取和弦：先就近对到 1/16，再按录音的规则放（抢拍 = 下一拍的和弦）
function chordBeat(b) { const q = Math.round(b * 4) / 4; if (!S.loop.on) return Math.max(0, q); const [ls, le] = loopRange(S); return ls + mod(q - ls, le - ls); }
// 录音时一律弹 / 记开始录的那一层（中途换层只换显示，见 selectLayer）
const playLayer = () => (rec.on && layerById(S, rec.layer)) || sel();

function noteDown(ev) {
  const L = playLayer(); if (!L) return false;
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
    held.set(code, { b: playedBeat(ev), t: heard, sound, layer: L.id, vel: v, recId: rec.on ? rec.pass : null });
  } else if (L.kind === 'synth' || L.kind === 'voice') {
    if (L.input === 'tap') {
      col = Math.max(ROW_A.indexOf(code), ROW_Q.indexOf(code), ROW_Z.indexOf(code));
      const oct = L.octave + (ROW_Q.includes(code) ? 1 : ROW_Z.includes(code) ? -1 : 0);
      const b = T.playing ? (playedBeat(ev) ?? 0) : 0;
      pitch = tapPitch(ctxObj(), T.playing ? chordBeat(b) : b, Math.floor(col / 2), oct, S.settings.tapPattern, tapCounter++);
    } else {
      const d = codeToDegree(S.key, code); if (d === null) return true;
      pitch = degreeToMidi(S.key, L.octave, d);
    }
    if (L.kind === 'voice' && L.input === 'hum') return true;
    // 琶音器：播放中按住 = 每个 1/16 自动弹一个和弦音（arpTick 负责），这里不出长音
    if (L.preset === 'arp' && T.playing) { held.set(code, arpHold(ev, L, pitch, vel)); litKey(code, true); return true; }
    const voice = E.note(L, pitch, vel, when);
    held.set(code, { voice, b: playedBeat(ev), t: heard, pitch, layer: L.id, vel, recId: rec.on ? rec.pass : null });
  } else return true;
  litKey(code, true);
  return true;
}
function noteUp(ev) {
  const h = held.get(ev.code); if (!h) return;
  held.delete(ev.code); litKey(ev.code, false);
  const when = E.now;
  h.voice?.release(when);
  if (h.arp) return arpRelease(h, ev);
  recordHeld(h, ev);
}
// 把一个按完的音记进这次录音（松键时；停录音那一刻还按着的也走这里，见 stopRec）
function recordHeld(h, ev) {
  // 只记这一次录音里按下的、录音那一层的
  if (!rec.on || h.recId !== rec.pass || h.b === null) return;
  const L = layerById(S, h.layer); if (!L || L.id !== rec.layer) return;
  let b = placeRecBeat(h.b); if (b === null) return;
  const holdSec = Math.max(0, ctxTimeOfEvent(ev) - h.t);
  const dRaw = Math.max(0.05, holdSec / spb(S));
  const [ls, le] = S.loop.on ? loopRange(S) : [0, Infinity];
  // 抢拍（音符；鼓是一下一下的不算）：循环头前 0.2 拍以内按下、而且按住过了循环头 = 想弹的就是循环头那一拍 → 放在循环头、时长照按住的
  // （不然会被截成循环尾巴上一个 10 ms 的「咔」）。只是轻轻点一下的（没按过循环头）还留在尾巴上。raw 照记真时刻
  if (S.loop.on && !h.sound && le - b < ANTICIPATE && dRaw > le - b) b = ls;
  const d = Math.min(dRaw, le - b);                  // 拖过循环尾巴的截在循环尾巴（不拆成两个音）；raw 留真按住的时长
  const vel = S.settings.velMode === 'hold' ? holdVelocity(holdSec) : h.vel;
  const input = L.input === 'tap' ? 'tap' : 'keyboard';
  // raw.b = 真按下的那一拍（不早于歌的开头：那之前没有时间）
  const n = { id: uid(), layer: L.id, b, d, vel, input, pass: rec.pass, raw: { b: Math.max(h.b, -S.anchor / spb(S)), d: dRaw, vel, hold_s: +holdSec.toFixed(3) } };
  if (h.sound) { n.sound = h.sound; n.d = Math.min(0.25, le - b); } else n.pitch = h.pitch;
  S.notes.push(n);
  rec.added.push(n.id);
}

// 琶音器：按住的键 → 从这个音往上的和弦音，每 1/16 一个；生成的每个音都是真值（录音时照记）。
// 一律按「听到的」时钟：第一个音落在按下的那一格（按晚一点点也算那一格），松开那一刻起的格子不算；
// 格子按段走（循环转回去、预备拍交给循环时接着弹，段尾那一格 = 下一段的第一格）
const ARP_TOL = 0.03;      // 秒：按晚 / 松早这么多还算同一格
function arpHold(ev, L, pitch, vel) {
  const sp = spb(S), th = ctxTimeOfEvent(ev) - (S.settings.kbOffsetMs || 0) / 1000, seg = T.segAt(th);
  const k = Math.ceil((seg.b0 + (th - seg.ctx0) / sp - ARP_TOL / sp) * 4 - 1e-6);
  return { arp: true, pitch, layer: L.id, vel, n: 0, seg, k: Math.max(k, Math.ceil(seg.b0 * 4 - 1e-6)), steps: [], recId: rec.on ? rec.pass : null };
}
function arpTick() {
  if (!T || !T.playing) return;
  const sp = spb(S), horizon = E.now + 0.03;            // 只排马上要响的格子（松手后最多多出这一点）
  for (const h of held.values()) {
    if (!h.arp) continue;
    const L = layerById(S, h.layer); if (!L) continue;
    for (let guard = 0; guard < 32; guard++) {
      let i = T.segs.indexOf(h.seg);
      if (i < 0) {                                      // 走带重新起过：从现在这一格接着
        h.seg = T.segAt(E.now); i = T.segs.indexOf(h.seg);
        h.k = Math.ceil((h.seg.b0 + (E.now - h.seg.ctx0) / sp) * 4 - 1e-6);
      }
      const seg = h.seg, nb = h.k / 4;
      if (nb >= seg.b1 - 1e-9) {                        // 段尾那一格 = 下一段的第一格（下一段还没排出来就下次再来）
        const nx = T.segs[i + 1]; if (!nx) break;
        h.seg = nx; h.k = Math.ceil(nx.b0 * 4 - 1e-6); continue;
      }
      const when = seg.ctx0 + (nb - seg.b0) * sp;
      if (when > horizon) break;
      h.k++;
      const deg = chordAtBar(S.prog, Math.floor(nb / S.beatsPerBar));
      const tones = chordTones(S.key, deg, h.pitch - 0.5, 4);
      const p = tones[h.n % 3] + (h.n % 6 >= 3 ? 12 : 0);
      h.n++;
      const st = { when, voice: E.note(L, p, h.vel, Math.max(when, E.now), sp * 0.2), id: null };
      const rb = rec.on && h.recId === rec.pass && L.id === rec.layer ? placeRecBeat(nb) : null;
      if (rb !== null) {
        const d = S.loop.on ? Math.min(0.2, loopRange(S)[1] - rb) : 0.2;
        const n = { id: uid(), layer: L.id, b: rb, d, pitch: p, vel: h.vel, input: 'keyboard', derived: 'arp', pass: rec.pass, raw: { b: nb, d: 0.2, held_pitch: h.pitch } };
        S.notes.push(n); rec.added.push(n.id); st.id = n.id;
      }
      h.steps.push(st); if (h.steps.length > 64) h.steps.shift();
    }
  }
}
setInterval(arpTick, 8);
// 松开：从松开那一刻起的格子不算 —— 记下的删掉，还没响的停掉（引擎给了 cancel 才停得掉）
function arpRelease(h, ev) {
  const tr = ctxTimeOfEvent(ev) - (S.settings.kbOffsetMs || 0) / 1000 - ARP_TOL, drop = new Set();
  for (const st of h.steps) if (st.when >= tr) { st.voice?.cancel?.(); if (st.id) drop.add(st.id); }
  if (drop.size) { S.notes = S.notes.filter(n => !drop.has(n.id)); rec.added = rec.added.filter(id => !drop.has(id)); }
}

// ---------- 录音开关 ----------
// layer = 录到哪一层（开始录时选的那层，中途换层不变）；startB = 从第几拍开始算录音；countEnd = 预备拍打到哪；loop0 = 开始录时的循环区
const rec = { on: false, pass: null, added: [], layer: null, startB: 0, countEnd: -Infinity, loop0: null, autoStop: null };
async function toggleRec() {
  if (rec.on) return stopRec();
  const L = sel();
  // 哼唱层切到键盘 / 拍 = 像合成层那样录音符，不开麦、不收哼唱
  const isHum = L.kind === 'voice' && L.input === 'hum';
  if (isHum && !(await micOn())) return;
  if (rec.on) return;                                    // 等麦克风权限时又按了一下
  closeVariants();                                       // 先收起变体（试听中的音色还原），再记撤销点
  snapshot();
  rec.on = true; rec.pass = uid('P'); rec.added = []; rec.layer = L.id; rec.loop0 = { ...S.loop };
  hum.active = isHum; hum.sessionTakes = []; hum.pass = null; hum.pickFor = null; hum.afterPick = null;
  $('rec').classList.add('rec-on');
  const [ls] = loopRange(S);
  if (!T.playing) {
    const pre = S.settings.countIn ? S.beatsPerBar : 0;
    rec.startB = S.loop.on ? ls : (TL.cursor || 0);
    rec.countEnd = rec.startB;
    T.start(rec.startB - pre, { countInEnd: rec.startB });
    if (isHum && !S.loop.on) humStartAt(rec.startB);
  } else {
    rec.countEnd = -Infinity;
    const hb = T.beatNow ?? 0;
    if (S.loop.on) { rec.startB = ls; if (isHum) status('从下一遍循环开头开始收音…', true); }
    else if (isHum) {
      // 关着循环、正在放：从下一小节开始收这一条
      rec.startB = Math.ceil(hb / S.beatsPerBar - 1e-6) * S.beatsPerBar;
      status(humStartAt(rec.startB) ? `从第 ${rec.startB / S.beatsPerBar + 1} 小节开始收音…（再按 \` 停）` : '离歌尾太近了，收不到一条', true);
    } else rec.startB = hb;
  }
}
function stopRec() {
  // 停的那一刻还按着的键：按到现在为止记下（声音照响，松键时不再记第二次）
  const nowEv = { timeStamp: performance.now() };
  for (const h of held.values()) if (!h.arp && h.recId === rec.pass) { recordHeld(h, nowEv); h.recId = null; }
  rec.on = false; clearTimeout(rec.autoStop); rec.autoStop = null;
  $('rec').classList.remove('rec-on');
  const wasHum = hum.active;
  if (hum.active && hum.pass) humPassEnd(E.now, true);
  hum.active = false; hum.pass = null;
  // 变体：键盘 / 拍 = 这次录进来的音；哼唱 = 这次的几遍（等每一遍都分析完）
  if (wasHum) { hum.pickFor = rec.layer; maybePick(); }
  else if (rec.added.length) openNoteVariants(rec.layer, rec.added);
}
// 打开别的会话时：录音直接作废（不开变体、晚到的几遍也不收）
function abortRec() {
  rec.on = false; clearTimeout(rec.autoStop); rec.autoStop = null; rec.added = [];
  hum.active = false; hum.pass = null; hum.pickFor = null;
  $('rec').classList.remove('rec-on');
}
// 录音中挪循环区：这一遍会对不上（哼唱整遍丢掉、键盘的音放错地方）→ 不让挪
const LOOP_LOCKED = '录音中不能挪循环区（这一遍会对不上）：先按 ` 停';

// ---------- 变体 ----------
// loop = 录音时的循环区：吸附到循环尾巴上的绕回循环头（再来一组也用同一个）
function openNoteVariants(layerId, ids, reroll = 0, loop = curLoop()) {
  if (variants?.kind === 'timbre') closeVariants();             // 试听中的音色先还原（新面板不能把它顶成「挑中了」）
  const L = layerById(S, layerId); if (!L) return;
  const idSet = new Set(ids);
  const src = S.notes.filter(n => idSet.has(n.id));
  if (!src.length) return;
  const isDrums = L.kind === 'drums';
  const opts = makeVariants(src, ctxObj(loop), { isDrums, reroll });
  const off = meanOffset(src);
  variants = { kind: 'notes', layer: layerId, removeIds: idSet, options: opts.map(o => ({ ...o, notes: o.notes.map(n => ({ ...n, layer: n.layer || layerId })) })), idx: 0, reroll, ids, loop };
  renderVariants();
  if (off.tooLarge) status('你这一段整体偏了将近半格：多半是延迟太大（比如蓝牙耳机）。吸附没扣偏移；先点「校准延迟」，或换有线耳机', true);
  else if (off.consistent) status(`你这一段平均${off.offset < 0 ? '提前' : '拖后'} ${Math.abs(off.offset * spb(S) * 1000).toFixed(0)} ms（很稳，多半是延迟）：吸附的变体里已经扣掉`, true);
}
function openTakeVariants(layerId) {
  if (variants?.kind === 'timbre') closeVariants();
  const takes = hum.sessionTakes.map(id => S.clips.find(c => c.id === id)).filter(Boolean);
  if (!takes.length) return;
  if (takes.length === 1) return openHumInterp(layerId, takes[0]);
  const last = takes.slice(-4);
  // 默认 = 现在激活的那一遍（停在一遍中间的残段不顶掉整遍，也不默认选它）
  const act = last.findIndex(c => c.active), def = act >= 0 ? act : last.length - 1;
  variants = {
    kind: 'takes', layer: layerId, takeIds: last.map(c => c.id), removeIds: new Set(), idx: def, playIdx: def,
    options: last.map(c => ({ label: `第 ${c.take} 遍${c.partial ? '（没唱完）' : ''}`, desc: `${MODE_PARAMS[c.analysis?.mode]?.label ?? '?'} · ${S.notes.filter(n => n.take === c.id).length} 个音`, clipId: c.id, notes: null })),
  };
  renderVariants();
}
// 一条哼唱的几种解释：不吸附 / 吸到调 / 吸到调 + 节拍 60% / 100%
function openHumInterp(layerId, clip, loop = curLoop()) {
  if (variants?.kind === 'timbre') closeVariants();
  const src = S.notes.filter(n => n.take === clip.id);
  if (!src.length) { status('这一遍没认出音符（太轻？太短？）'); return; }
  const mk = (label, desc, f) => ({ label, desc, notes: f(cloneNotes(src)) });
  // 只取半音：音分曲线原来相对吸附后的音，换了音高要跟着挪（音高 + 音分 = 唱的那个音）
  const unsnap = ns => ns.map(n => {
    const p = Math.round(n.raw.pitch_raw), dc = Math.round((p - n.pitch) * 100);
    return { ...n, pitch: p, ...(n.bend ? { bend: { ...n.bend, cents: n.bend.cents.map(c => c - dc) } } : {}) };
  });
  const off = meanOffset(src), ro = off.consistent ? off.offset : 0;
  variants = {
    kind: 'notes', layer: layerId, removeIds: new Set(src.map(n => n.id)), idx: 1, loop,
    options: [
      mk('原样（半音）', '音高只取最近的半音，不管调', unsnap),
      mk('吸到调', '调内 + 强拍偏和弦音（默认）', ns => ns),
      mk('吸到调 + 节拍 60%', '时刻靠向格子六成', ns => quantize(ns, { strength: 0.6, removeOffset: ro, loop })),
      mk('吸到调 + 节拍 100%', '完全对齐 1/16', ns => quantize(ns, { strength: 1, removeOffset: ro, loop })),
    ].map(o => ({ ...o, notes: o.notes.map(n => ({ ...n, layer: layerId, take: clip.id })) })),
  };
  renderVariants();
}
function openTimbreVariants() {
  const L = sel(); if (!L || L.kind !== 'synth') return status('音色变体只给合成层');
  if (variants) closeVariants();                                  // 上一个面板（含试听中的音色）先还原
  variants = { kind: 'timbre', layer: L.id, removeIds: new Set(), idx: 0, orig: { ...L.params }, options: [{ label: '原样', desc: '现在的音色', params: { ...L.params } }, ...TIMBRE_VARIANTS.slice(0, 4).map(v => ({ label: v.label, desc: v.desc, params: v.patch({ ...L.params }) }))] };
  renderVariants();
}
function previewVariant() {
  if (!variants) return;
  if (variants.kind === 'timbre') { const L = layerById(S, variants.layer); L.params = { ...variants.options[variants.idx].params }; E.updateLayer(L, true); renderParams(); }
  // 挑哪一遍：没在放 = 马上换；在放 = 下一段开头录音和音符一起换（clipsForPlay）
  if (variants.kind === 'takes') { if (!T.playing) variants.playIdx = variants.idx; else if (variants.playIdx !== variants.idx) status('下一遍循环开头换成这一遍（人声和音符一起换）'); }
  renderVariants(); TL && TL.draw(T.beatNow);
}
// 变体里挑出来的音：规则生成的（鬼音 / 八度叠 / 三度和声 / 琶音……）来源一律记成 pick、不编 raw；其余没有 raw 的补上
function committed(n, layer) {
  const c = { ...n, id: uid(), layer };
  delete c._preview;
  if (c.derived) c.input = 'pick';
  else { c.input = c.input || 'pick'; if (!c.raw) c.raw = { b: n.b, d: n.d }; }
  return c;
}
function commitVariant(action) {
  if (!variants) return;
  const v = variants, o = v.options[v.idx], L = layerById(S, v.layer);
  // 撤销过了的变体（那一层 / 那一遍 / 那些音已经不在了）不能再用：关掉，不记 pick
  if (!L) { closeVariants(false); return status('这一层已经不在了，变体关掉了'); }
  if (v.kind === 'takes' && !S.clips.some(c => c.id === o.clipId)) { closeVariants(false); return status('这一遍已经撤销了，变体关掉了'); }
  if (v.kind === 'notes' && v.removeIds.size && ![...v.removeIds].some(id => S.notes.some(n => n.id === id))) { closeVariants(false); return status('这些音已经撤销了，变体关掉了'); }
  // 叠成新层只对音符变体有意义；挑哪一遍 / 音色按「用这个」处理
  if (action === 'overlay' && v.kind !== 'notes') { action = 'replace'; status('这种变体不能叠成新层，按「用这个」处理'); }
  // 音色：撤销点要是试听之前的音色（试听改的是 L.params）；用这个 = 试听时的音色 + 试听时拧过的旋钮
  const chosen = v.kind === 'timbre' ? { ...L.params } : null;
  if (v.kind === 'timbre') L.params = { ...v.orig };
  snapshot();
  const [ls] = loopRange(S);
  S.picks.push({ at: new Date().toISOString(), t: +(S.anchor + ls * spb(S)).toFixed(3), layer: v.layer, kind: v.kind, options: v.options.map(x => x.label), chose: o.label, action, input: 'pick' });
  if (v.kind === 'timbre') {
    L.params = action === 'discard' ? { ...v.orig } : chosen;
    E.updateLayer(L, true);
  } else if (v.kind === 'takes') {
    // 同一层同一起点的哼唱只留挑中的这一遍（晚到的那遍也算在里面）
    if (action !== 'discard') { const ch = S.clips.find(c => c.id === o.clipId); for (const c of S.clips) if (sameTakeSlot(c, ch)) c.active = c.id === ch.id; }
  } else if (action === 'replace') {
    S.notes = S.notes.filter(n => !v.removeIds.has(n.id));
    for (const n of o.notes) S.notes.push(committed(n, n.layer || v.layer));
  } else if (action === 'overlay') {
    const nl = makeLayer(L.kind === 'voice' ? 'synth' : L.kind, L.kind === 'voice' ? 'lead' : L.preset, `${L.name}·${o.label}`, S.layers.length);
    if (L.kind !== 'voice') nl.params = { ...L.params }; nl.octave = L.octave;
    S.layers.push(nl); joinScenes(nl.id);
    for (const n of o.notes) { const c = committed(n, nl.id); delete c.take; S.notes.push(c); }
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
// runV / runN：每个旋钮上一次写的值、连着写了几次这个值（按住不动时只留一段的两头）
const xy = { down: false, x: 0.5, y: 0.5, lastB: null, live: {}, runV: {}, runN: {} };
function xyTargets() { return [$('xSel').value, $('ySel').value]; }
function xyPos(e) {
  const r = $('xy').getBoundingClientRect();
  xy.x = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
  xy.y = Math.max(0, Math.min(1, 1 - (e.clientY - r.top) / r.height));
  const [mx, my] = xyTargets();
  xy.live = { [mx]: xy.x, [my]: xy.y };
}
// 录音中把手上的值写进曲线：按住不动也一直写（每帧，见 frame）—— 听到的是它，真值也得是它
function xyWrite() {
  if (!(rec.on && T.playing)) return;
  const b = T.beatNow; if (b === null) return;
  const [ls, le] = loopRange(S);
  if (b < (S.loop.on ? ls : rec.startB)) { xy.lastB = null; return; }     // 预备拍里不写（声音照样跟着手）
  const rb = +b.toFixed(4), from = xy.lastB ?? rb, wrap = S.loop.on && rb < from - 0.5;
  for (const [id, v0] of Object.entries(xy.live)) {
    const v = +v0.toFixed(4);
    let pts = S.curves[id] || [];
    if (wrap) { pts = curveWrite(pts, from, le - 1e-3, v); pts = curveWrite(pts, ls, rb, v); xy.runN[id] = 0; }
    else {
      // 同一个值连着写：把上一个点挪到现在（不每帧加一个点）
      if (xy.runV[id] === v && xy.runN[id] >= 2 && rb > from) pts = pts.filter(p => Math.abs(p[0] - from) > 1e-6);
      pts = curveWrite(pts, from, rb, v);
    }
    xy.runN[id] = xy.runV[id] === v ? (xy.runN[id] || 0) + 1 : 1; xy.runV[id] = v;
    S.curves[id] = pts;
    (S.curvesInput ||= {})[id] = 'xy';
    shownLanes.add(id);
  }
  xy.lastB = rb;
}
function setupXY() {
  const cv = $('xy');
  cv.addEventListener('pointerdown', e => { xy.down = true; xy.lastB = null; xy.runV = {}; xy.runN = {}; try { cv.setPointerCapture(e.pointerId); } catch (err) { /* 合成事件没有指针 */ } if (rec.on) snapshot(); xyPos(e); xyWrite(); });
  cv.addEventListener('pointermove', e => { if (xy.down) { xyPos(e); xyWrite(); } });
  // 松手：先把按住到现在的这一段写完
  const up = () => { if (xy.down) xyWrite(); xy.down = false; xy.live = {}; xy.lastB = null; renderLaneChecks(); };
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
      else return selectLayer(L.id);
      syncAll();
    };
    d.ondblclick = ev => { if (ev.target.dataset.a) return; const nm = prompt('层的名字', L.name); if (nm) { snapshot(); L.name = nm; syncAll(); } };
    box.appendChild(d);
  });
  // 层列表自己会滚（定高布局）：数字键选的那层要露出来
  box.querySelector('.layer.sel')?.scrollIntoView?.({ block: 'nearest' });
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
// 重新识别：只换每一遍自己的音（take 指着它的）；贴出来 / 复制出去的音不挂在哪一遍上，不动。分析在后台线程，算完再一起换
async function reanalyze(L0) {
  const clips = S.clips.filter(c => c.layer === L0.id && c.input === 'hum' && audio.get(c.audio));
  if (!clips.length) return status('这一层还没有录音');
  const opts = { mode: L0.params.mode, snap: L0.params.snap }, gen = sessionGen, layerId = L0.id;
  status(`重新识别 ${clips.length} 条…`, true);
  const res = [];
  for (const c of clips) { const a = audio.get(c.audio); res.push(await analyzeAsync(a.pcm || a.buffer.getChannelData(0), a.sr, opts)); }
  if (gen !== sessionGen || !layerById(S, layerId)) return;
  snapshot();
  let n0 = 0;
  clips.forEach((c0, i) => {
    const c = S.clips.find(x => x.id === c0.id); if (!c) return;          // 等的时候撤销掉了
    const an = res[i];
    S.notes = S.notes.filter(n => n.take !== c.id);
    const notes = notesToBeats(an.segs, { t0Beat: c.b, spb: spb(S), key: S.key, prog: S.prog, beatsPerBar: S.beatsPerBar, snap: opts.snap });
    for (const n of notes) S.notes.push({ id: uid(), layer: layerId, input: 'hum', take: c.id, ...n, raw: { b: n.b, d: n.d, ...n.raw } });
    c.analysis = { ...c.analysis, mode: an.mode, guess: an.guess.mode, why: an.guess.why, tune_cents: +an.tuneCents.toFixed(1) };
    n0++;
  });
  status(`重新识别了 ${n0} 条`); syncAll();
}
function copyHumToSynth(L) {
  const act = new Set(S.clips.filter(c => c.active && c.layer === L.id).map(c => c.id));
  const src = S.notes.filter(n => n.layer === L.id && (!n.take || act.has(n.take)));
  if (!src.length) return status('没有音符可复制');
  snapshot();
  const nl = makeLayer('synth', 'lead', `${L.name}→合成`, S.layers.length);
  S.layers.push(nl); joinScenes(nl.id);
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
  const deg = chordAtBar(S.prog, bar), pcs = chordTonePcs(S.key, deg);
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
// 选层：录音中只换显示，录的还是开始录的那一层（D3）
function selectLayer(id, quiet = false) {
  if (S.sel !== id) {
    S.sel = id;
    if (rec.on && id !== rec.layer) status(`录音中：还在录「${layerById(S, rec.layer)?.name ?? '?'}」（键盘弹的也是它；换层只换显示，停了再录别的层）`);
  }
  if (quiet) { renderLayers(); renderParams(); renderKeys(); } else syncAll();
}

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
  if (rec.on) return status(LOOP_LOCKED);
  const next = S.scenes[i + 1];
  S.loop.start = sc.bar; S.loop.bars = Math.max(1, Math.min(16, (next ? next.bar : S.songBars) - sc.bar));
  syncHeader(); status(`循环区 → ${sc.name}（第 ${sc.bar + 1}–${sc.bar + S.loop.bars} 小节）`);
}

// ---------- 时间线编辑：删 / 复制 / 粘贴 ----------
function deleteSel() {
  // 删音以后开着的变体面板就过期了（它会把删掉的音加回来）→ 关掉
  if ((TL.sel.size || TL.range) && variants && variants.kind !== 'timbre') closeVariants(false);
  if (TL.sel.size) { snapshot(); S.notes = S.notes.filter(n => !TL.sel.has(n.id)); TL.sel.clear(); status('删了'); return syncAll(); }
  if (TL.range) {
    snapshot(); const [a, z] = TL.range;
    // 只删看得见的（没在用的那几遍哼唱的音不动，换回那一遍时它还在；挑哪一遍试听时按正在试听的那一遍算）
    const act = new Set(clipsNow().filter(c => c.active).map(c => c.id));
    S.notes = S.notes.filter(n => n.b < a || n.b >= z || (n.take && !act.has(n.take)));
    for (const k of Object.keys(S.curves)) S.curves[k] = S.curves[k].filter(p => p[0] < a || p[0] > z);
    status(`删了第 ${(a / S.beatsPerBar + 1).toFixed(2)}–${(z / S.beatsPerBar + 1).toFixed(2)} 小节里的音符和曲线（录音 / 分轨的声音不动）`); syncAll();
  }
}
// 复制出去的音：拍和原始拍位都相对复制起点；不再挂在哪一遍哼唱上（贴出来 = 普通音符：换一遍 / 重新识别都不动它）
function relNote(n, a) {
  const c = { ...n, b: n.b - a, ...(n.raw ? { raw: { ...n.raw, b: (n.raw.b ?? n.b) - a } } : {}) };
  delete c.take; delete c._preview;
  return c;
}
// 复制成了返回 true；什么都没选 / 选的已经不在了 → 只提示，剪贴板不动
function copySel() {
  if (TL.range) {
    const [a, z] = TL.range;
    const notes = events().filter(n => !n._preview && n.b >= a && n.b < z).map(n => relNote(n, a));
    const curves = Object.fromEntries(Object.entries(S.curves).map(([k, p]) => [k, p.filter(q => q[0] >= a && q[0] <= z).map(q => [q[0] - a, q[1]])]));
    if (!notes.length && !Object.values(curves).some(p => p.length)) { status('这一段里没有音符也没有曲线'); return false; }
    clipboard = { span: z - a, notes, curves };
  } else {
    const ns = S.notes.filter(n => TL.sel.has(n.id));
    if (!ns.length) { status('先选：拖标尺下条选一段，或框选音符'); return false; }
    const a = Math.min(...ns.map(n => n.b));
    clipboard = { span: Math.max(...ns.map(n => n.b + (n.d || 0.25))) - a, notes: ns.map(n => relNote(n, a)), curves: {} };
  }
  status(`复制了 ${clipboard.notes.length} 个音${Object.values(clipboard.curves).some(p => p.length) ? ' + 曲线' : ''}`);
  return true;
}
function pasteAt(b) {
  if (!clipboard) return status('剪贴板是空的');
  if (!Number.isFinite(b)) return status('贴到哪？先在标尺下半条点一下放光标');
  snapshot();
  const ids = [];
  for (const n of clipboard.notes) { const id = uid(); ids.push(id); S.notes.push({ ...n, id, b: n.b + b, ...(n.raw ? { raw: { ...n.raw, b: n.raw.b + b } } : {}) }); }
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
  if (!copySel()) return;                    // 没选东西：不拿旧剪贴板去贴（以前会贴到第 Infinity 拍）
  const start = TL.range ? TL.range[0] : Math.min(...S.notes.filter(n => TL.sel.has(n.id)).map(n => n.b));
  const span = TL.range ? clipboard.span : Math.ceil(clipboard.span / S.beatsPerBar) * S.beatsPerBar;
  pasteAt(start + span);
  if (TL.range) TL.range = [TL.range[0] + span, TL.range[1] + span];
}

// ---------- 原曲分轨 / 拖进来的文件 ----------
// 连不上服务器（不是服务器回了错）的错误带 offline 标记：保存时改成下载
const netFetch = (path, opt) => fetch(path, opt).catch(e => { const err = new Error(e?.message || String(e)); err.offline = true; throw err; });
async function api(path, opt) { const r = await netFetch(path, opt); if (!r.ok) throw new Error(`${r.status} ${await r.text()}`); return r; }
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
  let ok = 0;
  for (const nm of names) {
    let buf;
    try {
      const r = await api(`/api/stem?work=${encodeURIComponent(S.work)}&name=${encodeURIComponent(nm)}&t0=${t0}&t1=${t1}`);
      buf = await E.ctx.decodeAudioData(await r.arrayBuffer());
    } catch (e) { msgs.push(`⚠️${nm} 没载入：${e.message}`); continue; }
    const aid = addAudio(buf, { file: `分轨/${nm}` });
    const L = makeLayer('stem', 'stem', '原曲·' + nm.replace(/\.wav$/i, ''), S.layers.length);
    S.layers.push(L); joinScenes(L.id);
    S.clips.push({ id: uid('C'), layer: L.id, b: 0, dur: buf.duration, audio: aid, input: 'stem', active: true, fileOffset: t0, src: `分轨/${nm}@${t0.toFixed(3)}-${t1.toFixed(3)}` });
    if (/drum/i.test(nm)) msgs.push(buildGroove(buf));
    ok++;
  }
  msgs.push(checkGridDrift());
  status([`载入了 ${ok} / ${names.length} 条分轨`, ...msgs.filter(Boolean)].join('\n'), true);
  syncAll();
}
// 起点 / 速度 / 全曲长度变了：原曲分轨按新窗口重新切（不然分轨还放旧的那一段，和格子、真值都差一截）。
// 新切的用新的音频号：撤销回去的旧片段还指着旧的那段声音
let stemWant = '';
async function reloadStems() {
  const stems = S.clips.filter(c => c.input === 'stem' && c.src && c.src.includes('@'));
  if (!stems.length) return;
  if (!server) return status('⚠️起点 / 速度变了，但没连上 serve.py：分轨没重新切，和格子对不上了', true);
  const t0 = S.anchor, t1 = S.anchor + songBeats(S) * spb(S), want = `${S.work}|${t0}|${t1}`, gen = sessionGen, S0 = S;
  stemWant = want;
  const got = [];
  for (const c of stems) {
    const nm = c.src.split('@')[0].replace(/^分轨\//, '');
    try {
      const r = await api(`/api/stem?work=${encodeURIComponent(S.work)}&name=${encodeURIComponent(nm)}&t0=${t0}&t1=${t1}`);
      got.push({ id: c.id, nm, buf: await E.ctx.decodeAudioData(await r.arrayBuffer()) });
    } catch (e) { return status(`⚠️分轨 ${nm} 没重新切成（${e.message}）：分轨和格子对不上了，改回原来的起点或重新「载入原曲分轨」`, true); }
  }
  if (gen !== sessionGen || stemWant !== want) return;      // 等的时候又改了 / 换了会话：以最后那次为准
  // 等的时候撤销 / 重做了（S 换了一个）：起点 / 速度可能已经回去了 → 按现在的状态再切一次，不把旧的切片贴上去
  if (S !== S0) { if (`${S.work}|${S.anchor}|${S.anchor + songBeats(S) * spb(S)}` !== want) reloadStems(); return; }
  const msgs = [];
  for (const g of got) {
    const c = S.clips.find(x => x.id === g.id); if (!c) continue;
    c.audio = addAudio(g.buf, { file: `分轨/${g.nm}` }); c.dur = g.buf.duration; c.fileOffset = t0; c.src = `分轨/${g.nm}@${t0.toFixed(3)}-${t1.toFixed(3)}`;
    if (/drum/i.test(g.nm)) msgs.push(buildGroove(g.buf));
  }
  status([`分轨按新的起点重新切了（${t0.toFixed(2)}–${t1.toFixed(2)} s，${got.length} 条；正在放的话下一遍生效）`, ...msgs].join('\n'), true);
  syncAll();
}
// 改速度：哼唱的音符跟着它那一遍录音走（录音是秒，音符是拍）——按新速度换算回拍，跟唱的合成器不跑偏
function retimeHum(oldBpm) {
  if (oldBpm === S.bpm) return;
  const k = (60 / oldBpm) / spb(S), clips = new Map(S.clips.map(c => [c.id, c]));
  for (const n of S.notes) {
    const c = n.take && clips.get(n.take); if (!c) continue;
    n.b = c.b + (n.b - c.b) * k; n.d *= k;
    if (n.raw) { if (Number.isFinite(n.raw.b)) n.raw.b = c.b + (n.raw.b - c.b) * k; if (Number.isFinite(n.raw.d)) n.raw.d *= k; }
  }
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
    S.layers.push(L); joinScenes(L.id);
    const [ls] = loopRange(S);
    // fileOffset 0：存下来的就是这个文件本身，从头放（文件名里碰巧有「@128-」也不当切片号）
    S.clips.push({ id: uid('C'), layer: L.id, b: ls, dur: buf.duration, audio: aid, input: 'file', active: true, fileOffset: 0, src: file.name });
    const gm = /drum|鼓/i.test(file.name) ? '\n' + buildGroove(buf) : '';
    const long = buf.duration > FILE_KEEP_SEC ? `\n⚠️超过 ${FILE_KEEP_SEC} s：保存时不会把它存进会话（重新打开时这层没声音）。长伴奏请放进作品的 分轨/ 再「载入原曲分轨」` : '';
    status(`拖进来：${file.name}（${buf.duration.toFixed(1)} s，${buf.numberOfChannels} 声道，从循环起点开始放）${gm}${long}`, !!(gm || long));
    syncAll();
  } catch (e) { status('读不了这个文件：' + e.message); }
}

// ---------- 保存 / 打开 ----------
function f32ToB64(f) { const u = new Uint8Array(f.buffer, f.byteOffset, f.byteLength); let s = ''; for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode.apply(null, u.subarray(i, i + 0x8000)); return btoa(s); }
// 要存的声道：录音 = 单声道 pcm；拖进来的文件按它原来的声道（立体声不只存左声道）
const audioChannels = a => a.pcm ? [a.pcm] : Array.from({ length: a.buffer.numberOfChannels }, (_, i) => a.buffer.getChannelData(i));
// b64 = 第 1 个声道（旧的 serve.py 只认它）；extra_channels = 第 2 个起的声道，serve.py 叠成多声道 wav
function wavPayload(name, a) { const ch = audioChannels(a); return { name, sr: a.sr, b64: f32ToB64(ch[0]), ...(ch.length > 1 ? { extra_channels: ch.slice(1).map(f32ToB64) } : {}) }; }
async function save() {
  // 刚停的哼唱还在分析（最后一遍还没进来）→ 等它，最多 8 秒；音色还在试听 → 先还原（没挑的不存）
  for (let i = 0; i < 80 && hum.pending > 0; i++) { if (!i) status('最后一遍哼唱还在分析，等它进来再存…', true); await new Promise(r => setTimeout(r, 100)); }
  if (hum.pending > 0) return status('最后一遍哼唱还没分析完，没存：过一会儿再按保存');
  if (variants?.kind === 'timbre') { closeVariants(); status('音色变体还在试听：先还原了再存（要用它请先按 Enter）', true); }
  const errs = validateTruth(exportTruth(S));
  if (errs.length) { status('自检没过，没存：' + errs.join('；')); return; }
  if (!server) return downloadAll();
  try {
    // 版本号由服务器给；文件已存在服务器会拒（409），再要下一个号 —— 绝不覆盖
    for (let attempt = 0; attempt < 3; attempt++) {
      const { ver } = await (await api(`/api/next?work=${encodeURIComponent(S.work)}`)).json();
      // 要存的音频：麦克风录的 + 拖进来的短文件（原曲分轨只记来源，不重复存；已经存过的沿用旧文件）
      const files = {}, toSave = [], skipped = [];
      for (const c of S.clips) {
        const a = audio.get(c.audio); if (!a) continue;
        if (a.file) { files[c.audio] = a.file; continue; }
        if (c.input !== 'hum' && a.buffer.duration > FILE_KEEP_SEC) { skipped.push(c.src || c.id); continue; }
        const name = c.input === 'hum' ? `哼唱_${ver}_第${c.take}遍.wav` : `素材_${ver}_${c.id}.wav`;
        files[c.audio] = `作曲/${name}`;          // 路径一律相对作品文件夹（分轨是 分轨/xxx.wav）
        toSave.push(wavPayload(name, a));
      }
      // 会话里的片段也记上文件名（重新打开以后 exportTruth 和存下来的那份一样）
      const truth = exportTruth(S, { audioFiles: files }), session = exportSession({ ...S, clips: S.clips.map(c => files[c.audio] ? { ...c, file: files[c.audio] } : c) }, files);
      const r = await netFetch(`/api/save?work=${encodeURIComponent(S.work)}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ver, truth, session, audio: toSave }) });
      if (r.status === 409) continue;
      if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
      const j = await r.json();
      for (const [id, fn] of Object.entries(files)) { const a = audio.get(id); if (a) a.file = fn; }
      for (const c of S.clips) if (files[c.audio]) c.file = files[c.audio];
      const sk = skipped.length ? `\n⚠️没存声音：${skipped.join('、')}（拖进来的超过 ${FILE_KEEP_SEC} s 只记来源，重新打开会话时这层没声音；长伴奏请放进 分轨/）` : '';
      status(`存好了（${ver}）：${j.dir}\n${j.written.join(' · ')}${sk}`, true);
      lastSaved = { truth: j.truth, snap: JSON.stringify(S) };
      return j.truth;
    }
    status('保存失败：连续三次撞上已有的版本号');
  } catch (e) {
    // 连不上（serve.py 的窗口关了 / 崩了）：别丢，改成下载
    if (e.offline) { downloadAll(); status('⚠️连不上 serve.py（窗口关了？）：这次下载到浏览器的下载文件夹了。重新打开 serve.py 后再存一次就回到作品文件夹', true); return; }
    status('保存失败：' + e.message);
  }
}
// 服务器那边用同一个 exportTruth 的结果（页面先算好），这里给离线用
function downloadAll() {
  const dl = (name, blob) => { const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 3000); };
  const files = {};
  for (const c of S.clips) { const a = audio.get(c.audio); if (!a || a.file) continue; const nm = `哼唱_${c.take ?? c.id}.wav`; files[c.audio] = nm; dl(nm, wavBlob(audioChannels(a), a.sr)); }
  dl('作曲.json', new Blob([JSON.stringify(exportTruth(S, { audioFiles: files }), null, 1)], { type: 'application/json' }));
  dl('作曲_会话.json', new Blob([JSON.stringify(exportSession(S, files))], { type: 'application/json' }));
  status('没连上 serve.py：下载到浏览器的下载文件夹了');
}
// 16 bit wav；chans = 每个声道一条（交错写）
function wavBlob(chans, sr) {
  const C = chans.length, n = chans[0].length, bytes = n * C * 2, b = new ArrayBuffer(44 + bytes), v = new DataView(b);
  const w = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  w(0, 'RIFF'); v.setUint32(4, 36 + bytes, true); w(8, 'WAVEfmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, C, true);
  v.setUint32(24, sr, true); v.setUint32(28, sr * 2 * C, true); v.setUint16(32, 2 * C, true); v.setUint16(34, 16, true); w(36, 'data'); v.setUint32(40, bytes, true);
  for (let i = 0; i < n; i++) for (let k = 0; k < C; k++) v.setInt16(44 + (i * C + k) * 2, Math.max(-1, Math.min(1, chans[k][i])) * 32767, true);
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
  let list;
  try { list = await (await api(`/api/sessions?work=${encodeURIComponent(S.work)}`)).json(); } catch (e) { return status('读不到会话列表：' + e.message); }
  const box = $('sessList'); box.innerHTML = list.length ? '' : '<p class="dim">这个作品还没有存过会话</p>';
  for (const s of list) { const b = document.createElement('button'); b.textContent = `${s.name}  ·  ${s.mtime}`; b.onclick = () => loadSession(s.name); box.appendChild(b); }
  $('sessBox').classList.remove('hidden');
}
// 换作品时把录音 / 分轨带过去：存过的文件名都在上一个作品的文件夹里 → 作废，下次保存重新写进新作品；
// 原曲分轨也是上一个作品的 → 改成普通素材（存的就是现在这一段，从头放）。撤销记录清空（旧的撤销点还指着上一个作品的文件）
function moveAudioToWork() {
  for (const a of audio.values()) a.file = null;
  for (const c of S.clips) { delete c.file; if (c.input === 'stem') { c.input = 'file'; c.fileOffset = 0; } }
  undoStack = []; redoStack = [];
  const long = S.clips.filter(c => { const a = audio.get(c.audio); return c.input !== 'hum' && a && a.buffer.duration > FILE_KEEP_SEC; }).length;
  return long ? `⚠️有 ${long} 段超过 ${FILE_KEEP_SEC} s 的分轨 / 文件存不进新作品（重新打开会话时没声音）：要用的话把它放进新作品的 分轨/ 再载入` : '';
}
async function loadSession(name) {
  $('sessBox').classList.add('hidden');
  const unsavedEdits = (S.notes.length || S.clips.length) && !(lastSaved && lastSaved.snap === serial());
  if ((hasUnsavedAudio() || hum.pending || unsavedEdits) && !confirm('有还没存的录音 / 改动：打开别的会话会把它们丢掉（撤销也找不回来）。\n确定 = 丢掉、打开；取消 = 先回去存（Ctrl+S）')) return;
  const keep = { latencyMs: S.settings.latencyMs, latencyCal: S.settings.latencyCal };   // 麦克风延迟量的是这台设备，不跟会话走
  try {
    const j = await (await api(`/api/file?work=${encodeURIComponent(S.work)}&name=${encodeURIComponent(name)}`)).json();
    T.stop(true); abortRec(); closeVariants();
    sessionGen++;
    const { audioFiles = {}, ...st } = j;
    S = { ...makeState(), ...st };
    // 撤销记录清空：旧会话的录音音频下面就清掉了，撤回去只会剩没声音的空壳（存的时候还会丢）
    undoStack = []; redoStack = []; lastSaved = null; groove = null;
    audio.clear(); peaksCache.clear(); hum.sessionTakes = [];
    for (const c of S.clips) {
      const fn = audioFiles[c.audio] || c.file;
      if (fn) c.file = fn;                    // 片段上也记着文件名：exportTruth 不靠 audioFiles 也对
      try {
        let r;
        if (c.input === 'stem' && c.src && c.src.includes('@')) {
          const [p, rng] = c.src.split('@'), [a, z] = rng.split('-');
          r = await api(`/api/stem?work=${encodeURIComponent(S.work)}&name=${encodeURIComponent(p.replace('分轨/', ''))}&t0=${a}&t1=${z}`);
        } else if (fn) r = await api(`/api/file?work=${encodeURIComponent(S.work)}&name=${encodeURIComponent(fn.replace(/^作曲\//, ''))}`);
        else continue;
        const buf = await E.ctx.decodeAudioData(await r.arrayBuffer());
        audio.set(c.audio, { buffer: buf, pcm: c.input === 'hum' ? buf.getChannelData(0) : null, sr: buf.sampleRate, file: fn || null });
        if (c.input === 'stem' && /drum/i.test(c.src || '')) buildGroove(buf);
      } catch (e) { status(`有一条音频没读到：${fn || c.src}`); }
    }
    hum.takeNo = Math.max(hum.takeNo, ...S.clips.map(c => c.take || 0));
    Object.assign(S.settings, keep);
    // 旧版存的会话循环区可能跑到歌外面：拉回歌里
    S.loop.bars = Math.max(1, Math.min(S.loop.bars, S.songBars)); S.loop.start = Math.max(0, Math.min(S.loop.start, S.songBars - S.loop.bars));
    await loadSense();                         // 原曲重拍 / 速度的缓存（会话里不存，见 exportSession）
    syncAll();
    // 打开的这一版就是存好的那一版：马上「定稿」不用再存一份
    const m = /^(作曲_v\d+)_会话\.json$/.exec(name);
    lastSaved = m ? { truth: `${m[1]}.json`, snap: serial() } : null;
    status(`打开了 ${name}（撤销记录从这里重新开始）`);
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
  // 改速度：先记下排到哪一拍（按旧速度），哼唱音符跟着录音换算，播放中从那一拍按新速度接着走
  const setBpm = bpm => { const front = playFront(), old = S.bpm; S.bpm = bpm; retimeHum(old); tempoChanged(old, front); return old !== S.bpm; };
  $('bpm').onchange = () => {
    if (rec.on) { $('bpm').value = S.bpm; return status('录音中不能改速度：先按 ` 停'); }
    // 开着的变体面板是按旧速度算的（哼唱的拍位）→ 先关掉，免得 Enter 把音放回旧的拍上
    closeVariants();
    snapshot(); const ch = setBpm(Math.max(40, Math.min(240, +$('bpm').value || 120))); syncAll(); if (ch) reloadStems();
    const w = checkGridDrift(); if (w) status(w, true);
  };
  $('root').onchange = () => { snapshot(); S.key.root = +$('root').value; syncAll(); };
  $('mode').onchange = () => { snapshot(); S.key.mode = $('mode').value; syncAll(); };
  $('prog').onchange = () => { snapshot(); S.prog = parseProgression($('prog').value); syncAll(); };
  // 循环区永远在歌里面（全曲改短了也一样）；没在录音时播放中的循环立刻挪过去
  const clampLoop = () => {
    const before = JSON.stringify(S.loop);
    S.loop.bars = Math.max(1, Math.min(S.loop.bars, S.songBars));
    S.loop.start = Math.max(0, Math.min(S.loop.start, S.songBars - S.loop.bars));
    if (JSON.stringify(S.loop) !== before && T.playing && S.loop.on && !rec.on) T.start(loopRange(S)[0]);
  };
  $('loopBars').onchange = () => { if (rec.on) { syncHeader(); return status(LOOP_LOCKED); } S.loop.bars = +$('loopBars').value; clampLoop(); syncAll(); };
  $('songBars').onchange = () => {
    if (rec.on) { syncHeader(); return status('录音中不能改全曲长度：先按 ` 停'); }
    snapshot(); S.songBars = Math.max(1, Math.min(256, +$('songBars').value || 16)); clampLoop(); TL.fit('song'); syncAll(); reloadStems();
  };
  $('anchor').onchange = () => { snapshot(); S.anchor = +$('anchor').value || 0; const w = checkGridDrift(); if (w) status(w, true); syncAll(); reloadStems(); };
  $('anchorSel').onchange = () => {
    const v = $('anchorSel').value; if (v === '') return;
    if (rec.on && S._sense?.bpm && S._sense.bpm !== S.bpm) { $('anchorSel').value = ''; return status('录音中不能改速度：先按 ` 停'); }
    closeVariants(); snapshot(); S.anchor = +v; if (S._sense?.bpm) setBpm(S._sense.bpm); syncAll();
    status([`起点 = ${(+v).toFixed(2)} s（第 1 小节对齐原曲这个重拍）`, checkGridDrift()].filter(Boolean).join('\n'), true);
    reloadStems();
  };
  $('work').onchange = async () => {
    const nw = $('work').value, old = S.work;
    if (rec.on) { $('work').value = old; return status('录音中不能换作品：先按 ` 停'); }
    // 录音 / 分轨存的文件名都在上一个作品的文件夹里：带过去就得在新作品里重新存，不然真值指着不存在的文件
    if (S.clips.length && !confirm(`换到「${nw}」：这个会话里的录音 / 分轨是「${old}」的。\n确定 = 带过去（下次保存时录音重新存进「${nw}」；「${old}」的原曲分轨当成普通素材一起存）\n取消 = 不换`)) { $('work').value = old; return; }
    const warn = S.clips.length ? moveAudioToWork() : '';
    S.work = nw; lastSaved = null; await loadSense(); refreshFinal(); status(`作品：${S.work}${warn ? '\n' + warn : ''}`, !!warn);
  };
  $('play').onclick = togglePlay; $('rec').onclick = toggleRec;
  // 播放中开关循环：从排到的那一拍按新的循环状态重新接（不然还在跑旧的循环段，关了循环也会在旧循环尾绕回去）
  $('loopOn').onclick = () => { if (rec.on) return status(LOOP_LOCKED); S.loop.on = !S.loop.on; const f = playFront(); if (f !== null) T.start(f); syncHeader(); };
  $('metro').onclick = () => { S.settings.metro = !S.settings.metro; syncHeader(); };
  $('countIn').onclick = () => { S.settings.countIn = !S.settings.countIn; syncHeader(); };
  $('undo').onclick = undo; $('redo').onclick = redo; $('save').onclick = save; $('load').onclick = openSessions; $('final').onclick = finalize;
  $('help').onclick = () => $('helpBox').classList.remove('hidden'); $('helpClose').onclick = () => $('helpBox').classList.add('hidden');
  $('sessClose').onclick = () => $('sessBox').classList.add('hidden');
  $('velMode').onchange = () => { S.settings.velMode = $('velMode').value; };
  $('tapPat').onchange = () => { S.settings.tapPattern = $('tapPat').value; };
  $('tParent').onchange = () => { S.settings.tensionParent = $('tParent').checked; };
  $('snapEdit').onchange = () => { S.settings.snapEdit = +$('snapEdit').value; };
  $('inputSeg').onclick = ev => { const v = ev.target.dataset.v; if (!v || ev.target.disabled) return; if (rec.on) return status('录音中不能换输入方式：先按 ` 停'); sel().input = v; renderKeys(); };
  $('addLayer').onchange = () => {
    const [kind, preset] = $('addLayer').value.split(':'); $('addLayer').value = '';
    if (!kind) return;
    if (kind === 'stem') { $('fileIn').click(); return; }
    snapshot();
    const name = kind === 'synth' ? PRESETS[preset].label : kind === 'drums' ? '鼓' : '哼唱';
    const L = makeLayer(kind, preset, name, S.layers.length);
    S.layers.push(L); joinScenes(L.id); selectLayer(L.id);
    if (S.scenes.length) status(`新层「${name}」已经加进全部 ${S.scenes.length} 个场景（不要它响的场景在场景表里点掉）`);
  };
  $('fileIn').onchange = () => { const f = $('fileIn').files[0]; if (f) addFileLayer(f); $('fileIn').value = ''; };
  $('micBtn').onclick = micOn; $('calib').onclick = calibrate;
  $('reroll').onclick = () => { if (variants?.ids) openNoteVariants(variants.layer, variants.ids, (variants.reroll || 0) + 1, variants.loop); };
  $('timbreVar').onclick = openTimbreVariants;
  $('fitSong').onclick = () => TL.fit('song'); $('fitLoop').onclick = () => TL.fit('loop');
  $('copy').onclick = copySel; $('paste').onclick = () => pasteAt(TL.cursor); $('dup').onclick = duplicate; $('del').onclick = deleteSel;
  $('addScene').onclick = addScene; $('loadStems').onclick = loadStems;
  // 点完按钮把焦点还给页面：不然空格 / 字母键会去「按」那个按钮
  document.addEventListener('click', e => { const b = e.target.closest('button'); if (b) b.blur(); });
  // 改完顶栏 / 旋钮的下拉框、数字框：焦点也还给页面（不然字母键会去下拉框里跳选项，空格 / ` 没反应）
  document.addEventListener('change', e => { const t = e.target; if (t && (t.tagName === 'SELECT' || (t.tagName === 'INPUT' && t.type !== 'range'))) t.blur(); });
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
  // 输入框里：Esc 退出；数字框 / 单行框里按 Enter = 确定（触发 change）并把焦点还给页面
  // 下拉框：点开又没改也会留着焦点 → 字母 / 数字 / 空格会偷偷改掉调、调式、作品。只留方向键 / Tab / Enter / Esc 给它，其余还给页面
  if (tag === 'SELECT' && !['Tab', 'ArrowUp', 'ArrowDown', 'Enter', 'NumpadEnter', 'Escape'].includes(ev.code) && !ev.ctrlKey && !ev.metaKey && !ev.altKey) { ev.preventDefault(); ev.target.blur(); }
  else if ((tag === 'INPUT' && ev.target.type !== 'range' && ev.target.type !== 'checkbox') || tag === 'TEXTAREA' || tag === 'SELECT') { if (ev.code === 'Escape' || (ev.code === 'Enter' && tag === 'INPUT')) ev.target.blur(); return; }
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
    if (rec.on) return status('录音中不能换输入方式（哼唱 / 键盘 / 拍）：先按 ` 停');
    const order = L.kind === 'voice' ? ['hum', 'keyboard', 'tap'] : L.kind === 'synth' ? ['keyboard', 'tap'] : [L.input];
    L.input = order[(order.indexOf(L.input) + 1) % order.length]; renderKeys(); status(`输入方式：${{ keyboard: '键盘', tap: '拍', hum: '哼唱' }[L.input]}`); return;
  }
  if (/^Digit[1-9]$/.test(ev.code)) {
    const i = +ev.code.slice(5) - 1;
    if (ev.shiftKey) return jumpScene(i);
    if (S.layers[i]) selectLayer(S.layers[i].id);
    return;
  }
  if (ev.code === 'Digit0') { S.settings.metro = !S.settings.metro; syncHeader(); return; }
  if (ev.code === 'ArrowUp' || ev.code === 'ArrowDown') { ev.preventDefault(); const L = sel(); if (L.kind === 'synth' || L.kind === 'voice') { L.octave = Math.max(0, Math.min(7, L.octave + (ev.code === 'ArrowUp' ? 1 : -1))); renderKeys(); renderParams(); status(`八度 ${L.octave}`); } return; }
  if (variants && (ev.code === 'ArrowLeft' || ev.code === 'ArrowRight')) { ev.preventDefault(); variants.idx = (variants.idx + (ev.code === 'ArrowRight' ? 1 : -1) + variants.options.length) % variants.options.length; previewVariant(); return; }
  if (variants && ev.code === 'Enter') { ev.preventDefault(); return commitVariant(ev.shiftKey ? 'overlay' : 'replace'); }
  if (ev.code === 'Escape') { if (variants) { commitVariant('discard'); } else { TL.sel.clear(); TL.range = null; } return; }
  if (ev.code === 'Delete' || ev.code === 'Backspace') { ev.preventDefault(); return deleteSel(); }
  if (ev.code === 'Minus' || ev.code === 'Equal') { if (rec.on) return status(LOOP_LOCKED); const d = ev.code === 'Minus' ? -S.loop.bars : S.loop.bars; S.loop.start = Math.max(0, Math.min(S.songBars - S.loop.bars, S.loop.start + d)); syncHeader(); status(`循环区：第 ${S.loop.start + 1}–${S.loop.start + S.loop.bars} 小节`); return; }
  if (ev.repeat) { ev.preventDefault(); return; }
  if (noteDown(ev)) ev.preventDefault();
}

// ---------- 主循环 ----------
let lastKeysBar = -1;
function frame() {
  const b = T.playing ? T.beatNow : null;
  if (xy.down) xyWrite();                        // 录音中按住 XY 不动：值照样一帧一帧写进曲线
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
    // 预备拍：k = 排到的那一拍（走带传进来）；老走带不传就用听到的拍。只打到开始录的那一拍之前
    countingIn: k => rec.on && T.playing && (k ?? T.beatNow ?? 0) < rec.countEnd - 1e-9,
    onSegment: (ng, prev) => {
      // 关着循环：哼唱从开始录到停是一条（toggleRec / stopRec 管），键盘不按遍撤销
      if (!rec.on || !S.loop.on) return;
      const [ls, le] = loopRange(S);
      if (prev && prev.b0 <= ls + 1e-9 && prev.b1 >= le - 1e-9) {
        // 一遍循环结束：哼唱这一遍收起来（存的时候自己记撤销点）；键盘 / XY 的这一遍可单独撤销
        if (hum.active) humPassEnd(ng ? ng.ctx0 : E.now);
        else pushUndo();
      }
      if (ng && hum.active && Math.abs(ng.b0 - ls) < 1e-9) humPassStart(ng);
    },
    onStop: () => { if (rec.on) stopRec(); },
  });
  TL = new Timeline($('tl'), {
    getState: () => S, events, clips: clipsNow, peaks: peaksOf, shownLanes: () => shownLanes,
    snapshot, changed: k => {
      // 录音中在标尺上拖循环区：拖不动（这一遍会对不上），还原成开始录时的
      if (k === 'loop' && rec.on && rec.loop0 && JSON.stringify(S.loop) !== JSON.stringify(rec.loop0)) { S.loop = { ...rec.loop0 }; status(LOOP_LOCKED); }
      if (k === 'loop') syncHeader(); if (k === 'notes-done' || k === 'curve') renderLaneChecks();
    },
    selectLayer: (id, quiet) => { if (S.sel !== id) selectLayer(id, quiet); },
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
  // 关页面 / 刷新：还有没存进硬盘的录音（或还在分析的那一遍）就先问一声（浏览器自己的自动存档里没有录音）
  window.addEventListener('beforeunload', e => { if (hasUnsavedAudio() || hum.pending > 0) { e.preventDefault(); e.returnValue = ''; } });
  // 换了耳机 / 声卡：校准过的延迟多半不对了
  try { navigator.mediaDevices?.addEventListener?.('devicechange', () => { if (S.settings.latencyCal) status('音频设备变了（换了耳机？）：麦克风延迟补偿可能不对了，建议重新「校准延迟」', true); }); } catch (e) { /* 老浏览器 */ }
  syncAll();
  if (restoredNote && server) status(restoredNote);
  if (E.silent) { status('静音模式（网址带 ?silent）：一切照常运行，只是不从扬声器出声', true); document.title = '作曲器（静音）'; }
  try { watchOtherTabs(); } catch (e) { /* 没有 BroadcastChannel 就算了 */ }
  requestAnimationFrame(frame);
  window.__composer = { get S() { return S; }, E, T, TL, audio, exportTruth: () => exportTruth(S), validate: () => validateTruth(exportTruth(S)), processHumBuffer: (pcm, sr, b0) => injectHum(pcm, sr, b0) };
}
// 测试 / 调试：直接塞一段音频当哼唱（不经过麦克风）。分析在后台线程：返回 Promise，await 它再看结果
async function injectHum(pcm, sr, b0 = 0) {
  mic.sr = sr; mic.chunks = [{ t: 100, x: pcm }];
  const L = sel(); if (L.kind !== 'voice') { S.sel = 'voice'; }
  const sess = hum.sessionTakes = [];
  await processHumTake(sel().id, b0, 100, 100 + pcm.length / sr, sess);
  openTakeVariants(sel().id);
}

$('gateBtn').onclick = async () => { $('gate').classList.add('hidden'); await boot(); };
