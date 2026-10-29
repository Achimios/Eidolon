// >v<🎛作曲器-状态与真值导出-内部时间用「拍」（改 BPM 不乱），导出一律用秒（歌曲播放时间 = anchor + 拍 × 每拍秒数）。
// 作曲.json 的格式见 _设计/设计_作曲器.md §二；本页加的字段：input（keyboard / hum / tap / xy / draw / pick）、raw（原始演奏）、
// layers、clips、chords、picks、macro_rules、derived、audible（音符 / 打击：按 mute + 场景算，solo 不算）。字段只增不改名。纯函数，不碰 DOM。
import { MODES, NOTE_NAMES, chordAtBar, chordName, chordPcs } from './theory.js';
import { PRESETS, DRUM_PARAMS } from './engine.js';

export const MACROS = [
  { id: 'tension',   label: '张力',   word: '情绪曲线：所有旋钮的总父级', def: 0.5 },
  { id: 'cutoff',    label: '截止',   word: '滤波器（母线低通）', def: 1 },
  { id: 'reso',      label: '共振',   word: '滤波器', def: 0.1 },
  { id: 'drive',     label: '失真',   word: '失真 / 饱和', def: 0 },
  { id: 'space',     label: '混响',   word: '延迟 / 混响', def: 0.25 },
  { id: 'pole_g',    label: '回声 g', word: '延迟的反馈 = 极点（> 1 失稳，限幅接住）', def: 0.35 },
  { id: 'sidechain', label: '侧链',   word: '侧链（底鼓压别的层）', def: 0.4 },
];
export const MACRO_BY_ID = Object.fromEntries(MACROS.map(m => [m.id, m]));
// 张力当父级：有效值 = v × (a + b × 张力)。写进 作曲.json，Python 渲染照这个算
export const MACRO_RULES = { cutoff: [0.55, 0.45], reso: [1, 0], drive: [0.35, 0.65], space: [0.5, 0.5], pole_g: [0.75, 0.35], sidechain: [0.35, 0.65] };

export function effectiveMacros(vals, parentOn = true) {
  const t = vals.tension ?? 0.5, out = { tension: t };
  for (const [k, [a, b]] of Object.entries(MACRO_RULES)) out[k] = (vals[k] ?? MACRO_BY_ID[k].def) * (parentOn ? a + b * t : 1);
  return out;
}

// ---- 曲线：稀疏点 [[拍, 值], …]，线性插值 ----
export function curveAt(pts, b, def) {
  if (!pts || !pts.length) return def;
  if (b <= pts[0][0]) return pts[0][1];
  if (b >= pts[pts.length - 1][0]) return pts[pts.length - 1][1];
  let lo = 0, hi = pts.length - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (pts[m][0] <= b) lo = m; else hi = m; }
  const [b0, v0] = pts[lo], [b1, v1] = pts[hi];
  return b1 > b0 ? v0 + (v1 - v0) * (b - b0) / (b1 - b0) : v1;
}
// 写一段：b0 = 上一个点（留着），b1 = 现在的位置（新点放这里）；删掉两者之间的旧点（XY 推子 / 手画都走这里）
// 两端先按存的精度（4 位小数）取整再比：不然上一个点存成进位后的值，会落进 (b0, b1] 被自己删掉（快画只剩一半的点）。
// 往左画也一样：新点放在 b1，不是两端里靠右的那个
export function curveWrite(pts, b0, b1, v) {
  if (!Number.isFinite(b0) || !Number.isFinite(b1) || !Number.isFinite(v)) return pts;
  const from = +b0.toFixed(4), at = +b1.toFixed(4);
  const lo = Math.min(from, at), hi = Math.max(from, at);
  const keep = pts.filter(p => !(p[0] > lo + 1e-9 && p[0] < hi - 1e-9) && Math.abs(p[0] - at) > 1e-4);
  keep.push([at, +v.toFixed(4)]);
  keep.sort((a, b) => a[0] - b[0]);
  return keep;
}

let _id = 0;
export const uid = (p = 'n') => `${p}${Date.now().toString(36)}${(_id++).toString(36)}`;

const COLORS = ['#4dff7c', '#ff8a4c', '#5cc8ff', '#e8d44d', '#d77bff', '#ff5c8a', '#7dffd8', '#b0b8ff'];
export function makeLayer(kind, preset, name, idx = 0, extra = {}) {
  const id = extra.id || uid('L');
  const base = { id, name, kind, preset, mute: false, solo: false, gainDb: 0, color: COLORS[idx % COLORS.length] };
  if (kind === 'synth') return { ...base, input: 'keyboard', octave: PRESETS[preset].octave, params: { ...PRESETS[preset].p }, ...extra };
  if (kind === 'drums') return { ...base, input: 'keyboard', octave: 0, params: { ...DRUM_PARAMS }, ...extra };
  if (kind === 'voice') return { ...base, input: 'hum', octave: 4, params: { mode: 'auto', snap: 'scale', rawGainDb: 0, synthOn: true, synthPreset: 'sing', bendOn: true, drive: 0, delaySend: 0.1, reverbSend: 0.2, duck: false, pan: 0 }, ...extra };
  return { ...base, input: 'none', octave: 0, params: { drive: 0, delaySend: 0, reverbSend: 0, duck: false, pan: 0 }, ...extra };   // stem
}

export function makeState() {
  const layers = [
    makeLayer('drums', 'drums', '鼓', 0, { id: 'drums' }),
    makeLayer('synth', 'bass', '贝斯', 1, { id: 'bass' }),
    makeLayer('synth', 'lead', '主旋律', 2, { id: 'lead' }),
    makeLayer('voice', 'voice', '哼唱', 3, { id: 'voice' }),
  ];
  return {
    format: 'composer_web/1',
    work: 'MV_Cant_Kill_Us',
    bpm: 150, beatsPerBar: 4, anchor: 0,
    key: { root: 4, mode: 'aeolian' },
    prog: [0, 5, 2, 6],                    // i VI III VII（Em C G D）
    songBars: 16,
    loop: { start: 0, bars: 4, on: true },
    layers, sel: 'lead',
    notes: [],                              // 合成层 / 鼓（鼓有 sound 没有 pitch）
    clips: [],                              // 哼唱录音 / 分轨
    curves: { tension: [] },
    scenes: [],
    picks: [],
    settings: { velMode: 'flat', latencyMs: null, kbOffsetMs: 0, countIn: true, metro: true, tensionParent: true, tapPattern: 'position', snapEdit: 0.25 },
  };
}

export const spb = s => 60 / s.bpm;
export const loopRange = s => [s.loop.start * s.beatsPerBar, (s.loop.start + s.loop.bars) * s.beatsPerBar];
export const songBeats = s => s.songBars * s.beatsPerBar;
export const layerById = (s, id) => s.layers.find(l => l.id === id);

// 某拍上哪一个场景在管（场景 = 从它的小节到下一个场景）
export function sceneAt(s, b) {
  let cur = null;
  for (const sc of s.scenes) if (sc.bar * s.beatsPerBar <= b + 1e-9) cur = sc;
  return cur;
}
// 层在这一拍进不进成片：静音 + 场景（独奏只是页面试听，不算）。导出真值的 audible 用这个
export function inMixAt(s, layer, b) {
  if (layer.mute) return false;
  const sc = sceneAt(s, b);
  return !sc || sc.layers.includes(layer.id);
}
// 层在这一拍响不响（页面里）：全局静音 / 独奏 / 场景
export function audibleAt(s, layer, b) {
  if (s.layers.some(l => l.solo) && !layer.solo) return false;
  return inMixAt(s, layer, b);
}

export function chordList(s) {
  const out = [];
  for (let bar = 0; bar < s.songBars; bar++) {
    const d = chordAtBar(s.prog, bar);
    out.push({ bar, degree: d, name: chordName(s.key, d), pcs: chordPcs(s.key, d) });
  }
  return out;
}

const r3 = v => Math.round(v * 1000) / 1000, r4 = v => Math.round(v * 10000) / 10000;
// 凭空生成、没有对应演奏的 derived（八度 / 和声 / 琶音都从一个真弹的音来，带着那个音的 raw）
const NO_SOURCE = new Set(['ghost']);

// ---- 真值导出：作曲.json ----
// audioFiles：audioId → 文件名（保存时服务器给）
export function exportTruth(s, { audioFiles = {}, macroFps = 60, created = new Date().toISOString() } = {}) {
  const T = b => r4(s.anchor + b * spb(s));
  const D = d => r4(d * spb(s));
  const total = songBeats(s);
  const L = id => layerById(s, id);
  const activeClipIds = new Set(s.clips.filter(c => c.active).map(c => c.id));
  const live = s.notes.filter(n => !n.take || activeClipIds.has(n.take)).filter(n => L(n.layer));
  const notes = [], hits = [];
  for (const n of [...live].sort((a, b) => a.b - b.b)) {
    // raw 只导出真弹 / 真唱过的：没有原始拍位就不编一个；凭空生成的（鬼音）没有原始演奏
    const hasRaw = n.raw && Number.isFinite(n.raw.b) && !NO_SOURCE.has(n.derived);
    const raw = hasRaw ? { t: T(n.raw.b), ...(n.raw.d !== undefined ? { dur: D(n.raw.d) } : {}), ...(n.raw.pitch_raw !== undefined ? { pitch_raw: n.raw.pitch_raw } : {}), ...(n.raw.held_pitch !== undefined ? { held_pitch: n.raw.held_pitch } : {}), ...(n.raw.vel !== undefined ? { vel: n.raw.vel } : {}), ...(n.raw.db !== undefined ? { db: n.raw.db } : {}) } : undefined;
    const input = n.input || (n.derived ? 'pick' : n.input);          // 规则生成的音：来源 = 👤在变体里挑的
    const audible = inMixAt(s, L(n.layer), n.b);
    if (n.sound) hits.push({ t: T(n.b), layer: n.layer, sound: n.sound, vel: r3(n.vel), input, audible, ...(raw ? { raw } : {}), ...(n.derived ? { derived: n.derived } : {}) });
    else {
      // 不吸附时音高是小数：导出取整，小数部分并进音分曲线（曲线永远相对导出的整数音高）
      const p = Math.round(n.pitch), frac = Math.round((n.pitch - p) * 100);
      const bend = n.bend ? (frac ? { ...n.bend, cents: n.bend.cents.map(c => c + frac) } : n.bend) : null;
      notes.push({ t: T(n.b), dur: D(n.d), pitch: p, vel: r3(n.vel), layer: n.layer, input, audible, ...(n.bright !== undefined ? { bright: n.bright } : {}), ...(bend ? { bend } : {}), ...(raw ? { raw } : {}), ...(n.derived ? { derived: n.derived } : {}) });
    }
  }
  // 旋钮：逐帧（从 anchor 开始，覆盖整首）；张力另存稀疏点（和 remix_cku 的 作曲.json 一样是 [t, v] 对）
  const nF = Math.ceil(total * spb(s) * macroFps) + 1;
  const macros = {};
  for (const m of MACROS) {
    if (m.id === 'tension') continue;
    const pts = s.curves[m.id];
    if (!pts || !pts.length) continue;
    const v = new Array(nF);
    for (let i = 0; i < nF; i++) v[i] = r3(curveAt(pts, i / macroFps / spb(s), m.def));
    macros[m.id] = { fps: macroFps, t0: r4(s.anchor), v, label: m.label, word: m.word, input: s.curvesInput?.[m.id] || 'xy', rule: s.settings.tensionParent ? { eff: 'v * (a + b * tension)', a: MACRO_RULES[m.id][0], b: MACRO_RULES[m.id][1] } : null };
  }
  const tensionPts = (s.curves.tension || []).map(([b, v]) => [T(b), r3(v)]);
  const scenes = s.scenes.map(sc => ({ t: T(sc.bar * s.beatsPerBar), name: sc.name, layers: [...sc.layers], ...(sc.text ? { '她此刻': sc.text } : {}) }));
  // offset = 从 file 开头算的片内起点（秒）：分轨是按窗口切片载入的，片内 0 = 文件里的切片起点。
  // 只有分轨的 src 带「@t0-t1」切片号（旧会话没存 fileOffset 时从这里解析）；拖进来的文件名里碰巧有 @128- 不算
  const fileOffset = c => { if (c.fileOffset !== undefined) return c.fileOffset; if (c.input !== 'stem') return 0; const m = /@(-?[\d.]+)-/.exec(c.src || ''); return m ? +m[1] : 0; };
  // audible_spans：这段录音在成片里真正发声的时间段（秒，歌曲时间）。按场景边界切开，逐段用 inMixAt（mute + 场景，solo 不算）判断，
  // 相邻的连起来；没激活的 take = []。渲染直接读它，判断只留一份
  const spans = c => {
    if (!c.active) return [];
    const a = c.b, z = c.b + c.dur / spb(s), lay = L(c.layer);
    const cuts = [a, ...s.scenes.map(sc => sc.bar * s.beatsPerBar).filter(x => x > a && x < z), z];
    const out = [];
    for (let i = 0; i < cuts.length - 1; i++) {
      if (!inMixAt(s, lay, cuts[i] + 1e-6)) continue;
      const last = out[out.length - 1];
      if (last && Math.abs(last[1] - cuts[i]) < 1e-9) last[1] = cuts[i + 1]; else out.push([cuts[i], cuts[i + 1]]);
    }
    return out.map(([p0, p1]) => [T(p0), T(p1)]);
  };
  const clips = s.clips.filter(c => L(c.layer)).map(c => ({
    t: T(c.b), dur: r4(c.dur), offset: r4(fileOffset(c) + (c.offset || 0)), layer: c.layer, input: c.input, active: !!c.active, take: c.take ?? null, audible_spans: spans(c),
    file: audioFiles[c.audio] || c.file || null, ...(c.src ? { src: c.src } : {}), ...(c.analysis ? { analysis: c.analysis } : {}),
  }));
  return {
    _说明: 'HTML 作曲器导出的真值。t = 歌曲播放时间（秒，= grid.anchor + 拍 × 60 / bpm）；raw = 原始演奏（没吸附），其余是解释后的；clips.file 相对作品文件夹，clips.offset = 从这个文件开头算的片内起点（秒）；layers.mute 进成片，独奏（solo）只是页面试听、不导出；notes / hits 的 audible 是导出时按 mute + scenes 算的快照，solo 不参与；clips 的 audible_spans = 这段录音在成片里真正发声的时间段（同样按 mute + scenes，solo 不参与）；derived = 规则生成的（不是弹 / 唱出来的），凭空生成的没有 raw；raw.pitch_raw = 真弹 / 真唱的音高（键盘是整数，哼唱是小数），琶音的 raw.held_pitch = 按住的那个键；bend.cents 相对导出的整数 pitch；tension 为空 = 全程 0.5。macros：逐帧，第 0 帧在 t0（= grid.anchor，不是 0 秒）；存的是旋钮原值，只列动过的，没列的用 macro_defaults；生效值 = 原值 × (a + b × tension)（a、b 见 macro_rules），没列出的旋钮也照这个随张力变。页面里的声音只是预览，最终音质由 Python 读这份渲染。',
    generator: 'composer_web/1',
    created,
    work: s.work,
    bpm: s.bpm,
    key: `${NOTE_NAMES[s.key.root]} ${MODES[s.key.mode].en}`,
    key_detail: { root_pc: s.key.root, mode: s.key.mode, steps: MODES[s.key.mode].steps },
    grid: { anchor: r4(s.anchor), beats_per_bar: s.beatsPerBar, downbeats: Array.from({ length: s.songBars + 1 }, (_, i) => T(i * s.beatsPerBar)) },
    window: [T(0), T(total)],
    chords: chordList(s).map(c => ({ t: T(c.bar * s.beatsPerBar), name: c.name, degree: c.degree, pcs: c.pcs })),
    layers: s.layers.map(l => ({ id: l.id, name: l.name, kind: l.kind, preset: l.preset, octave: l.octave, gain_db: l.gainDb, mute: l.mute, input: l.input, params: { ...l.params } })),
    notes, hits,
    clips,
    scenes,
    macros,
    macro_rules: s.settings.tensionParent ? Object.fromEntries(Object.entries(MACRO_RULES).map(([k, [a, b]]) => [k, { a, b }])) : null,
    macro_defaults: Object.fromEntries(MACROS.map(m => [m.id, m.def])),
    tension: tensionPts,
    reveals: [],
    mood: scenes.filter(sc => sc['她此刻']).map(sc => ({ t: sc.t, text: sc['她此刻'] })),
    picks: s.picks.map(p => ({ ...p })),
    latency_ms: { keyboard_offset: s.settings.kbOffsetMs, mic: s.settings.latencyMs },
  };
}

// 会话（能接着玩的那份）：内部状态原样 + 音频文件名
export function exportSession(s, audioFiles = {}) {
  const out = JSON.parse(JSON.stringify(s));
  delete out._sense;            // 感官.json 的缓存不存（下游现读上游）
  return { ...out, audioFiles };
}

// 校验真值（自检 / 测试用）：返回问题列表，空 = 通过
export function validateTruth(j) {
  const errs = [];
  const need = ['bpm', 'grid', 'notes', 'hits', 'scenes', 'macros', 'tension', 'reveals', 'mood'];
  for (const k of need) if (!(k in j)) errs.push(`缺字段 ${k}`);
  // 数一律要是有限数：Infinity / NaN 过 JSON 会变成 null，下游一读就崩
  const F = Number.isFinite, t0 = x => F(x) && x >= 0, v01 = x => F(x) && x >= 0 && x <= 1;
  const rawOk = r => r === undefined || (r && F(r.t) && (r.dur === undefined || F(r.dur)));
  for (const n of j.notes || []) {
    if (!t0(n.t) || !(F(n.dur) && n.dur > 0) || !Number.isInteger(n.pitch) || !v01(n.vel) || !n.layer || !n.input || !rawOk(n.raw)) { errs.push(`音符不合法 ${JSON.stringify(n).slice(0, 120)}`); break; }
  }
  for (const h of j.hits || []) if (!t0(h.t) || !v01(h.vel) || !h.layer || !h.sound || !rawOk(h.raw)) { errs.push(`打击不合法 ${JSON.stringify(h).slice(0, 120)}`); break; }
  for (const [k, m] of Object.entries(j.macros || {})) if (!m.fps || !Array.isArray(m.v) || !m.v.every(F)) errs.push(`旋钮 ${k} 格式不对`);
  for (const p of j.tension || []) if (!Array.isArray(p) || !F(p[0]) || !v01(p[1])) { errs.push(`张力点不合法 ${JSON.stringify(p)}`); break; }
  for (const sc of j.scenes || []) if (!t0(sc.t)) { errs.push(`场景时间不合法 ${JSON.stringify(sc).slice(0, 80)}`); break; }
  for (const c of j.clips || []) if (!F(c.t) || !(F(c.dur) && c.dur >= 0) || !F(c.offset)) { errs.push(`音频片段不合法 ${JSON.stringify(c).slice(0, 120)}`); break; }
  const ts = (j.notes || []).map(n => n.t);
  for (let i = 1; i < ts.length; i++) if (ts[i] < ts[i - 1]) { errs.push('音符没按时间排序'); break; }
  return errs;
}
