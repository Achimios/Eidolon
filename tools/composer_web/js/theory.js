// >v<🎛作曲器-乐理-调式、和弦、锁调键盘的键位 → 音高。纯函数，不碰 DOM（node 测试直接 import）。

export const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

export const MODES = {
  aeolian:   { label: '自然小调', steps: [0, 2, 3, 5, 7, 8, 10], en: 'minor' },
  harmonic:  { label: '和声小调', steps: [0, 2, 3, 5, 7, 8, 11], en: 'harmonic minor' },
  dorian:    { label: '多利亚',   steps: [0, 2, 3, 5, 7, 9, 10], en: 'dorian' },
  phrygian:  { label: '弗里几亚', steps: [0, 1, 3, 5, 7, 8, 10], en: 'phrygian' },
  ionian:    { label: '大调',     steps: [0, 2, 4, 5, 7, 9, 11], en: 'major' },
  minpenta:  { label: '小调五声', steps: [0, 3, 5, 7, 10], en: 'minor pentatonic' },
  majpenta:  { label: '大调五声', steps: [0, 2, 4, 7, 9], en: 'major pentatonic' },
};

// 三排字母键（e.code，和键盘布局 / Shift 无关）。A 排 = 本八度，Q 排 = 高八度（同列同级），Z 排 = 低八度。
export const ROW_Q = ['KeyQ', 'KeyW', 'KeyE', 'KeyR', 'KeyT', 'KeyY', 'KeyU', 'KeyI', 'KeyO', 'KeyP', 'BracketLeft', 'BracketRight'];
export const ROW_A = ['KeyA', 'KeyS', 'KeyD', 'KeyF', 'KeyG', 'KeyH', 'KeyJ', 'KeyK', 'KeyL', 'Semicolon', 'Quote'];
export const ROW_Z = ['KeyZ', 'KeyX', 'KeyC', 'KeyV', 'KeyB', 'KeyN', 'KeyM', 'Comma', 'Period', 'Slash'];
export const KEY_LABEL = c => ({ BracketLeft: '[', BracketRight: ']', Semicolon: ';', Quote: "'", Comma: ',', Period: '.', Slash: '/' }[c] || c.replace('Key', ''));

export const midiHz = m => 440 * Math.pow(2, (m - 69) / 12);
export const hzMidi = f => 69 + 12 * Math.log2(f / 440);
export const noteName = m => NOTE_NAMES[((Math.round(m) % 12) + 12) % 12] + (Math.floor(Math.round(m) / 12) - 1);

export function keyLabel(key) {
  return `${NOTE_NAMES[key.root]} ${MODES[key.mode].label}`;
}

// 音阶第 d 级（可以是负数 / 超过一个八度）→ midi。octave = 第 0 级所在的八度（C4 = 60 的那种记法）。
export function degreeToMidi(key, octave, d) {
  const st = MODES[key.mode].steps, n = st.length;
  const o = Math.floor(d / n), i = ((d % n) + n) % n;
  return 12 * (octave + 1) + key.root + st[i] + 12 * o;
}

// 键 → 音阶级数；不是音符键返回 null
export function codeToDegree(key, code) {
  const n = MODES[key.mode].steps.length;
  let i = ROW_A.indexOf(code);
  if (i >= 0) return i;
  i = ROW_Q.indexOf(code);
  if (i >= 0) return i + n;
  i = ROW_Z.indexOf(code);
  if (i >= 0) return i - n;
  return null;
}

export function inScale(key, m) {
  const pc = (((Math.round(m) - key.root) % 12) + 12) % 12;
  return MODES[key.mode].steps.includes(pc);
}

// 调内最近的音（平局取下面那个）
export function snapToScale(key, m) {
  let best = null, bd = 1e9;
  for (let c = Math.floor(m) - 3; c <= Math.ceil(m) + 3; c++) {
    if (!inScale(key, c)) continue;
    const d = Math.abs(c - m);
    if (d < bd - 1e-9) { bd = d; best = c; }
  }
  return best;
}

// ---- 和弦：每小节一个级数（0 = i 级），三和弦按调内三度叠 ----
const ROMAN = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII'];
export function chordPcs(key, degree) {
  const st = MODES[key.mode].steps, n = st.length;
  // 五声音阶叠三度没意义：退回到自然小调 / 大调来算和弦
  const base = n === 7 ? st : (key.mode === 'majpenta' ? MODES.ionian.steps : MODES.aeolian.steps);
  return [0, 2, 4].map(k => (key.root + base[(degree + k) % 7] + 12 * Math.floor((degree + k) / 7)) % 12);
}
export function chordName(key, degree) {
  const pcs = chordPcs(key, degree);
  const third = (pcs[1] - pcs[0] + 12) % 12, fifth = (pcs[2] - pcs[0] + 12) % 12;
  const q = third === 3 ? (fifth === 6 ? 'dim' : 'm') : (fifth === 8 ? 'aug' : '');
  return NOTE_NAMES[pcs[0]] + q;
}
export function chordRoman(key, degree) {
  const pcs = chordPcs(key, degree);
  const minor = (pcs[1] - pcs[0] + 12) % 12 === 3;
  const r = ROMAN[degree % 7];
  return minor ? r.toLowerCase() : r;
}
// 解析「1 6 3 7」或「i VI III VII」→ [0,5,2,6]
export function parseProgression(s) {
  const out = [];
  for (const tok of String(s).trim().split(/[\s,，|-]+/)) {
    if (!tok) continue;
    const t = tok.replace(/[^ivxIVX0-9]/g, '');
    if (/^[1-7]$/.test(t)) { out.push(+t - 1); continue; }
    const i = ROMAN.indexOf(t.toUpperCase());
    if (i >= 0) out.push(i);
  }
  return out.length ? out : [0];
}
export function chordAtBar(prog, bar) {
  return prog[((Math.floor(bar) % prog.length) + prog.length) % prog.length];
}
// 和弦音（midi），从 lo 往上铺 count 个
// 能弹的和弦音：七声调式 = 三和弦；五声调式里三和弦会带调外音（E 小调五声的 VI = C E G，C 不在调里）→ 只留调内的；
// 剩不到两个就用离和弦根音最近的调内音 + 往上隔一个的调内音。拍 / 琶音 / 吸和弦音都走这里，保证「按哪个都不会错音」
export function chordTonePcs(key, degree) {
  const pcs = chordPcs(key, degree), steps = MODES[key.mode].steps;
  if (steps.length === 7) return pcs;
  const inS = pcs.filter(pc => inScale(key, pc));
  if (inS.length >= 2) return inS;
  const rel = ((pcs[0] - key.root) % 12 + 12) % 12;
  const i = steps.reduce((bi, s, k) => Math.abs(s - rel) < Math.abs(steps[bi] - rel) ? k : bi, 0);
  return [i, i + 2].map(k => (key.root + steps[k % steps.length]) % 12);
}
export function chordTones(key, degree, lo, count) {
  const pcs = chordTonePcs(key, degree), out = [];
  for (let m = Math.ceil(lo); out.length < count && m < lo + 60; m++) if (pcs.includes(((m % 12) + 12) % 12)) out.push(m);
  return out;
}
export function isChordTone(key, degree, m) {
  return chordTonePcs(key, degree).includes(((Math.round(m) % 12) + 12) % 12);
}

// 带和弦偏好的吸附：调内候选按距离排，强拍上和弦音有一点优惠；离两个候选差不多远时，跟上一个音走（惯性）
export function snapMelodic(key, m, { degree = null, strong = 0, prev = null } = {}) {
  const cands = [];
  for (let c = Math.floor(m) - 2; c <= Math.ceil(m) + 2; c++) {
    if (!inScale(key, c)) continue;
    let s = Math.abs(c - m);
    if (degree !== null && isChordTone(key, degree, c)) s -= 0.12 * strong;
    cands.push([s, c]);
  }
  cands.sort((a, b) => a[0] - b[0]);
  if (!cands.length) return Math.round(m);
  if (prev !== null && cands.length > 1 && Math.abs(cands[0][0] - cands[1][0]) < 0.25 && cands[1][1] === prev) return prev;
  return cands[0][1];
}

// 节拍强度：0 = 小节第一拍（最强）… 用来定规则力度和强拍吸附
export function beatStrength(b, beatsPerBar = 4) {
  const eps = 1e-3, inBar = ((b % beatsPerBar) + beatsPerBar) % beatsPerBar;
  if (Math.abs(inBar) < eps || Math.abs(inBar - beatsPerBar) < eps) return 1;
  if (beatsPerBar === 4 && Math.abs(inBar - 2) < eps) return 0.7;
  const f = inBar - Math.floor(inBar);
  if (f < eps || f > 1 - eps) return 0.5;
  if (Math.abs(f - 0.5) < eps) return 0.25;
  return 0;
}
