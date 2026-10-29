// >v<🎛作曲器-解释层-原始演奏 → 去延迟 / 吸附节拍（强度、摇摆、律动）/ 规则力度 / 变体。哼唱和键盘共用这一层。
// 原则：原始（raw）永远不改；这里只产出「解释后」的新音符，变体 = 解释强度不同，👤挑。纯函数，不碰 DOM。
import { beatStrength, chordAtBar, chordTones, degreeToMidi, inScale, isChordTone, snapToScale } from './theory.js';

const median = a => { if (!a.length) return 0; const s = [...a].sort((p, q) => p - q), m = s.length >> 1; return s.length % 2 ? s[m] : 0.5 * (s[m - 1] + s[m]); };
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const fin = (v, def) => Number.isFinite(v) ? v : def;        // 力度 / 时值缺了或是 NaN → 用默认（NaN 进引擎会让走带卡死）
export const cloneNotes = ns => ns.map(n => ({ ...n, raw: n.raw ? { ...n.raw } : undefined }));

// ---- 0. 循环里的位置：录音时开着循环 → 音一律放进 [ls, le)（对循环长度取模）；时值超过 le 就截到 le（不拆成两段）----
// loop = [ls, le] 或 null（没开循环 = 不动）。原始值（raw）不在这里改
export function wrapToLoop(b, d, loop) {
  if (!loop || !(loop[1] > loop[0]) || !Number.isFinite(b)) return { b, d };
  const [ls, le] = loop, L = le - ls;
  let nb = b >= ls && b < le - 1e-9 ? b : ls + (((b - ls) % L) + L) % L;
  if (nb >= le - 1e-9) nb = ls;                               // 落在 le 上（或浮点误差贴着 le）= 下一遍的开头
  return { b: nb, d: d === undefined ? d : Math.min(d, le - nb) };
}
// 变体出口统一收一遍：放进循环 + 力度不是数就给默认（试听和定下来的都不会把 NaN 送进引擎 / 真值）
const tidyNotes = (ns, loop) => ns.map(n => {
  const w = wrapToLoop(n.b, n.d, loop), vel = fin(n.vel, 0.8);
  return w.b === n.b && w.d === n.d && vel === n.vel ? n : { ...n, b: w.b, d: w.d, vel };
});

// ---- 1. 稳定的提前 / 拖后（多半是延迟，不是你的意图）：对 16 分格的偏移取中值，够一致才减 ----
export function meanOffset(notes, grid = 0.25) {
  if (notes.length < 4) return { offset: 0, consistent: false };
  const offs = notes.map(n => { const g = Math.round(n.b / grid) * grid; return n.b - g; });
  const m = median(offs), mad = median(offs.map(o => Math.abs(o - m)));
  // 偏移接近半格时，取余会绕成反号（蓝牙耳机的延迟一定超过半格）→ 不可信，不扣，提示先校准
  const tooLarge = Math.abs(m) > grid * 0.38;
  return { offset: m, consistent: !tooLarge && Math.abs(m) > 0.04 && mad < Math.abs(m) * 0.8 + 0.02, mad, tooLarge };
}

// ---- 2. 自动选格子：8 分 / 16 分 / 8 分三连，误差 + 复杂度惩罚（格子越细越不划算）----
export const GRIDS = { '1/8': 0.5, '1/16': 0.25, '1/8T': 1 / 3, '1/16T': 1 / 6 };
export function autoGrid(notes) {
  if (!notes.length) return '1/16';
  let best = '1/16', bs = 1e9;
  const pen = { '1/8': 0, '1/16': 0.012, '1/8T': 0.02, '1/16T': 0.04 };
  for (const [k, g] of Object.entries(GRIDS)) {
    const err = median(notes.map(n => Math.abs(n.b - Math.round(n.b / g) * g)));
    const s = err + pen[k];
    if (s < bs) { bs = s; best = k; }
  }
  return best;
}

// ---- 3. 律动模板：从鼓轨起音里取每个 16 分位置的平均偏移（拍为单位），吸到原曲的「手感」上 ----
export function grooveFromOnsets(beats, { slots = 16, beatsPerBar = 4, spb = 0.4 } = {}) {
  const per = beatsPerBar / slots;           // 一格多少拍
  const acc = Array.from({ length: slots }, () => []);
  for (const b of beats) {
    const k = Math.round(b / per), off = b - k * per;
    if (Math.abs(off) > per * 0.45) continue;
    acc[((k % slots) + slots) % slots].push(off);
  }
  // 去噪：① 拍点（四分音符那几格）一起偏的那一份不是律动（起音检测的固定滞后 / 起点没对准），减掉 —— 律动 = 相对拍点的偏移；
  // ② 起音少的格子往 0 收（n / (n + 2)）；③ 剩下不到 4 ms 的当 0（听不出来，只是噪声）
  const raw = acc.map(a => a.length >= 2 ? median(a) : null);
  const step = Math.max(1, Math.round(slots / beatsPerBar));
  const onBeat = acc.filter((a, i) => i % step === 0).flat(), all = acc.flat();
  const common = onBeat.length >= 4 ? median(onBeat) : all.length >= 4 ? median(all) : 0;
  const offsets = raw.map((o, i) => {
    if (o === null) return 0;
    const v = (o - common) * acc[i].length / (acc[i].length + 2);
    return Math.abs(v) * spb < 0.004 ? 0 : v;
  });
  return { slots, per, offsets, counts: acc.map(a => a.length), common };
}

// ---- 4. 吸附：起音按强度靠向格子（+ 摇摆 / 律动），时值的尾巴吸得弱一点 ----
// loop = [ls, le]：吸到 le 上（或之后）的绕回 ls，时值截到 le（循环里放不到 le 之后的东西）
export function quantize(notes, { grid = '1/16', strength = 1, swing = 0, groove = null, removeOffset = 0, minDur = 0.125, loop = null } = {}) {
  const g = typeof grid === 'number' ? grid : GRIDS[grid] || 0.25;
  return notes.map(n0 => {
    const n = Number.isFinite(n0.d) ? n0 : { ...n0, d: 0.25 };
    const b0 = n.b - removeOffset;
    let k = Math.round(b0 / g), target = k * g;
    // 摇摆：8 分格时每个偶数后的那一格往后推（16 分格推奇数 16 分）
    if (swing && (g === 0.5 || g === 0.25) && (((k % 2) + 2) % 2) === 1) target += swing * g;
    if (groove) {
      const slot = Math.round(target / groove.per);
      target += groove.offsets[((slot % groove.slots) + groove.slots) % groove.slots] || 0;
    }
    const b = b0 + (target - b0) * strength;
    const end0 = n.b + n.d - removeOffset;
    const endT = Math.max(target + g * 0.5, Math.round(end0 / g) * g);
    const end = end0 + (endT - end0) * strength * 0.6;
    const w = wrapToLoop(b, Math.max(minDur * strength + n.d * (1 - strength) * 0.5, end - b, 0.06), loop);
    return { ...n, b: w.b, d: w.d };
  });
}

// ---- 5. 力度规则：节拍位置（重拍强）+ 句子走向（上行渐强）+ 张力（总父级）。键盘没有力度，就让规则给 ----
// 鼓没有音高：走向只拿有音高的音算（鼓 = 0，只看节拍位置和张力）
export function ruleVelocity(notes, { beatsPerBar = 4, tensionAt = () => 0.5, amount = 1 } = {}) {
  const sorted = [...notes].sort((a, b) => a.b - b.b);
  const out = new Map();
  for (let i = 0; i < sorted.length; i++) {
    const n = sorted[i], vel0 = fin(n.vel, 0.8);
    const s = fin(beatStrength(Math.round(n.b * 4) / 4, beatsPerBar), 0);
    // 走向：和前面 4 个有音高的音的平均音高比
    const prev = Number.isFinite(n.pitch) ? sorted.slice(0, i).filter(p => Number.isFinite(p.pitch)).slice(-4) : [];
    const contour = prev.length ? clamp((n.pitch - prev.reduce((a, p) => a + p.pitch, 0) / prev.length) / 7, -1, 1) : 0;
    const t = fin(tensionAt(n.b), 0.5);
    const v = (0.62 + 0.22 * s + 0.08 * contour) * (0.8 + 0.35 * t);
    out.set(n, clamp(fin(vel0 + (v - vel0) * amount, vel0), 0.15, 1));
  }
  return notes.map(n => ({ ...n, vel: +out.get(n).toFixed(3) }));
}

// 按住时长 → 力度（短促 = 轻，按住 = 重），用于键盘的「按住时长」力度模式
export const holdVelocity = holdSec => +clamp(0.45 + 0.55 * Math.min(1, fin(holdSec, 0) / 0.6), 0.3, 1).toFixed(3);

// ---- 6. 鼓轨起音（给律动模板）：高频能量的正向差分，峰值挑选 ----
export function detectOnsets(ch, sr, { hop = 256, win = 1024, thresh = 1.6 } = {}) {
  const env = [];
  for (let st = 0; st + win <= ch.length; st += hop) {
    let e = 0;
    // 一阶差分 ≈ 高通，压住贝斯那种慢变化
    for (let i = st + 1; i < st + win; i++) { const d = ch[i] - ch[i - 1]; e += d * d; }
    env.push(Math.log1p(e * 1000));
  }
  const flux = env.map((v, i) => i ? Math.max(0, v - env[i - 1]) : 0);
  const out = [];
  const W = 16;
  for (let i = 1; i < flux.length - 1; i++) {
    const lo = Math.max(0, i - W), hi = Math.min(flux.length, i + W);
    const loc = flux.slice(lo, hi), m = loc.reduce((a, b) => a + b, 0) / loc.length;
    if (flux[i] > thresh * m + 0.05 && flux[i] >= flux[i - 1] && flux[i] >= flux[i + 1]) {
      const t = (i * hop + win / 2) / sr;
      if (!out.length || t - out[out.length - 1] > 0.06) out.push(t);
    }
  }
  return out;
}

// ---- 7. 变体 ----
// ctx = { key, prog, beatsPerBar, tensionAt, groove, octave, loop }；loop = [ls, le]（录音时开着循环）或 null
const VARIANT_POOL = {
  raw:     { label: '原样',            desc: '你弹 / 唱的，一点不改' },
  q60:     { label: '吸附 60%',        desc: '靠向格子六成，留人味' },
  q100:    { label: '吸附 100%',       desc: '完全对齐格子（机器感）' },
  groove:  { label: '吸到原曲律动',     desc: '对齐原曲鼓轨的手感，不是数学格子' },
  swing:   { label: '摇摆',            desc: '后半拍往后推，晃起来' },
  velrule: { label: '规则力度',        desc: '重拍强、上行渐强、跟张力走' },
  sparse:  { label: '稀疏',            desc: '只留强拍附近的音，留白' },
  octave:  { label: '八度叠',          desc: '每个音加高八度' },
  harm3:   { label: '三度和声',        desc: '调内平行三度（往上）' },
  legato:  { label: '连奏',            desc: '每个音拖到下一个音' },
  stacc:   { label: '断奏',            desc: '每个音只留一半' },
  chordy:  { label: '强拍吸和弦音',     desc: '强拍上的音换成最近的和弦音' },
  ghost:   { label: '加鬼音',          desc: '空的 16 分格补轻踩镲', drums: true },
};
export const variantInfo = k => VARIANT_POOL[k];

function applyVariant(kind, notes, ctx, isDrums) {
  const g = autoGrid(notes);
  const off = meanOffset(notes);
  const ro = off.consistent ? off.offset : 0;
  const loop = ctx.loop || null;
  const pitched = notes.filter(n => !n.sound && Number.isFinite(n.pitch));     // 叠八度 / 和声只给有音高的音
  switch (kind) {
    case 'raw': return cloneNotes(notes);
    case 'q60': return quantize(notes, { grid: g, strength: 0.6, removeOffset: ro, loop });
    case 'q100': return quantize(notes, { grid: g, strength: 1, removeOffset: ro, loop });
    case 'groove': return quantize(notes, { grid: '1/16', strength: 1, removeOffset: ro, groove: ctx.groove, loop });
    case 'swing': return quantize(notes, { grid: g === '1/16' ? '1/16' : '1/8', strength: 1, swing: 0.33, removeOffset: ro, loop });
    case 'velrule': return ruleVelocity(quantize(notes, { grid: g, strength: 0.6, removeOffset: ro, loop }), ctx);
    case 'sparse': {
      const q = quantize(notes, { grid: g, strength: 1, removeOffset: ro, loop });
      const keep = q.filter(n => beatStrength(Math.round(n.b * 2) / 2, ctx.beatsPerBar) >= 0.5);
      return keep.length ? keep : q.filter((_, i) => i % 2 === 0);
    }
    case 'octave': return [...cloneNotes(notes), ...pitched.map(n => ({ ...n, pitch: n.pitch + 12, vel: +(fin(n.vel, 0.8) * 0.7).toFixed(3), derived: 'octave' }))];
    case 'harm3': return [...cloneNotes(notes), ...pitched.map(n => ({ ...n, pitch: scaleStep(ctx.key, n.pitch, 2), vel: +(fin(n.vel, 0.8) * 0.75).toFixed(3), derived: 'harm3' }))];
    case 'legato': {
      const s = [...notes].sort((a, b) => a.b - b.b);
      return s.map((n, i) => ({ ...n, d: i < s.length - 1 ? Math.max(0.1, s[i + 1].b - n.b) : n.d }));
    }
    case 'stacc': return notes.map(n => ({ ...n, d: Math.max(0.06, fin(n.d, 0.25) * 0.5) }));
    case 'chordy': {
      const q = quantize(notes, { grid: g, strength: 1, removeOffset: ro, loop });
      return q.map(n => {
        if (n.sound || !Number.isFinite(n.pitch)) return n;
        if (beatStrength(Math.round(n.b * 4) / 4, ctx.beatsPerBar) < 0.5) return n;
        const deg = chordAtBar(ctx.prog, Math.floor(n.b / ctx.beatsPerBar));
        if (isChordTone(ctx.key, deg, n.pitch)) return n;
        const c = chordTones(ctx.key, deg, n.pitch - 6, 6).sort((a, b) => Math.abs(a - n.pitch) - Math.abs(b - n.pitch))[0];
        return { ...n, pitch: c ?? n.pitch };
      });
    }
    case 'ghost': {
      if (!isDrums) return cloneNotes(notes);
      const q = quantize(notes, { grid: '1/16', strength: 1, removeOffset: ro, loop });
      const taken = new Set(q.map(n => Math.round(n.b * 4)));
      let lo = Math.floor(Math.min(...q.map(n => n.b))), hi = Math.ceil(Math.max(...q.map(n => n.b + 0.25)));
      if (loop) { lo = Math.max(lo, loop[0]); hi = Math.min(hi, loop[1]); }      // 鬼音只补在循环里（不补到 le 之后）
      const add = [];
      for (let k = lo * 4; k < hi * 4; k++) if (!taken.has(k) && k % 2 === 1) add.push({ b: k / 4, d: 0.125, sound: 'hat', vel: 0.28, derived: 'ghost' });
      return [...q, ...add];
    }
  }
  return cloneNotes(notes);
}

// 调内往上走 steps 级
export function scaleStep(key, m, steps) {
  let p = snapToScale(key, m), k = 0;
  while (k < steps) { p++; if (inScale(key, p)) k++; }
  return p;
}

// 出一组变体：第一个永远是「原样」；默认 = 原样 / 吸附 60% / 吸附 100% / 律动（有律动模板时）或一个随机
// ctx.loop = [ls, le] 时每个变体的音都放进循环里（位置取模、时值截到 le），试听和定下来的是同一份
export function makeVariants(notes, ctx, { isDrums = false, reroll = 0 } = {}) {
  let kinds;
  if (!reroll) kinds = ['raw', 'q60', 'q100', ctx.groove ? 'groove' : (isDrums ? 'ghost' : 'velrule')];
  else {
    const pool = Object.keys(VARIANT_POOL).filter(k => k !== 'raw' && (!VARIANT_POOL[k].drums || isDrums) && (k !== 'groove' || ctx.groove)
      && !(isDrums && ['octave', 'harm3', 'chordy', 'legato'].includes(k)));
    // 可复现的洗牌（reroll 当种子）
    let s = reroll * 7919 + 13;
    const rnd = () => { s = (s * 16807) % 2147483647; return s / 2147483647; };
    for (let i = pool.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [pool[i], pool[j]] = [pool[j], pool[i]]; }
    kinds = ['raw', ...pool.slice(0, 3)];
  }
  return kinds.map(k => ({ kind: k, ...VARIANT_POOL[k], notes: tidyNotes(applyVariant(k, notes, ctx, isDrums), ctx.loop || null) }));
}

// 音色变体（合成层）：只改参数，不改音
export const TIMBRE_VARIANTS = [
  { kind: 'bright', label: '更亮', desc: '截止开大、共振高一点', patch: p => ({ ...p, cutoff: Math.min(16000, p.cutoff * 2.2), reso: Math.min(0.8, p.reso + 0.15) }) },
  { kind: 'dark', label: '更暗', desc: '截止收小、包络慢一点', patch: p => ({ ...p, cutoff: Math.max(200, p.cutoff * 0.45), attack: Math.max(p.attack, 0.02) }) },
  { kind: 'wide', label: '更宽（齐奏）', desc: '5 个声部、失谐 20 音分', patch: p => ({ ...p, voices: 5, detune: 20 }) },
  { kind: 'fm', label: '金属（FM）', desc: 'FM 比 3.5、深度 2', patch: p => ({ ...p, fmRatio: 3.5, fmIndex: 2 }) },
  { kind: 'dirty', label: '更脏', desc: '失真 0.55', patch: p => ({ ...p, drive: Math.max(p.drive, 0.55) }) },
];

// 拍的音高：按键的左右位置 → 当前和弦的第几个音（永远是和弦音）
export function tapPitch(ctx, b, col, octave, pattern = 'position', counter = 0) {
  const deg = chordAtBar(ctx.prog, Math.floor(b / ctx.beatsPerBar));
  const lo = degreeToMidi(ctx.key, octave, 0) - 0.5;
  const tones = chordTones(ctx.key, deg, lo, 12);
  if (pattern === 'root') return tones[0];
  if (pattern === 'arp') return tones[counter % 4];
  if (pattern === 'random') return tones[Math.floor(((Math.sin(counter * 12.9898 + b * 78.233) * 43758.5453) % 1 + 1) % 1 * 4)];
  return tones[clamp(col, 0, tones.length - 1)];
}
