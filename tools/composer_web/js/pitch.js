// >v<🎛作曲器-哼唱识别-YIN 音高 → 逐帧特征 → 猜「人声 / 哼唱 / 旋律」→ 切音符（去漂移、跳过进音滑音、颤音不切）。
// 纯函数，不碰 DOM。原始音高曲线一并留下（滑音、颤音是情绪，写进真值；Python 那边可以再用 crepe 重算）。
import { hzMidi, snapMelodic, snapToScale, beatStrength, chordAtBar } from './theory.js';

// ---- YIN（de Cheveigné & Kawahara 2002）：差分函数 → 累积均值归一化 → 绝对阈值 → 抛物线插值 ----
export function yin(x, sr, { fmin = 65, fmax = 1600, thresh = 0.15 } = {}) {
  const W = x.length >> 1;
  const tauMin = Math.max(2, Math.floor(sr / fmax)), tauMax = Math.min(W - 1, Math.ceil(sr / fmin));
  const d = new Float32Array(tauMax + 2);
  for (let tau = 1; tau <= tauMax + 1; tau++) {
    let s = 0;
    for (let i = 0; i < W; i++) { const v = x[i] - x[i + tau]; s += v * v; }
    d[tau] = s;
  }
  // 累积均值归一化
  let run = 0;
  const c = new Float32Array(tauMax + 2); c[0] = 1;
  for (let tau = 1; tau <= tauMax + 1; tau++) { run += d[tau]; c[tau] = run > 0 ? d[tau] * tau / run : 1; }
  let tau = -1, best = 1, bestTau = -1;
  for (let t = tauMin; t <= tauMax; t++) {
    if (c[t] < best) { best = c[t]; bestTau = t; }
    if (c[t] < thresh) {
      while (t + 1 <= tauMax && c[t + 1] < c[t]) t++;
      tau = t; break;
    }
  }
  if (tau < 0) return { f0: 0, ap: best, tau: bestTau };
  // 抛物线插值
  const a = c[tau - 1], b = c[tau], e = c[tau + 1];
  const den = a + e - 2 * b;
  const shift = den !== 0 ? 0.5 * (a - e) / den : 0;
  const t = tau + Math.max(-1, Math.min(1, shift));
  return { f0: sr / t, ap: c[tau], tau: t };
}

// 降采样到 ~16 kHz（盒式低通 + 抽取；只给音高用）
export function decimate(pcm, sr, target = 16000) {
  const k = Math.max(1, Math.round(sr / target));
  if (k === 1) return { x: pcm, sr };
  const n = Math.floor(pcm.length / k), x = new Float32Array(n);
  for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < k; j++) s += pcm[i * k + j]; x[i] = s / k; }
  return { x, sr: sr / k };
}

// 逐帧特征：时刻 / 响度 dB / 音高 midi（0 = 没音高）/ 非周期度 / 过零率
export function track(pcm, sr, { hopSec = 0.01, winSec = 0.04, fmin = 65, fmax = 3000 } = {}) {
  const { x: x0, sr: s2 } = decimate(pcm, sr);
  const hop = Math.round(hopSec * s2);
  // YIN 需要窗长 ≥ 2 × 最长周期
  const win = Math.max(Math.round(winSec * s2), 2 * Math.ceil(s2 / fmin) + 4);
  // YIN 只积分窗的前一半（x[i] − x[i+τ]，i < win/2）→ 音高的时刻 = 前一半的中心 st + win/4，不是整窗中心（晚 10 ms）。
  // 开头补 win/4 个零：第一帧的时刻 = 0，录音一开头就唱的音和中间的音一样起（不补的话开头的音晚 +15 ms）
  const pad = win >> 2, x = new Float32Array(x0.length + pad); x.set(x0, pad);
  const frames = [];
  for (let st = 0; st + win <= x.length; st += hop) {
    const fr = x.subarray(st, st + win);
    let e = 0, e1 = 0, zc = 0;
    for (let i = 0; i < win; i++) { if (i === win >> 1) e1 = e; e += fr[i] * fr[i]; if (i && (fr[i] >= 0) !== (fr[i - 1] >= 0)) zc++; }
    const rms = Math.sqrt(e / win), db = 20 * Math.log10(rms + 1e-9);
    // YIN 积分的前一半也得有声音：起音帧的前一半是静音时，静音里微弱的周期信号会被测成音高（起点早 ~25 ms）
    const db1 = 20 * Math.log10(Math.sqrt(e1 / (win >> 1)) + 1e-9);
    const y = db > -60 && db1 > -60 ? yin(fr, s2, { fmin, fmax }) : { f0: 0, ap: 1 };
    frames.push({ t: (st - pad + (win >> 2)) / s2, db, midi: y.f0 > 0 ? hzMidi(y.f0) : 0, ap: y.ap, zcr: zc / win * s2 / 2 });
  }
  return frames;
}

const median = a => { if (!a.length) return 0; const s = [...a].sort((p, q) => p - q), m = s.length >> 1; return s.length % 2 ? s[m] : 0.5 * (s[m - 1] + s[m]); };
const pct = (a, p) => { if (!a.length) return 0; const s = [...a].sort((u, v) => u - v); return s[Math.min(s.length - 1, Math.max(0, Math.round(p * (s.length - 1))))]; };

// 噪声门：以这一条录音自己的响度分布定（安静的环境 / 吵的环境都能用）
export function gateDb(frames) {
  const dbs = frames.map(f => f.db);
  // 辅音比元音低 10–15 dB，门要开在它下面：本底 + 6 dB，最多比最响处低 35 dB。
  // ⚠️整段连着哼、几乎没有停顿时「最安静的 5%」也是响的 → 门最高只到最响处往下 20 dB（不然整段都被门掉）
  const floor = pct(dbs, 0.05), top = pct(dbs, 0.95);
  return Math.min(Math.max(floor + 6, top - 35, -55), top - 20);
}

// 浊音判定 + 中值滤波 + 八度错误折回
export function cleanPitch(frames, { apMax = 0.25 } = {}) {
  const g = gateDb(frames);
  const out = frames.map(f => ({ ...f, loud: f.db > g, voiced: f.db > g && f.midi > 0 && f.ap < apMax }));
  const raw = out.map(f => f.voiced ? f.midi : 0);
  for (let i = 0; i < out.length; i++) {
    if (!out[i].voiced) continue;
    const w = [];
    for (let j = Math.max(0, i - 3); j <= Math.min(out.length - 1, i + 3); j++) if (raw[j]) w.push(raw[j]);
    const m = median(w);
    let v = raw[i];
    // 八度错误：离邻居中值差 ±12（YIN 常见）→ 折回
    while (v - m > 9) v -= 12;
    while (m - v > 9) v += 12;
    out[i].midi = v;
  }
  // 再来一遍 3 点中值，去掉单帧尖刺
  const m2 = out.map(f => f.midi);
  for (let i = 1; i < out.length - 1; i++) {
    if (out[i].voiced && out[i - 1].voiced && out[i + 1].voiced) out[i].midi = median([m2[i - 1], m2[i], m2[i + 1]]);
  }
  return { frames: out, gate: g };
}

// 整体跑调（音分）：浊音帧离最近半音的圆周平均，范围 ±50
export function tuningOffset(frames) {
  let sx = 0, sy = 0, n = 0;
  for (const f of frames) {
    if (!f.voiced) continue;
    const a = 2 * Math.PI * (f.midi - Math.round(f.midi));
    sx += Math.cos(a); sy += Math.sin(a); n++;
  }
  if (!n) return 0;
  // 滑音 / 颤音多时相位几乎均匀分布（合成长度 R = |Σe^{iφ}|/n 很小）→ 算出来的 ±50 是噪声，不扣
  if (Math.hypot(sx, sy) / n < 0.5) return 0;
  return Math.atan2(sy, sx) / (2 * Math.PI) * 100;
}

// 猜类型：人声（有词，辅音多）/ 哼唱（嗯、啦）/ 旋律（口哨、纯音）
export function classify(frames) {
  const loud = frames.filter(f => f.loud);
  if (loud.length < 5) return { mode: 'hum', why: '太短，按哼唱处理', stats: {} };
  const voiced = loud.filter(f => f.voiced);
  const vr = voiced.length / loud.length;
  const f0 = median(voiced.map(f => f.midi));
  const ap = median(voiced.map(f => f.ap));
  // 清音段（响但没音高）= 辅音；数段数，不数帧
  let bursts = 0, inB = false;
  for (const f of frames) {
    const u = f.loud && !f.voiced;
    if (u && !inB) bursts++;
    inB = u;
  }
  const dur = frames.length ? frames[frames.length - 1].t - frames[0].t : 1;
  const burstRate = bursts / Math.max(dur, 0.1);
  const stats = { voicedRatio: +vr.toFixed(2), medianMidi: +f0.toFixed(1), medianAp: +ap.toFixed(3), consonantsPerSec: +burstRate.toFixed(2) };
  if (f0 > 76 && ap < 0.08) return { mode: 'melody', why: `音高很高（中位 ${f0.toFixed(0)}）、几乎没有泛音噪声：像口哨 / 纯音`, stats };
  if (vr < 0.92 && burstRate >= 1.0) return { mode: 'voice', why: `浊音只占 ${(vr * 100).toFixed(0)}%、每秒 ${burstRate.toFixed(1)} 个辅音：像在唱词`, stats };
  return { mode: 'hum', why: `浊音占 ${(vr * 100).toFixed(0)}%、辅音少：像哼唱`, stats };
}

// 三种类型的切音参数（「要留下什么」不同）
export const MODE_PARAMS = {
  hum:    { label: '哼唱', minNote: 0.07, split: 0.75, splitHold: 0.05, gap: 0.04, scoop: 0.05, reartic: 5, consonantSplits: false },
  voice:  { label: '人声', minNote: 0.08, split: 0.9, splitHold: 0.07, gap: 0.12, scoop: 0.06, reartic: 7, consonantSplits: true },
  melody: { label: '旋律', minNote: 0.05, split: 0.6, splitHold: 0.035, gap: 0.03, scoop: 0.03, reartic: 4, consonantSplits: false },
};

// 切音符。返回秒级的音符（相对这条录音的开头），带原始音高曲线（音分，相对吸附后的音）
export function segment(frames, mode = 'hum', { tuneCents = 0 } = {}) {
  const P = MODE_PARAMS[mode] || MODE_PARAMS.hum;
  if (frames.length < 2) return [];
  const hop = frames[1].t - frames[0].t;
  const holdN = Math.max(1, Math.round(P.splitHold / hop)), gapN = Math.max(1, Math.round(P.gap / hop));
  const scoopN = Math.round(P.scoop / hop);
  const notes = [];
  const corr = f => f.midi - tuneCents / 100;
  const emit = fr => {
    // 起点 = 第一个浊音帧往前半帧；录音开头之前没有东西 → 不早于 0（终点不动）
    const t0 = fr.length ? Math.max(0, fr[0].t - hop / 2) : 0;
    const dur = fr.length ? fr[fr.length - 1].t + hop / 2 - t0 : 0;
    if (dur < P.minNote - 1e-6) return;          // 浮点：正好 minNote 长的（帧数 × 帧距算出来差 1e-16）也算
    const body = fr.length > scoopN + 3 ? fr.slice(scoopN) : fr;
    notes.push({ t: t0, dur, pitchRaw: median(body.map(corr)), db: median(fr.map(f => f.db)), zcr: median(fr.map(f => f.zcr)), curve: fr.map(f => ({ t: f.t, m: corr(f) })) });
  };
  // 同音重唱（嗯、嗯、嗯）：音高不变，响度先掉 reartic dB 再起来 → 在谷底切开
  const minN = Math.max(2, Math.round(P.minNote / hop)), look = Math.round(0.15 / hop);
  const splitReartic = fr => {
    for (let k = minN; k < fr.length - minN; k++) {
      const d = fr[k].db;
      if (d > fr[k - 1].db || d > fr[k + 1].db) continue;
      const before = Math.max(...fr.slice(Math.max(0, k - look), k).map(q => q.db));
      const after = Math.max(...fr.slice(k + 1, k + 1 + look).map(q => q.db));
      if (before - d >= P.reartic && after - d >= P.reartic) return [...splitReartic(fr.slice(0, k)), ...splitReartic(fr.slice(k))];
    }
    return [fr];
  };
  // gapAp = 这段空隙里最深的 YIN 谷（f.ap）：真辅音（噪声）≈ 0.8；元音里 YIN 断续的帧只有 0.15–0.35，不算辅音
  let cur = null, gap = 0, gapLoud = 0, gapAp = 0, dev = 0;
  const close = () => { if (cur) for (const piece of splitReartic(cur)) emit(piece); cur = null; };
  for (let i = 0; i < frames.length; i++) {
    const f = frames[i];
    if (f.voiced) {
      // 人声：中间隔了一段响的清音（辅音）→ 新音节。⚠️要至少一帧明显没周期（ap ≥ 0.4）：
      // 只数「响」的话元音里的断续也会切，碎片 < minNote 被丢 → 整句没音（真人声实测）；
      // 也不能每帧都要求 ap ≥ 0.4 —— 20 ms 的短辅音只有一帧那么深，两边的帧窗里混着元音
      if (cur && P.consonantSplits && gapLoud >= 2 && gapAp >= 0.4) close();
      if (!cur) { cur = []; dev = 0; }
      gap = 0; gapLoud = 0; gapAp = 0;
      if (cur.length > scoopN + 2) {
        // 参考音高 = 进音以后、最近 0.2 s 的中值；持续偏到同一侧才算换音（颤音一来一回不会）
        const ref = median(cur.slice(Math.max(scoopN, cur.length - 20)).map(corr));
        const side = Math.sign(corr(f) - ref);
        dev = Math.abs(corr(f) - ref) > P.split ? (dev && Math.sign(dev) === side ? dev + side : side) : 0;
        if (Math.abs(dev) >= holdN) {
          const n = Math.abs(dev) - 1, tail = cur.splice(cur.length - n, n);
          close();
          cur = tail; dev = 0;
        }
      }
      cur.push(f);
    } else if (cur) {
      gap++;
      if (f.loud) { gapLoud++; gapAp = Math.max(gapAp, f.ap); }
      const bridging = P.consonantSplits && f.loud && gap < 12;   // 辅音不算断，等下一个元音
      if (gap > gapN && !bridging) { close(); gap = 0; gapLoud = 0; gapAp = 0; }
    }
  }
  close();
  return notes;
}

// 秒级的音符 → 节拍上的音符：吸附音高（调内 + 和弦音偏好 + 惯性）、力度来自响度、亮度来自过零率
export function notesToBeats(segs, { t0Beat, spb, key, prog, beatsPerBar = 4, snap = 'scale', octaveShift = 0 }) {
  if (!segs.length) return [];
  const dbs = segs.map(s => s.db), lo = pct(dbs, 0.1), hi = pct(dbs, 0.95);
  const zs = segs.map(s => s.zcr), zlo = pct(zs, 0.1), zhi = pct(zs, 0.9);
  let prev = null;
  return segs.map(s => {
    const b = t0Beat + s.t / spb;
    const raw = s.pitchRaw + 12 * octaveShift;
    let pitch;
    if (snap === 'off') pitch = raw;
    else if (snap === 'semitone') pitch = Math.round(raw);
    else {
      const bar = Math.floor(b / beatsPerBar);
      pitch = snapMelodic(key, raw, { degree: chordAtBar(prog, bar), strong: beatStrength(Math.round(b * 4) / 4, beatsPerBar), prev });
      if (pitch === null) pitch = snapToScale(key, raw);
    }
    prev = pitch;
    const vel = hi > lo ? 0.45 + 0.55 * Math.min(1, Math.max(0, (s.db - lo) / (hi - lo))) : 0.8;
    const bright = zhi > zlo ? Math.min(1, Math.max(0, (s.zcr - zlo) / (zhi - zlo))) : 0.5;
    // 音分曲线（相对吸附后的音），每 20 ms 一个
    const cents = [];
    let nextT = s.curve.length ? s.curve[0].t : 0;
    for (const c of s.curve) if (c.t >= nextT - 1e-6) { cents.push(Math.round((c.m + 12 * octaveShift - pitch) * 100)); nextT += 0.02; }
    return { b, d: s.dur / spb, pitch, vel: +vel.toFixed(3), bright: +bright.toFixed(3), raw: { t: s.t, dur: s.dur, pitch_raw: +raw.toFixed(3), db: +s.db.toFixed(1) }, bend: { dt: 0.02, cents } };
  });
}

// 每一遍往前多读 pre 秒（预读）：只用来看清第一个音的起音（早唱一点点的强拍不再少了头）。
// segs 的时刻相对预读开头 → 换成相对这一遍开头：预读里就结束了的（上一遍的尾巴）丢掉；跨过开头的从 0 开始，
// early = 它比这一遍开头早了多少秒（早到预读开头还在响的，是上一遍拖过来的长音，不算抢拍、不标 early）
export function trimPreroll(segs, pre) {
  const out = [];
  for (const s of segs) {
    const t = s.t - pre, end = t + s.dur;
    if (end <= 0.02) continue;
    const curve = s.curve.filter(c => c.t >= pre - 1e-9).map(c => ({ ...c, t: c.t - pre }));
    if (t >= 0) { out.push({ ...s, t, curve }); continue; }
    const carried = -t >= pre - 0.02;
    out.push({ ...s, t: 0, dur: end, curve: curve.length ? curve : s.curve.slice(-1).map(c => ({ ...c, t: 0 })), ...(carried ? {} : { early: +(-t).toFixed(4) }) });
  }
  return out;
}

// 一条哼唱的完整分析
// 音高上限 3 kHz：口哨常在 1–2.5 kHz。⚠️不能先按人声范围测再看「测不出就放宽」——纯音的周期 2τ 处也是零，
// 上限太低时 YIN 会稳稳地测成低一个八度，而不是测不出（合成口哨 C7 实测）。人声的高八度误判靠 cleanPitch 折回
export const FMAX = 3000;
export function analyzeHum(pcm, sr, opts = {}) {
  const { frames, gate } = cleanPitch(track(pcm, sr, { ...opts, fmax: FMAX }));
  const guess = classify(frames);
  const mode = opts.mode && opts.mode !== 'auto' ? opts.mode : guess.mode;
  const tune = tuningOffset(frames);
  const segs = segment(frames, mode, { tuneCents: opts.snap === 'off' ? 0 : tune });
  return { frames, gate, guess, mode, tuneCents: tune, segs };
}
