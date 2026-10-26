// 测试用的合成「人声」：谐波 + 共振峰感的滚降 + 可选滑音 / 颤音 / 辅音噪声
export function voiceLike(notes, sr = 48000, { vib = 0, scoop = 0, noise = 0.002, consonant = 0, seed = 1, gapSec = 0 } = {}) {
  let rnd = seed;
  const rand = () => { rnd = (rnd * 1103515245 + 12345) & 0x7fffffff; return rnd / 0x7fffffff - 0.5; };
  const total = notes.reduce((s, n) => s + n.dur + gapSec, 0) + 0.3;
  const x = new Float32Array(Math.ceil(total * sr));
  let t0 = 0.15, ph = 0;
  for (const n of notes) {
    const N = Math.floor(n.dur * sr), C = Math.floor(consonant * sr);
    for (let i = -C; i < 0; i++) { const k = Math.floor(t0 * sr) + i; if (k >= 0) x[k] += 0.15 * rand() * 2; }
    for (let i = 0; i < N; i++) {
      const t = i / sr;
      let m = n.midi + (n.cents || 0) / 100;
      if (scoop && t < 0.06) m -= scoop * (1 - t / 0.06);
      if (vib) m += vib * Math.sin(2 * Math.PI * 5.5 * t);
      const f = 440 * Math.pow(2, (m - 69) / 12);
      ph += 2 * Math.PI * f / sr;
      const env = Math.min(1, i / (0.01 * sr), (N - i) / (0.01 * sr)) * (n.amp || 0.3);
      let s = 0;
      for (let h = 1; h <= 8; h++) s += Math.sin(h * ph) / (h * h * 0.6 + 0.4);
      x[Math.floor(t0 * sr) + i] += env * s;
    }
    t0 += n.dur + gapSec;
  }
  for (let i = 0; i < x.length; i++) x[i] += noise * rand();
  return x;
}
