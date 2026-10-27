// >v<🎛作曲器-时间线-全部层（音符块 / 鼓点 / 录音波形）+ 旋钮曲线。拖动 = 移、框选、删、复制一段；曲线道上直接画。
// 标尺上半条拖 = 设循环区；下半条拖 = 选一段时间（复制整段用），单击 = 放光标（粘贴的位置）。
import { MACROS, curveAt, curveWrite, loopRange, songBeats, spb } from './state.js';
import { DRUM_SOUNDS } from './engine.js';
import { inScale, chordName, chordAtBar } from './theory.js';

const GUT = 112, RULER = 38, ROW = 50, LANE = 34;

export class Timeline {
  constructor(canvas, api) {
    this.cv = canvas; this.api = api; this.g = canvas.getContext('2d');
    this.sel = new Set(); this.range = null; this.cursor = 0; this.view = null;
    this.drag = null; this.hover = null;
    canvas.addEventListener('mousedown', e => this._down(e));
    window.addEventListener('mousemove', e => this._move(e));
    window.addEventListener('mouseup', () => this._up());
    canvas.addEventListener('dblclick', e => this._dbl(e));
    canvas.addEventListener('wheel', e => this._wheel(e), { passive: false });
    canvas.addEventListener('contextmenu', e => e.preventDefault());
    new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
    this.resize();
  }
  get S() { return this.api.getState(); }
  lanes() { return MACROS.filter(m => m.id === 'tension' || (this.S.curves[m.id] && this.S.curves[m.id].length) || this.api.shownLanes().has(m.id)); }
  height() { return RULER + this.S.layers.length * ROW + this.lanes().length * LANE + 8; }
  resize() {
    const dpr = window.devicePixelRatio || 1, w = this.cv.parentElement.clientWidth, h = this.height();
    this.cv.width = w * dpr; this.cv.height = h * dpr; this.cv.style.width = w + 'px'; this.cv.style.height = h + 'px';
    this.g.setTransform(dpr, 0, 0, dpr, 0, 0); this.W = w; this.H = h;
  }
  _v() { const total = songBeats(this.S); if (!this.view || this.view[1] > total + 1e-9) this.view = [0, total]; return this.view; }
  bx(b) { const [v0, v1] = this._v(); return GUT + (b - v0) / (v1 - v0) * (this.W - GUT - 6); }
  xb(x) { const [v0, v1] = this._v(); return v0 + (x - GUT) / (this.W - GUT - 6) * (v1 - v0); }
  // 编辑吸附：0 =「自由」（不吸）；旧会话没有这个设置 → 1/16
  snapQ() { const q = this.S.settings.snapEdit ?? 0.25; return q > 0 ? q : 0; }
  snap(b, alt) { const q = this.snapQ(); return alt || !q ? b : Math.round(b / q) * q; }
  rowOf(y) {
    const S = this.S;
    if (y < RULER) return { kind: 'ruler', top: y < RULER / 2 };
    const r = Math.floor((y - RULER) / ROW);
    if (r < S.layers.length) return { kind: 'layer', i: r, layer: S.layers[r], y0: RULER + r * ROW };
    const li = Math.floor((y - RULER - S.layers.length * ROW) / LANE), lanes = this.lanes();
    if (li < lanes.length) return { kind: 'lane', macro: lanes[li], y0: RULER + S.layers.length * ROW + li * LANE };
    return { kind: 'none' };
  }
  // 一行里的音高范围（至少一个八度）
  _pitchRange(layer, notes) {
    const ps = notes.filter(n => n.layer === layer.id && !n.sound).map(n => n.pitch);
    const c = 12 * (layer.octave + 1) + this.S.key.root;
    let lo = Math.min(c - 2, ...ps), hi = Math.max(c + 12, ...ps);
    if (hi - lo < 14) { const m = (hi + lo) / 2; lo = m - 7; hi = m + 7; }
    return [lo - 1, hi + 1];
  }
  _noteRect(n, layer, y0, notes, cache) {
    const x0 = this.bx(n.b), x1 = Math.max(x0 + 3, this.bx(n.b + (n.sound ? 0.12 : n.d)));
    if (n.sound) {
      const k = Math.max(0, DRUM_SOUNDS.indexOf(n.sound)), h = (ROW - 8) / DRUM_SOUNDS.length;
      return [x0, y0 + 4 + k * h, Math.max(4, x1 - x0), Math.max(3, h - 1)];
    }
    const [lo, hi] = cache[layer.id] || (cache[layer.id] = this._pitchRange(layer, notes));
    const h = Math.max(2.5, (ROW - 8) / (hi - lo));
    const y = y0 + 4 + (hi - n.pitch) / (hi - lo) * (ROW - 8) - h / 2;
    return [x0, y, x1 - x0, h];
  }
  hitNote(x, y) {
    const row = this.rowOf(y);
    if (row.kind !== 'layer') return null;
    const notes = this.api.events(), cache = {};
    for (let i = notes.length - 1; i >= 0; i--) {
      const n = notes[i]; if (n.layer !== row.layer.id || n._preview) continue;
      const [rx, ry, rw, rh] = this._noteRect(n, row.layer, row.y0, notes, cache);
      if (x >= rx - 2 && x <= rx + rw + 2 && y >= ry - 3 && y <= ry + rh + 3) return n;
    }
    return null;
  }
  _xy(e) { const r = this.cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }

  _down(e) {
    const [x, y] = this._xy(e), S = this.S, row = this.rowOf(y);
    if (x < GUT) { if (row.kind === 'layer') this.api.selectLayer(row.layer.id); return; }
    const b = this.xb(x);
    if (row.kind === 'ruler') {
      if (row.top) { this.drag = { kind: 'loop', b0: Math.floor(b / S.beatsPerBar) }; }
      else { this.drag = { kind: 'range', b0: this.snap(b) }; this.range = null; }
      return;
    }
    if (row.kind === 'lane') {
      if (e.button === 2) { this.drag = { kind: 'erase', macro: row.macro, b0: b }; return; }
      this.api.snapshot();
      const v = this._laneV(y, row), bc = Math.max(0, Math.min(songBeats(S), b));
      S.curves[row.macro.id] = curveWrite(S.curves[row.macro.id] || [], bc, bc, v);
      (S.curvesInput ||= {})[row.macro.id] = 'draw';
      this.drag = { kind: 'draw', macro: row.macro, row, lastB: bc };
      this.api.changed('curve');
      return;
    }
    if (row.kind === 'layer') {
      this.api.selectLayer(row.layer.id, true);
      const n = this.hitNote(x, y);
      if (n) {
        if (e.shiftKey) { if (this.sel.has(n.id)) this.sel.delete(n.id); else this.sel.add(n.id); }
        else if (!this.sel.has(n.id)) { this.sel = new Set([n.id]); }
        this.drag = { kind: 'move', x0: x, y0: y, b0: b, started: false, orig: new Map(S.notes.filter(q => this.sel.has(q.id)).map(q => [q.id, { b: q.b, pitch: q.pitch, sound: q.sound }])) };
      } else {
        if (!e.shiftKey) this.sel.clear();
        this.drag = { kind: 'box', x0: x, y0: y, x1: x, y1: y };
      }
      this.api.changed('sel');
    }
  }
  _laneV(y, row) { return Math.max(0, Math.min(1, 1 - (y - row.y0 - 3) / (LANE - 6))); }
  _move(e) {
    const d = this.drag; if (!d) return;
    const [x, y] = this._xy(e), S = this.S, b = this.xb(x);
    if (d.kind === 'loop') {
      const bar = Math.floor(b / S.beatsPerBar);
      const a = Math.max(0, Math.min(d.b0, bar)), z = Math.min(S.songBars, Math.max(d.b0, bar) + 1);
      S.loop.start = a; S.loop.bars = Math.max(1, z - a); d.moved = true; this.api.changed('loop');
    } else if (d.kind === 'range') {
      const b1 = this.snap(b); this.range = [Math.min(d.b0, b1), Math.max(d.b0, b1)]; d.moved = true;
    } else if (d.kind === 'draw') {
      const row = d.row, v = this._laneV(y, row);
      // 拖得快会跳格：上一个点和这一个点之间连成直线（中间的旧点删掉）；往左画也是新点落在鼠标这里。
      // 鼠标拖出画布：钉在 0 – 歌尾（不写负拍）
      const bc = Math.max(0, Math.min(songBeats(S), b));
      S.curves[d.macro.id] = curveWrite(S.curves[d.macro.id] || [], d.lastB, bc, v);
      d.lastB = bc; this.api.changed('curve');
    } else if (d.kind === 'erase') {
      d.b1 = b;
    } else if (d.kind === 'box') {
      d.x1 = x; d.y1 = y;
    } else if (d.kind === 'move') {
      if (!d.started && Math.abs(x - d.x0) + Math.abs(y - d.y0) < 4) return;
      if (!d.started) { this.api.snapshot(); d.started = true; }
      const db = this.snap(b - d.b0, e.altKey);
      const dSteps = Math.round((d.y0 - y) / 7);
      for (const n of S.notes) {
        const o = d.orig.get(n.id); if (!o) continue;
        n.b = Math.max(0, o.b + db);
        if (n.sound) { const k = DRUM_SOUNDS.indexOf(o.sound) - dSteps; n.sound = DRUM_SOUNDS[Math.max(0, Math.min(DRUM_SOUNDS.length - 1, k))]; }
        else n.pitch = stepScale(S.key, o.pitch, dSteps);
      }
      this.api.changed('notes');
    }
  }
  _up() {
    const d = this.drag; if (!d) return;
    this.drag = null;
    const S = this.S;
    if (d.kind === 'loop' && !d.moved) { S.loop.start = Math.max(0, Math.min(S.songBars - S.loop.bars, d.b0)); this.api.changed('loop'); }
    if (d.kind === 'range' && !d.moved) { this.cursor = d.b0; this.range = null; this.api.changed('cursor'); }
    if (d.kind === 'erase' && d.b1 !== undefined) {
      this.api.snapshot();
      const [a, z] = [Math.min(d.b0, d.b1), Math.max(d.b0, d.b1)];
      S.curves[d.macro.id] = (S.curves[d.macro.id] || []).filter(p => p[0] < a || p[0] > z);
      this.api.changed('curve');
    }
    if (d.kind === 'box') {
      const [x0, x1] = [Math.min(d.x0, d.x1), Math.max(d.x0, d.x1)], [y0, y1] = [Math.min(d.y0, d.y1), Math.max(d.y0, d.y1)];
      if (x1 - x0 > 3 || y1 - y0 > 3) {
        const notes = this.api.events(), cache = {};
        S.layers.forEach((L, i) => {
          const ry = RULER + i * ROW;
          for (const n of notes) {
            if (n.layer !== L.id || n._preview) continue;
            const [rx, rY, rw, rh] = this._noteRect(n, L, ry, notes, cache);
            if (rx + rw >= x0 && rx <= x1 && rY + rh >= y0 && rY <= y1) this.sel.add(n.id);
          }
        });
      }
      this.api.changed('sel');
    }
    if (d.kind === 'move' && d.started) this.api.changed('notes-done');
  }
  _dbl(e) {
    const [x, y] = this._xy(e), row = this.rowOf(y), S = this.S;
    if (x < GUT) return;
    if (row.kind === 'layer' && !this.hitNote(x, y) && (row.layer.kind === 'synth' || row.layer.kind === 'drums')) {
      // 双击空白 = 加一个音（调内 / 鼓的那一行）
      this.api.snapshot();
      const b = this.snap(this.xb(x));
      const notes = this.api.events();
      if (row.layer.kind === 'drums') {
        const k = Math.floor((y - row.y0 - 4) / ((ROW - 8) / DRUM_SOUNDS.length));
        this.api.addNote({ layer: row.layer.id, b, d: 0.25, sound: DRUM_SOUNDS[Math.max(0, Math.min(7, k))], vel: 0.8, input: 'draw' });
      } else {
        const [lo, hi] = this._pitchRange(row.layer, notes);
        let p = Math.round(hi - (y - row.y0 - 4) / (ROW - 8) * (hi - lo));
        while (!inScale(S.key, p)) p--;
        this.api.addNote({ layer: row.layer.id, b, d: this.snapQ() || 0.25, pitch: p, vel: 0.8, input: 'draw' });   // 自由时音长默认 1/16
      }
    }
  }
  _wheel(e) {
    // 竖着滚、底栏还能往下滚（下面还有曲线道 / 场景）→ 让底栏滚；横滚 / Shift / Ctrl 才平移、缩放时间
    const box = this.cv.closest('#bottom');
    if (!e.ctrlKey && !e.metaKey && !e.shiftKey && !e.deltaX && box && box.scrollHeight > box.clientHeight + 1) {
      const down = e.deltaY > 0, atEnd = down ? box.scrollTop + box.clientHeight >= box.scrollHeight - 1 : box.scrollTop <= 0;
      if (!atEnd) return;
    }
    e.preventDefault();
    const total = songBeats(this.S), [v0, v1] = this._v(), span = v1 - v0;
    if (e.ctrlKey || e.metaKey) {
      const [x] = this._xy(e), c = this.xb(x), k = Math.exp(e.deltaY * 0.0015);
      let ns = Math.max(2, Math.min(total, span * k));
      let a = c - (c - v0) * ns / span; a = Math.max(0, Math.min(total - ns, a));
      this.view = [a, a + ns];
    } else {
      const sh = (e.deltaX || e.deltaY) / this.W * span;
      const a = Math.max(0, Math.min(total - span, v0 + sh)); this.view = [a, a + span];
    }
  }
  fit(which) { const S = this.S; if (which === 'loop') { const [a, z] = loopRange(S); this.view = [a, z]; } else this.view = [0, songBeats(S)]; }

  draw(play) {
    const S = this.S, g = this.g, W = this.W;
    if (Math.abs(this.H - this.height()) > 0.5) this.resize();
    const H = this.H, bpb = S.beatsPerBar, [v0, v1] = this._v();
    const css = getComputedStyle(document.documentElement);
    const C = k => css.getPropertyValue(k).trim();
    g.clearRect(0, 0, W, H);
    g.fillStyle = C('--panel'); g.fillRect(0, 0, W, H);
    const [ls, le] = loopRange(S);
    // 循环区底色
    g.fillStyle = C('--loop'); g.fillRect(this.bx(ls), 0, this.bx(le) - this.bx(ls), H);
    // 选中的时间段
    if (this.range) { g.fillStyle = C('--range'); g.fillRect(this.bx(this.range[0]), RULER / 2, this.bx(this.range[1]) - this.bx(this.range[0]), H); }
    // 网格
    const pxPerBeat = (W - GUT) / (v1 - v0);
    for (let b = Math.floor(v0); b <= v1; b++) {
      const x = this.bx(b), bar = b % bpb === 0;
      if (!bar && pxPerBeat < 9) continue;
      g.strokeStyle = bar ? C('--line2') : C('--line'); g.lineWidth = 1;
      g.beginPath(); g.moveTo(x + 0.5, RULER / 2); g.lineTo(x + 0.5, H); g.stroke();
    }
    // 标尺
    g.fillStyle = C('--bg'); g.fillRect(GUT, 0, W - GUT, RULER);
    g.fillStyle = C('--g'); g.globalAlpha = 0.85; g.fillRect(this.bx(ls), 3, this.bx(le) - this.bx(ls), 9); g.globalAlpha = 1;
    g.font = '11px Consolas, monospace'; g.textBaseline = 'middle';
    for (let bar = Math.floor(v0 / bpb); bar * bpb <= v1; bar++) {
      const x = this.bx(bar * bpb);
      g.fillStyle = C('--dim'); g.fillText(String(bar + 1), x + 3, RULER - 12);
      if (pxPerBeat * bpb > 60) { g.fillStyle = C('--o'); g.fillText(chordName(S.key, chordAtBar(S.prog, bar)), x + 22, RULER - 12); }
    }
    for (const sc of S.scenes) {
      const x = this.bx(sc.bar * bpb);
      g.fillStyle = C('--o'); g.fillRect(x, 0, 2, H);
      g.fillStyle = C('--o'); g.fillText('▸' + sc.name, x + 4, 8);
    }
    // 层
    const notes = this.api.events(), cache = {};
    const peaks = this.api.peaks;
    S.layers.forEach((L, i) => {
      const y0 = RULER + i * ROW;
      g.fillStyle = L.id === S.sel ? C('--selrow') : 'transparent'; g.fillRect(0, y0, W, ROW);
      g.strokeStyle = C('--line'); g.beginPath(); g.moveTo(0, y0 + ROW - 0.5); g.lineTo(W, y0 + ROW - 0.5); g.stroke();
      // 录音波形（挑 take 时画正在试听的那一条，和音符对得上）
      for (const c of (this.api.clips ? this.api.clips() : S.clips)) {
        if (c.layer !== L.id || !c.active) continue;
        const pk = peaks(c.audio); if (!pk) continue;
        const x0 = this.bx(c.b), x1 = this.bx(c.b + c.dur / spb(S));
        g.fillStyle = L.color; g.globalAlpha = 0.22;
        const n = pk.length, mid = y0 + ROW / 2;
        for (let px = Math.max(GUT, Math.floor(x0)); px < Math.min(W, x1); px++) {
          const k = Math.floor((px - x0) / (x1 - x0) * n), a = pk[k] || 0;
          g.fillRect(px, mid - a * (ROW / 2 - 3), 1, 2 * a * (ROW / 2 - 3) + 1);
        }
        g.globalAlpha = 1;
      }
      for (const n of notes) {
        if (n.layer !== L.id) continue;
        if (n.b + (n.d || 0.1) < v0 || n.b > v1) continue;
        const [x, y, w, h] = this._noteRect(n, L, y0, notes, cache);
        if (x + w < GUT) continue;
        g.globalAlpha = n._preview ? 0.95 : (n._ghost ? 0.25 : 0.35 + 0.65 * (n.vel ?? 0.8));
        if (n._preview) { g.strokeStyle = C('--ink'); g.lineWidth = 1.2; g.strokeRect(x + 0.5, y + 0.5, w, h); }
        else { g.fillStyle = this.sel.has(n.id) ? C('--ink') : L.color; g.fillRect(x, y, w, h); }
        g.globalAlpha = 1;
      }
      // 左边的层名
      g.fillStyle = C('--bg'); g.fillRect(0, y0, GUT - 4, ROW - 1);
      g.fillStyle = L.color; g.fillRect(0, y0, 3, ROW - 1);
      g.fillStyle = L.mute ? C('--dim') : C('--ink'); g.font = '13px "Microsoft YaHei", sans-serif';
      g.fillText(`${i + 1} ${L.name}`, 10, y0 + 16);
      g.fillStyle = C('--dim'); g.font = '11px Consolas, monospace';
      g.fillText(`${L.kind === 'synth' ? L.preset : L.kind}${L.mute ? ' · 静音' : ''}${L.solo ? ' · 独奏' : ''}`, 10, y0 + 34);
    });
    // 曲线道
    const lanes = this.lanes(), ly0 = RULER + S.layers.length * ROW;
    lanes.forEach((m, k) => {
      const y0 = ly0 + k * LANE, pts = S.curves[m.id] || [];
      g.strokeStyle = C('--line'); g.beginPath(); g.moveTo(0, y0 + LANE - 0.5); g.lineTo(W, y0 + LANE - 0.5); g.stroke();
      g.strokeStyle = m.id === 'tension' ? C('--o') : C('--g'); g.lineWidth = 1.6; g.beginPath();
      for (let px = GUT; px < W; px += 2) {
        const v = curveAt(pts, this.xb(px), m.def), y = y0 + 3 + (1 - v) * (LANE - 6);
        if (px === GUT) g.moveTo(px, y); else g.lineTo(px, y);
      }
      g.globalAlpha = pts.length ? 1 : 0.35; g.stroke(); g.globalAlpha = 1;
      g.fillStyle = C('--bg'); g.fillRect(0, y0, GUT - 4, LANE - 1);
      g.fillStyle = m.id === 'tension' ? C('--o') : C('--dim'); g.font = '12px "Microsoft YaHei", sans-serif';
      g.fillText(`${m.label}${pts.length ? '' : '（默认）'}`, 10, y0 + LANE / 2);
    });
    // 框选
    if (this.drag && this.drag.kind === 'box') {
      const d = this.drag; g.strokeStyle = C('--ink'); g.setLineDash([4, 3]);
      g.strokeRect(Math.min(d.x0, d.x1), Math.min(d.y0, d.y1), Math.abs(d.x1 - d.x0), Math.abs(d.y1 - d.y0)); g.setLineDash([]);
    }
    // 光标（粘贴位置）+ 播放头
    g.fillStyle = C('--dim'); g.fillRect(this.bx(this.cursor), RULER / 2, 1, H);
    if (play !== null && play !== undefined) { g.fillStyle = C('--ink'); g.fillRect(this.bx(play), 0, 1.5, H); }
  }
}

// 调内走 k 级（正 = 上）
export function stepScale(key, m, k) {
  let p = m;
  const dir = Math.sign(k);
  for (let i = 0; i < Math.abs(k); i++) { do { p += dir; } while (!inScale(key, p)); }
  return p;
}
