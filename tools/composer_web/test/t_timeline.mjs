// >v<🎛作曲器-自测-时间线交互（node 跑，不开浏览器）：滚轮分工（👤2026-10-29：滚轮上下 / Alt 左右 / Ctrl 以播放头缩放）、拖底边改层高。
// 用法：node tools/composer_web/test/t_timeline.mjs
import assert from 'node:assert/strict';

const winL = {};
globalThis.window = { devicePixelRatio: 1, addEventListener: (k, f) => (winL[k] ||= []).push(f) };
globalThis.ResizeObserver = class { observe() {} };
const { Timeline, rowH } = await import('../js/timeline.js');

let n = 0, failed = 0;
const t = (name, f) => { n++; try { f(); } catch (e) { failed++; console.error(`✗ ${name}\n  ${e.stack || e.message}`); } };

function make() {
  const S = {
    layers: [{ id: 'a', kind: 'synth', octave: 4 }, { id: 'b', kind: 'drums', octave: 3 }, { id: 'c', kind: 'synth', octave: 3 }],
    clips: [], notes: [], curves: {}, settings: {}, songBars: 64, beatsPerBar: 4, bpm: 120, loop: { start: 0, bars: 4, on: true }, key: { root: 4, mode: 'aeolian' },
  };
  const cv = {
    style: {}, width: 0, height: 0, parentElement: { clientWidth: 1112 },
    addEventListener() {}, getContext: () => ({ setTransform() {} }),
    getBoundingClientRect: () => ({ left: 0, top: 0 }),
  };
  const log = { changed: [], said: [], snaps: 0 };
  const tl = new Timeline(cv, {
    getState: () => S, shownLanes: () => new Set(), events: () => S.notes, snapshot() { log.snaps++; }, selectLayer() {},
    changed: (k, arg) => log.changed.push([k, arg]), say: m => log.said.push(m),
    clipGroup: c => c.input === 'stem' ? { err: '原曲分轨不能挪' } : { clips: S.clips.filter(q => q.b === c.b && q.layer === c.layer), notes: S.notes.filter(n => n.take === c.id) },
  });
  tl.view = [0, 64];
  return { S, tl, cv, log };
}
const wheel = (o = {}) => { const e = { deltaX: 0, deltaY: 100, clientX: 600, clientY: 60, ctrlKey: false, metaKey: false, altKey: false, shiftKey: false, prevented: false, ...o }; e.preventDefault = () => { e.prevented = true; }; return e; };

t('普通滚轮 = 交给页面上下滚：不拦、不动时间', () => {
  const { tl } = make(); const e = wheel();
  tl._wheel(e);
  assert.equal(e.prevented, false); assert.deepEqual(tl.view, [0, 64]);
});
t('Alt+滚轮 = 左右滚（往下滚 = 往后）；松开 Alt 那一下被拦（不跳到浏览器菜单）', () => {
  const { tl } = make(); const e = wheel({ altKey: true });
  tl._wheel(e);
  assert.equal(e.prevented, true); assert.ok(tl.view[0] > 0); assert.ok(Math.abs(tl.view[1] - tl.view[0] - 64) < 1e-9);
  const up = { key: 'Alt', prevented: false, preventDefault() { this.prevented = true; } };
  for (const f of winL.keyup) f(up);
  assert.equal(up.prevented, true);
  const up2 = { key: 'Alt', prevented: false, preventDefault() { this.prevented = true; } };
  for (const f of winL.keyup) f(up2);
  assert.equal(up2.prevented, false, '没用过 Alt+滚轮的 Alt 不拦');
});
t('Shift+滚轮（Chrome 给 deltaX）/ 触摸板横划 = 左右滚', () => {
  const { tl } = make(); tl.view = [0, 16];
  tl._wheel(wheel({ shiftKey: true, deltaX: 100, deltaY: 0 })); assert.ok(tl.view[0] > 0);
  const m = make(); m.tl.view = [0, 16]; m.tl._wheel(wheel({ deltaX: 100, deltaY: 0 })); assert.ok(m.tl.view[0] > 0);
});
t('Ctrl+滚轮 = 以播放头为中心缩放：播放头在屏幕上的位置不变；停着 = 光标；都不在画面里 = 鼠标', () => {
  const { tl } = make();
  tl.play = 40; tl._wheel(wheel({ ctrlKey: true, deltaY: -300 }));
  let [a, z] = tl.view; assert.ok(z - a < 64);
  assert.ok(Math.abs((40 - a) / (z - a) - 40 / 64) < 1e-9, '播放头在画面里的比例不变');
  tl.play = null; tl.cursor = a + (z - a) / 4; const c = tl.cursor, f = (c - a) / (z - a);
  tl._wheel(wheel({ ctrlKey: true, deltaY: -300 }));
  [a, z] = tl.view; assert.ok(Math.abs((c - a) / (z - a) - f) < 1e-9, '停着 = 光标不动');
  tl.cursor = -5; const mb = tl.xb(600), mf = (mb - a) / (z - a);
  tl._wheel(wheel({ ctrlKey: true, deltaY: -300 }));
  [a, z] = tl.view; assert.ok(Math.abs((mb - a) / (z - a) - mf) < 1e-9, '光标不在画面里 = 鼠标那里');
});
t('拖层的底边 = 改这一层高度；下面的层跟着往下；拖回默认就不记 h；有上下限', () => {
  const { S, tl, cv } = make();
  const y = tl.rowY(1);                                         // 第 1 层底边
  tl._hover({ clientX: 30, clientY: y }); assert.equal(cv.style.cursor, 'ns-resize');
  tl._hover({ clientX: 30, clientY: y + 20 }); assert.equal(cv.style.cursor, '');
  tl._down({ button: 0, clientX: 30, clientY: y });              // 左边层名那里也能拖
  tl._move({ clientX: 30, clientY: y + 40 }); tl._up();
  assert.equal(S.layers[0].h, 90); assert.equal(tl.rowY(1), y + 40);
  assert.equal(tl.rowOf(y + 30).layer.id, 'a', '变高以后那一块还是第 1 层');
  assert.equal(tl.rowOf(y + 45).layer.id, 'b');
  assert.equal(tl.height() - make().tl.height(), 40);
  tl._down({ button: 0, clientX: 600, clientY: y + 40 }); tl._move({ clientX: 600, clientY: y }); tl._up();
  assert.equal('h' in S.layers[0], false, '拖回 50 = 默认');
  tl._down({ button: 0, clientX: 600, clientY: tl.rowY(2) }); tl._move({ clientX: 600, clientY: 5000 }); tl._up();
  assert.equal(rowH(S.layers[1]), 260);
  tl._down({ button: 0, clientX: 600, clientY: tl.rowY(2) }); tl._move({ clientX: 600, clientY: -5000 }); tl._up();
  assert.equal(rowH(S.layers[1]), 28);
});
t('音符位置跟着层高缩放：高的那层音符也铺满', () => {
  const { S, tl } = make();
  S.layers[0].h = 150;
  const nt = { layer: 'a', b: 0, d: 1, pitch: 71 };
  const [, y, , h] = tl._noteRect(nt, S.layers[0], tl.rowY(0), [nt], {});
  assert.ok(y >= tl.rowY(0) && y + h <= tl.rowY(1)); assert.ok(h > (50 - 8) / 20);
});

// ---------- 👤2026-10-29：拖播放头 / 挪录音片段 / 拖文件落点 ----------
t('拖标尺上的播放头 = 挪播放头，松手才报（放着时从那里接着放）；标尺下半条单击 = 放播放头；放着时抓的是正在放的那根', () => {
  const { tl, log } = make();
  tl.cursor = 8;
  const hx = tl.bx(8);
  tl._hover({ clientX: hx, clientY: 30 }); assert.equal(tl.cv.style.cursor, 'ew-resize');
  tl._down({ button: 0, clientX: hx + 3, clientY: 30 });
  assert.equal(tl.drag.kind, 'scrub');
  tl._move({ clientX: tl.bx(20), clientY: 30 }); assert.equal(tl.cursor, 20);
  assert.equal(log.changed.length, 0, '拖的时候不跳（不然一路重启走带）');
  tl._up(); assert.deepEqual(log.changed.at(-1), ['cursor', undefined]);
  assert.equal(tl.range, null, '不是选一段');
  tl._down({ button: 0, clientX: tl.bx(40), clientY: 30 }); tl._up();
  assert.equal(tl.cursor, 40); assert.equal(log.changed.at(-1)[0], 'cursor');
  tl.play = 50;                                                   // 放着：抓手在正在放的那一拍上
  tl._down({ button: 0, clientX: tl.bx(50), clientY: 30 }); assert.equal(tl.drag.kind, 'scrub'); assert.equal(tl.cursor, 50);
  tl._move({ clientX: tl.bx(12), clientY: 30 }); tl._up(); assert.equal(tl.cursor, 12);
});
t('拖录音片段的头条 = 挪它：同一位置的几遍、它们的音和 raw、剪掉的段一起走；吸格子；头条以下照旧是音符 / 框选', () => {
  const { S, tl, log } = make();
  S.layers[0].kind = 'voice';
  S.clips.push({ id: 'C1', layer: 'a', b: 4, dur: 2, audio: 'x', input: 'hum', take: 1, active: true, gaps: [[5, 5.5]] },
    { id: 'C2', layer: 'a', b: 4, dur: 2, audio: 'y', input: 'hum', take: 2, active: false });
  S.notes.push({ id: 'n1', layer: 'a', b: 5, d: 1, pitch: 64, take: 'C1', raw: { b: 5.1, d: 1 } });
  const y0 = tl.rowY(0);
  assert.equal(tl.hitClip(tl.bx(5), y0 + 6)?.id, 'C1');
  assert.equal(tl.hitClip(tl.bx(5), y0 + 30), null, '头条以下不是片段');
  assert.equal(tl.hitClip(tl.bx(9), y0 + 6), null, '片段外面');
  tl._hover({ clientX: tl.bx(5), clientY: y0 + 6 }); assert.equal(tl.cv.style.cursor, 'grab');
  tl._down({ button: 0, clientX: tl.bx(5), clientY: y0 + 6 });
  tl._move({ clientX: tl.bx(5) + 2, clientY: y0 + 6 }); assert.equal(log.snaps, 0, '手抖不算挪');
  tl._move({ clientX: tl.bx(9.03), clientY: y0 + 6 }); tl._up();
  assert.equal(log.snaps, 1);
  assert.deepEqual(S.clips.map(c => c.b), [8, 8], '吸到 1/16 格，两遍一起');
  assert.equal(S.notes[0].b, 9); assert.ok(Math.abs(S.notes[0].raw.b - 9.1) < 1e-9);
  assert.deepEqual(S.clips[0].gaps, [[9, 9.5]]);
  assert.deepEqual(log.changed.at(-1), ['clips-done', 4]);
  tl._down({ button: 0, clientX: tl.bx(9), clientY: y0 + 6 }); tl._move({ clientX: tl.bx(-20), clientY: y0 + 6 }); tl._up();
  assert.equal(S.clips[0].b, 0, '不能挪到 0 拍前面');
  // 有片段的层：音符让出头条
  const nt = { layer: 'a', b: 0, d: 1, pitch: 80 };
  const [, ny] = tl._noteRect(nt, S.layers[0], y0, [nt], {});
  assert.ok(ny >= y0 + 13);
});
t('原曲分轨的头条：不让挪，说为什么', () => {
  const { S, tl, log } = make();
  S.clips.push({ id: 'S1', layer: 'a', b: 0, dur: 8, audio: 'x', input: 'stem', active: true });
  tl._down({ button: 0, clientX: tl.bx(3), clientY: tl.rowY(0) + 6 });
  assert.equal(tl.drag, null); assert.match(log.said[0], /原曲分轨/); assert.equal(S.clips[0].b, 0);
});
t('拖文件的落点：哪一层、哪一拍（吸格子）；在标尺 / 曲线道上 = 只给拍；画布外 = 没有', () => {
  const { tl } = make();
  const at = tl.dropAt(tl.bx(10.02), tl.rowY(1) + 20);
  assert.equal(at.layer.id, 'b'); assert.equal(at.b, 10);
  assert.equal(tl.dropAt(tl.bx(6), 10).layer, null);
  assert.equal(tl.dropAt(-5, 50), null);
});

console.log(`${n - failed}/${n} 通过`);
if (failed) process.exit(1);
