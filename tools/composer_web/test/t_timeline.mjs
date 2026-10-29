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
    curves: {}, settings: {}, songBars: 64, beatsPerBar: 4, bpm: 120, loop: { start: 0, bars: 4, on: true }, key: { root: 4, mode: 'aeolian' },
  };
  const cv = {
    style: {}, width: 0, height: 0, parentElement: { clientWidth: 1112 },
    addEventListener() {}, getContext: () => ({ setTransform() {} }),
    getBoundingClientRect: () => ({ left: 0, top: 0 }),
  };
  const tl = new Timeline(cv, { getState: () => S, shownLanes: () => new Set(), events: () => [], changed() {}, snapshot() {}, selectLayer() {} });
  tl.view = [0, 64];
  return { S, tl, cv };
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

console.log(`${n - failed}/${n} 通过`);
if (failed) process.exit(1);
