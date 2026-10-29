// >v<🎛作曲器-自测-页面骨架（node 跑，不开浏览器）：main.js 要的 id 都在、定高布局的几条关键规则还在。
// 用法：node tools/composer_web/test/layout.mjs
// 真正的像素检查（1440×900 键盘和时间线同屏、1100×800 XY 不放大）要在浏览器里量，这里只守住规则别被改丢（C39 / C42）。
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { keycapHTML } from '../js/keycaps.js';

const here = new URL('..', import.meta.url);
const html = readFileSync(new URL('index.html', here), 'utf8');
const css = readFileSync(new URL('style.css', here), 'utf8').replace(/\/\*[\s\S]*?\*\//g, '');
const mainJs = readFileSync(new URL('js/main.js', here), 'utf8');

let n = 0, failed = 0;
const t = (name, f) => { n++; try { f(); } catch (e) { failed++; console.error(`✗ ${name}\n  ${e.message}`); } };

// 顶层规则（不进 @media）里某个选择器的声明；@media 块单独取
const stripMedia = s => { let out = '', i = 0; for (;;) { const k = s.indexOf('@media', i); if (k < 0) return out + s.slice(i); out += s.slice(i, k); let d = 0, j = s.indexOf('{', k); for (; j < s.length; j++) { if (s[j] === '{') d++; else if (s[j] === '}' && --d === 0) break; } i = j + 1; } };
const top = stripMedia(css);
const decls = sel => [...top.matchAll(/([^{}]+)\{([^}]*)\}/g)].filter(m => m[1].split(',').map(x => x.trim()).includes(sel)).map(m => m[2]).join(';');
const media = q => { const k = css.indexOf(q); assert.ok(k >= 0, `缺 ${q}`); let d = 0, j = css.indexOf('{', k); const a = j; for (; j < css.length; j++) { if (css[j] === '{') d++; else if (css[j] === '}' && --d === 0) break; } return css.slice(a + 1, j); };
const px = (s, prop) => { const m = s.match(new RegExp(`(?:^|[;\\s])${prop}\\s*:\\s*([^;]+)`)); return m ? m[1].trim() : null; };

t('main.js 用到的 id 都在 index.html 里', () => {
  const want = new Set([...mainJs.matchAll(/\$\('([A-Za-z_][\w-]*)'\)/g)].map(m => m[1]));
  const have = new Set([...html.matchAll(/\sid="([^"]+)"/g)].map(m => m[1]));
  assert.ok(want.size > 50, `只找到 ${want.size} 个 id，正则失效？`);
  assert.deepEqual([...want].filter(i => !have.has(i)), []);
});
t('时间线画布的父元素是 #tlWrap（timeline.js 用 parentElement 量宽）', () => {
  assert.match(html, /<div id="tlWrap"><canvas id="tl"><\/canvas><\/div>/);
});
t('整页定高：body 100vh 不滚，中间带 flex 撑满、时间线带有上限且自己滚', () => {
  const b = decls('body');
  assert.equal(px(b, 'height'), '100vh'); assert.equal(px(b, 'overflow'), 'hidden'); assert.equal(px(b, 'flex-direction'), 'column');
  const m = decls('main');
  assert.match(px(m, 'flex'), /^1 1 0/); assert.ok(px(m, 'min-height'), 'main 要有最小高度（键盘不被挤没）');
  const bot = decls('#bottom');
  assert.ok(px(bot, 'max-height'), '#bottom 要有上限'); assert.equal(px(bot, 'overflow-y'), 'auto');
});
t('三栏各自滚：旋钮、中栏内容、右栏', () => {
  assert.equal(px(decls('#params'), 'overflow-y'), 'auto');
  assert.equal(px(decls('.cbody'), 'overflow-y'), 'auto');
  assert.equal(px(decls('#right'), 'overflow-y'), 'auto');
  assert.match(html, /<div class="cbody">[\s\S]*id="keys"[\s\S]*id="variants"[\s\S]*id="hum"[\s\S]*<\/div>\s*<div id="status"><\/div>/);
});
t('XY 和示波器封顶 280（= 画布像素，不放大），XY 保持正方形', () => {
  const xy = decls('#xy'), sc = decls('#scope');
  assert.match(xy, /aspect-ratio:\s*1\b/);
  assert.match(xy, /max-width:\s*280px/); assert.match(xy, /clamp\([^;]*280px\)/);
  assert.equal(px(sc, 'max-width'), '280px');
  assert.match(html, /id="xy" width="280" height="280"/); assert.match(html, /id="scope" width="280" height="200"/);
});
t('窄 / 矮屏退回整页滚，但画布照样封顶', () => {
  const q = media('@media (max-width: 999px), (max-height: 559px)');
  assert.match(q, /body\s*\{[^}]*height:\s*auto[^}]*overflow:\s*visible/);
  assert.match(q, /#xy\s*\{[^}]*max-width:\s*280px/);
  assert.match(q, /#right\s*\{[^}]*container-type:\s*normal/);
});
t('键帽（照 Synapsis）：修饰键各有颜色、鼠标动作是虚线框、「/」是「或」；键位页的每个键都能画', () => {
  const h = keycapHTML('Ctrl+Shift+Z');
  assert.match(h, /class="k ctrl">Ctrl<[\s\S]*class="k shift">Shift<[\s\S]*class="k ">Z</);
  assert.match(keycapHTML('Alt+滚轮'), /class="k alt">Alt<[\s\S]*class="k mouse">滚轮</);
  assert.match(keycapHTML('← / →'), /class="k arrow">←<\/kbd><span class="ksp"><\/span><span class="kor">\/<\/span>/);
  assert.match(keycapHTML('Space'), /class="k space"/); assert.match(keycapHTML('Tab'), /class="k tab"/);
  assert.match(keycapHTML('A S D F … L'), /<span class="kop">…<\/span>/);
  assert.match(keycapHTML('<b>'), /&lt;b&gt;/);
  const kx = [...html.matchAll(/class="kx">([^<]+)</g)];
  assert.ok(kx.length > 40, `键位页只找到 ${kx.length} 个键`);
  for (const css of ['.kcombo .k.ctrl', '.kcombo .k.shift', '.kcombo .k.alt', '.kcombo .k.tab', '.kcombo .k.space']) assert.ok(top.includes(css), `缺 ${css}`);
});

console.log(`${n - failed}/${n} 通过`);
if (failed) process.exit(1);
