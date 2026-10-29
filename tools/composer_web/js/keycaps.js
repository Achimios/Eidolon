// 键帽：照 Synapsis 快捷页（shell.css 的 .kcombo）——每个键一个带色外框，Ctrl 蓝 / Shift 黄 / Alt 红 / Tab 绿 / 方向键青
// 写法：<span class="kx">Ctrl+Shift+Z</span>；空格分开 = 几个键并排；「/」= 或；鼠标动作（滚轮、拖、双击……）画成虚线框
const KIND = {
  ctrl: 'ctrl', shift: 'shift', alt: 'alt', tab: 'tab', space: 'space', 空格: 'space',
  enter: 'act', esc: 'act', delete: 'act', '←': 'arrow', '→': 'arrow', '↑': 'arrow', '↓': 'arrow',
};
const MOUSE = /^(滚轮|拖|单击|双击|右键拖|点)/;
const esc = s => s.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

export function keycapHTML(str) {
  return '<span class="kcombo">' + str.trim().split(/\s+/).map(tok => {
    if (tok === '/') return '<span class="kor">/</span>';
    if (tok === '–' || tok === '…') return `<span class="kop">${tok}</span>`;
    const parts = tok.length > 1 ? tok.split('+') : [tok];
    return parts.map(p => {
      const cls = MOUSE.test(p) ? 'mouse' : KIND[p.toLowerCase()] || '';
      return `<kbd class="k ${cls}">${esc(p)}</kbd>`;
    }).join('<span class="kop">+</span>');
  }).join('<span class="ksp"></span>') + '</span>';
}

export function paintKeycaps(root = document) {
  for (const el of root.querySelectorAll('.kx:not(.kdone)')) { el.innerHTML = keycapHTML(el.textContent); el.classList.add('kdone'); }
}
