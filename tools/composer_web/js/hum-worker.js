// >v<🎛作曲器-哼唱识别-后台线程：analyzeHum 搬出主线程（16 小节 / 慢速的一遍要分析几百 ms，会卡住走带的 120 ms 提前量）。
// pitch.js / theory.js 是纯函数，直接 import。收 { id, pcm, sr, opts }，回 { id, an }（不带逐帧数据，太大又用不上）或 { id, err }。
import { analyzeHum } from './pitch.js';

self.onmessage = ev => {
  const { id, pcm, sr, opts } = ev.data;
  try {
    const { frames, ...an } = analyzeHum(pcm, sr, opts);
    self.postMessage({ id, an });
  } catch (e) {
    self.postMessage({ id, err: String((e && e.message) || e) });
  }
};
