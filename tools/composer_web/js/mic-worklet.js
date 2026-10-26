// >v<🎛作曲器-麦克风-每攒 1024 个样本发一次给主线程，带上第一个样本的音频时钟（录音对齐靠它）。
class MicRec extends AudioWorkletProcessor {
  constructor() { super(); this.buf = new Float32Array(1024); this.n = 0; this.t0 = 0; }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      if (this.n === 0) this.t0 = currentTime + i / sampleRate;
      this.buf[this.n++] = ch[i];
      if (this.n === this.buf.length) { this.port.postMessage({ t: this.t0, x: this.buf.slice(0) }); this.n = 0; }
    }
    return true;
  }
}
registerProcessor('mic-rec', MicRec);
