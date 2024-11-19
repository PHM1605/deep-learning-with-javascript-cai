const { AudioUtils, nextPowerOfTwo } = require("./utils/audio_utils");

class WavFileFeatureExtractor {
  #features;
  targetSr = 16000;
  bufferLength = 480; // how many samples per window
  melCount = 40;
  hopLength = 160; // how much one window overlap with the previous
  duration = 1.0;
  isMfccEnabled = true; // mfcc or mel
  fftSize = 512;
  bufferCount;
  melFilterBank;
  audioUtils = new AudioUtils();

  config(params) {
    Object.assign(this, params);
    this.bufferCount = Math.floor((this.duration * this.targetSr - this.bufferLength) / this.hopLength) + 1;
    if (this.hopLength > this.bufferLength) {
      console.error("Hop length must be smaller than buffer length.");
    }
    // Mel filterbank size = half the number of samples, as fft array is complex value
    this.fftSize = nextPowerOfTwo(this.bufferLength);
    this.melFilterBank = this.audioUtils.createMelFilterBank(this.fftSize/2+1, this.melCount);
    console.log("AFTER: ", this.melFilterBank)
  }

  // samples: 1 sound file
  start(samples) {
    this.#features = [];
    const buffers = this.#getFullBuffers(samples);
    // buffers: list of windows of 1 sound file; each window has length of 480 samples
    for (const buffer of buffers) {
      const fft = this.audioUtils.fft(buffer); // Float32Array of 513 elements
      const fftEnergies = this.audioUtils.fftEnergies(fft); // Float32Array of 513/2=257 elements
      const melEnergies = this.audioUtils.applyFilterBank(fftEnergies, this.melFilterBank)
      break
    }
    return this.#features;
  }

  // return list of windows from 1 sound file
  #getFullBuffers(sample) {
    const out = [];
    let index = 0;
    while (index <= sample.length - this.bufferLength) {
      const buffer = sample.slice(index, index + this.bufferLength);
      index += this.hopLength;
      out.push(buffer);
    }
    return out;
  }
}

module.exports = {
  WavFileFeatureExtractor
}