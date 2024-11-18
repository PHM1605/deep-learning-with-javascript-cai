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
  }
}

module.exports = {
  WavFileFeatureExtractor
}