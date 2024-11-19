const DCT = require("dct");
const KissFFT = require("kissfft-js");
const SR = 16000;
const hannWindowMap = {} // {15: [3,4,1], 20: [],... }
let context;

class AudioUtils {
  startIndex = 0;
  endIndex = 0;
  bandMapper = [];
  context;

  constructor() {}

  getPeriodicHann(windowLength) {
    if (!hannWindowMap[windowLength]) {
      const window = [];
      for (let i=0; i<windowLength; ++i) {
        window[i] = 0.5 - 0.5* Math.cos((2*Math.PI*i) / windowLength);
      }
      hannWindowMap[windowLength] = window;
    }
    return hannWindowMap[windowLength];
  }

  fft(y) {
    const window = this.getPeriodicHann(y.length);
    y = y.map((v, index) => v * window[index]);
    const fftSize = nextPowerOfTwo(y.length);
    for (let i=y.length; i<fftSize; i++) {
      y[i] = 0;
    }
    const fftr = new KissFFT.FFTR(fftSize);
    const transform = fftr.forward(y);
    fftr.dispose();
    transform[fftSize] = transform[1];
    transform[fftSize+1] = 0;
    transform[1] = 0;
    return transform;
  }

  fftEnergies(y) {
    const out = new Float32Array(y.length / 2);
    for (let i=0; i<y.length; i++) {
      out[i] = y[i*2] * y[i*2] + y[i*2+1] * y[i*2+1];
    }
    return out;
  }

  createMelFilterBank(fftSize, melCount = 40, lowHz = 20, highHz = 4000, sr=SR) {
    const lowMel = this.#hzToMel(lowHz);
    const hightMel = this.#hzToMel(highHz);
    const mels = []
    const melSpan = highMel - lowMel;
    const melSpacing = melSpan / (melCount + 1);
    for (let i=0; i<melCount+1; i++) {
      mels[i] = lowMel + melSpacing * (i+1);
    }
    const hzPerSbin = 0.5 * sr / (fftSize-1);
    this.startIndex = Math.floor(1.5 + lowHz / hzPerBin);
    this.endIndex = Math.ceil(highHz/hzPerSbin);
    // Maps the input spectrum bin indices to filter bank channels/indices
    let channel = 0;
    for (let i=0; i< fftSize; i++) {
      const melf = this.#hzToMel(i * hzPerSbin);
      if (i < this.startIndex || i > this.endIndex) {
        this.bandMapper[i] = -2; // unused Fourier coefficients
      } else {
        while (mels[channel] < melf && channel < melCount) {
          ++channel;
        }
        this.bandMapper[i] = channel - 1;
      }
    }
    const weights = new Float32Array(fftSize);
    for (let i = 0; i < fftSize; ++i) {
      channel = this.bandMapper[i];
      if (i<this.startIndex || i>this.endIndex) {
        weights[i] = 0.0;
      } else {
        if (channel >= 0) {
          weights[i] = (mels[channel+1] - this.#hzToMel(i*hzPerSbin)) / (mels[channel+1] - mels[channel]);
        } else {
          weights[i] = (mels[0] - this.#hzToMel(i * hzPerSbin)) / (mels[0] - lowMel);
        }
      }
    }
    return weights;
  }

  #hzToMel(hz) {
    return 1127.0 * Math.log(1.0 + hz/700.0);
  }


}

// e.g. 8 = 2^3 => exponent = 3 => return 2^4
function nextPowerOfTwo(value) {
  const exponent = Math.ceil(Math.log2(value)); 
  return 1 << exponent;
}

module.exports = {AudioUtils, nextPowerOfTwo}