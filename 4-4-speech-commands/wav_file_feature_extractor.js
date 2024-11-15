class WavFileFeatureExtractor {
  #features;
  targetSr = 16000;
  bufferLength = 480;
  melCount = 40;
  hopLength = 160;
  duration = 1.0;
  isMfccEnabled = true; // mfcc or mel
  fftSize = 512;
  bufferCount;
  melFilterBank;

}

module.exports = {
  WavFileFeatureExtractor
}