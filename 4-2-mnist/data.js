import * as tf from '@tensorflow/tfjs';
export const IMAGE_H = 28;
export const IMAGE_W = 28;
const IMAGE_SIZE = IMAGE_H * IMAGE_W;
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 65000;
const NUM_TRAIN_ELEMENTS = 55000;
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;
const MNIST_IMAGES_SPRITE_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const MNIST_LABELS_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

export class MnistData {
  constructor() {}
  async load() {
    const img = new Image();
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const imgRequest = new Promise((resolve, reject) => {
      img.crossOrigin = '';
      img.onload = () => {
        img.width = img.naturalWidth; // 784 = 28*28
        img.height = img.naturalHeight; // 65000 = total number of samples
        const datasetBytesBuffer = new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4); // "RGBA"; if grayscale then we only use "R" channel
        const chunkSize = 5000;
        canvas.width = img.width; // 784 = 28*28
        canvas.height = chunkSize; // 5000 = number of samples in 1 batch
        for (let i=0; i<NUM_DATASET_ELEMENTS/chunkSize; i++) {
          const datasetBytesView = new Float32Array(datasetBytesBuffer, i*IMAGE_SIZE*4*chunkSize, IMAGE_SIZE*chunkSize); // 5000*28*28
          // first 4 integers: source image; last 4 integers: canvas
          ctx.drawImage(img, 0, i*chunkSize, img.width, chunkSize, 0, 0, img.width, chunkSize);
          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height); // 5000*28*28*4
          // grayscale => we use only the "R channel"
          for (let j=0; j<imageData.data.length/4; j++) {
            datasetBytesView[j] = imageData.data[j*4] /255;
          }
        }
        this.datasetImages = new Float32Array(datasetBytesBuffer); // 65000*28*28
        resolve();
      };
      img.src = MNIST_IMAGES_SPRITE_PATH;
    });

    const labelsRequest = fetch(MNIST_LABELS_PATH);
    const [ imgResponse, labelsResponse ] = await Promise.all([imgRequest, labelsRequest])
    this.datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer());
    this.trainImages = this.datasetImages.slice(0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS); // 55000*28*28
    this.testImages = this.datasetImages.slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS); // 10000*28*28
    this.trainLabels = this.datasetLabels.slice(0, NUM_CLASSES * NUM_TRAIN_ELEMENTS); // 55000*10
    this.testLabels = this.datasetLabels.slice(NUM_CLASSES * NUM_TRAIN_ELEMENTS); // 10000*10
  }

  getTrainData() {
    const xs = tf.tensor4d(this.trainImages, [this.trainImages.length/IMAGE_SIZE, IMAGE_H, IMAGE_W, 1]);
    const labels = tf.tensor2d(this.trainLabels, [this.trainLabels.length/NUM_CLASSES, NUM_CLASSES]);
    return {xs, labels};
  }
  // if numExamples == null we take all test data
  getTestData(numExamples) {
    let xs = tf.tensor4d(this.testImages, [this.testImages.length / IMAGE_SIZE, IMAGE_H, IMAGE_W, 1]);
    let labels = tf.tensor2d(this.testLabels, [this.testLabels.length / NUM_CLASSES, NUM_CLASSES]);
    if (numExamples) {
      xs = xs.slice([0,0,0,0], [numExamples, IMAGE_H, IMAGE_W, 1]);
      labels = labels.slice([0,0], [numExamples, NUM_CLASSES]);
    }
    return {xs, labels};
  }
}