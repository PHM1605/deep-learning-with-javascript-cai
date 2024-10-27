import * as tf from "@tensorflow/tfjs";
import * as utils from "./utils";

const TRAIN_DATA = 'train-data';
const TRAIN_TARGET = 'train-target';
const TEST_DATA = 'test-data';
const TEST_TARGET = 'test-target';

export class WebsitePhishingDataset { 
  constructor() {
    this.dataset = null;
    this.trainSize = 0;
    this.testSize = 0;
    this.trainBatchIndex = 0;
    this.testBatchIndex = 0;
    this.NUM_FEATURES = 30;
    this.NUM_CLASSES = 2;
  }

  get numFeatures() {
    return this.NUM_FEATURES;
  }

  async loadData() {
    this.dataset = await Promise.all([
      utils.loadCsv(TRAIN_DATA), utils.loadCsv(TRAIN_TARGET), utils.loadCsv(TEST_DATA), utils.loadCsv(TEST_TARGET)
    ]);
    console.log(this.dataset)
  }
}