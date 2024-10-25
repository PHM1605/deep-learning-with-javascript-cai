import { BostonHousingDataset } from "./data.js";
import * as tf from "@tensorflow/tfjs";

const bostonData = new BostonHousingDataset();
const tensors = {}

export const arraysToTensors = () => {
  tensors.rawTrainFeatures = tf.tensor2d(bostonData.trainFeatures);
  tensors.trainTarget = tf.tensor2d(bostonData.trainTarget);
  tensors.rawTestFeatures = tf.tensor2d(bostonData.testFeatures);
  tensors.testTarget = tf.tensor2d(bostonData.testTarget);
}

document.addEventListener('DOMContentLoaded', async () => {
  await bostonData.loadData();
  arraysToTensors();
  console.log("TENSORS:", tensors)
}, false)