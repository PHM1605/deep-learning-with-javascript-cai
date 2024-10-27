import { BostonHousingDataset, featureDescriptions } from "./data.js";
import * as tf from "@tensorflow/tfjs";
import * as normalization from "./normalization.js";
import * as ui from "./ui.js";

const NUM_EPOCHS = 200;
const BATCH_SIZE = 40;
const LEARNING_RATE = 0.01;

const bostonData = new BostonHousingDataset();
const tensors = {}

export const arraysToTensors = () => {
  tensors.rawTrainFeatures = tf.tensor2d(bostonData.trainFeatures);
  tensors.trainTarget = tf.tensor2d(bostonData.trainTarget);
  tensors.rawTestFeatures = tf.tensor2d(bostonData.testFeatures);
  tensors.testTarget = tf.tensor2d(bostonData.testTarget);
  let {dataMean, dataStd} = normalization.determineMeanAndStddev(tensors.rawTrainFeatures);
  tensors.trainFeatures = normalization.normalizeTensor(tensors.rawTrainFeatures, dataMean, dataStd);
  tensors.testFeatures = normalization.normalizeTensor(tensors.rawTestFeatures, dataMean, dataStd);
}

export function linearRegressionModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({inputShape:[bostonData.numFeatures], units:1}));
  model.summary();
  return model;
}

export function multiLayerPerceptronRegressionModel1Hidden() {
  const model = tf.sequential();
  model.add(tf.layers.dense({
    inputShape: [bostonData.numFeatures],
    units: 50, 
    activation: 'sigmoid',
    kernelInitializer: 'leCunNormal'
  }));
  model.add(tf.layers.dense({units: 1}));
  model.summary();
  return model;
}

export function multiLayerPerceptronRegressionModel2Hidden() {
  const model = tf.sequential();
  model.add(tf.layers.dense({
    inputShape: [bostonData.numFeatures], 
    units: 50, 
    activation: 'sigmoid', 
    kernelInitializer: 'leCunNormal'
  }));
  model.add(tf.layers.dense({
    units: 50,
    activation: 'sigmoid',
    kernelInitializer: 'leCunNormal'
  }));
  model.add(tf.layers.dense({units: 1}));
  model.summary();
  return model;
}

export function describeKernelElements(kernel) {
  tf.util.assert(kernel.length === 12, `kernel must be an array of length 12, got ${kernel.length}`);
  const outList = [];
  for (let idx = 0; idx < kernel.length; idx++) {
    outList.push({description: featureDescriptions[idx], value: kernel[idx]})
  }
  return outList;
}

export async function run(model, modelName, weightsIllustration) {
  model.compile({
    optimizer: tf.train.sgd(LEARNING_RATE),
    loss: 'meanSquaredError'
  });
  let trainLogs = [];
  const container = document.querySelector(`#${modelName} .chart`);
  ui.updateStatus('Starting training process...');
  await model.fit(tensors.trainFeatures, tensors.trainTarget, {
    batchSize: BATCH_SIZE,
    epochs: NUM_EPOCHS, 
    validationSplit: 0.2,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        await ui.updateModelStatus(`Epoch ${epoch+1} of ${NUM_EPOCHS} completed.`, modelName);
        trainLogs.push(logs);
        ui.plotData(container, NUM_EPOCHS, trainLogs);
        if (weightsIllustration) {
          model.layers[0].getWeights()[0].data().then(kernelAsArr => {
            const weightsList = describeKernelElements(kernelAsArr);
            ui.updateWeightDescription(weightsList);
          })
        }
      }
    }
  });

  ui.updateStatus('Running on test data...')
  const result = model.evaluate(tensors.testFeatures, tensors.testTarget, {batchSize: BATCH_SIZE});
  const testLoss = result.dataSync()[0];
  const trainLoss = trainLogs[trainLogs.length - 1].loss;
  const valLoss = trainLogs[trainLogs.length - 1].val_loss;
  await ui.updateModelStatus(`Final train-set loss: ${trainLoss.toFixed(4)}\nFinal validation-set loss: ${valLoss.toFixed(4)}\nTest-set loss: ${testLoss.toFixed(4)}`, modelName);
}

export function computeBaseline() {
  const avgPrice = tensors.trainTarget.mean();
  console.log(`Average price: ${avgPrice.dataSync()}`);
  const baseline = tensors.testTarget.sub(avgPrice).square().mean();
  console.log(`Baseline loss: ${baseline.dataSync()[0]}`);
  const baselineMsg = `Baseline loss (mean squared error) is ${baseline.dataSync()[0].toFixed(2)}`;
  ui.updateBaselineStatus(baselineMsg);
}

document.addEventListener('DOMContentLoaded', async () => {
  await bostonData.loadData();
  ui.updateStatus('Data loaded, converting to tensors');
  arraysToTensors();
  ui.updateStatus('Data is now available as tensors.\nClick a train button to begin.');
  ui.updateBaselineStatus('Estimating baseline loss');
  computeBaseline();
  await ui.setup();
}, false)