import * as tf from "@tensorflow/tfjs";
import * as ui from "./ui";
import { IMAGE_H, IMAGE_W, MnistData } from "./data";

let data;
async function load() {
  data = new MnistData();
  await data.load();
}

function createConvModel() {
  const model = tf.sequential();
  model.add(tf.layers.conv2d({inputShape: [IMAGE_H, IMAGE_W, 1], kernelSize: 3, filters: 16, activation: 'relu'}));
  model.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));
  model.add(tf.layers.conv2d({kernelSize: 3, filters: 32, activation: 'relu'}));
  model.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));
  model.add(tf.layers.conv2d({kernelSize: 3, filters: 32, activation: 'relu'}));
  model.add(tf.layers.flatten({}));
  model.add(tf.layers.dense({units: 64, activation:'relu'}));
  model.add(tf.layers.dense({units:10, activation:'softmax'}));
  return model;
}

// only to demonstrate a worse performing model, with the same number of parameters
function createDenseModel() {
  const model = tf.sequential();
  model.add(tf.layers.flatten({inputShape: [IMAGE_H, IMAGE_W, 1]}));
  model.add(tf.layers.dense({units: 42, activation: 'relu'}));
  model.add(tf.layers.dense({units: 10, activation: 'softmax'}));
  return model;
}

async function train(model, onIteration) {
  ui.logStatus("Training model...");
  const optimizer = 'rmsprop';
  model.compile({optimizer, loss:'categoricalCrossentropy', metrics:['accuracy']});
  const batchSize = 320;
  const validationSplit = 0.15;
  const trainEpochs = ui.getTrainEpochs();
  let trainBatchCount = 0; // current training batch

  const trainData = data.getTrainData();
  const testData = data.getTestData();
  // total number of batches summing from all epochs
  const totalNumBatches = Math.ceil(trainData.xs.shape[0] * (1-validationSplit) / batchSize) * trainEpochs;
  let valAcc;
  await model.fit(trainData.xs, trainData.labels, {batchSize, validationSplit, epochs: trainEpochs, callbacks: {
    onBatchEnd: async (batch, logs) => {
      trainBatchCount++;
      ui.logStatus(`Training... (${(trainBatchCount/totalNumBatches*100).toFixed(1)}% completed). To stop training, refresh or close page.`);
      ui.plotLoss(trainBatchCount, logs.loss, 'train');
      ui.plotAccuracy(trainBatchCount, logs.acc, 'train');
      if (onIteration && batch%10===0) {
        onIteration();
      }
      await tf.nextFrame();
    },
    onEpochEnd: async (epoch, logs) => {
      valAcc = logs.val_acc;
      ui.plotLoss(trainBatchCount, logs.val_loss, 'validation');
      ui.plotAccuracy(trainBatchCount, logs.val_acc, 'validation');
      if (onIteration) {
        onIteration();
      }
      await tf.nextFrame();
    }
  }});
  const testResult = model.evaluate(testData.xs, testData.labels);
  const testAccPercent = testResult[1].dataSync()[0] * 100;
  const finalValAccPercent = valAcc * 100;
  ui.logStatus(`Final validation accuracy: ${finalValAccPercent.toFixed(1)}%; Final test accuracy: ${testAccPercent.toFixed(1)}%`);
}

async function showPredictions(model) {
  const testExamples = 10;
  const examples = data.getTestData(testExamples);
  tf.tidy(() => {
    const output = model.predict(examples.xs);
    const axis = -1;
    const labels = Array.from(examples.labels.argMax(axis).dataSync());
    const predictions = Array.from(output.argMax(axis).dataSync());
    ui.showTestResults(examples, predictions, labels);
  })
}

function createModel() {
  let model;
  const modelType = ui.getModelTypeId();
  if (modelType === "ConvNet") {
    console.log("!")
    model = createConvModel();
  } else if (modelType === "DenseNet") {
    console.log("@")
    model = createDenseModel();
  } else {
    throw new Error(`Invalid model type: ${modelType}`);
  }
  return model;
}

ui.setTrainButtonCallback(async ()=>{
  ui.logStatus("Loading MNIST data...");
  await load();
  ui.logStatus('Creating model...');
  const model = createModel();
  model.summary();
  ui.logStatus('Starting model training...');
  await train(model, ()=>showPredictions(model));
})