import * as tf from "@tensorflow/tfjs";
import { WebsitePhishingDataset } from "./data";
import * as ui from "./ui"
import * as utils from "./utils"

const epochs = 400;
const batchSize = 350;

function falsePositives(yTrue, yPred) {
  return tf.tidy(() => {
    const one = tf.scalar(1);
    const zero = tf.scalar(0);
    return tf.logicalAnd(yTrue.equal(zero), yPred.equal(one)).sum().cast('float32');
  })
}

function trueNegatives(yTrue, yPred) {
  return tf.tidy(()=>{
    const zero = tf.scalar(0);
    return tf.logicalAnd(yTrue.equal(zero), yPred.equal(zero)).sum().cast('float32');
  })
}

function falsePositiveRate(yTrue, yPred) {
  return tf.tidy(()=>{
    const fp = falsePositives(yTrue, yPred);
    const tn = trueNegatives(yTrue, yPred);
    return fp.div(fp.add(tn));
  })
}

function drawROC(targets, probs, epoch) {
  return tf.tidy(() => {
    const thresholds = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0];
    const tprs = [], fprs = [];
    let area = 0;
    for (let i=0; i<thresholds.length; ++i) {
      const threshold = thresholds[i];
      const threshPredictions = utils.binarize(probs, threshold).as1D();
      const fpr = falsePositiveRate(targets, threshPredictions).dataSync()[0];
      const tpr = tf.metrics.recall(targets, threshPredictions).dataSync()[0];
      fprs.push(fpr);
      tprs.push(tpr);
      // accumulate to area for AUC calculation
      if (i>0) {
        area += 1/2 * (tprs[i] + tprs[i] * (fprs[i-1] - fprs[i]))
      }
    }
    ui.plotROC(fprs, tprs, epoch);
    return area;
  });
}

const data = new WebsitePhishingDataset();
data.loadData().then(async () => {
  await ui.updateStatus("Getting training and testing data...");
  const trainData = data.getTrainData();
  const testData = data.getTestData();
  await ui.updateStatus('Building model...');
  const model = tf.sequential();
  model.add(tf.layers.dense({
    inputShape:[data.numFeatures],
    units: 100,
    activation: 'sigmoid'
  }));
  model.add(tf.layers.dense({
    units: 100,
    activation: 'sigmoid'
  }));
  model.add(tf.layers.dense({
    units: 1,
    activation: 'sigmoid'
  }));
  model.compile({
    optimizer: 'adam',
    loss: 'binaryCrossentropy',
    metrics: ['accuracy']
  });

  const trainLogs = [];
  let auc;
  await ui.updateStatus('Training starting...');
  await model.fit(trainData.data, trainData.target, {batchSize, epochs, validationSplit: 0.2, callbacks: {
    onEpochBegin: async (epoch) => {
      if ((epoch+1)%100 ===0 || epoch===0 || epoch===2 || epoch===4) {
        const probs = model.predict(testData.data);
        auc = drawROC(testData.target, probs, epoch)
      }
    },
    onEpochEnd: async (epoch, logs) => {
      await ui.updateStatus(`Epoch ${epoch+1} of ${epochs} completed.`);
      trainLogs.push(logs);
      ui.plotLosses(trainLogs);
      ui.plotAccuracies(trainLogs);
    }
  }})
})