import * as tf from "@tensorflow/tfjs";
import { WebsitePhishingDataset } from "./data";
import * as ui from "./ui"

const epochs = 400;
const batchSize = 350;



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