import * as tf from "@tensorflow/tfjs"
import * as data from "./data"
import * as ui from "./ui"
import * as tfvis from "./tfvis"
import * as loader from "./loader"

async function trainModel(xTrain, yTrain, xTest, yTest) {
  ui.status('Training model... Please wait');
  const params = ui.loadTrainParametersFromUI();
  const model = tf.sequential();
  model.add(tf.layers.dense({units:10, activation:'sigmoid', inputShape:[xTrain.shape[1]]}));
  model.add(tf.layers.dense({units:3, activation:'softmax'}));
  model.summary();
  const optimizer = tf.train.adam(params.learningRate);
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });
  const trainLogs = [];
  const lossContainer = document.getElementById('lossCanvas');
  const accContainer = document.getElementById('accuracyCanvas');
  const beginMs = performance.now();

  const history = await model.fit(xTrain, yTrain, {
    epochs: params.epochs,
    validationData: [xTest, yTest],
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        const secPerEpoch = (performance.now() - beginMs) / (1000 * (epoch+1));
        ui.status(`Training model...Approximately ${secPerEpoch.toFixed(4)} seconds per epoch`);
        trainLogs.push(logs);
        tfvis.show.history(lossContainer, trainLogs, ['loss', 'val_loss']);
        tfvis.show.history(accContainer, trainLogs, ['acc', 'val_acc']);
        calculateAndDrawConfusionMatrix(model, xTest, yTest)
      }
    }
  });
  const secPerEpoch = (performance.now() - beginMs) / (1000 * params.epochs);
  ui.status(`Model training complete: ${secPerEpoch.toFixed(4)} seconds per epoch`);
  return model;
}

async function predictOnManualInput(model) {
  if(model === null) {
    ui.setManualInputWinnerMessage('ERROR: Please load or train model first');
    return;
  }
  tf.tidy(()=>{
    const inputData = ui.getManualInputData();
    const input = tf.tensor2d([inputData], [1,4]);
    const predictOut = model.predict(input);
    const logits = Array.from(predictOut.dataSync());
    const winner = data.IRIS_CLASSES[predictOut.argMax(-1).dataSync()[0]]
    ui.setManualInputWinnerMessage(winner);
    ui.renderLogitsForManualInput(logits);
  })
}

async function calculateAndDrawConfusionMatrix(model, xTest, yTest) {
  const [preds, labels] = tf.tidy(() => {
    const preds = model.predict(xTest).argMax(-1);
    const labels = yTest.argMax(-1);
    return [preds, labels];
  });
  const confMatrixData = tfvis.metrics.confusionMatrix(labels, preds);
  const container = document.getElementById('confusion-matrix');
  tfvis.render.confusionMatrix(container, {values: confMatrixData, labels: data.IRIS_CLASSES}, {shadeDiagonal: true});
  tf.dispose([preds, labels]);
}

async function evaluateModelOnTestData(model, xTest, yTest) {
  ui.clearEvaluateTable();
  tf.tidy(() => {
    const xData = xTest.dataSync();
    const yTrue = yTest.argMax(-1).dataSync();
    const predictOut = model.predict(xTest);
    const yPred = predictOut.argMax(-1);
    ui.renderEvaluateTable(xData, yTrue, yPred.dataSync(), predictOut.dataSync());
    calculateAndDrawConfusionMatrix(model, xTest, yTest);
  });
  predictOnManualInput(model);
}

let model;
const HOSTED_MODEL_JSON_URL = 'https://storage.googleapis.com/tfjs-models/tfjs/iris_v1/model.json';
async function iris() {
  // each [numSamples, 4] or [numSamples, 1]
  const [xTrain, yTrain, xTest, yTest] = data.getIrisData(0.15);
  const localLoadButton = document.getElementById('load-local');
  const localSaveButton = document.getElementById('save-local');
  const localRemoveButton = document.getElementById('remove-local');
  document.getElementById('train-from-scratch').addEventListener('click', async () => {
    model = await trainModel(xTrain, yTrain, xTest, yTest);
    await evaluateModelOnTestData(model, xTest, yTest);
    localSaveButton.disabled = false;
  })
  if (await loader.urlExists(HOSTED_MODEL_JSON_URL)) {
    ui.status("Model available: " + HOSTED_MODEL_JSON_URL);
    const button = document.getElementById('load-pretrained-remote');
    button.addEventListener('click', async () => {
      ui.clearEvaluateTable();
      model = await loader.loadHostedPretrainedModel(HOSTED_MODEL_JSON_URL);
      await predictOnManualInput(model);
      localSaveButton.disabled = false;
    });
    localLoadButton.addEventListener('click', async ()=>{
      // model = await loader.
    });
    localSaveButton.addEventListener('click', async () => {
      await loader.saveModelLocally(model);
      await loader.updateLocalModelStatus();
    });
    localRemoveButton.addEventListener('click', async ()=>{
      await loader.removeModelLocally();
      await loader.updateLocalModelStatus();
    });
    await loader.updateLocalModelStatus();
    ui.status('Standing by.')
    ui.wireUpEvaluateTableCallbacks(()=>predictOnManualInput(model))
  }
}

iris()