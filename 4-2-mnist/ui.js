import * as tfvis from "./tfvis"
const statusElement = document.getElementById('status')
const messageElement = document.getElementById('message')
const imagesElement = document.getElementById('images')

export function logStatus(message) {
  statusElement.innerText = message;
}

export function trainingLog(message) {
  messageElement.innerText = `${message}\n`;
}

export function showTestResults(batch, predictions, labels) {
  const testExamples = batch.xs.shape[0];
  imagesElement.innerHTML = "";
  for (let i=0; i< testExamples; i++) {
    const image = batch.xs.slice([i, 0], [1, batch.xs.shape[1]]);
    const div = document.createElement('div');
    div.className = 'pred-container';
    const canvas = document.createElement('canvas');
    canvas.className = 'prediction-canvas';
    draw(image.flatten(), canvas);
    const pred = document.createElement('div');
    const prediction = predictions[i];
    const label = labels[i];
    const correct = prediction === label;
    pred.className = `pred ${(correct ? 'pred-correct' : 'pred-incorrect')}`;
    pred.innerText = `pred: ${prediction}`;
    div.appendChild(pred);
    div.appendChild(canvas);
    imagesElement.appendChild(div);
  }
}

const lossLabelElement = document.getElementById('loss-label');
const accuracyLabelElement = document.getElementById('accuracy-label');
const lossValues = [[], []];
// batch: index number; loss: loss of that batch (one number); set: 'train'/'validation'
export function plotLoss(batch, loss, set) {
  const series = set === 'train' ? 0 : 1;
  lossValues[series].push({x:batch, y:loss});
  const lossContainer = document.getElementById('loss-canvas')
  tfvis.render.linechart(lossContainer, {values: lossValues, series: ['train', 'validation']},
    {xLabel: 'Batch #', yLabel: "Loss", width: 400, height: 300}
  );
  lossLabelElement.innerText = `last loss: ${loss.toFixed(3)}`;
}

const accuracyValues = [[], []];
export function plotAccuracy(batch, accuracy, set) {
  const series = set ==='train' ? 0 : 1;
  const accuracyContainer = document.getElementById('accuracy-canvas');
  accuracyValues[series].push({x:batch, y:accuracy});
  tfvis.render.linechart(accuracyContainer, {values: accuracyValues, series:['train', 'validation']}, 
    {xLabel: 'Batch #', yLabel: 'Accuracy', width: 400, height: 300}
  );
  accuracyLabelElement.innerText = `last accuracy: ${(accuracy*100).toFixed(1)}%`;
}

export function draw(image, canvas) {
  const [width, height] = [28,28];
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  const imageData = new ImageData(width, height);
  const data = image.dataSync();
  for (let i=0; i<height*width; ++i) {
    const j=i*4;
    imageData.data[j+0] = data[i] * 255;
    imageData.data[j+1] = data[i] * 255;
    imageData.data[j+2] = data[i] * 255;
    imageData.data[j+3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}

export function getModelTypeId() {
  return document.getElementById('model-type').value;
}

export function getTrainEpochs() {
  return Number.parseInt(document.getElementById('train-epochs').value)
}

export function setTrainButtonCallback(callback) {
  const trainButton = document.getElementById('train');
  const modelType = document.getElementById('model-type');
  trainButton.addEventListener('click', ()=>{
    trainButton.setAttribute('disabled', true);
    modelType.setAttribute('disabled', true);
    callback();
  })
}