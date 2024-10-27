import { linearRegressionModel, multiLayerPerceptronRegressionModel1Hidden, multiLayerPerceptronRegressionModel2Hidden, run } from ".";

const statusElement = document.getElementById("status");
export function updateStatus(message) {
  statusElement.innerText = message;
}

const baselineStatusElement = document.getElementById('baselineStatus');
export function updateBaselineStatus(message) {
  baselineStatusElement.innerText = message;
}

export function updateModelStatus(message, modelName) {
  const statElement = document.querySelector(`#${modelName} .status`);
  statElement.innerText = message;
}

const NUM_TOP_WEIGHTS_TO_DISPLAY = 5;
export function updateWeightDescription(weightsList) {
  const inspectionHeadlineElement = document.getElementById('inspectionHeadline');
  inspectionHeadlineElement.innerText = `Top ${NUM_TOP_WEIGHTS_TO_DISPLAY} weights by magnitude`;
  weightsList.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  var table = document.getElementById('myTable');
  table.innerHTML = '';
  weightsList.forEach((weight, i) => {
    if (i<NUM_TOP_WEIGHTS_TO_DISPLAY) {
      let row = table.insertRow(-1);
      let cell1 = row.insertCell(0);
      let cell2 = row.insertCell(1);
      if (weight.value < 0) {
        cell2.setAttribute('class', 'negativeWeight');
      } else {
        cell2.setAttribute('class', 'positiveWeight');
      }
      cell1.innerHTML = weight.description;
      cell2.innerHTML = weight.value.toFixed(4);
    }
  })
}

export async function setup() {
  const trainSimpleLinearRegression = document.getElementById('simple-mlr');
  const trainNeuralNetworkLinearRegression1Hidden = document.getElementById('nn-mlr-1hidden');
  const trainNeuralNetworkLinearRegression2Hidden = document.getElementById('nn-mlr-2hidden');
  trainSimpleLinearRegression.addEventListener('click', async ()=>{
    const model = linearRegressionModel();
    await run(model, 'linear', true);
  }, false);
  trainNeuralNetworkLinearRegression1Hidden.addEventListener('click', async ()=>{
    const model = multiLayerPerceptronRegressionModel1Hidden();
    await run(model, 'oneHidden', false);
  }, false);
  trainNeuralNetworkLinearRegression2Hidden.addEventListener('click', async()=>{
    const model = multiLayerPerceptronRegressionModel2Hidden();
    await run(model, 'twoHidden', false);
  }, false);
}

export function plotData(container, numEpochs, trainLogs) {
  let x=[], loss=[], valLoss = [];
  for (let epoch=0; epoch < trainLogs.length; epoch++) {
    x.push(epoch+1);
    loss.push(trainLogs[epoch]['loss']);
    valLoss.push(trainLogs[epoch]['val_loss']);
  }
  const lossTrace = {x, y:loss, name:"loss"}
  const valLossTrace = {x, y:valLoss, name:"valLoss"}
  Plotly.newPlot(container, [lossTrace, valLossTrace], {
    height:300, width:300, 
    title:'lossPlot', 
    xaxis: {
      title: "Epoch",
      range: [0,numEpochs+1]
    }, 
    yaxis: {
      title:"Loss",
      range: [0,300]
    }})
}