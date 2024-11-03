import { IRIS_CLASSES, IRIS_NUM_CLASSES } from "./data";

export function clearEvaluateTable() {
  const tableBody = document.getElementById('evaluate-tbody');
  while(tableBody.children.length > 1) {
    tableBody.removeChild(tableBody.children[1]);
  }
}

export function getManualInputData() {
  return [
    Number(document.getElementById('petal-length').value), 
    Number(document.getElementById('petal-width').value),
    Number(document.getElementById('sepal-length').value),
    Number(document.getElementById('sepal-width').value)
  ]
}

export function setManualInputWinnerMessage(message) {
  const winnerElement = document.getElementById('winner');
  winnerElement.textContent = message;
}

// logit: e.g. [0.1, 0.7, 0.2] = class probs
function logitsToSpans(logits) {
  let idxMax = -1
  let maxLogit = Number.NEGATIVE_INFINITY
  for (let i=0; i < logits.length; ++i) {
    if (logits[i] > maxLogit ) {
      maxLogit = logits[i]
      idxMax = i; 
    }
  }
  const spans = []
  for (let i=0; i<logits.length; ++i) {
    const logitSpan = document.createElement('span');
    logitSpan.textContent = logits[i].toFixed(3); // <span>0.78
    if (i===idxMax) {
      logitSpan.style['font-weight'] = 'bold';
    }
    logitSpan.classList = ['logit-span'];
    spans.push(logitSpan);
  }
  return spans;
}

function renderLogits(logits, parentElement) {
  while(parentElement.firstChild) {
    parentElement.removeChild(parentElement.firstChild)
  }
  logitsToSpans(logits).map(logitSpan =>{
    parentElement.appendChild(logitSpan)
  })
}

export function renderLogitsForManualInput(logits) {
  const logitsElement = document.getElementById('logits')
  renderLogits(logits, logitsElement)
}

export function renderEvaluateTable(xData, yTrue, yPred, logits) {
  const tableBody = document.getElementById('evaluate-tbody');
  for (let i=0; i<yTrue.length; ++i) {
    const row = document.createElement('tr');
    for (let j=0; j<4; ++j) {
      const cell = document.createElement('td');
      cell.textContent = xData[4*i+j].toFixed(1);
      row.appendChild(cell);
    }
    const truthCell = document.createElement('td');
    truthCell.textContent = IRIS_CLASSES[yTrue[i]];
    row.appendChild(truthCell);
    const predCell = document.createElement('td');
    predCell.textContent = IRIS_CLASSES[yPred[i]];
    predCell.classList = yPred[i]===yTrue[i] ? ['correct-prediction'] : ['wrong-prediction']
    row.appendChild(predCell);
    const logitsCell = document.createElement('td')
    const exampleLogits = logits.slice(i*IRIS_NUM_CLASSES, (i+1)*IRIS_NUM_CLASSES);
    logitsToSpans(exampleLogits).map(logitSpan => {
      logitsCell.appendChild(logitSpan);
    })
    row.appendChild(logitsCell);
    tableBody.appendChild(row)
  }
}

export function wireUpEvaluateTableCallbacks(predictOnManualInputCallback) {
  const petalLength = document.getElementById('petal-length');
  const petalWidth = document.getElementById('petal-width');
  const sepalLength = document.getElementById('sepal-length');
  const sepalWidth = document.getElementById('sepal-width');
  const increment = 0.1;
  document.getElementById('petal-length-inc').addEventListener('click', ()=>{
    petalLength.value = (Number(petalLength.value) + increment).toFixed(1);
    predictOnManualInputCallback();
  });
  document.getElementById('petal-length-dec').addEventListener('click', ()=>{
    petalLength.value = (Number(petalLength.value) - increment).toFixed(1);
    predictOnManualInputCallback();
  });
  document.getElementById('petal-width-inc').addEventListener('click', ()=>{
    petalWidth.value = (Number(petalWidth.value) + increment).toFixed(1);
    predictOnManualInputCallback();
  });
  document.getElementById('petal-width-dec').addEventListener('click', ()=>{
    petalWidth.value = (Number(petalWidth.value) - increment).toFixed(1);
    predictOnManualInputCallback();
  });
  document.getElementById('sepal-length-inc').addEventListener('click', ()=>{
    sepalLength.value = (Number(sepalLength.value) + increment).toFixed(1);
    predictOnManualInputCallback();
  });
  document.getElementById('sepal-length-dec').addEventListener('click', ()=>{
    sepalLength.value = (Number(sepalLength.value) - increment).toFixed(1);
    predictOnManualInputCallback();
  });
  document.getElementById('sepal-width-inc').addEventListener('click', ()=>{
    sepalWidth.value = (Number(sepalWidth.value) + increment).toFixed(1);
    predictOnManualInputCallback();
  });
  document.getElementById('sepal-width-dec').addEventListener('click', ()=>{
    sepalWidth.value = (Number(sepalWidth.value) - increment).toFixed(1);
    predictOnManualInputCallback();
  });
  document.getElementById('petal-length').addEventListener('change', ()=>{
    predictOnManualInputCallback();
  });
  document.getElementById('petal-width').addEventListener('change', ()=>{
    predictOnManualInputCallback();
  })
  document.getElementById('sepal-length').addEventListener('change', ()=>{
    predictOnManualInputCallback();
  });
  document.getElementById('sepal-width').addEventListener('change', ()=>{
    predictOnManualInputCallback();
  })
}

export function loadTrainParametersFromUI() {
  return {
    epochs: Number(document.getElementById('train-epochs').value),
    learningRate: Number(document.getElementById('learning-rate').value)
  }
}

export function status(statusText) {
  document.getElementById('demo-status').textContent = statusText;
}