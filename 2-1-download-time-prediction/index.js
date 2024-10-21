const trainData = {
  sizeMB: [0.08, 9, 0.001,0.1, 8, 5, 0.1, 6, 0.05, 0.5, 0.002, 2, 0.005, 10, 0.01, 7, 6, 5, 1, 1],
  timeSec: [0.135, 0.739, 0.067, 0.126, 0.646, 0.435, 0.069, 0.497, 0.068, 0.116, 0.07, 0.289, 0.076, 0.744, 0.083, 0.56, 0.48, 0.399, 0.153, 0.149]
};
const testData = {
  sizeMB: [5.000, 0.200, 0.001, 9.000, 0.002, 0.020, 0.008, 4.000, 0.001, 1.000, 0.005, 0.080, 0.800, 0.200, 0.050, 7.000, 0.005, 0.002, 8.000, 0.008],
  timeSec: [0.425, 0.098, 0.052, 0.686, 0.066, 0.078, 0.070, 0.375, 0.058, 0.136, 0.052, 0.063, 0.183, 0.087, 0.066, 0.558, 0.066, 0.068, 0.610, 0.057]
};

trainXs = tf.tensor2d(trainData.sizeMB, [20, 1]);
trainYs = tf.tensor2d(trainData.timeSec, [20, 1]);
testXs = tf.tensor2d(testData.sizeMB, [20, 1]);
testYs = tf.tensor2d(testData.timeSec, [20, 1]);

const dataTraceTrain = {
  x: trainData.sizeMB, y: trainData.timeSec,
  name: 'trainData', mode: 'markers', type: 'scatter', marker: {symbol: 'circle', size: 8}
};
const dataTraceTest = {
  x: testData.sizeMB, y: testData.timeSec,
  name: 'testData', mode: 'markers', type: 'scatter', marker: {symbol: 'triangle-up', size: 10}
};
// // plot new 
// Plotly.newPlot(
//   'dataSpace', 
//   [dataTraceTrain, dataTraceTest], 
//   {width: 700, title: 'File download duration', xaxis: {title: 'size (MB)'}, yaxis: {title: "time (sec)"}}
// )

const dataTrace10Epochs = {
  x: [0, 2], y: [0, 0.01],
  name: 'model after 10 epochs',
  mode: 'lines',
  line: {color: 'blue', width:1, dash:"dot"}
}
const dataTrace20Epochs = {
  x: [0, 2], y: [0, 0.01],
  name: 'model after 20 epochs',
  mode: 'lines',
  line: {color: 'green', width:2, dash:"dash"}
}
const dataTrace200Epochs = {
  x: [0, 2], y: [0, 0.01],
  name: 'model after 200 epochs',
  mode: 'lines',
  line: {color: 'red', width:3, dash:"solid"}
}

Plotly.newPlot(
  'dataSpace',
  [dataTraceTrain, dataTraceTest, dataTrace10Epochs, dataTrace20Epochs, dataTrace200Epochs],
  {width: 700, title:"Model fit result", xaxis:{title:"size (MB)"}, yaxis:{title:"time (sec)"}}
)

const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}))
model.compile({optimizer: tf.train.sgd(0.0005), loss:"meanAbsoluteError"});

