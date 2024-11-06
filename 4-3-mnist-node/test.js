const tf = require("@tensorflow/tfjs-node");

async function loadAndSummarizeModel() {
  const loadedModel = await tf.loadLayersModel('file://./tfjs-mnist/model.json');
  loadedModel.summary();
}

loadAndSummarizeModel()