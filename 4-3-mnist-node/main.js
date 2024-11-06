const tf = require("@tensorflow/tfjs-node");
const argparse = require("argparse");
const data = require('./data')
const model = require('./model')

async function run(epochs, batchSize, modelSavePath) {
  await data.loadData();
  const {images: trainImages, labels: trainLabels} = data.getTrainData();
  model.summary();
  let epochBeginTime;
  let millisPerStep;
  const validationSplit = 0.15;
  const numTrainExamplesPerEpoch = trainImages.shape[0]*(1-validationSplit);
  const numTrainBatchesPerEpoch = Math.ceil(numTrainExamplesPerEpoch/batchSize);
  await model.fit(trainImages, trainLabels, {epochs, batchSize, validationSplit});
  const {images: testImages, labels: testLabels} = data.getTestData();
  const evalOutput = model.evaluate(testImages, testLabels);
  console.log('\nEvaluation result:');
  console.log(` Loss = ${evalOutput[0].dataSync()[0].toFixed(3)}; Accuracy = ${evalOutput[1].dataSync()[0].toFixed(3)}`)
  if (modelSavePath) {
    await model.save(`file://${modelSavePath}`);
    console.log(`Saved model to path: ${modelSavePath}`);
  }
}

const parser = new argparse.ArgumentParser({
  description: "TensorFlow.js-Node MNIST Example.",
  add_help: true
});
parser.add_argument("--epochs", {type: 'int', default: 20, help: "Number of epochs to train the model for."});
parser.add_argument("--batch_size", {type: "int", default: 128, help: "Batch size to be used during model training."});
parser.add_argument("--model_save_path", {type: "string", default: "./tfjs-mnist", help: "Path to which model will be saved after training."});
const args = parser.parse_args();

run(args.epochs, args.batch_size, args.model_save_path);