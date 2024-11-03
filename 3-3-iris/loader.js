import * as tf from '@tensorflow/tfjs'
import * as ui from "./ui"

export async function urlExists(url) {
  ui.status('Testing url ' + url);
  try {
    const response = await fetch(url, {method: "HEAD"});
    return response.ok;
  } catch(err) {
    return false;
  }
}

export async function loadHostedPretrainedModel(url) {
  ui.status('Loading pretrained model from ' + url);
  try {
    const model = await tf.loadLayersModel(url);
    ui.status('Done loading pretrained model.');
    return model;
  } catch(err) {
    console.error(err);
    ui.status('Loading pretrained model failed.');
  }
}

const LOCAL_MODEL_URL = 'indexeddb://tfjs-iris-demo-model/v1';
export async function saveModelLocally(model) {
  const saveResult = await model.save(LOCAL_MODEL_URL);
}
export async function loadModelLocally() {
  return await tf.loadLayersModel(LOCAL_MODEL_URL);
}

export async function removeModelLocally() {
  return await tf.io.removeModel(LOCAL_MODEL_URL);
}

export async function updateLocalModelStatus() {
  const localModelStatus = document.getElementById('local-model-status');
  const localLoadButton = document.getElementById('load-local');
  const localRemoveButton = document.getElementById('remove-local');
  const modelsInfo = await tf.io.listModels();
  if (LOCAL_MODEL_URL in modelsInfo) {
    localModelStatus.textContent = 'Found locally stored model saved at ' + modelsInfo[LOCAL_MODEL_URL].dateSaved.toDateString();
    localLoadButton.disabled = false;
    localRemoveButton.disabled = false;
  } else {
    localModelStatus.textContent = 'No locally-stored model is found.';
    localLoadButton.disabled = true;
    localRemoveButton.disabled = true;
  }
}