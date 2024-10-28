import * as tf from "@tensorflow/tfjs";
import * as Papa from "papaparse";
const BASE_URL = 'https://gist.githubusercontent.com/ManrajGrover/6589d3fd3eb9a0719d2a83128741dfc1/raw/d0a86602a87bfe147c240e87e6a9641786cafc19/';

async function parseCsv(data) {
  return new Promise(resolve => {
    data = data.map(row => (
      Object.keys(row).sort().map(key => parseFloat(row[key]))
    ));
    resolve(data);
  });
}

export async function loadCsv(fileName) {
  return new Promise(resolve => {
    const url = `${BASE_URL}${fileName}.csv`;
    console.log(`* Downloading data from: ${url}`);
    Papa.parse(url, {
      download: true, 
      header: true,
      complete: (results) => {
        resolve(parseCsv(results['data']))
      }
    })
  });
}

export async function shuffle(data, label) {
  let counter = data.length;
  let temp = 0;
  let index = 0;
  while(counter > 0) {
    index = (Math.random() * counter) | 0;
    counter--;
    temp = data[counter];
    data[counter] = data[index];
    data[index] = temp;
    temp = label[counter];
    label[counter] = label[index];
    label[index] = temp;
  }
}

function mean(vector) {
  let sum = 0;
  for (const x of vector) {
    sum += x;
  }
  return sum/vector.length;
}

function stddev(vector) {
  let squareSum = 0;
  const vectorMean = mean(vector);
  for (const x of vector) {
    squareSum += (x - vectorMean) * (x - vectorMean);
  }
  return Math.sqrt(squareSum / (vector.length - 1));
}

const normalizeVector = (vector, vectorMean, vectorStddev) => {
  return vector.map(x => (x - vectorMean) / vectorStddev);
}

// dataset: [numSamples, numFeatures], vectorMeans: [numFeatures], vectorStddevs: [numFeatures]
export function normalizeDataset(dataset, isTrainData = true, vectorMeans = [], vectorStddevs = []) {
  const numFeatures = dataset[0].length;
  // holds ONE value of ONE feature's  mean/std
  let vectorMean, vectorStddev;
  for(let i=0; i<numFeatures; i++) {
    // vector: [numSamples] = feature i of all samples
    const vector = dataset.map(row => row[i]);
    if (isTrainData) {
      vectorMean = mean(vector);
      vectorStddev = stddev(vector);
      vectorMeans.push(vectorMean);
      vectorStddevs.push(vectorStddev);
    } else {
      vectorMean = vectorMeans[i];
      vectorStddev = vectorStddevs[i];
    }
    // vectorNormalized: [numSamples]
    const vectorNormalized = normalizeVector(vector, vectorMean, vectorStddev);
    vectorNormalized.forEach((value, index) =>{
      dataset[index][i] = value;
    })
  }

  return {dataset, vectorMeans, vectorStddevs};
}