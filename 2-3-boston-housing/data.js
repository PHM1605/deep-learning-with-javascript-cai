import Papa from "papaparse"
const BASE_URL = 'https://storage.googleapis.com/tfjs-examples/multivariate-linear-regression/data/';
const TRAIN_FEATURES_FN = 'train-data.csv';
const TRAIN_TARGET_FN = 'train-target.csv';
const TEST_FEATURES_FN = 'test-data.csv';
const TEST_TARGET_FN = 'test-target.csv';

export const featureDescriptions = ["Crime rate", "Land zone size", "Industrial proportion", "Next to river", 
  "Nitric oxide concentration", "Number of rooms per house", "Age of housing", "Distance to commute", "Distance to highway", 
  "Tax rate", "School class size", "School drop-out rate"
];

const parseCsv = async (data) => {
  return new Promise(resolve=>{
    data = data.map(row => Object.keys(row).map(key=>parseFloat(row[key])));
    resolve(data);
  })
};

export const loadCsv = async (filename) => {
  return new Promise(resolve => {
    const url = `${BASE_URL}${filename}`;
    console.log(`** Downloading data from: ${url}`);
    Papa.parse(url, {download: true, header: true, complete: (results => {
      resolve(parseCsv(results['data']))
    })})
  })
}

export class BostonHousingDataset {
  constructor() {
    this.trainFeatures = null;
    this.trainTarget = null;
    this.testFeatures = null;
    this.testTarget = null;
  }

  get numFeatures() {
    if (this.trainFeatures === null) {
      throw new Error('\'loadData()\' must be called before numFeatures');
    }
    return this.trainFeatures[0].length;
  }
  
  async loadData() {
    [this.trainFeatures, this.trainTarget, this.testFeatures, this.testTarget] = 
      await Promise.all([loadCsv(TRAIN_FEATURES_FN), loadCsv(TRAIN_TARGET_FN), loadCsv(TEST_FEATURES_FN), loadCsv(TEST_TARGET_FN)])
    shuffle(this.trainFeatures, this.trainTarget);
    shuffle(this.testFeatures, this.testTarget);
  }
}

function shuffle(data, target) {
  let counter = data.length;
  let temp = 0, index = 0;
  while (counter > 0) {
    index = (Math.random() * counter) | 0
    counter--;
    // swap place of last-index (backward move) and a random element
    temp = data[counter];
    data[counter] = data[index];
    data[index] = temp;
    // shuffle target
    temp = target[counter];
    target[counter] = target[index];
    target[index] = temp;
  }
}