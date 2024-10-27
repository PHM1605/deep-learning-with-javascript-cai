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