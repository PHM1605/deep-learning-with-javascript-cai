import * as tfvis from "./tfvis";

const statusElement = document.getElementById("status");
export function updateStatus(message) {
  statusElement.innerText = message;
}
export async function plotLosses(trainLogs) {
  return tfvis.show.history(
    document.getElementById('plotLoss'),
    trainLogs,
    ['loss', 'val_loss'],
    {width: 450, height:320, xLabel:"Epoch", yLabel:"Loss"}
  )
}
export async function plotAccuracies(trainLogs) {
  return tfvis.show.history(
    document.getElementById('plotAccuracy'),
    trainLogs,
    ['acc', 'val_acc'],
    {width: 450, height:320, xLabel:"Epoch", yLabel:"Accuracy"}
  );
}

const rocSeries = [], rocValues = [];
export async function plotROC(fprs, tprs, epoch) {
  epoch++;
  const seriesName = 'epoch ' + (epoch<10 ? `00${epoch}` : (epoch<100 ? `0${epoch}` : `${epoch}`));
  rocSeries.push(seriesName);
  // newSeries: [{x:1,y:4}, {x:2,y:3}, {}, ...]
  const newSeries = []
  for (let i=0; i<fprs.length; i++) {
    newSeries.push({x: fprs[i], y: tprs[i]});
  }
  // rocValues: [[], [], ] with each element is a list of points
  rocValues.push(newSeries);
  return tfvis.render.linechart(
    document.getElementById('rocCurve'),
    {values: rocValues, series: rocSeries},
    {width: 450, height: 450}
  )
}