import * as tf from "@tensorflow/tfjs"

// config: {width:300, height:300, xLabel:"xxx", yLabel:"yyy"}
// trainLogs: [{val_loss:..., val_acc:..., loss:..., acc:...}, {}, ...]
// traceList: ['loss', 'val_loss']
export const show = {
  history: (container, trainLogs, traceNames) => {
    const traceList = traceNames.map(traceName => (
      {
        x: Array.from(Array(trainLogs.length).keys()), 
        y: trainLogs.map(oneLog => oneLog[traceName]), 
        name: traceName}
    ));
    Plotly.newPlot(container, traceList, {
      xaxis: {title: "Epoch", range: [0,100]},
    })
  }
}

export const metrics = {
  // labels: Tensor [0,2,1,1,0,...]
  // preds: Tensor [0,2,1,0,0,...]
  confusionMatrix: (labels, preds) => {
    const classes = tf.unique(labels).values.dataSync();
    let confMatrix = [];
    for (let i=0; i<classes.length; i++) {
      confMatrix.push([]);
      for (let j=0; j<classes.length; j++) {
        confMatrix[i].push(0);
      }
    }
    const arrayLabels = labels.dataSync();
    const arrayPreds = preds.dataSync();
    arrayLabels.forEach((sample, idx) => {
      confMatrix[sample][arrayPreds[idx]]++;
    })
    return tf.tensor2d(confMatrix);
  }
}

// dataConfig: {values: tensor2D, labels: list of strings}
export const render = {
  confusionMatrix: (container, dataConfig, config) => {
    const confusionMat = dataConfig.values;
    const w = container.width;
    const h = container.height;
    const ctx = container.getContext('2d');
    ctx.clearRect(0, 0, w, h);
    const n = confusionMat.shape[0];
    const rawConfusion = confusionMat.dataSync();
    const normalizedConfusion = confusionMat.div(confusionMat.sum(-1).expandDims(1)).dataSync();
    for (let i=0; i<n; ++i) {
      for (let j=0; j<n; ++j) {
        const rgbValue = Math.round(255 * (1) - normalizedConfusion[i*n+j]);
        ctx.fillStyle = `rgb(${rgbValue}, ${rgbValue}, ${rgbValue})`
        ctx.fillRect(w/n*j, h/n*i, w/n, h/n);
        ctx.stroke();
        ctx.strokeStyle = '#808080';
        ctx.rect(w/n*j, h/n*i, w/n, h/n);
        ctx.stroke();
        ctx.font = '18px Arial';
        ctx.fillStyle = '#ff00ff';
        ctx.fillText(`${rawConfusion[i*n+j]}`, w/n*(j+0.4), h/n*(i+0.66));
        ctx.stroke();
      }
    }
  }
}
