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

const trainTensors = {
  sizeMB: tf.tensor2d(trainData.sizeMB, [20, 1]),
  timeSec: tf.tensor2d(trainData.timeSec, [20, 1])
};
const testTensors = {
  sizeMB: tf.tensor2d(testData.sizeMB, [20, 1]),
  timeSec: tf.tensor2d(testData.timeSec, [20, 1])
}

const dataTraceTrain = {
  x: trainData.sizeMB, y: trainData.timeSec,
  name: 'trainData', mode: 'markers', type: 'scatter', marker: {symbol: 'circle', size: 8}
};
const dataTraceTest = {
  x: testData.sizeMB, y: testData.timeSec,
  name: 'testData', mode: 'markers', type: 'scatter', marker: {symbol: 'triangle-up', size: 10}
};

function updateScatterWithLines(dataTrace, k, b, N, traceIndex) {
  dataTrace.x = [0, 10];
  dataTrace.y = [b, b+k*10];
  var update = {x: [dataTrace.x], y: [dataTrace.y], name: "model after " + N + " epochs"}
  Plotly.restyle('dataSpace', update, traceIndex)
}

function setModelWeights(k, b) {
  model.setWeights([tf.tensor2d([k], [1,1]), tf.tensor1d([b])])
}

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

// Plot loss plot
const lossTrace = {x:[], y:[]}
lossPlotAnnotationFont = {family: "sans serif", size:18, color:"#000000"}
function plotLoss(epoch, loss) {
  lossTrace.x.push(epoch);
  lossTrace.y.push(loss);
  Plotly.newPlot("lossPlot", [lossTrace], {
    width: 500,
    height: 500,
    title: "Loss vs. Epoch",
    xaxis: {title: "epoch #", range: [0,201]}, 
    yaxis: {title: "loss", range:[0,0.31]},
    annotations: [
      {x:1, y:0.295, xref:"x", yref:"y", text:"Start", showarrow:true, arrowhead:6, arrowcolor:"#000000", font:lossPlotAnnotationFont, ax:60, ay:0},
      {x:20, y:0.198, xref:"x", yref:"y", text:"Epoch 20", showarrow:true, arrowhead:6, arrowcolor:"#000000", font:lossPlotAnnotationFont, ax:60, ay:0},
      {x:100, y:0.035, xref:"x", yref:"y", text:"Epoch 100", showarrow:true, arrowhead:6, arrowcolor:"#000000", font:lossPlotAnnotationFont, ax:0, ay:-60},
      {x:200, y:0.0206, xref:"x", yref:"y", text:"Epoch 200", showarrow:true, arrowhead:6, arrowcolor:"#000000", font:lossPlotAnnotationFont, ax:0, ay:-60}
    ]
  })
}

// Prepare loss surface 2d
const landscape = {x:[], y:[], z:[], type:'contour'}
function genLossLandscape() {
  const kMin=-0.05, kMax=0.15, kStep=0.0035;
  const bMin=-0.1, bMax=0.15, bStep=0.06;
  for(let k=kMin; k<=kMax; k+=kStep) {
    for(let b=bMin; b<=bMax; b+=bStep) {
      tf.tidy(()=>{
        setModelWeights(k, b);
        const loss = model.evaluate(testXs, testYs).dataSync()[0];
        landscape.x.push(k);
        landscape.y.push(b);
        landscape.z.push(loss);
      })
    }
  }
}

// Plot loss surface 2d with trajectory
const trajectory = {x:[], y:[], type:"scatter"}
function recordTrajectory(k, b, loss) {
  trajectory.x.push(k);
  trajectory.y.push(b);
  contourAnnotationFont = {family:"sans serif", size:18, color:"#ffffff"};
  Plotly.newPlot('lossSurface2dWithTraj', [landscape, trajectory], {
    height:500, width:500, title:"Loss surface", xaxis:{title:"kernel"}, yaxis:{title:"bias"}, showlegend:false,
    annotations:[]
  })
}

const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}))
model.compile({optimizer: tf.train.sgd(0.0005), loss:"meanAbsoluteError"});
let k = 0;
let b = 0;
setModelWeights(0, 0);
genLossLandscape();


(async () => {
  await model.fit(trainTensors.sizeMB, trainTensors.timeSec, {
    epochs: 200, 
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        plotLoss(epoch+1, logs.loss);
        recordTrajectory(k, b, logs.loss)
        k = model.getWeights()[0].dataSync()[0]
        b = model.getWeights()[1].dataSync()[0]
        if (epoch === 9) {
          updateScatterWithLines(dataTrace10Epochs, k, b, 10, 2);
          console.log("wrote model 10")
        }
        if (epoch===19) {
          updateScatterWithLines(dataTrace20Epochs, k, b, 20, 3);
          console.log("wrote model 20")
        }
        if (epoch === 199) {
          updateScatterWithLines(dataTrace200Epochs, k, b, 200, 4);
          console.log("wrote model 200")
        }
        await tf.nextFrame();
      }
    }
  })
})();
