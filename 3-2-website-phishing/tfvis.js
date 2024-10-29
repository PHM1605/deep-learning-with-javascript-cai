// config: {width:300, height:300, xLabel:"xxx", yLabel:"yyy"}
// trainLogs: [{val_loss:..., val_acc:..., loss:..., acc:...}, {}, ...]
// traceList: ['loss', 'val_loss']
export const show = {
  history: (container, trainLogs, traceNames, config) => {
    const traceList = traceNames.map(traceName => (
      {
        x: Array.from(Array(trainLogs.length).keys()), 
        y: trainLogs.map(oneLog => oneLog[traceName]), 
        name: traceName}
    ));
    Plotly.newPlot(container, traceList, {
      width: config.width,
      height: config.height,
      xaxis: {title: config.xLabel},
      yaxis: {title: config.yLabel}
    })
  }
}

export const render = {
  // seriesConfig: {values: [[{x:1,y:3}, {x:2,y:4},... ],[],...], series: ['name1', 'name2',...]}
  // config: {width:450, height:320,...}
  linechart: (container, seriesConfig, config) => {
    // Plotly addtraces
    const traceList = seriesConfig.series.map((seriesName, idx) => (
      {
        x: seriesConfig.values[idx].map(onePoint => onePoint.x),
        y: seriesConfig.values[idx].map(onePoint => onePoint.y),
        name: seriesName
      }
    ));
    Plotly.newPlot(container, traceList, {
      width: config.width,
      height: config.height,
      xaxis: {title: "FPR"},
      yaxis: {title: "TPR"}
    })
  }
}

