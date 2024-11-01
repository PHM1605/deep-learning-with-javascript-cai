import * as tf from "@tensorflow/tfjs"

const test = tf.tensor2d([[1,2],[3,4]])
const res = test.div(test.sum(-1).expandDims(1)).dataSync()
console.log(res)