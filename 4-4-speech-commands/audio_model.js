const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const wav = require('node-wav');
const path = require('path');
const {Dataset} = require('./dataset');

class AudioModel {
  #model;
  #labels;
  #dataset;
  #featureExtractor;
  constructor(inputShape, labels, dataset, featureExtractor) {
    this.#labels = labels;
    this.#dataset = dataset;
    this.#featureExtractor = featureExtractor 
    this.#model = this.#createModel(inputShape);
  }

  #createModel(inputShape) {
    const model = tf.sequential();
    model.add(tf.layers.conv2d({filters: 8, kernelSize: [4,2], activation:'relu', inputShape}));
    model.add(tf.layers.maxPooling2d({poolSize: [2,2], strides: [2,2]}));
    model.add(tf.layers.conv2d({filters: 32, kernelSize: [4,2], activation:'relu'}));
    model.add(tf.layers.maxPooling2d({poolSize: [2,2], strides: [2,2]}));
    model.add(tf.layers.conv2d({filters: 32, kernelSize: [4,2], activation:'relu'}));
    model.add(tf.layers.maxPooling2d({poolSize: [2,2], strides: [2,2]}));
    model.add(tf.layers.conv2d({filters: 32, kernelSize: [4,2], activation:'relu'}));
    model.add(tf.layers.maxPooling2d({poolSize: [2,2], strides: [1,2]}));
    model.add(tf.layers.flatten({}));
    model.add(tf.layers.dropout({rate:0.25}));
    model.add(tf.layers.dense({units: 2000, activation: 'relu'}));
    model.add(tf.layers.dropout({rate:0.5}));
    model.add(tf.layers.dense({units: this.#labels.length, activation:'softmax'}));
    model.compile({loss: 'categoricalCrossentropy', optimizer: tf.train.sgd(0.01), metrics: ['accuracy']});
    model.summary();
    return model;
  }

  async loadAll(dir, callback) {
    const promises = [];
    // ('zero', 0), ('call', 1)
    this.labels.forEach(async (label, index) =>{
      callback(`loading label: ${label} (${index})`);
      promises.push(

      );
    })
    let allSpecs = await Promise.all(promises);
  }
}

module.exports = {AudioModel};