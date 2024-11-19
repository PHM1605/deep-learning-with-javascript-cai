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
    this.#featureExtractor.config({
      melCount: 40,
      bufferLength: 480,
      hopLength: 160,
      targetSr: 16000,
      isMfccEnabled: true,
      duration: 1.0
    })
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

  async loadData(dir, label, callback) {
    const index = this.#labels.indexOf(label);
    const specs = await this.#loadDataArray(dir, callback);
    console.log("DEBUG", specs)
    this.#dataset.addExamples(
      this.#melSpectrogramToInput(specs), 
      tf.oneHot(tf.fill([specs.length], index, 'int32'), this.#labels.length) // all specs has the label of "index", in oneHot form
    )
  }

  #loadDataArray(dir, callback) {
    return new Promise((resolve, reject) =>{
      fs.readdir(dir, (err, filenames) => {
        if (err) {
          reject(err);
        }
        let specs = [];
        filenames.forEach(filename => {
          callback('decoding ' + dir + '/' + filename);
          console.log("ERROR HERE: ", this.#decode(dir + '/' + filename))
          const spec = this.#splitSpecs(this.#decode(dir + '/' + filename));
          if (spec) {
            specs = specs.concat(spec);
          }
          callback('decoding ' + dir + '/' + filename + '...done');
        });
        resolve(specs);
      })
    })
  }

  #decode(filename) {
    const result = wav.decode(fs.readFileSync(filename));
    return this.#featureExtractor.start(result.channelData[0]);
  }

  async train(epochs, trainCallback) {
    return this.#model.fit(
      this.#dataset.xs,
      this.#dataset.ys,
      { batchSize: 64, epochs: epochs||100, shuffle:true, validationSplit:0.1, callbacks: trainCallback }
    )
  }

  #splitSpecs(spec) {
    if (spec.length >=98) {
      const output = [];
      for (let i=0; i<=(spec.length-98); i+=32) {
        output.push(spec.slice(i, i+98));
      }
      return output;
    }
    return undefined;
  }
  // specs: [batch, time, mels]
  #melSpectrogramToInput(specs) {
    const batch = specs.length;
    const times = specs[0].length;
    const freqs = specs[0][0].length;
    const data = new Float32Array(batch*times*freqs);
    for (let j=0; j<batch; j++) {
      const spec = specs[j];
      for (let i=0; i<times; i++) {
        const mel = spec[i];
        const offset = j*freqs* times + i*freqs;
        data.set(mel, offset);
      }
    }
    const shape = [batch, times, freqs, 1];
    return tf.tensor4d(data, shape);
  }
}

module.exports = {AudioModel};