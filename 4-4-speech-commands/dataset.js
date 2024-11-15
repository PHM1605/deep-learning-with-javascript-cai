const tf = require("@tensorflow/tfjs-core");

class Dataset {
  #xs;
  #ys;
  constructor(numClass) {}
  addExamples(examples, labels) {
    if (this.#xs === null) {
      // to keep Dataset's xs and ys; to ensure when calling 'addExamples' in tf.tidy(), we still not dispose 2 Tensors
      this.#xs = tf.keep(examples);
      this.#ys = tf.keep(labels);
    } else {
      const oldX = this. #xs;
      this.#xs = tf.keep(oldX.concat(examples, 0));
      const oldY = this.#ys;
      this.#ys = tf.keep(oldY.concat(labels, 0));
      oldX.dispose();
      oldY.dispose();
    }
  }
}

module.exports = {Dataset};