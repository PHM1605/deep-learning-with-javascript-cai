const vorpal = require('vorpal')();
const {AudioModel} = require('./audio_model');
const {Dataset} = require('./dataset');
const {WavFileFeatureExtractor} = require('./wav_file_feature_extractor');

let labels;
const MODEL_SHAPE = [98, 40, 1];

vorpal.command('create_model [labels...]')
  .alias('c')
  .action((args, cb) =>{
    console.log(args.labels)
    labels = args.labels;
    model = new AudioModel(MODEL_SHAPE, labels, new Dataset(labels.length), new WavFileFeatureExtractor());
    cb();
  })


  
vorpal.show();

module.exports = {
  MODEL_SHAPE
}