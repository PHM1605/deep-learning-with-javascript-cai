const vorpal = require('vorpal')();
const {AudioModel} = require('./audio_model');
const {Dataset} = require('./dataset');
const {WavFileFeatureExtractor} = require('./wav_file_feature_extractor');

let labels;
const MODEL_SHAPE = [98, 40, 1];
let model; 

vorpal.command('create_model [labels...]')
  .alias('c')
  .action((args, cb) =>{
    console.log(args.labels)
    labels = args.labels;
    model = new AudioModel(MODEL_SHAPE, labels, new Dataset(labels.length), new WavFileFeatureExtractor());
    cb();
  })

vorpal.command('load_dataset <dir> <label>', 'Load dataset from the directory with the label')
  .alias('l')
  .action(args => {
    return model.loadData(args.dir, args.label, (text) => {
      console.log('Load dataset: ' + text);
    });
  })

vorpal.command('load_dataset all <dir>', "Load all data from the root directory by the labels")
  .alias('la')
  .action((args) => {
    console.log("Load dataset...")
    return model.loadAll(args.dir, (text, finished) => {
      if (finished) {
        console.log(`Succeed load all: ${text}`);
      }
    })
  })

vorpal.command('dataset size', 'Show the size of the dataset')
  .alias('d')
  .action((args, cb) => {
    console.log(`dataset size = ${model.size()}`);
    cb();
  })

vorpal.command('train [epoch]')
  .alias('t')
  .description('train all audio dataset')
  .action(args => {
    console.log("...Training model")
    // 10 is radix
    return model.train(parseInt(args.epoch, 10) || 20, {
      onBatchEnd: async (batch, logs) => {
        console.log(`loss: ${logs.loss.toFixed(5)}`);
      },
      onEpochEnd: async (epoch, logs) => {
        console.log(`epoch: ${epoch}, loss: ${logs.loss.toFixed(5)}, accuracy: ${logs.acc.toFixed(5)}, validation accuracy: ${logs.val_acc.toFixed(5)}`);
      } 
    })
  })

vorpal.command('save_model <filename>')
  .alias('s')
  .description('save the audio model')
  .action(args => {
    console.log(`saving to ${args.filename}...`)
    return model.save(args.filename).then(() => {
      console.log(`${args.filename} saved.`);
    }, () => console.log(`failed to save ${args.filename}`))
  })
vorpal.show();

module.exports = {
  MODEL_SHAPE
}