const vorpal = require('vorpal')();

let labels;

vorpal.command('create_model [labels...]')
  .alias('c')
  .action((args, cb) =>{
    console.log(args.labels)
    labels = args.labels;
    
  })

vorpal.show();