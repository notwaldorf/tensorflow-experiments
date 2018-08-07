// Small hack so that this file can be used both in the web demo and in node
let IS_RUNNING_IN_NODE = (typeof window === 'undefined');
const tf = IS_RUNNING_IN_NODE ? require('@tensorflow/tfjs') : window.tf;

// Got data from https://www.zillow.com/research/data/
// 1 bedroom time series, by city.
let Data = {};
let Model = {};

let LEARNING_RATE = 0.001;
let EPOCHS = 100;
let MAX_LOSS = 0.00000001;
let UNITS = 60;

// Small hack so that this file can be used both in the web demo and in node.
if (IS_RUNNING_IN_NODE) {
  require('@tensorflow/tfjs-node');
  var fs = require('fs');

  fs.readFile('./data.json', 'utf8', async (err, data) => {
    if (err) throw err;
    await dataReceived(JSON.parse(data));
    train();
  });
} else {
  fetch('./data.json')
  .then(function(response) {
    return response.json();
  })
  .then(dataReceived);
}

async function dataReceived(obj) {
  let start = Date.now();
  Data = parseData(obj);
  console.log(`Parsing data: ${Date.now() - start} ms`);

  start = Date.now();
  initModel();
  console.log(`Initializing model: ${Date.now() - start} ms`);
  replot();

  start = Date.now();
  const prediction = await getNormalizedPrediction(Data.trainingData[0]);
  console.log(`First inference: ${Date.now() - start} ms`);
  replot(prediction);
}

function parseData(obj) {
  // The data is:
  // - 11 entries
  // - columns: RegionName/State/SizeRank/1996-04/1996-05...

  // Names of cities
  let cities = [];
  let dates = [];
  let prices = [];

  // min/max price
  let minPrice = 100000;
  let maxPrice = -1;
  const ignoredCols = ['RegionName', 'State', 'SizeRank'];

  for (let entry of obj) {
    // Save this just in case we need it.
    cities.push(entry['RegionName']);

    // Add it to the select dropdown if in the web demo.
    if (!IS_RUNNING_IN_NODE) {
      const opt = document.createElement('option');
      opt.textContent = entry['RegionName'];
      select.add(opt);
    }

    const theseCols = Object.keys(entry);
    // If this entry doesn't have any new columns, move on.
    if (ignoredCols.length + dates.length !== theseCols.length) {
      // Go through all the columns, and save all the ones that are
      // the dates (i.e. aren't names) and we haven't seen before
      for (let column of Object.keys(entry)) {
         if (!ignoredCols.includes(column) &&
             !dates.includes(column)) {
           dates.push(column);
         }
      }
    }

    // Save the prices for this city. We're assuming not all cities have all the dates.
    const values = [];
    for (let date of dates) {
      values.push(entry[date]);
    }
    prices.push(values);
    minPrice = Math.min(minPrice, Math.min(...values));
    maxPrice = Math.max(maxPrice, Math.max(...values));
  }
  return {cities, dates, prices, minPrice, maxPrice};
}

/*
 LSTM stuff
*/
function initModel() {
  // Normalize the data.
  Data.trainingData = getTrainingData(Data);

  Model = tf.sequential();
  Model.add(tf.layers.lstm({
      units: UNITS,
      inputShape: [Data.trainingData[0].length, 1],
      //returnSequences: true
  }));

  Model.add(tf.layers.dense({
    units: 1,
    activation: 'sigmoid'   // try softplus?
  }));

  Model.compile({
    optimizer: tf.train.adam(LEARNING_RATE),  // TODO:change this.
    loss: 'meanSquaredError'
  });
}

async function train() {
  console.log('Starting training...');
  let start = Date.now();

  for(let e = 0; e < EPOCHS; e++) {
    var totalLoss = 0;
    for(let i = 0; i < Data.trainingData.length; i++) {
      var info = await Model.fit(Data.trainingData[i].input, Data.trainingData[i].output, {epochs: 1});
      totalLoss += info.history.loss[0];
     // console.log(`training data ${i}, totalLoss: ${totalLoss}`)
    }
    var avgLoss = totalLoss/Data.trainingData.length;
    console.log(`[${e}/${EPOCHS}] Average Loss: ${avgLoss};`);

    if(MAX_LOSS >= avgLoss)
      break;

    // Use tf.nextFrame to not block the browser.
    await tf.nextFrame();
  }

  console.log(`Training: ${Date.now() - start} ms`);

  start = Date.now();
  const prediction = await getNormalizedPrediction(Data.trainingData[0]);
  console.log(`Inference: ${Date.now() - start} ms`);
  replot(prediction);
}

async function getNormalizedPrediction(trainingData) {
  let prediction = await Model.predict(trainingData.input);
  prediction = prediction.get(0,0);
  return denormalize(prediction, Data.minPrice, Data.maxPrice);
}

/*
Plotting stuff
*/
async function replot(prediction) {
  // No plotting in node.
  if (IS_RUNNING_IN_NODE) {
    return;
  }
  const index = select.selectedIndex;
  plot(Data.dates, Data.prices[index], prediction);
}


function plot(xs, ys, prediction) {
  const trace1 = {
    x: xs,
    y: ys,
    mode: 'lines',
    marker: { size: 12, color:'#29B6F6' },
    name: 'Training Data'
  };

  const trace2 = {
    x: [xs[xs.length - 1]],
    y: [prediction],
    mode: 'markers',
    marker: { size: 12, color:'#EE018A' },
    name: 'Prediction'
  };
  const layout = {

  };
  Plotly.newPlot('graph', [trace1, trace2], layout, {displayModeBar: false});
}

/*
General utils
*/
function getTrainingData(data) {
  const trainingData = [];
  for(let i = 0; i < data.cities.length; i++) {
    let ys = data.prices[i];
    let normalized = [];

    for(let j = 0; j < ys.length - 1; j++) {
      normalized.push(normalize(ys[j], data.minPrice, data.maxPrice));
    }
    const last = normalize(ys[ys.length - 1], data.minPrice, data.maxPrice);
    trainingData.push({
      input : tf.tensor3d(normalized, [1, normalized.length, 1]),
      output : tf.tensor2d([last], [1, 1])
    });
  }
  return trainingData;
}

// Converts values to the range between values 0 and 1;
function normalize(num, min, max) {
	return (num - min) * (1/(max - min));
}
// Reconverts values from range between values 0 and 1 to range between Min and Max;
function denormalize(num, min, max) {
	return (num / (1/(max - min))) + min;
}
