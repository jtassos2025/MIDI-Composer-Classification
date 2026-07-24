/* Train equivalent Conv1D and LSTM classifiers with TensorFlow.js on CPU. */

const fs = require('fs');
const path = require('path');
const tf = require('/tmp/composer_tfjs_node/node_modules/@tensorflow/tfjs-node');

const projectDir = __dirname;
const artifactDir = path.join(projectDir, 'artifacts');
const data = JSON.parse(fs.readFileSync(path.join(artifactDir, 'preprocessed_data.json'), 'utf8'));

function makeTensors() {
  const nFeatures = data.feature_columns.length;
  const shape = (rows) => [rows.length, nFeatures, 1];
  if (nFeatures % 3 !== 0) throw new Error('The grouped LSTM input requires a feature count divisible by 3.');
  const lstmShape = (rows) => [rows.length, nFeatures / 3, 3];
  return {
    nFeatures,
    nClasses: data.class_names.length,
    xTrain: tf.tensor3d(data.X_train.flat(2), shape(data.X_train)),
    xVal: tf.tensor3d(data.X_val.flat(2), shape(data.X_val)),
    xTest: tf.tensor3d(data.X_test.flat(2), shape(data.X_test)),
    xTrainLstm: tf.tensor3d(data.X_train.flat(2), lstmShape(data.X_train)),
    xValLstm: tf.tensor3d(data.X_val.flat(2), lstmShape(data.X_val)),
    xTestLstm: tf.tensor3d(data.X_test.flat(2), lstmShape(data.X_test)),
    yTrain: tf.tensor1d(data.y_train, 'float32'),
    yVal: tf.tensor1d(data.y_val, 'float32'),
    yTest: tf.tensor1d(data.y_test, 'float32'),
  };
}

function buildCNN(nFeatures, nClasses) {
  const model = tf.sequential();
  model.add(tf.layers.conv1d({
    inputShape: [nFeatures, 1], filters: 24, kernelSize: 3,
    padding: 'same', activation: 'relu',
  }));
  model.add(tf.layers.conv1d({filters: 32, kernelSize: 3, padding: 'same', activation: 'relu'}));
  model.add(tf.layers.maxPooling1d({poolSize: 2}));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dropout({rate: 0.30}));
  model.add(tf.layers.dense({units: 48, activation: 'relu'}));
  model.add(tf.layers.dropout({rate: 0.20}));
  model.add(tf.layers.dense({units: nClasses, activation: 'softmax'}));
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'sparseCategoricalCrossentropy',
    metrics: ['accuracy'],
  });
  return model;
}

function buildLSTM(nFeatures, nClasses) {
  const model = tf.sequential();
  model.add(tf.layers.lstm({inputShape: [nFeatures / 3, 3], units: 32, returnSequences: false}));
  model.add(tf.layers.dropout({rate: 0.25}));
  model.add(tf.layers.dense({units: 48, activation: 'relu'}));
  model.add(tf.layers.dropout({rate: 0.20}));
  model.add(tf.layers.dense({units: nClasses, activation: 'softmax'}));
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'sparseCategoricalCrossentropy',
    metrics: ['accuracy'],
  });
  return model;
}

async function saveModel(model, outputDir) {
  fs.mkdirSync(outputDir, {recursive: true});
  await model.save(tf.io.withSaveHandler(async (artifacts) => {
    const weightsFile = 'model.weights.bin';
    fs.writeFileSync(path.join(outputDir, weightsFile), Buffer.from(artifacts.weightData));
    const modelJson = {
      format: 'layers-model',
      generatedBy: `TensorFlow.js tfjs-layers v${tf.version.tfjs}`,
      convertedBy: null,
      modelTopology: artifacts.modelTopology,
      weightsManifest: [{paths: [weightsFile], weights: artifacts.weightSpecs}],
    };
    fs.writeFileSync(path.join(outputDir, 'model.json'), JSON.stringify(modelJson, null, 2));
    return {modelArtifactsInfo: tf.io.getModelArtifactsInfoForJSON(artifacts)};
  }));
}

function confusionMatrix(yTrue, yPred, nClasses) {
  const cm = Array.from({length: nClasses}, () => Array(nClasses).fill(0));
  yTrue.forEach((actual, i) => { cm[actual][yPred[i]] += 1; });
  return cm;
}

function metricsFromPredictions(yTrue, yPred, classNames) {
  const cm = confusionMatrix(yTrue, yPred, classNames.length);
  const perClass = {};
  const values = [];
  for (let i = 0; i < classNames.length; i += 1) {
    const tp = cm[i][i];
    const fp = cm.reduce((sum, row, r) => sum + (r === i ? 0 : row[i]), 0);
    const fn = cm[i].reduce((sum, value, c) => sum + (c === i ? 0 : value), 0);
    const support = cm[i].reduce((a, b) => a + b, 0);
    const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
    const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
    const f1 = precision + recall === 0 ? 0 : 2 * precision * recall / (precision + recall);
    perClass[classNames[i]] = {precision, recall, f1, support};
    values.push({precision, recall, f1});
  }
  const accuracy = yTrue.filter((v, i) => v === yPred[i]).length / yTrue.length;
  const macro = (key) => values.reduce((sum, item) => sum + item[key], 0) / values.length;
  return {
    accuracy,
    macro_precision: macro('precision'),
    macro_recall: macro('recall'),
    macro_f1: macro('f1'),
    confusion_matrix: cm,
    per_class: perClass,
  };
}

async function trainOne(name, buildModel, tensors) {
  console.log(`\nTraining ${name.toUpperCase()} model...`);
  const model = buildModel(tensors.nFeatures, tensors.nClasses);
  model.summary();
  const start = Date.now();
  let bestValLoss = Infinity;
  let bestEpoch = 0;
  let epochsWithoutImprovement = 0;
  let bestWeights = null;
  const moderatedClassWeights = Object.fromEntries(
    Object.entries(data.class_weights).map(([key, value]) => [key, Math.sqrt(value)])
  );
  const selectedClassWeights = name === 'lstm' ? data.class_weights : moderatedClassWeights;
  const history = await model.fit(tensors.xTrain, tensors.yTrain, {
    epochs: name === 'lstm' ? 40 : 30,
    batchSize: 64,
    shuffle: true,
    classWeight: selectedClassWeights,
    validationData: [tensors.xVal, tensors.yVal],
    callbacks: [
      new tf.CustomCallback({
        onEpochEnd: async (epoch, logs) => {
          console.log(`${name} epoch ${epoch + 1}: loss=${logs.loss.toFixed(4)} acc=${logs.acc.toFixed(4)} val_loss=${logs.val_loss.toFixed(4)} val_acc=${logs.val_acc.toFixed(4)}`);
          if (logs.val_loss < bestValLoss - 0.0005) {
            if (bestWeights) bestWeights.forEach((weight) => weight.dispose());
            bestWeights = model.getWeights().map((weight) => weight.clone());
            bestValLoss = logs.val_loss;
            bestEpoch = epoch + 1;
            epochsWithoutImprovement = 0;
          } else {
            epochsWithoutImprovement += 1;
            if (epochsWithoutImprovement >= 6) model.stopTraining = true;
          }
          await tf.nextFrame();
        },
      }),
    ],
  });
  if (bestWeights) model.setWeights(bestWeights);
  const probabilities = model.predict(tensors.xTest);
  const predictionsTensor = probabilities.argMax(-1);
  const yPred = Array.from(await predictionsTensor.data());
  const metrics = metricsFromPredictions(data.y_test, yPred, data.class_names);
  metrics.epochs_trained = history.epoch.length;
  metrics.best_epoch = bestEpoch;
  metrics.training_seconds = (Date.now() - start) / 1000;
  metrics.parameter_count = model.countParams();
  metrics.history = history.history;
  fs.writeFileSync(path.join(artifactDir, `${name}_metrics.json`), JSON.stringify(metrics, null, 2));
  fs.writeFileSync(path.join(artifactDir, `${name}_predictions.json`), JSON.stringify({actual: data.y_test, predicted: yPred}, null, 2));
  await saveModel(model, path.join(artifactDir, `${name}_tfjs_model`));
  probabilities.dispose();
  predictionsTensor.dispose();
  model.dispose();
  console.log(`${name.toUpperCase()} test accuracy=${metrics.accuracy.toFixed(4)} macro_f1=${metrics.macro_f1.toFixed(4)}`);
  return metrics;
}

async function main() {
  await tf.setBackend('tensorflow');
  await tf.ready();
  const tensors = makeTensors();
  const modelFilter = process.env.MODEL_FILTER || 'both';
  const cnn = modelFilter === 'lstm'
    ? JSON.parse(fs.readFileSync(path.join(artifactDir, 'cnn_metrics.json'), 'utf8'))
    : await trainOne('cnn', buildCNN, tensors);
  const lstmInputs = {
    ...tensors,
    xTrain: tensors.xTrainLstm,
    xVal: tensors.xValLstm,
    xTest: tensors.xTestLstm,
  };
  const lstm = modelFilter === 'cnn'
    ? JSON.parse(fs.readFileSync(path.join(artifactDir, 'lstm_metrics.json'), 'utf8'))
    : await trainOne('lstm', buildLSTM, lstmInputs);
  const comparison = {
    implementation: 'TensorFlow.js CPU validation run',
    class_names: data.class_names,
    feature_count: data.feature_columns.length,
    split_sizes: {train: data.y_train.length, validation: data.y_val.length, test: data.y_test.length},
    cnn,
    lstm,
  };
  fs.writeFileSync(path.join(artifactDir, 'model_comparison.json'), JSON.stringify(comparison, null, 2));
  Object.values(tensors).forEach((value) => { if (value && value.dispose) value.dispose(); });
  console.log('\nTraining complete.');
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
