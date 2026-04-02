let session;
let metadata;

function softmax(values) {
  const maxVal = Math.max(...values);
  const exps = values.map(v => Math.exp(v - maxVal));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / sum);
}

function standardize(raw, mean, scale) {
  return raw.map((v, i) => (v - mean[i]) / scale[i]);
}

function getInputs() {
  return [
    parseFloat(document.getElementById('bill_length_mm').value),
    parseFloat(document.getElementById('bill_depth_mm').value),
    parseFloat(document.getElementById('flipper_length_mm').value),
    parseFloat(document.getElementById('body_mass_g').value)
  ];
}

function showResult(text) {
  document.getElementById('result').textContent = text;
}

async function loadModel() {
  try {
    metadata = await fetch('./metadata.json').then(r => {
      if (!r.ok) throw new Error('Could not load metadata.json');
      return r.json();
    });

    if (!window.ort) {
      throw new Error('ONNX Runtime script did not load');
    }

    session = await window.ort.InferenceSession.create('./penguin_model.onnx');
    document.getElementById('status').textContent =
      `Model ready. Test accuracy: ${metadata.test_accuracy}`;
  } catch (err) {
    document.getElementById('status').textContent =
      'Model failed to load: ' + err.message;
    console.error(err);
  }
}

async function predict() {
  if (!session || !metadata) {
    showResult('Model is not loaded yet.');
    return;
  }

  const raw = getInputs();
  if (raw.some(v => Number.isNaN(v))) {
    showResult('Please enter valid numeric values.');
    return;
  }

  const scaled = standardize(raw, metadata.scaler_mean, metadata.scaler_scale);
  const inputTensor = new window.ort.Tensor('float32', Float32Array.from(scaled), [1, 4]);
  const output = await session.run({ input: inputTensor });
  const logits = Array.from(output.logits.data);
  const probs = softmax(logits);

  let bestIdx = 0;
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > probs[bestIdx]) bestIdx = i;
  }

  const label = metadata.class_names[bestIdx];
  const conf = (probs[bestIdx] * 100).toFixed(2);
  showResult(`${label} (${conf}% confidence)`);
}

function loadSample() {
  document.getElementById('bill_length_mm').value = 46.5;
  document.getElementById('bill_depth_mm').value = 14.5;
  document.getElementById('flipper_length_mm').value = 213;
  document.getElementById('body_mass_g').value = 4400;
}

document.getElementById('predictBtn').addEventListener('click', predict);
document.getElementById('sampleBtn').addEventListener('click', loadSample);

loadModel();