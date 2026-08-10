/* ═══════════════════════════════════════════════════════════
   ISL TRANSLATOR — FILE UPLOAD & TESTER CONTROLLER
═══════════════════════════════════════════════════════════ */

'use strict';

let BACKEND = window.location.origin;
let currentMode = 'image';
let frameBuffer = [];
let isProcessingVideo = false;

let fileInput, previewWrap, previewImg, previewVid, canvas, ctx, processBtn, modeSelect, resultBox, dropZone;

document.addEventListener('DOMContentLoaded', () => {
  fileInput   = document.getElementById('file-input');
  previewWrap = document.getElementById('preview-wrap');
  previewImg  = document.getElementById('preview-img');
  previewVid  = document.getElementById('preview-vid');
  canvas      = document.getElementById('preview-canvas');
  if (canvas) ctx = canvas.getContext('2d');
  processBtn  = document.getElementById('process-btn');
  modeSelect  = document.getElementById('mode-select');
  resultBox   = document.getElementById('result-box');
  dropZone    = document.getElementById('drop-zone');

  setupUploadListeners();
});

// MediaPipe Setup
const hands = new Hands({ locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}` });
hands.setOptions({ maxNumHands: 2, modelComplexity: 1, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });

function flattenLandmarks(results) {
  const flat = new Float32Array(126).fill(0);
  const h = results.multiHandLandmarks || [];
  const s = results.multiHandedness || [];
  for (let i = 0; i < Math.min(h.length, 2); i++) {
    const isRight = s[i] && s[i].label === 'Right';
    const offset = isRight ? 63 : 0;
    for (let j = 0; j < 21; j++) {
      flat[offset+j*3] = h[i][j].x; flat[offset+j*3+1] = h[i][j].y; flat[offset+j*3+2] = h[i][j].z;
    }
  }
  return Array.from(flat);
}

hands.onResults(results => {
  if (!canvas || !ctx) return;
  ctx.save(); ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (results.multiHandLandmarks) {
    for (let i = 0; i < results.multiHandLandmarks.length; i++) {
      drawConnectors(ctx, results.multiHandLandmarks[i], HAND_CONNECTIONS, { color:'#4f8ef7', lineWidth:3 });
      drawLandmarks(ctx, results.multiHandLandmarks[i], { color:'#22c55e', lineWidth:1, radius:3 });
    }
  }
  ctx.restore();

  const flat = flattenLandmarks(results);
  if (currentMode === 'image') {
    sendPrediction('/predict', { landmarks: flat, mode: modeSelect.value, engine: 'auto' });
  } else if (currentMode === 'video' && isProcessingVideo) {
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) frameBuffer.push(flat);
  }
});

function setupUploadListeners() {
  if (!dropZone) return;
  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault(); dropZone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });

  fileInput.addEventListener('change', e => {
    const file = e.target.files[0];
    if (file) handleFile(file);
  });

  processBtn.addEventListener('click', async () => {
    processBtn.disabled = true;
    processBtn.innerHTML = '<div class="spinner"></div> Processing…';
    canvas.width  = previewWrap.clientWidth;
    canvas.height = previewWrap.clientHeight;
    resultBox.style.display = 'none';

    if (currentMode === 'image') {
      await hands.send({ image: previewImg });
    } else {
      frameBuffer = [];
      isProcessingVideo = true;
      previewVid.play();
      const processVideoFrame = async () => {
        if (previewVid.paused || previewVid.ended) {
          isProcessingVideo = false;
          sendPrediction('/predict_sequence', { frames: frameBuffer });
          return;
        }
        await hands.send({ image: previewVid });
        requestAnimationFrame(processVideoFrame);
      };
      processVideoFrame();
    }
  });
}

function handleFile(file) {
  previewWrap.style.display = 'block';
  resultBox.style.display   = 'none';
  processBtn.disabled       = false;
  processBtn.innerHTML      = '<span class="material-icons-round">memory</span> Run Neural Prediction';
  if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);

  const chip = document.getElementById('file-chip');
  const name = document.getElementById('file-name');
  if (chip) chip.style.display = 'inline-flex';
  if (name) name.textContent   = file.name.length > 28 ? file.name.slice(0, 25) + '…' : file.name;

  if (file.type.startsWith('image/')) {
    currentMode = 'image';
    if (modeSelect.value === 'word') modeSelect.value = 'alphabet';
    previewImg.style.display = 'block';
    previewVid.style.display = 'none';
    previewImg.src = URL.createObjectURL(file);
  } else if (file.type.startsWith('video/')) {
    currentMode = 'video';
    modeSelect.value = 'word';
    previewImg.style.display = 'none';
    previewVid.style.display = 'block';
    previewVid.src = URL.createObjectURL(file);
  }
}

async function sendPrediction(endpoint, payload) {
  try {
    const res    = await fetch(BACKEND + endpoint, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
    const result = await res.json();

    resultBox.className = 'result-card';
    resultBox.style.display = 'flex';
    document.getElementById('res-tag').textContent  = 'Detected Sign';
    document.getElementById('res-val').textContent  = result.label || '---';
    document.getElementById('res-conf').textContent = `CONF: ${Math.round((result.confidence || 0) * 100)}%`;
    document.getElementById('res-eng').textContent  = `ENG: ${(result.engine || 'none').toUpperCase()}`;
  } catch(e) {
    resultBox.className = 'result-card error-card';
    resultBox.style.display = 'flex';
    document.getElementById('res-tag').textContent  = 'Error';
    document.getElementById('res-val').textContent  = 'ERR';
    document.getElementById('res-conf').textContent = 'Connection Failed';
    document.getElementById('res-eng').textContent  = 'N/A';
  }
  processBtn.disabled = false;
  processBtn.innerHTML = '<span class="material-icons-round">memory</span> Run Neural Prediction';
}
