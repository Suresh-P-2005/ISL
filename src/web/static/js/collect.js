/* ═══════════════════════════════════════════════════════════
   ISL TRANSLATOR — DATASET COLLECTOR CONTROLLER
═══════════════════════════════════════════════════════════ */

'use strict';

let BACKEND = window.location.origin;

const SIGN_DB = {
  "Hello":{hands:1},"Yes":{hands:1},"No":{hands:1},"Help":{hands:1},"Good":{hands:1},"Bad":{hands:1},
  "Thank You":{hands:1},"Sorry":{hands:1},"Please":{hands:1},"Wait":{hands:1},"Water":{hands:1},"Food":{hands:1},
  "Eat":{hands:1},"Drink":{hands:1},"Sick":{hands:1},"Mother":{hands:1},"Father":{hands:1},"Medicine":{hands:1},
  "Toilet":{hands:1},"Call":{hands:1},"Pain":{hands:1},"Who":{hands:1},"More":{hands:2},"Finished":{hands:2},
  "Again":{hands:2},"Baby":{hands:2},"Friend":{hands:2},"Family":{hands:2},"Stop":{hands:2},"Danger":{hands:2},
  "Come":{hands:2},"Where":{hands:2},"What":{hands:2},"When":{hands:2},"Which":{hands:2},"Why":{hands:2},
  "I Love You":{hands:2},"Understand":{hands:2},"Sister":{hands:2},"Brother":{hands:2},"Sleep":{hands:2},
  "Tired":{hands:2},"Home":{hands:2},"Fire":{hands:2}
};

const LABELS = {
  alphabet: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split(''),
  number:   '0123456789'.split(''),
  word: [],
  static_word: []
};

let customWordHands = {};

function loadCustomWords() {
  try {
    const saved = localStorage.getItem('isl_custom_words');
    if (saved) {
      const obj = JSON.parse(saved);
      LABELS.word        = obj.words || [];
      LABELS.static_word = obj.static_words || [];
      customWordHands    = obj.hands || {};
    }
  } catch(e) {}
}

function saveCustomWords() {
  try {
    localStorage.setItem('isl_custom_words', JSON.stringify({
      words:        LABELS.word,
      static_words: LABELS.static_word,
      hands:        customWordHands
    }));
  } catch(e) {}
}

loadCustomWords();

const MODE_CONFIG = {
  alphabet:    { type:'image', target:1200, label:'IMAGE MODE — 200 photos per sign' },
  number:      { type:'image', target:400,  label:'IMAGE MODE — 200 photos per sign' },
  static_word: { type:'image', target:400,  label:'IMAGE MODE — 200 photos per static word' },
  word:        { type:'video', target:20,   label:'VIDEO MODE — 20 recordings × 5s per sign' }
};

let currentMode = 'alphabet', currentLabel = '', isCameraOn = false;
let isAutoCapture = false, autoInterval = null, autoSpeed = 200;
let isRecording = false, isCountdown = false, isAutoRecord = false;
let autoRecordTarget = 20, autoRecordRemaining = 0;
let handDetected = false, numHandsDetected = 0;
let samplesCollected = {};
let lastLandmarks = null;
let videoFrameBuffer = [];
let recordingTimer = null, recordStartTime = 0;
const VIDEO_DURATION  = 5000;
const COUNTDOWN_SECS  = 3;
let wmHandChoice = 1;

let videoEl, canvasEl, ctx, camBtn, labelSel, logBox, progressFill, progressText, handStatus, handsCount, flash, recOverlay, cntOverlay;

document.addEventListener('DOMContentLoaded', () => {
  videoEl     = document.getElementById('video');
  canvasEl    = document.getElementById('canvas');
  if (canvasEl) ctx = canvasEl.getContext('2d');
  camBtn      = document.getElementById('cam-btn');
  labelSel    = document.getElementById('label-select');
  logBox      = document.getElementById('log-box');
  progressFill= document.getElementById('progress-fill');
  progressText= document.getElementById('progress-text');
  handStatus  = document.getElementById('hand-status');
  handsCount  = document.getElementById('hands-count');
  flash       = document.getElementById('flash');
  recOverlay  = document.getElementById('rec-overlay');
  cntOverlay  = document.getElementById('countdown-overlay');

  setupCollectEventListeners();
  switchMode('alphabet');
  renderWordChips();
  setTimeout(loadStats, 500);
});

function log(msg, type = 'info') {
  if (!logBox) return;
  const s = document.createElement('span');
  s.className = `log-${type}`;
  s.textContent = `[${new Date().toLocaleTimeString()}] ${msg}\n`;
  logBox.appendChild(s);
  logBox.scrollTop = logBox.scrollHeight;
}

function updateLabelList() {
  const cfg = MODE_CONFIG[currentMode];
  if (!cfg) return;
  const target = cfg.target;
  const isVid = cfg.type === 'video';
  const labels = LABELS[currentMode] || [];
  const listEl = document.getElementById('label-list');
  if (!listEl) return;
  let total = 0, done = 0;
  listEl.innerHTML = labels.map(label => {
    const count = samplesCollected[label] || 0; total += count;
    const isDone = count >= target; if (isDone) done++;
    const pct  = Math.min(count / target * 100, 100);
    const hands = customWordHands[label] || (SIGN_DB[label]?.hands) || 1;
    return `
      <div class="label-item ${label === currentLabel ? 'current' : ''}">
        <span class="label-name" style="color:${isDone?'var(--success)':'var(--text-primary)'}">${label}</span>
        <span class="hand-mini">${hands===2?'✋✋':'✋'}</span>
        <span class="type-mini">${isVid?'VID':'IMG'}</span>
        <div class="label-bar-wrap"><div class="label-bar" style="width:${pct}%"></div></div>
        <span class="label-count">${count}</span>
      </div>`;
  }).join('');
  if (document.getElementById('stat-total')) document.getElementById('stat-total').textContent = total;
  if (document.getElementById('stat-done')) document.getElementById('stat-done').textContent  = done;
  if (document.getElementById('train-btn')) document.getElementById('train-btn').disabled = total < 5;
}

function updateProgress() {
  const cfg   = MODE_CONFIG[currentMode];
  if (!cfg) return;
  const count = samplesCollected[currentLabel] || 0;
  const target = cfg.target;
  if (progressFill) progressFill.style.width = `${Math.min(count / target * 100, 100)}%`;
  if (progressText) progressText.textContent = `${count} / ${target}`;
  if (document.getElementById('progress-label')) {
    document.getElementById('progress-label').textContent = `${currentLabel} — ${cfg.type === 'video' ? 'Recordings' : 'Photos'}`;
  }
}

function renderWordChips() {
  const list = document.getElementById('wm-word-list');
  if (!list) return;
  list.innerHTML = '';
  const currentList = LABELS[currentMode] || [];
  if (currentList.length === 0) {
    list.innerHTML = `<span style="font-size:0.75rem;color:var(--text-muted)">No custom words yet for ${currentMode.toUpperCase()}.</span>`;
    return;
  }
  currentList.forEach(word => {
    const hands = customWordHands[word] || 1;
    const chip = document.createElement('div');
    chip.className = 'wm-chip';
    chip.style.cssText = 'display:inline-flex;align-items:center;gap:6px;padding:4px 10px;border-radius:12px;background:var(--bg-glass);border:1px solid var(--border-mid);font-size:0.75rem;margin:3px;';
    chip.innerHTML = `<span>${word}</span><span style="opacity:0.6">${hands===2?'✋✋':'✋'}</span>`;
    list.appendChild(chip);
  });
}

function setupCollectEventListeners() {
  document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      switchMode(btn.dataset.mode);
    });
  });

  if (labelSel) {
    labelSel.addEventListener('change', () => {
      currentLabel = labelSel.value;
      updateSelectedHandUI();
      updateProgress(); updateLabelList();
    });
  }

  document.querySelectorAll('input[name="sign-hand-req"]').forEach(radio => {
    radio.addEventListener('change', () => {
      if (!currentLabel) return;
      const hands = parseInt(radio.value, 10);
      customWordHands[currentLabel] = hands;
      saveCustomWords();
      updateLabelList();
      log(`✋ Set ${currentLabel} requirement to ${hands} Hand${hands===2?'s':''}`, 'info');
    });
  });

  const capBtn = document.getElementById('capture-btn');
  if (capBtn) capBtn.addEventListener('click', captureImage);

  const recBtn = document.getElementById('record-btn');
  if (recBtn) recBtn.addEventListener('click', startCountdownThenRecord);

  const trainBtn = document.getElementById('train-btn');
  if (trainBtn) {
    trainBtn.addEventListener('click', async () => {
      trainBtn.disabled = true;
      trainBtn.innerHTML = '<span class="material-icons-round">hourglass_top</span> Training…';
      log(`Training ${currentMode.toUpperCase()}…`, 'info');
      try {
        const res = await fetch(BACKEND + '/retrain', {
          method:'POST', headers:{'Content-Type':'application/json'},
          body: JSON.stringify({ mode: currentMode })
        });
        if (res.ok) { log('✅ Training complete!', 'ok'); }
        else { log('Training failed', 'err'); }
      } catch(e) { log(`Error: ${e.message}`, 'err'); }
      trainBtn.disabled = false;
      trainBtn.innerHTML = '<span class="material-icons-round">model_training</span> Train Model Now';
    });
  }

  const addBtn = document.getElementById('add-word-btn');
  if (addBtn) {
    addBtn.addEventListener('click', () => {
      const input = document.getElementById('new-sign-input');
      const descInput = document.getElementById('new-sign-desc');
      const radio = document.querySelector('input[name="hand-req-radio"]:checked');

      const word = (input ? input.value : '').trim();
      const desc = (descInput ? descInput.value : '').trim() || 'Custom ISL sign gesture.';
      const hands = radio ? parseInt(radio.value, 10) : 1;

      if (!word) {
        log('Please enter a sign word name.', 'warn');
        return;
      }

      if (!LABELS[currentMode]) LABELS[currentMode] = [];
      if (!LABELS[currentMode].includes(word)) {
        LABELS[currentMode].push(word);
      }
      customWordHands[word] = hands;

      // Persist in localStorage for Tutorial Dictionary sync
      saveCustomWords();
      try {
        const descs = JSON.parse(localStorage.getItem('isl_custom_word_descs') || '{}');
        descs[word] = desc;
        localStorage.setItem('isl_custom_word_descs', JSON.stringify(descs));

        // Sync with backend custom_signs.json database
        fetch(BACKEND + '/add_custom_sign', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ label: word, mode: currentMode, hands: hands, description: desc })
        }).catch(() => {});
      } catch(e) {}

      populateLabels(currentMode);
      if (labelSel) labelSel.value = word;
      currentLabel = word;
      updateSelectedHandUI();
      updateProgress();
      updateLabelList();

      if (input) input.value = '';
      if (descInput) descInput.value = '';

      log(`➕ Added "${word}" (${hands} Hand${hands===2?'s':''}) to ${currentMode.toUpperCase()}`, 'ok');
    });
  }

  if (camBtn) {
    camBtn.addEventListener('click', () => {
      if (!isCameraOn) {
        isCameraOn = true;
        camBtn.className = 'rec-btn stop';
        camBtn.innerHTML = '<span class="material-icons-round">stop_circle</span> Stop Collector Camera';
        camera.start().then(() => log('📷 Camera started', 'ok')).catch(e => { isCameraOn = false; });
      } else {
        isCameraOn = false;
        const stream = videoEl.srcObject;
        if (stream) stream.getTracks().forEach(t => t.stop());
        videoEl.srcObject = null;
        camBtn.className = 'rec-btn photo';
        camBtn.innerHTML = '<span class="material-icons-round">videocam</span> Start Collector Camera';
      }
    });
  }
}

function switchMode(mode) {
  currentMode = mode;
  const cfg = MODE_CONFIG[mode];
  if (!cfg) return;

  const imgCtrl = document.getElementById('image-controls');
  const vidCtrl = document.getElementById('video-controls');
  if (imgCtrl && vidCtrl) {
    if (cfg.type === 'video') {
      imgCtrl.style.display = 'none';
      vidCtrl.style.display = 'block';
    } else {
      imgCtrl.style.display = 'block';
      vidCtrl.style.display = 'none';
    }
  }

  // Toggle Hand Requirement Selector visibility
  const handReqGroup = document.getElementById('hand-req-selector-group');
  if (handReqGroup) {
    if (mode === 'word' || mode === 'static_word') {
      handReqGroup.style.display = 'none';
    } else {
      handReqGroup.style.display = 'block';
    }
  }

  // Only show Add Custom Sign card for Static & Dynamic Word modes
  const addCard = document.getElementById('add-word-card');
  if (addCard) {
    addCard.style.display = (mode === 'static_word' || mode === 'word') ? 'block' : 'none';
  }

  populateLabels(mode);
  loadStats();
}

function updateSelectedHandUI() {
  const sign = currentLabel;
  if (!sign) return;
  const hands = customWordHands[sign] || (SIGN_DB[sign]?.hands) || 1;
  const r1 = document.getElementById('sign-hand-1');
  const r2 = document.getElementById('sign-hand-2');
  if (hands === 2) { if (r2) r2.checked = true; }
  else { if (r1) r1.checked = true; }
}

function populateLabels(mode) {
  if (!labelSel) return;
  labelSel.innerHTML = '';
  (LABELS[mode] || []).forEach(label => {
    const opt = document.createElement('option');
    opt.value = opt.textContent = label;
    labelSel.appendChild(opt);
  });
  currentLabel = (LABELS[mode] || [])[0] || '';
  labelSel.value = currentLabel;
  updateSelectedHandUI();
  updateProgress(); updateLabelList();
}

async function captureImage() {
  if (!lastLandmarks || numHandsDetected === 0) { log('No hand detected', 'warn'); return; }
  try {
    const res = await fetch(BACKEND + '/collect', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ landmarks: Array.from(lastLandmarks), label: currentLabel, mode: currentMode })
    });
    if (res.ok) {
      const data = await res.json();
      samplesCollected[currentLabel] = data.count;
      updateProgress(); updateLabelList();
      log(`📸 ${currentLabel}: ${data.count}`, 'ok');
    }
  } catch(e) {}
}

async function startCountdownThenRecord() {
  if (isRecording || isCountdown) return;
  isCountdown = true;
  if (cntOverlay) cntOverlay.classList.add('active');
  for (let i = COUNTDOWN_SECS; i >= 1; i--) {
    if (document.getElementById('countdown-num')) document.getElementById('countdown-num').textContent = i;
    await new Promise(r => setTimeout(r, 1000));
  }
  if (cntOverlay) cntOverlay.classList.remove('active');
  isCountdown = false;
  startRecording();
}

function startRecording() {
  isRecording = true; videoFrameBuffer = [];
  recordStartTime = Date.now();
  if (recOverlay) recOverlay.classList.add('active');
  log(`🎬 Recording ${currentLabel}…`, 'info');

  recordingTimer = setInterval(() => {
    if (lastLandmarks && numHandsDetected > 0) videoFrameBuffer.push(Array.from(lastLandmarks));
    const elapsed = Date.now() - recordStartTime;
    if (elapsed >= VIDEO_DURATION) { clearInterval(recordingTimer); finishRecording(); }
  }, 50);
}

async function finishRecording() {
  isRecording = false;
  if (recOverlay) recOverlay.classList.remove('active');
  log(`✅ Captured ${videoFrameBuffer.length} frames — saving…`, 'ok');
  try {
    const res = await fetch(BACKEND + '/collect_video', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frames: videoFrameBuffer, label: currentLabel, mode: currentMode, fps: 20, duration: VIDEO_DURATION/1000 })
    });
    if (res.ok) {
      const data = await res.json();
      samplesCollected[currentLabel] = data.count;
      updateProgress(); updateLabelList();
    }
  } catch(e) {}
}

const hands = new Hands({ locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}` });
hands.setOptions({ maxNumHands:2, modelComplexity:0, minDetectionConfidence:0.6, minTrackingConfidence:0.5 });
hands.onResults(onResults);

function flattenLandmarks(results) {
  const flat = new Float32Array(126).fill(0);
  const h = results.multiHandLandmarks || [];
  const s = results.multiHandedness || [];
  for (let i = 0; i < Math.min(h.length, 2); i++) {
    if (!h[i]) continue;
    const isRight = s[i] && s[i].label === 'Right';
    const offset  = isRight ? 63 : 0;
    for (let j = 0; j < 21; j++) {
      if (!h[i][j]) continue;
      flat[offset+j*3]   = h[i][j].x;
      flat[offset+j*3+1] = h[i][j].y;
      flat[offset+j*3+2] = h[i][j].z;
    }
  }
  return flat;
}

function onResults(results) {
  if (!canvasEl || !ctx) return;
  ctx.save(); ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
  numHandsDetected = (results.multiHandLandmarks || []).length;
  handDetected     = numHandsDetected > 0;

  if (results.multiHandLandmarks) {
    for (let i = 0; i < results.multiHandLandmarks.length; i++) {
      const lm   = results.multiHandLandmarks[i];
      const side = results.multiHandedness[i]?.label;
      drawConnectors(ctx, lm, HAND_CONNECTIONS, { color: side==='Right'?'#4f8ef7':'#a78bfa', lineWidth:2.5 });
      drawLandmarks(ctx, lm, { color: side==='Right'?'#22c55e':'#f9a8d4', lineWidth:1, radius:3 });
    }
    lastLandmarks = flattenLandmarks(results);
  } else { lastLandmarks = null; }
  ctx.restore();

  if (handsCount) handsCount.textContent = `${numHandsDetected} HAND${numHandsDetected !== 1 ? 'S' : ''}`;
  if (handStatus) {
    handStatus.textContent = handDetected ? 'HAND DETECTED ✅' : 'NO HAND';
    handStatus.className   = handDetected ? 'hand-status ok' : 'hand-status bad';
  }

  const capBtn = document.getElementById('capture-btn');
  const recBtn = document.getElementById('record-btn');
  if (capBtn) capBtn.disabled = !handDetected;
  if (recBtn) recBtn.disabled = !handDetected;
}

async function loadStats() {
  try {
    const res = await fetch(BACKEND + '/collect_stats');
    if (res.ok) {
      const data = await res.json();
      ['alphabet','number','word','static_word'].forEach(m => {
        if (data[m]?.per_label) Object.assign(samplesCollected, data[m].per_label);
      });
      updateProgress(); updateLabelList();
    }
  } catch(e) {}
}

const camera = new Camera(document.getElementById('video') || document.createElement('video'), {
  onFrame: async () => { if (isCameraOn && videoEl) await hands.send({ image: videoEl }); },
  width: 640, height: 480
});
