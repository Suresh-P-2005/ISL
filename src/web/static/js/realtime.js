/* ═══════════════════════════════════════════════════════════
   ISL TRANSLATOR — REAL-TIME WEBCAM CONTROLLER
═══════════════════════════════════════════════════════════ */

'use strict';

// ─────────────────────────────────────────────────────────────
// CONFIG
// ─────────────────────────────────────────────────────────────
let BACKEND = window.location.origin;
const KEYFRAMES           = 30;
const N_FEAT              = 126;
const FRAME_BUFFER_MAX    = 60;
const PRED_INTERVAL_WORD  = 500;
const PRED_INTERVAL_STATIC= 150;
const STABILITY_NEEDED        = 3;
const STABILITY_NEEDED_STATIC = 6;
const CONF_MIN_WORD    = 0.85;
const CONF_MIN_STATIC  = 0.70;
const CONF_SPEAK       = 0.78;
const PAUSE_WORD_MS    = 1000;

// STATE
let currentMode   = 'word';
let currentEngine = 'auto';
let isCameraOn    = false;
let isProcessing  = false;
let ttsEnabled    = true;
let handRequirements = {};
let translationCache = {};
let frameBuffer   = [];
let lastPredTime  = 0;
let stabilityLabel = '';
let stabilityCount = 0;
let stabilityConfs = [];
let lastDetectedWord = '';
let lastDetectedTime = 0;
let sentenceWords = [];
let lastWordTime  = 0;
let pauseTimer    = null;
let totalDetections = 0;
let confSum = 0;
let wordCount = 0;
let fpsFrames = 0, fpsLast = Date.now();

// DOM REFS
let videoEl, canvasEl, ctx, resultChar, resultOrig, resultTr, resultLang;
let confBar, confVal, engineVal, latencyVal, fpsDisplay, langSelect, speedSel;
let sentDisp, sentTr, wordChips, bufferBar, bufferText, visionBtn;

// Init DOM elements when DOM ready
document.addEventListener('DOMContentLoaded', () => {
  videoEl    = document.getElementById('video');
  canvasEl   = document.getElementById('canvas');
  if (canvasEl) ctx = canvasEl.getContext('2d');
  resultChar = document.getElementById('result-char');
  resultOrig = document.getElementById('result-orig');
  resultTr   = document.getElementById('result-tr');
  resultLang = document.getElementById('result-lang');
  confBar    = document.getElementById('conf-bar');
  confVal    = document.getElementById('conf-val');
  engineVal  = document.getElementById('engine-val');
  latencyVal = document.getElementById('latency-val');
  fpsDisplay = document.getElementById('fps-display');
  langSelect = document.getElementById('lang-select');
  speedSel   = document.getElementById('speed-sel');
  sentDisp   = document.getElementById('sentence-display');
  sentTr     = document.getElementById('sentence-tr');
  wordChips  = document.getElementById('word-chips');
  bufferBar  = document.getElementById('buffer-bar');
  bufferText = document.getElementById('buffer-frames-text');
  visionBtn  = document.getElementById('vision-btn');

  setupEventListeners();
  updateModeUI();
  updatePauseIndicator(0);
  checkBackend();
});

// BACKEND CHECK
async function checkBackend() {
  const pill = document.getElementById('status-pill');
  const dot  = document.getElementById('status-dot');
  const txt  = document.getElementById('status-text');
  if (!pill) return;
  pill.className = 'status-pill checking';
  txt.textContent = 'LINKING...';
  try {
    const r = await fetch(BACKEND + '/status', { signal: AbortSignal.timeout(3000) });
    if (r.ok) {
      const d = await r.json();
      pill.className = 'status-pill online';
      const engine = d.lstm_ready ? 'LSTM' : (d.rf_models?.length ? 'RF' : 'GEO');
      txt.textContent = `ONLINE · ${engine}`;
      const hr = await fetch(BACKEND + '/hand_requirements');
      if (hr.ok) handRequirements = await hr.json();
    } else {
      pill.className = 'status-pill offline';
      txt.textContent = 'OFFLINE';
    }
  } catch {
    if (pill) {
      pill.className = 'status-pill offline';
      txt.textContent = 'OFFLINE';
    }
  }
}

// TRANSLATION & TTS
async function getTranslation(word, lang) {
  if (!word || lang === 'en-US' || lang === 'en-GB') return word;
  const key = `${word}_${lang}`;
  if (translationCache[key]) return translationCache[key];
  try {
    const r = await fetch(BACKEND + '/translate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ word, lang }),
      signal: AbortSignal.timeout(2000)
    });
    if (r.ok) {
      const d = await r.json();
      translationCache[key] = d.translated;
      return d.translated;
    }
  } catch {}
  return word;
}

function getLangName(lang) {
  const m = {
    'ta-IN':'Tamil','hi-IN':'Hindi','te-IN':'Telugu','kn-IN':'Kannada',
    'ml-IN':'Malayalam','mr-IN':'Marathi','bn-IN':'Bengali','fr-FR':'French',
    'de-DE':'German','es-ES':'Spanish','ar-SA':'Arabic'
  };
  return m[lang] || lang;
}

async function speak(text, lang) {
  if (!ttsEnabled || !text || text === '---') return;
  try {
    const r = await fetch(BACKEND + '/tts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, lang })
    });
    if (r.ok) {
      const blob = await r.blob();
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      audio.playbackRate = parseFloat(speedSel.value);
      audio.play();
      return;
    }
  } catch (e) {}
  const u = new SpeechSynthesisUtterance(text);
  u.lang = lang;
  u.rate = parseFloat(speedSel.value);
  speechSynthesis.cancel();
  speechSynthesis.speak(u);
}

// LANDMARK FLATTENING
function flattenLandmarks(results) {
  const flat = new Float32Array(126).fill(0);
  const h = results.multiHandLandmarks || [];
  const s = results.multiHandedness || [];
  for (let i = 0; i < Math.min(h.length, 2); i++) {
    if (!h[i]) continue;
    const isRight = s[i] && s[i].label === 'Right';
    const offset = isRight ? 63 : 0;
    for (let j = 0; j < 21; j++) {
      if (!h[i][j]) continue;
      flat[offset + j*3]   = h[i][j].x;
      flat[offset + j*3+1] = h[i][j].y;
      flat[offset + j*3+2] = h[i][j].z;
    }
  }
  return flat;
}

// SENTENCE BUILDER
function buildSentenceText() { return sentenceWords.map(w => w.word).join(' '); }
function buildSentenceTr()   { return sentenceWords.map(w => w.translated || w.word).join(' '); }

async function addWordToSentence(word) {
  const lastReal = [...sentenceWords].reverse().find(w => !w.punct);
  if (lastReal && lastReal.word === word) { resetPauseTimer(); return; }
  const lang       = langSelect.value;
  const translated = await getTranslation(word, lang);
  sentenceWords.push({ word, translated, time: Date.now() });
  renderSentence();
  wordCount++;
  if (document.getElementById('stat-words')) document.getElementById('stat-words').textContent = wordCount;
}

function renderSentence() {
  if (!sentDisp) return;
  const txt = buildSentenceText();
  const tr  = buildSentenceTr();
  sentDisp.textContent = txt || '—';
  sentDisp.className   = `sentence-display ${txt ? 'active' : ''}`;
  const lang = langSelect.value;
  sentTr.textContent = (lang !== 'en-US' && lang !== 'en-GB' && tr !== txt) ? tr : '';

  wordChips.innerHTML = sentenceWords.map((w, i) =>
    `<div class="word-chip ${w.punct ? 'punct' : ''}" data-idx="${i}">${w.word}</div>`
  ).join('');

  wordChips.querySelectorAll('.word-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      sentenceWords.splice(parseInt(chip.dataset.idx), 1);
      renderSentence();
    });
  });
}

function resetPauseTimer() {
  if (pauseTimer) clearTimeout(pauseTimer);
  lastWordTime = Date.now();
  updatePauseIndicator(0);
  pauseTimer = setTimeout(() => updatePauseIndicator(1), PAUSE_WORD_MS);
}

function updatePauseIndicator(state) {
  const el  = document.getElementById('pause-indicator');
  const txt = document.getElementById('pause-text');
  const ico = document.getElementById('pause-icon');
  const bar = document.getElementById('pause-bar');
  if (!el) return;
  if (state === 0) {
    el.className = 'pause-indicator signing'; txt.textContent = 'SIGNING...'; ico.textContent = 'gesture'; bar.style.width = '0%';
  } else if (state === 1) {
    el.className = 'pause-indicator pause1'; txt.textContent = '1s PAUSE — SPACE'; ico.textContent = 'space_bar'; bar.style.width = '33%';
  } else {
    el.className = 'pause-indicator pause3'; txt.textContent = '3s PAUSE — END'; ico.textContent = 'stop_circle'; bar.style.width = '100%';
  }
}

// EVENT LISTENERS SETUP
function setupEventListeners() {
  const linkBtn = document.getElementById('link-btn');
  if (linkBtn) {
    linkBtn.addEventListener('click', () => {
      BACKEND = document.getElementById('backend-input').value.trim().replace(/\/$/, '');
      checkBackend();
    });
  }

  const ttsBtn = document.getElementById('tts-btn');
  if (ttsBtn) {
    ttsBtn.addEventListener('click', () => {
      ttsEnabled = !ttsEnabled;
      ttsBtn.className = `tts-btn ${ttsEnabled ? 'on' : 'off'}`;
      document.getElementById('tts-icon').textContent = ttsEnabled ? 'volume_up' : 'volume_off';
      document.getElementById('tts-label').textContent = ttsEnabled ? 'TTS ON' : 'TTS OFF';
    });
  }

  document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentMode = btn.dataset.mode;
      frameBuffer = [];
      stabilityLabel = ''; stabilityCount = 0; stabilityConfs = [];
      updateModeUI();
    });
  });

  document.querySelectorAll('.eng-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.eng-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentEngine = btn.dataset.eng;
    });
  });

  const speakBtn = document.getElementById('speak-sentence-btn');
  if (speakBtn) {
    speakBtn.addEventListener('click', () => {
      const txt = buildSentenceText();
      if (txt) speak(txt, langSelect.value);
    });
  }

  const clearBtn = document.getElementById('clear-btn');
  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      sentenceWords = []; renderSentence(); updatePauseIndicator(0);
    });
  }

  const copyBtn = document.getElementById('copy-btn');
  if (copyBtn) {
    copyBtn.addEventListener('click', async () => {
      const txt = buildSentenceText();
      if (txt) { await navigator.clipboard.writeText(txt); }
    });
  }

  const magicBtn = document.getElementById('magic-btn');
  if (magicBtn) {
    magicBtn.addEventListener('click', async () => {
      const rawText = buildSentenceText();
      if (!rawText) return;
      sentDisp.innerHTML = '<span style="color:var(--text-muted);font-size:1rem">✨ Generating…</span>';
      try {
        const r = await fetch(BACKEND + '/make_sentence', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ words: rawText, lang: langSelect.value })
        });
        if (r.ok) {
          const data = await r.json();
          const engWords = (data.english_sentence || '').split(/\s+/).filter(Boolean);
          const trWords  = (data.translated_sentence || '').split(/\s+/).filter(Boolean);
          sentenceWords  = engWords.map((w, i) => ({ word: w, translated: trWords[i] || w }));
          renderSentence();
        }
      } catch (e) { sentDisp.textContent = rawText; }
    });
  }

  // Camera vision button
  if (visionBtn) {
    visionBtn.addEventListener('click', () => {
      if (!isCameraOn) {
        isCameraOn = true;
        visionBtn.className = 'vision-btn stop';
        visionBtn.innerHTML = '<span class="material-icons-round">stop_circle</span> Stop Vision';
        frameBuffer = [];
        camera.start().then(() => checkBackend()).catch(() => { isCameraOn = false; resetVisionBtn(); });
      } else {
        isCameraOn = false;
        const stream = videoEl.srcObject;
        if (stream) stream.getTracks().forEach(t => t.stop());
        videoEl.srcObject = null;
        ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
        resetVisionBtn();
      }
    });
  }
}

function resetVisionBtn() {
  if (visionBtn) {
    visionBtn.className = 'vision-btn start';
    visionBtn.innerHTML = '<span class="material-icons-round">play_circle</span> Initialize Vision';
  }
}

function updateModeUI() {
  const badge = document.getElementById('cam-mode-badge');
  const bwrap = document.getElementById('buffer-wrap');
  if (!badge) return;
  const modeMap = {
    word:         ['WORD · LSTM',   'word',        'block'],
    static_word:  ['S-WORD · RF+CNN','static_word','none'],
    alphabet:     ['ALPHA · RF+CNN', 'alpha',      'none'],
    number:       ['NUM · RF+CNN',   'num',        'none'],
  };
  const [label, cls, disp] = modeMap[currentMode] || ['WORD · LSTM','word','block'];
  badge.textContent = label; badge.className = `cam-mode-badge ${cls}`;
  if (bwrap) bwrap.style.display = disp;
}

// MEDIAPIPE PIPELINE
const hands = new Hands({ locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}` });
hands.setOptions({ maxNumHands: 2, modelComplexity: 1, minDetectionConfidence: 0.65, minTrackingConfidence: 0.55 });
hands.onResults(onResults);

const camera = new Camera(document.getElementById('video') || document.createElement('video'), {
  onFrame: async () => { if (isCameraOn && videoEl) await hands.send({ image: videoEl }); },
  width: 640, height: 480
});

function onResults(results) {
  if (!canvasEl || !videoEl) return;
  canvasEl.width  = videoEl.clientWidth;
  canvasEl.height = videoEl.clientHeight;
  ctx.save();
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

  const numHands = (results.multiHandLandmarks || []).length;
  if (results.multiHandLandmarks) {
    for (let i = 0; i < results.multiHandLandmarks.length; i++) {
      const lm   = results.multiHandLandmarks[i];
      const side = results.multiHandedness[i]?.label;
      const col  = side === 'Right' ? '#4f8ef7' : '#a78bfa';
      drawConnectors(ctx, lm, HAND_CONNECTIONS, { color: col, lineWidth: 2.5 });
      drawLandmarks(ctx, lm, { color: '#22c55e', lineWidth: 1, radius: 3 });
    }
  }
  ctx.restore();

  fpsFrames++;
  const now = Date.now();
  if (now - fpsLast >= 1000) {
    if (fpsDisplay) fpsDisplay.textContent = `${fpsFrames} FPS`;
    fpsFrames = 0; fpsLast = now;
  }

  const flat = flattenLandmarks(results);
  if (numHands > 0) {
    frameBuffer.push(Array.from(flat));
    if (frameBuffer.length > FRAME_BUFFER_MAX) frameBuffer.shift();
  } else {
    frameBuffer.push(new Array(N_FEAT).fill(0));
    if (frameBuffer.length > FRAME_BUFFER_MAX) frameBuffer.shift();
  }

  const bufLen = Math.min(frameBuffer.filter(f => f.some(v => v !== 0)).length, KEYFRAMES);
  if (bufferBar) bufferBar.style.width = `${bufLen / KEYFRAMES * 100}%`;
  if (bufferText) bufferText.textContent = `${bufLen} / ${KEYFRAMES}`;

  const interval = currentMode === 'word' ? PRED_INTERVAL_WORD : PRED_INTERVAL_STATIC;
  if (now - lastPredTime > interval && !isProcessing) {
    lastPredTime = now;
    if (currentMode === 'word') runLSTMPrediction(numHands, flat);
    else runStaticPrediction(flat, numHands);
  }
}

async function runLSTMPrediction(numHands, latestFlat) {
  if (isProcessing) return;
  const realFrames = frameBuffer.filter(f => f.some(v => v !== 0));
  if (realFrames.length < KEYFRAMES / 2) return;
  isProcessing = true;
  const t0 = Date.now();
  try {
    const frames = frameBuffer.slice(-KEYFRAMES);
    const r = await fetch(BACKEND + '/predict_sequence', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frames, num_hands: numHands }),
      signal: AbortSignal.timeout(800)
    });
    if (r.ok) {
      const pred = await r.json();
      if (latencyVal) latencyVal.textContent = `${Date.now() - t0}ms`;
      handleWordPrediction(pred, numHands);
    }
  } catch {}
  isProcessing = false;
}

async function runStaticPrediction(flat, numHands) {
  if (isProcessing || flat.every(v => v === 0)) return;
  isProcessing = true;
  const t0 = Date.now();
  try {
    const r = await fetch(BACKEND + '/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ landmarks: Array.from(flat), mode: currentMode, engine: currentEngine }),
      signal: AbortSignal.timeout(500)
    });
    if (r.ok) {
      const pred = await r.json();
      if (latencyVal) latencyVal.textContent = `${Date.now() - t0}ms`;
      handleStaticPrediction(pred, numHands);
    }
  } catch {}
  isProcessing = false;
}

function handleWordPrediction(pred, numHands) {
  const conf  = pred.confidence || 0;
  const label = pred.label || '---';
  updateConfUI(conf, pred.engine || 'lstm');

  if (label === '---' || conf < CONF_MIN_WORD) {
    stabilityLabel = ''; stabilityCount = 0; stabilityConfs = [];
    if (resultChar) { resultChar.style.opacity = '0.2'; resultChar.textContent = '---'; }
    updateStability(0); return;
  }

  if (label === stabilityLabel) { stabilityCount++; stabilityConfs.push(conf); }
  else { stabilityLabel = label; stabilityCount = 1; stabilityConfs = [conf]; }

  updateStability(stabilityCount);
  if (resultChar) {
    resultChar.textContent = label;
    resultChar.style.opacity = Math.min(0.3 + stabilityCount / STABILITY_NEEDED * 0.7, 0.9).toFixed(2);
  }
  if (stabilityCount < STABILITY_NEEDED) return;
  const avgConf = stabilityConfs.reduce((a, b) => a + b, 0) / stabilityConfs.length;
  if (avgConf < CONF_SPEAK) return;

  resultChar.style.opacity = '1';
  if (label === lastDetectedWord && Date.now() - lastDetectedTime < 2000) {
    stabilityLabel = ''; stabilityCount = 0; stabilityConfs = []; return;
  }
  lastDetectedWord = label; lastDetectedTime = Date.now();
  stabilityLabel = ''; stabilityCount = 0; stabilityConfs = [];
  commitWord(label, avgConf, pred.engine || 'lstm');
}

function handleStaticPrediction(pred, numHands) {
  const conf  = pred.confidence || 0;
  const label = pred.label || '---';
  updateConfUI(conf, pred.engine || 'rf');

  if (label === '---' || conf < CONF_MIN_STATIC) {
    stabilityLabel = ''; stabilityCount = 0; stabilityConfs = [];
    if (resultChar) { resultChar.style.opacity = '0.2'; resultChar.textContent = '---'; }
    updateStability(0); return;
  }

  if (label === stabilityLabel) { stabilityCount++; stabilityConfs.push(conf); }
  else { stabilityLabel = label; stabilityCount = 1; stabilityConfs = [conf]; }

  updateStability(stabilityCount);
  if (resultChar) {
    resultChar.textContent = label;
    resultChar.style.opacity = Math.min(0.2 + stabilityCount / STABILITY_NEEDED_STATIC * 0.8, 0.9).toFixed(2);
  }
  if (stabilityCount < STABILITY_NEEDED_STATIC) return;
  const avgConf = stabilityConfs.reduce((a, b) => a + b, 0) / stabilityConfs.length;
  if (avgConf < CONF_SPEAK) return;

  resultChar.style.opacity = '1';
  if (label === lastDetectedWord && Date.now() - lastDetectedTime < 1500) {
    stabilityLabel = ''; stabilityCount = 0; stabilityConfs = []; return;
  }
  lastDetectedWord = label; lastDetectedTime = Date.now();
  stabilityLabel = ''; stabilityCount = 0; stabilityConfs = [];
  commitWord(label, avgConf, pred.engine || 'rf');
}

async function commitWord(word, conf, engine) {
  const lang       = langSelect.value;
  const translated = await getTranslation(word, lang);
  const isEn       = lang === 'en-US' || lang === 'en-GB';

  if (resultOrig) resultOrig.textContent = word;
  if (resultChar) resultChar.textContent = word;
  if (!isEn && translated !== word) {
    if (resultTr) resultTr.textContent = translated;
    if (resultLang) resultLang.textContent = `[ ${getLangName(lang)} ]`;
  } else {
    if (resultTr) resultTr.textContent = '';
    if (resultLang) resultLang.textContent = '';
  }

  speak(isEn ? word : translated, lang);
  await addWordToSentence(word);
  resetPauseTimer();

  totalDetections++; confSum += conf;
  if (document.getElementById('stat-total')) document.getElementById('stat-total').textContent = totalDetections;
  if (document.getElementById('stat-conf')) document.getElementById('stat-conf').textContent = `${Math.round(confSum / totalDetections * 100)}%`;
  addHistory(word, conf, engine);
}

function updateConfUI(conf, engine) {
  const pct = Math.round(conf * 100);
  if (confBar) {
    confBar.style.width = `${pct}%`;
    confBar.style.background = pct > 85 ? 'var(--success)' : pct > 65 ? 'var(--warning)' : 'var(--danger)';
  }
  if (confVal) confVal.textContent = `${pct}%`;
  if (engineVal) engineVal.textContent = engine.toUpperCase();
}

function updateStability(count) {
  const max = currentMode === 'word' ? STABILITY_NEEDED : STABILITY_NEEDED_STATIC;
  const normalised = Math.round((count / max) * 5);
  for (let i = 0; i < 5; i++) {
    const el = document.getElementById(`stab${i}`);
    if (el) el.classList.toggle('active', i < normalised);
  }
  if (document.getElementById('stability-text')) document.getElementById('stability-text').textContent = `${count}/${max}`;
}

function addHistory(word, conf, engine) {
  const list = document.getElementById('history-list');
  if (!list) return;
  const div  = document.createElement('div');
  div.className = 'hist-item';
  div.innerHTML = `
    <span class="hist-word">${word}</span>
    <span class="hist-conf">${Math.round(conf * 100)}%</span>
    <span class="hist-eng">${engine.toUpperCase()}</span>
    <span class="hist-time">${new Date().toLocaleTimeString()}</span>`;
  list.insertBefore(div, list.firstChild);
  while (list.children.length > 15) list.removeChild(list.lastChild);
}
