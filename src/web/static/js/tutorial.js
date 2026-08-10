/* ═══════════════════════════════════════════════════════════
   ISL TRANSLATOR — TUTORIAL & SIGN DICTIONARY CONTROLLER
═══════════════════════════════════════════════════════════ */

'use strict';

let BACKEND = window.location.origin;

const KNOWN_POSTURES = {
  "A": "Fist with thumb on side.", "B": "Four fingers straight up, thumb tucked.",
  "C": "Curved hand like holding a ball.", "D": "Index finger up, circle formed by thumb and fingers.",
  "E": "Fingers curled touching thumb.", "F": "Index & thumb touch, 3 fingers up.",
  "G": "Thumb & index extended horizontally.", "H": "Index & middle extended horizontally.",
  "I": "Pinky finger only extended up.", "J": "Pinky finger traces J in air.",
  "K": "Thumb between index & middle fingers.", "L": "L-shape formed by thumb & index.",
  "M": "Three fingers over thumb.", "N": "Two fingers over thumb.",
  "O": "All fingertips touch thumb in O shape.", "P": "K-shape inverted pointing down.",
  "Q": "G-shape inverted pointing down.", "R": "Index and middle fingers crossed.",
  "S": "Fist with thumb over fingers.", "T": "Thumb between index and middle.",
  "U": "Index & middle fingers up together.", "V": "V-shape with index & middle spread.",
  "W": "W-shape with 3 fingers spread.", "X": "Index finger hooked/bent.",
  "Y": "Thumb & pinky extended (Shaka).", "Z": "Index finger traces Z in air.",
  "0": "Compact O shape.", "1": "Index finger up.", "2": "Index and middle fingers up.",
  "3": "Thumb, index, middle fingers up.", "4": "Four fingers up, thumb tucked.",
  "5": "All 5 fingers open and extended.", "6": "Pinky touches thumb tip.",
  "7": "Ring finger touches thumb tip.", "8": "Middle finger touches thumb tip.",
  "9": "Index finger touches thumb tip."
};

let systemData = {
  static_words: ['Bad', 'Call', 'Good', 'Love', 'Me', 'No', 'Yes', 'You'],
  dynamic_words: ['Bye', 'Food', 'Hello', 'Help', 'Indian', 'Man', 'School', 'Sick', 'Sorry', 'Stop', 'ThankYou', 'Time', 'Tired', 'Toilet', 'Wait', 'What', 'When', 'Where', 'Why', 'Woman'],
  hand_requirements: {
    "Help": 2, "School": 2, "Stop": 2, "What": 2, "Where": 2, "When": 2, "Why": 2, "Tired": 2,
    "Hello": 1, "ThankYou": 1, "Wait": 1, "Food": 1, "Sick": 1, "Sorry": 1, "Time": 1, "Toilet": 1
  }
};

let activeFilter = 'all';
let searchQuery = '';

document.addEventListener('DOMContentLoaded', () => {
  setupTutorialListeners();
  renderDictionary();
  fetchSystemData();
});

async function fetchSystemData() {
  try {
    const r = await fetch(BACKEND + '/system_words');
    if (r.ok) {
      const res = await r.json();
      if (res.static_words) systemData.static_words = res.static_words;
      if (res.dynamic_words) systemData.dynamic_words = res.dynamic_words;
      if (res.hand_requirements) Object.assign(systemData.hand_requirements, res.hand_requirements);
    }
  } catch(e) {}
  renderDictionary();
}

function getSystemSignsMap() {
  const map = {};

  'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('').forEach(ch => {
    map[ch] = {
      category: 'alpha',
      hands: systemData.hand_requirements[ch] || 1,
      desc: KNOWN_POSTURES[ch] || 'Standard ISL Alphabet sign gesture.'
    };
  });

  '0123456789'.split('').forEach(num => {
    map[num] = {
      category: 'num',
      hands: 1,
      desc: KNOWN_POSTURES[num] || 'Standard ISL Number sign gesture.'
    };
  });

  (systemData.static_words || []).forEach(w => {
    if (!w || map[w]) return;
    map[w] = {
      category: 'static_word',
      hands: systemData.hand_requirements[w] || 1,
      desc: KNOWN_POSTURES[w] || 'Static hand gesture classified via Random Forest / CNN.'
    };
  });

  (systemData.dynamic_words || []).forEach(w => {
    if (!w || map[w]) return;
    map[w] = {
      category: 'word',
      hands: systemData.hand_requirements[w] || 1,
      desc: KNOWN_POSTURES[w] || 'Dynamic sequence gesture classified via LSTM Neural Model.'
    };
  });

  // Merge custom user-added signs & hand requirements from localStorage
  try {
    const savedObj = JSON.parse(localStorage.getItem('isl_custom_words') || '{}');
    const customHands = savedObj.hands || {};
    const customDescs = JSON.parse(localStorage.getItem('isl_custom_word_descs') || '{}');

    (savedObj.words || []).forEach(w => {
      if (!w) return;
      map[w] = {
        category: 'word',
        hands: customHands[w] || systemData.hand_requirements[w] || 1,
        desc: customDescs[w] || KNOWN_POSTURES[w] || 'User added dynamic sequence sign.'
      };
    });

    (savedObj.static_words || []).forEach(w => {
      if (!w) return;
      map[w] = {
        category: 'static_word',
        hands: customHands[w] || systemData.hand_requirements[w] || 1,
        desc: customDescs[w] || KNOWN_POSTURES[w] || 'User added static posture sign.'
      };
    });
  } catch(e) {}

  return map;
}

function renderDictionary() {
  const grid = document.getElementById('dict-grid');
  if (!grid) return;
  const systemMap = getSystemSignsMap();
  const keys = Object.keys(systemMap);

  const filtered = keys.filter(key => {
    const item = systemMap[key];
    const nameMatch = key.toLowerCase().includes(searchQuery.toLowerCase());
    const descMatch = (item.desc || '').toLowerCase().includes(searchQuery.toLowerCase());
    if (!nameMatch && !descMatch) return false;

    if (activeFilter === 'all') return true;
    if (activeFilter === 'alpha') return item.category === 'alpha';
    if (activeFilter === 'num')   return item.category === 'num';
    if (activeFilter === 'static_word') return item.category === 'static_word';
    if (activeFilter === 'word')  return item.category === 'word';
    if (activeFilter === '1h')    return item.hands === 1;
    if (activeFilter === '2h')    return item.hands === 2;
    return true;
  });

  if (filtered.length === 0) {
    grid.innerHTML = `<div style="grid-column:1/-1;text-align:center;padding:2rem;color:var(--text-muted)">No system signs found.</div>`;
    return;
  }

  grid.innerHTML = filtered.map(name => {
    const sign = systemMap[name];
    const catLabel = sign.category.toUpperCase();
    const handBadgeHTML = sign.hands === 2 ? '<span class="hand-badge two">✋✋ 2 Hands</span>' : '<span class="hand-badge one">✋ 1 Hand</span>';

    return `
      <div class="sign-card">
        <div class="sign-card-top">
          <span class="sign-title">${name}</span>
          ${handBadgeHTML}
        </div>
        <span class="category-tag ${sign.category}">${catLabel}</span>
        <div class="sign-desc">${sign.desc}</div>
      </div>`;
  }).join('');
}

function setupTutorialListeners() {
  const searchInput = document.getElementById('dict-search');
  if (searchInput) {
    searchInput.addEventListener('input', e => {
      searchQuery = e.target.value;
      renderDictionary();
    });
  }

  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activeFilter = btn.dataset.filter;
      renderDictionary();
    });
  });
}
