/* ═══════════════════════════════════════════════════════════
   ISL TRANSLATOR — SHARED LANDMARK NORMALIZATION UTILITY
   Flattens MediaPipe 21 3D hand landmarks per hand (up to 2 hands)
   into a unified 126-element Float32Array.
═══════════════════════════════════════════════════════════ */

'use strict';

/**
 * Normalizes and flattens MediaPipe multiHandLandmarks & multiHandedness
 * @param {Object} results - MediaPipe hands onResults object
 * @returns {Float32Array} 126-element normalized landmark array
 */
function flattenLandmarks(results) {
  const flat = new Float32Array(126).fill(0);
  const h = (results && results.multiHandLandmarks) ? results.multiHandLandmarks : [];
  const s = (results && results.multiHandedness) ? results.multiHandedness : [];

  for (let i = 0; i < Math.min(h.length, 2); i++) {
    if (!h[i]) continue;
    const isRight = s[i] && s[i].label === 'Right';
    const offset  = isRight ? 63 : 0;
    for (let j = 0; j < 21; j++) {
      if (!h[i][j]) continue;
      flat[offset + j * 3]     = h[i][j].x;
      flat[offset + j * 3 + 1] = h[i][j].y;
      flat[offset + j * 3 + 2] = h[i][j].z;
    }
  }
  return flat;
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { flattenLandmarks };
}
