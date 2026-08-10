import os
import pickle
import numpy as np
import pandas as pd
from src.ml.data.extractors import normalise_landmarks

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    CNN_AVAILABLE = True
except Exception:
    CNN_AVAILABLE = False

import json

class InferenceService:
    def __init__(self, models_dir: str, config: dict):
        self.models_dir = models_dir
        self.config = config
        self.rf = {}
        self.cnn = {}
        self.le = {}
        self.lstm = None
        self.lstm_le = None
        self.manifest_path = os.path.join(self.models_dir, 'models_manifest.json')
        self.manifest = self.load_manifest()
        self.load_models()

    def load_manifest(self) -> dict:
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {"active_version": "v1.0.0", "updated_at": "2026-07-30"}

    def load_models(self):
        for mode in ["alphabet", "number", "static_word"]:
            p = os.path.join(self.models_dir, f'isl_{mode}_rf.pkl')
            if os.path.exists(p):
                try:
                    with open(p, 'rb') as f:
                        self.rf[mode] = pickle.load(f)
                except Exception as e:
                    print(f"Error loading RF model for {mode}: {e}")

        if CNN_AVAILABLE:
            for mode in ["alphabet", "number", "static_word"]:
                cp = os.path.join(self.models_dir, f'isl_{mode}_cnn.keras')
                lp = os.path.join(self.models_dir, f'isl_{mode}_le.pkl')
                if os.path.exists(cp) and os.path.exists(lp):
                    try:
                        self.cnn[mode] = load_model(cp)
                        with open(lp, 'rb') as f:
                            self.le[mode] = pickle.load(f)
                    except Exception as e:
                        print(f"Error loading CNN model for {mode}: {e}")

            lp = os.path.join(self.models_dir, 'isl_word_lstm.keras')
            ep = os.path.join(self.models_dir, 'isl_word_lstm_le.pkl')
            if os.path.exists(lp) and os.path.exists(ep):
                try:
                    self.lstm = load_model(lp)
                    with open(ep, 'rb') as f:
                        self.lstm_le = pickle.load(f)
                except Exception as e:
                    print(f"Error loading LSTM model: {e}")

    def predict_static(self, landmarks: list, mode: str = "alphabet", engine: str = "auto") -> dict:
        arr = np.array(landmarks)
        norm = normalise_landmarks(arr)
        if np.sum(np.abs(norm)) < 1e-6:
            return {"label": "---", "confidence": 0.0, "engine": "none"}

        hand_reqs = self.config.get("HAND_REQUIREMENTS", {})

        # Try RF prediction
        r = self._predict_rf(norm, mode)
        if r and r["confidence"] > 0.80:
            r["hands_required"] = hand_reqs.get(r.get("label", "---"), 1)
            return r

        # Try CNN prediction
        c = self._predict_cnn(norm, mode)
        if c and c["confidence"] > 0.75:
            c["hands_required"] = hand_reqs.get(c.get("label", "---"), 1)
            return c

        if r:
            r["hands_required"] = hand_reqs.get(r.get("label", "---"), 1)
            return r

        return {"label": "---", "confidence": 0.0, "engine": "none"}

    def _predict_rf(self, norm, mode):
        if mode not in self.rf:
            return None
        try:
            n_feat = self.config.get("N_FEAT", 126)
            inp = pd.DataFrame([norm], columns=[f"p_{i}" for i in range(n_feat)])
            pred = self.rf[mode].predict(inp)[0]
            conf = float(np.max(self.rf[mode].predict_proba(inp)[0]))
            return {"label": str(pred), "confidence": conf, "engine": "rf"}
        except Exception:
            return None

    def _predict_cnn(self, norm, mode):
        if mode not in self.cnn or mode not in self.le:
            return None
        try:
            n_feat = self.config.get("N_FEAT", 126)
            probs = self.cnn[mode].predict(norm.astype('float32').reshape(1, n_feat, 1), verbose=0)[0]
            top_idx = np.argmax(probs)
            label = self.le[mode].inverse_transform([top_idx])[0]
            return {"label": str(label), "confidence": float(probs[top_idx]), "engine": "cnn"}
        except Exception:
            return None

    def predict_sequence(self, frames: list, num_hands: int = 1) -> dict:
        if self.lstm is None or self.lstm_le is None:
            return {"label": "---", "confidence": 0.0, "engine": "no_lstm", "error": "LSTM not trained"}

        frames_arr = np.array(frames)
        if len(frames_arr) == 0:
            return {"label": "---", "confidence": 0.0, "engine": "lstm"}

        keyframes = self.config.get("KEYFRAMES", 30)
        n_feat = self.config.get("N_FEAT", 126)
        hand_reqs = self.config.get("HAND_REQUIREMENTS", {})

        if len(frames_arr) >= keyframes:
            idx = np.linspace(0, len(frames_arr) - 1, keyframes, dtype=int)
            seq = frames_arr[idx]
        else:
            pad = np.tile(frames_arr[-1], (keyframes - len(frames_arr), 1))
            seq = np.vstack([frames_arr, pad])

        norm_seq = np.array([normalise_landmarks(f) for f in seq])
        probs = self.lstm.predict(norm_seq.astype('float32').reshape(1, keyframes, n_feat), verbose=0)[0]
        top_idx = np.argmax(probs)
        top_label = self.lstm_le.inverse_transform([top_idx])[0]

        return {
            "label": top_label,
            "confidence": float(probs[top_idx]),
            "engine": "lstm",
            "hands_required": hand_reqs.get(top_label, 1),
            "num_hands": num_hands,
        }
