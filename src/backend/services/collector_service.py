import os
import csv
import json
import sqlite3
import shutil
import numpy as np
import pandas as pd

class CollectorService:
    def __init__(self, data_dir: str, video_dir: str, keyframes: int = 30):
        self.data_dir = data_dir
        self.video_dir = video_dir
        self.keyframes = keyframes
        self.db_path = os.path.join(os.path.dirname(os.path.abspath(data_dir)), "data", "isl_dataset.db")
        self.custom_file = os.path.join(os.path.dirname(os.path.abspath(data_dir)), "custom_signs.json")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.custom_signs = self._load_custom_signs()
        self._init_sqlite_db()

    def _init_sqlite_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS dataset_samples (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        mode TEXT NOT NULL,
                        label TEXT NOT NULL,
                        sample_type TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_mode_label ON dataset_samples(mode, label)")
        except Exception as e:
            print(f"SQLite DB init error: {e}")

    def _load_custom_signs(self) -> dict:
        if os.path.exists(self.custom_file):
            try:
                with open(self.custom_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {"words": [], "static_words": [], "hands": {}, "descriptions": {}}

    def _save_custom_signs(self):
        try:
            with open(self.custom_file, 'w', encoding='utf-8') as f:
                json.dump(self.custom_signs, f, indent=2)
        except Exception as e:
            print(f"Error saving custom_signs.json: {e}")

    def add_custom_sign(self, label: str, mode: str, hands: int = 1, description: str = "") -> dict:
        if not label:
            return self.custom_signs
        mode_key = "word" if mode == "word" else "static_words"
        if mode == "word":
            if label not in self.custom_signs["words"]:
                self.custom_signs["words"].append(label)
        elif mode == "static_word":
            if label not in self.custom_signs["static_words"]:
                self.custom_signs["static_words"].append(label)
        
        self.custom_signs["hands"][label] = hands
        if description:
            self.custom_signs["descriptions"][label] = description
        self._save_custom_signs()
        return self.custom_signs

    def save_image_sample(self, landmarks: list, label: str, mode: str) -> int:
        path = os.path.join(self.data_dir, f'{mode}_landmarks.csv')
        exists = os.path.isfile(path)
        with open(path, 'a', newline='') as f:
            w = csv.writer(f)
            if not exists:
                w.writerow([f"p_{i}" for i in range(len(landmarks))] + ["label"])
            w.writerow(list(landmarks) + [label])
        df = pd.read_csv(path)
        return int((df['label'].astype(str) == str(label)).sum())

    def save_video_sample(self, frames: list, label: str, mode: str) -> int:
        frames_arr = np.array(frames)
        if len(frames_arr) >= self.keyframes:
            idx = np.linspace(0, len(frames_arr) - 1, self.keyframes, dtype=int)
            sampled = frames_arr[idx]
        else:
            pad = np.tile(frames_arr[-1], (self.keyframes - len(frames_arr), 1))
            sampled = np.vstack([frames_arr, pad])
        flat = sampled.flatten()
        path = os.path.join(self.video_dir, f'{mode}_video.csv')
        exists = os.path.isfile(path)
        with open(path, 'a', newline='') as f:
            w = csv.writer(f)
            if not exists:
                w.writerow([f"f_{i}" for i in range(len(flat))] + ["label"])
            w.writerow(list(flat) + [label])
        df = pd.read_csv(path)
        return int((df['label'].astype(str) == str(label)).sum())

    def get_stats(self) -> dict:
        stats = {}
        for mode in ["alphabet", "number", "word", "static_word"]:
            img_path = os.path.join(self.data_dir, f'{mode}_landmarks.csv')
            vid_path = os.path.join(self.video_dir, f'{mode}_video.csv')
            per_label = {}
            total = 0
            if os.path.exists(img_path):
                df = pd.read_csv(img_path)
                for label, cnt in df.groupby('label').size().items():
                    per_label[str(label)] = int(cnt)
                    total += int(cnt)
            if os.path.exists(vid_path):
                df = pd.read_csv(vid_path)
                for label, cnt in df.groupby('label').size().items():
                    per_label[str(label)] = per_label.get(str(label), 0) + int(cnt)
                    total += int(cnt)
            stats[mode] = {"total": total, "classes": len(per_label), "per_label": per_label}
        stats["custom_signs"] = self.custom_signs
        return stats

    def delete_label(self, mode: str, label: str):
        if mode and label:
            p1 = os.path.join(self.data_dir, f'{mode}_landmarks.csv')
            p2 = os.path.join(self.video_dir, f'{mode}_video.csv')
            for path in [p1, p2]:
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    df = df[df['label'].astype(str) != str(label)]
                    df.to_csv(path, index=False)

    def delete_all(self, mode: str = None):
        if mode:
            for path in [os.path.join(self.data_dir, f'{mode}_landmarks.csv'), os.path.join(self.video_dir, f'{mode}_video.csv')]:
                if os.path.exists(path):
                    os.remove(path)
        else:
            if os.path.exists(self.data_dir):
                shutil.rmtree(self.data_dir)
                os.makedirs(self.data_dir)
            if os.path.exists(self.video_dir):
                shutil.rmtree(self.video_dir)
                os.makedirs(self.video_dir)
