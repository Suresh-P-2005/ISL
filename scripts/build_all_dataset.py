import os
import cv2
import glob
import pandas as pd
import numpy as np
import mediapipe as mp
from src.ml.data.extractors import extract_mediapipe_landmarks, normalise_landmarks

mp_hands = mp.solutions.hands

def process_static_category(input_dir, output_csv):
    """
    Processes static images (alphabets, numbers, static words).
    Extracts 126-length landmarks and saves to CSV.
    Skips images where no hands are detected.
    """
    if not os.path.exists(input_dir):
        print(f"Skipping {input_dir} (does not exist)")
        return

    data = []
    classes = os.listdir(input_dir)
    
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        for class_name in classes:
            class_dir = os.path.join(input_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            image_paths = glob.glob(os.path.join(class_dir, "*.*"))
            for img_path in image_paths:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)
                
                # Rule 1: Automatically ignore if no hands detected
                if not results.multi_hand_landmarks:
                    continue
                    
                raw_landmarks = extract_mediapipe_landmarks(results)
                norm_landmarks = normalise_landmarks(raw_landmarks)
                
                row = [class_name] + list(norm_landmarks)
                data.append(row)
                
    if data:
        columns = ["label"] + [f"p_{i}" for i in range(126)]
        df = pd.DataFrame(data, columns=columns)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Saved {len(df)} samples to {output_csv}")
    else:
        print(f"No valid hand samples found in {input_dir}")

def process_dynamic_category(input_dir, output_csv, keyframes=30):
    """
    Processes video files (dynamic words).
    Extracts sequences of landmarks, normalizes length to exactly 'keyframes' (30).
    Skips frames with no hands. Skips videos with zero valid frames.
    """
    if not os.path.exists(input_dir):
        print(f"Skipping {input_dir} (does not exist)")
        return

    data = []
    classes = os.listdir(input_dir)
    
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
        for class_name in classes:
            class_dir = os.path.join(input_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            video_paths = glob.glob(os.path.join(class_dir, "*.*"))
            for vid_path in video_paths:
                cap = cv2.VideoCapture(vid_path)
                frames_landmarks = []
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(img_rgb)
                    
                    # Rule 1: Automatically ignore frame if no hands detected
                    if not results.multi_hand_landmarks:
                        continue
                        
                    raw_landmarks = extract_mediapipe_landmarks(results)
                    norm_landmarks = normalise_landmarks(raw_landmarks)
                    frames_landmarks.append(norm_landmarks)
                    
                cap.release()
                
                # If no hands detected in the entire video, skip it
                if len(frames_landmarks) == 0:
                    continue
                    
                # Rule 2: Fixed sequence length (30 frames)
                frames_arr = np.array(frames_landmarks)
                if len(frames_arr) >= keyframes:
                    # Downsample
                    idx = np.linspace(0, len(frames_arr) - 1, keyframes, dtype=int)
                    seq = frames_arr[idx]
                else:
                    # Pad by repeating the last valid frame
                    pad = np.tile(frames_arr[-1], (keyframes - len(frames_arr), 1))
                    seq = np.vstack([frames_arr, pad])
                    
                # Flatten the 30x126 array into a single row of 3780 features
                row = [class_name] + list(seq.flatten())
                data.append(row)
                
    if data:
        columns = ["label"] + [f"p_{i}" for i in range(keyframes * 126)]
        df = pd.DataFrame(data, columns=columns)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Saved {len(df)} sequence samples to {output_csv}")
    else:
        print(f"No valid video samples found in {input_dir}")

if __name__ == "__main__":
    print("Starting Dataset Extraction Pipeline...")
    
    # Static Signs
    process_static_category(
        "raw_dataset/alphabet", 
        "real_landmark_data/alphabet_landmarks.csv"
    )
    process_static_category(
        "raw_dataset/number", 
        "real_landmark_data/number_landmarks.csv"
    )
    process_static_category(
        "raw_dataset/static_word", 
        "real_landmark_data/static_word_landmarks.csv"
    )
    
    # Dynamic Signs
    process_dynamic_category(
        "raw_dataset/word", 
        "video_landmark_data/word_video.csv",
        keyframes=30
    )
    
    print("Dataset extraction complete!")
