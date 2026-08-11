import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Import model builders
from src.ml.models.random_forest import build_random_forest
from src.ml.models.cnn_1d import build_1d_cnn
from src.ml.models.lstm_bidi import build_bidi_lstm

def train_static_models(mode, csv_path, models_dir):
    """
    Trains Random Forest and 1D CNN models for static gestures.
    """
    if not os.path.exists(csv_path):
        print(f"Skipping static {mode} (dataset not found: {csv_path})")
        return

    print(f"\n--- Training Static Models for: {mode.upper()} ---")
    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"Dataset {csv_path} is empty.")
        return

    X = df.drop('label', axis=1).values
    y = df['label'].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)

    # Save LabelEncoder
    le_path = os.path.join(models_dir, f'isl_{mode}_le.pkl')
    with open(le_path, 'wb') as f:
        pickle.dump(le, f)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # 1. Train Random Forest
    print("Training Random Forest...")
    rf_model = build_random_forest()
    rf_model.fit(X_train, y_train)
    rf_acc = rf_model.score(X_test, y_test)
    print(f"Random Forest Accuracy: {rf_acc:.4f}")
    
    rf_path = os.path.join(models_dir, f'isl_{mode}_rf.pkl')
    with open(rf_path, 'wb') as f:
        pickle.dump(rf_model, f)

    # 2. Train 1D CNN
    print("Training 1D CNN...")
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)
    
    # Reshape X for CNN (samples, features, 1)
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1).astype('float32')
    X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1).astype('float32')

    cnn_model = build_1d_cnn(n_features=126, num_classes=num_classes)
    cnn_model.fit(X_train_cnn, y_train_cat, epochs=30, batch_size=32, validation_data=(X_test_cnn, y_test_cat), verbose=1)
    
    loss, cnn_acc = cnn_model.evaluate(X_test_cnn, y_test_cat, verbose=0)
    print(f"1D CNN Accuracy: {cnn_acc:.4f}")

    cnn_path = os.path.join(models_dir, f'isl_{mode}_cnn.keras')
    cnn_model.save(cnn_path)


def train_dynamic_models(csv_path, models_dir, keyframes=30, n_features=126):
    """
    Trains Bi-LSTM model for dynamic video sequences.
    """
    if not os.path.exists(csv_path):
        print(f"Skipping dynamic words (dataset not found: {csv_path})")
        return

    print(f"\n--- Training Dynamic Model (Bi-LSTM) for: WORDS ---")
    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"Dataset {csv_path} is empty.")
        return

    X_flat = df.drop('label', axis=1).values
    y = df['label'].values

    # Reshape from flat to (samples, keyframes, features)
    # The flat array has keyframes * 126 columns
    if X_flat.shape[1] != keyframes * n_features:
        print(f"Error: Expected {keyframes * n_features} columns, got {X_flat.shape[1]}")
        return

    X = X_flat.reshape(-1, keyframes, n_features).astype('float32')

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)

    # Save LabelEncoder
    le_path = os.path.join(models_dir, 'isl_word_lstm_le.pkl')
    with open(le_path, 'wb') as f:
        pickle.dump(le, f)

    y_cat = to_categorical(y_encoded, num_classes)
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

    print("Training Bi-LSTM...")
    lstm_model = build_bidi_lstm(keyframes=keyframes, n_features=n_features, num_classes=num_classes)
    lstm_model.fit(X_train, y_train, epochs=40, batch_size=16, validation_data=(X_test, y_test), verbose=1)

    loss, lstm_acc = lstm_model.evaluate(X_test, y_test, verbose=0)
    print(f"Bi-LSTM Accuracy: {lstm_acc:.4f}")

    lstm_path = os.path.join(models_dir, 'isl_word_lstm.keras')
    lstm_model.save(lstm_path)


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    print("Starting Model Training Pipeline...")

    # Static Models
    train_static_models("alphabet", "real_landmark_data/alphabet_landmarks.csv", "models")
    train_static_models("number", "real_landmark_data/number_landmarks.csv", "models")
    train_static_models("static_word", "real_landmark_data/static_word_landmarks.csv", "models")

    # Dynamic Models
    train_dynamic_models("video_landmark_data/word_video.csv", "models")

    print("\nAll models trained and saved to the 'models/' directory!")
