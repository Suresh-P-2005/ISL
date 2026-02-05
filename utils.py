# Utility Functions

import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from config import ISL_GESTURES

def preprocess_sequence(sequence):
    """
    Preprocess landmark sequence for model input
    
    Args:
        sequence: Numpy array of shape (sequence_length, num_features)
        
    Returns:
        preprocessed_sequence: Normalized and scaled sequence
    """
    # Normalize each frame
    normalized_sequence = []
    
    for frame_landmarks in sequence:
        # Reshape to (21, 3)
        landmarks = frame_landmarks.reshape(21, 3)
        
        # Get wrist position (landmark 0)
        wrist = landmarks[0].copy()
        
        # Translate to wrist origin
        landmarks = landmarks - wrist
        
        # Calculate hand size
        distances = np.linalg.norm(landmarks, axis=1)
        hand_size = np.max(distances)
        
        if hand_size > 0:
            landmarks = landmarks / hand_size
        
        normalized_sequence.append(landmarks.flatten())
    
    normalized_sequence = np.array(normalized_sequence)
    
    # Additional standardization
    scaler = StandardScaler()
    standardized = scaler.fit_transform(normalized_sequence)
    
    return standardized

def decode_prediction(prediction_idx, confidence, all_probabilities=None):
    """
    Decode model prediction to gesture name and details
    
    Args:
        prediction_idx: Predicted class index
        confidence: Prediction confidence
        all_probabilities: All class probabilities (optional)
        
    Returns:
        result: Dictionary with prediction details
    """
    gesture_names = list(ISL_GESTURES.keys())
    
    if prediction_idx >= len(gesture_names):
        return {
            'gesture': 'Unknown',
            'confidence': 0.0,
            'translation': 'Gesture not recognized'
        }
    
    gesture = gesture_names[prediction_idx]
    translation = ISL_GESTURES[gesture]
    
    result = {
        'gesture': gesture,
        'confidence': float(confidence),
        'translation': translation
    }
    
    if all_probabilities is not None:
        top_5_indices = np.argsort(all_probabilities)[-5:][::-1]
        top_5 = {
            gesture_names[i]: float(all_probabilities[i])
            for i in top_5_indices if i < len(gesture_names)
        }
        result['top_5'] = top_5
    
    return result

def augment_sequence(sequence):
    """
    Augment landmark sequence for training
    
    Augmentation techniques:
    - Add random noise
    - Scale variation
    - Rotation
    - Time warping
    
    Args:
        sequence: Original sequence
        
    Returns:
        augmented_sequences: List of augmented sequences
    """
    augmented = []
    
    # Original
    augmented.append(sequence)
    
    # Add noise
    noise = np.random.normal(0, 0.02, sequence.shape)
    augmented.append(sequence + noise)
    
    # Scale variation
    scale = np.random.uniform(0.9, 1.1)
    augmented.append(sequence * scale)
    
    # Rotation (2D rotation in x-y plane)
    angle = np.random.uniform(-0.2, 0.2)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[cos_a, -sin_a, 0],
                                [sin_a, cos_a, 0],
                                [0, 0, 1]])
    
    rotated = sequence.copy()
    for i in range(len(sequence)):
        landmarks = sequence[i].reshape(21, 3)
        rotated_landmarks = landmarks @ rotation_matrix.T
        rotated[i] = rotated_landmarks.flatten()
    augmented.append(rotated)
    
    # Time warping (speed variation)
    indices = np.linspace(0, len(sequence) - 1, len(sequence))
    warped_indices = np.sort(np.random.uniform(0, len(sequence) - 1, len(sequence)))
    warped = np.interp(indices, warped_indices, np.arange(len(sequence)))
    warped = warped.astype(int)
    augmented.append(sequence[warped])
    
    return augmented

def calculate_accuracy_metrics(y_true, y_pred, y_proba=None):
    """
    Calculate various accuracy metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities (optional)
        
    Returns:
        metrics: Dictionary of accuracy metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix
    )
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics

def save_training_history(history, filepath):
    """
    Save training history to JSON file
    
    Args:
        history: Keras training history object
        filepath: Path to save file
    """
    import json
    
    history_dict = {
        'loss': [float(x) for x in history.history['loss']],
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']]
    }
    
    with open(filepath, 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    print(f"Training history saved to {filepath}")

def load_training_history(filepath):
    """Load training history from JSON file"""
    import json
    
    with open(filepath, 'r') as f:
        history_dict = json.load(f)
    
    return history_dict

def visualize_landmarks(landmarks, image_shape=(480, 640, 3)):
    """
    Visualize hand landmarks on blank image
    
    Args:
        landmarks: Landmark array (63 features)
        image_shape: Output image shape
        
    Returns:
        image: Image with drawn landmarks
    """
    image = np.zeros(image_shape, dtype=np.uint8)
    height, width = image_shape[:2]
    
    # Reshape landmarks
    landmarks_2d = landmarks.reshape(21, 3)[:, :2]  # Take only x, y
    
    # Scale to image size
    landmarks_2d[:, 0] *= width
    landmarks_2d[:, 1] *= height
    landmarks_2d = landmarks_2d.astype(int)
    
    # Draw connections (hand skeleton)
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
        (5, 9), (9, 13), (13, 17)  # Palm
    ]
    
    # Draw connections
    for start, end in connections:
        if start < len(landmarks_2d) and end < len(landmarks_2d):
            cv2.line(image, 
                    tuple(landmarks_2d[start]), 
                    tuple(landmarks_2d[end]), 
                    (0, 255, 0), 2)
    
    # Draw landmarks
    for point in landmarks_2d:
        cv2.circle(image, tuple(point), 4, (0, 0, 255), -1)
    
    return image

def create_sequence_video(sequence, output_path, fps=10):
    """
    Create video from landmark sequence
    
    Args:
        sequence: Sequence of landmarks
        output_path: Path to save video
        fps: Frames per second
    """
    frame_height, frame_width = 480, 640
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    for landmarks in sequence:
        frame = visualize_landmarks(landmarks, (frame_height, frame_width, 3))
        out.write(frame)
    
    out.release()
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test sequence preprocessing
    dummy_sequence = np.random.rand(30, 63)
    processed = preprocess_sequence(dummy_sequence)
    print(f"Preprocessed sequence shape: {processed.shape}")
    
    # Test augmentation
    augmented = augment_sequence(dummy_sequence)
    print(f"Generated {len(augmented)} augmented sequences")
    
    print("Utilities test completed!")
