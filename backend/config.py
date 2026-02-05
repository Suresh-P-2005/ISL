# Configuration Settings

import os

# ISL Gestures Dictionary (30 gestures for comprehensive recognition)
ISL_GESTURES = {
    'A': 'Letter A',
    'B': 'Letter B',
    'C': 'Letter C',
    'D': 'Letter D',
    'E': 'Letter E',
    'F': 'Letter F',
    'G': 'Letter G',
    'H': 'Letter H',
    'I': 'Letter I',
    'J': 'Letter J',
    'K': 'Letter K',
    'L': 'Letter L',
    'M': 'Letter M',
    'N': 'Letter N',
    'O': 'Letter O',
    'P': 'Letter P',
    'Q': 'Letter Q',
    'R': 'Letter R',
    'S': 'Letter S',
    'T': 'Letter T',
    'Hello': 'Hello / Greetings',
    'Thank You': 'Thank you',
    'Sorry': 'Sorry',
    'Please': 'Please',
    'Help': 'Help me',
    'Yes': 'Yes / Agree',
    'No': 'No / Disagree',
    'Water': 'I need water',
    'Food': 'I need food',
    'Emergency': 'Emergency'
}

# ISL English Translations (detailed)
ISL_TRANSLATIONS = {
    'A': 'Letter A - First letter of the alphabet',
    'B': 'Letter B - Second letter of the alphabet',
    'C': 'Letter C - Third letter of the alphabet',
    'D': 'Letter D - Fourth letter of the alphabet',
    'E': 'Letter E - Fifth letter of the alphabet',
    'F': 'Letter F - Sixth letter of the alphabet',
    'G': 'Letter G - Seventh letter of the alphabet',
    'H': 'Letter H - Eighth letter of the alphabet',
    'I': 'Letter I - Ninth letter of the alphabet',
    'J': 'Letter J - Tenth letter of the alphabet',
    'K': 'Letter K - Eleventh letter of the alphabet',
    'L': 'Letter L - Twelfth letter of the alphabet',
    'M': 'Letter M - Thirteenth letter of the alphabet',
    'N': 'Letter N - Fourteenth letter of the alphabet',
    'O': 'Letter O - Fifteenth letter of the alphabet',
    'P': 'Letter P - Sixteenth letter of the alphabet',
    'Q': 'Letter Q - Seventeenth letter of the alphabet',
    'R': 'Letter R - Eighteenth letter of the alphabet',
    'S': 'Letter S - Nineteenth letter of the alphabet',
    'T': 'Letter T - Twentieth letter of the alphabet',
    'Hello': 'Greeting: Hello, How are you? Nice to meet you.',
    'Thank You': 'Expression of gratitude and appreciation.',
    'Sorry': 'Apology or expression of regret.',
    'Please': 'Polite request or asking for permission.',
    'Help': 'Request for assistance or support.',
    'Yes': 'Affirmative response, agreement, or confirmation.',
    'No': 'Negative response, disagreement, or denial.',
    'Water': 'Request: I need water to drink.',
    'Food': 'Request: I need food to eat.',
    'Emergency': 'Urgent situation requiring immediate attention or help.'
}

# Model Configuration
MODEL_CONFIG = {
    'sequence_length': 30,      # Number of frames per gesture
    'num_features': 63,         # 21 landmarks * 3 coordinates (x, y, z)
    'lstm_units': [128, 64, 32],  # LSTM layer units
    'dropout_rate': 0.3,        # Dropout rate for regularization
    'learning_rate': 0.001,     # Initial learning rate
    'batch_size': 32,           # Training batch size
    'epochs': 100,              # Maximum training epochs
    'validation_split': 0.2     # Validation data split
}

# MediaPipe Configuration
MEDIAPIPE_CONFIG = {
    'static_image_mode': False,
    'max_num_hands': 1,
    'min_detection_confidence': 0.7,
    'min_tracking_confidence': 0.5
}

# Data Paths
DATA_PATHS = {
    'raw_data': '../data/raw/',
    'processed_data': '../data/processed/',
    'models': '../data/models/',
    'logs': '../logs/'
}

# Training Configuration
TRAINING_CONFIG = {
    'train_test_split': 0.8,
    'augmentation': True,
    'normalize': True,
    'shuffle': True,
    'random_state': 42
}

# Server Configuration
SERVER_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': True,
    'threaded': True
}

# Create directories if they don't exist
for path in DATA_PATHS.values():
    os.makedirs(path, exist_ok=True)

# Dataset Information (for training)
DATASET_INFO = {
    'name': 'ISL Hand Gesture Dataset',
    'samples_per_gesture': 100,  # Minimum samples needed per gesture
    'total_gestures': len(ISL_GESTURES),
    'video_format': '.mp4',
    'image_format': '.jpg',
    'fps': 30
}

# Performance Thresholds
PERFORMANCE_THRESHOLDS = {
    'min_confidence': 0.7,      # Minimum confidence for prediction
    'high_confidence': 0.9,     # High confidence threshold
    'sequence_buffer_size': 30  # Number of frames to buffer
}

if __name__ == "__main__":
    print("Configuration loaded successfully!")
    print(f"Total ISL Gestures: {len(ISL_GESTURES)}")
    print(f"Model Sequence Length: {MODEL_CONFIG['sequence_length']}")
    print(f"Model Features: {MODEL_CONFIG['num_features']}")
    print(f"LSTM Units: {MODEL_CONFIG['lstm_units']}")
