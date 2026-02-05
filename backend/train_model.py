# Model Training Script

import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from datetime import datetime

from model import ISLModel
from mediapipe_handler import MediaPipeHandler
from utils import preprocess_sequence, augment_sequence, save_training_history
from config import ISL_GESTURES, MODEL_CONFIG, DATA_PATHS, TRAINING_CONFIG

class ISLTrainer:
    """Trainer class for ISL Recognition Model"""
    
    def __init__(self):
        self.model = ISLModel()
        self.mediapipe_handler = MediaPipeHandler()
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        
    def load_dataset(self, data_path):
        """
        Load preprocessed dataset from numpy files
        
        Expected structure:
        data_path/
            gesture_name_1/
                sequence_001.npy
                sequence_002.npy
                ...
            gesture_name_2/
                ...
        
        Args:
            data_path: Path to dataset directory
            
        Returns:
            X: Feature sequences
            y: Labels
        """
        X = []
        y = []
        gesture_names = list(ISL_GESTURES.keys())
        
        print("Loading dataset...")
        
        for gesture_idx, gesture_name in enumerate(gesture_names):
            gesture_path = os.path.join(data_path, gesture_name)
            
            if not os.path.exists(gesture_path):
                print(f"Warning: No data found for gesture '{gesture_name}'")
                continue
            
            # Load all sequences for this gesture
            sequence_files = [f for f in os.listdir(gesture_path) if f.endswith('.npy')]
            
            for seq_file in sequence_files:
                seq_path = os.path.join(gesture_path, seq_file)
                sequence = np.load(seq_path)
                
                # Preprocess
                processed_seq = preprocess_sequence(sequence)
                
                X.append(processed_seq)
                y.append(gesture_idx)
                
                # Apply augmentation if enabled
                if TRAINING_CONFIG['augmentation']:
                    augmented_seqs = augment_sequence(processed_seq)
                    for aug_seq in augmented_seqs[1:]:  # Skip original
                        X.append(aug_seq)
                        y.append(gesture_idx)
            
            print(f"Loaded {len(sequence_files)} sequences for '{gesture_name}'")
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"\nDataset loaded: {len(X)} sequences, {len(gesture_names)} classes")
        
        return X, y
    
    def prepare_data(self, X, y):
        """
        Prepare data for training
        
        Args:
            X: Features
            y: Labels
        """
        # Shuffle data
        if TRAINING_CONFIG['shuffle']:
            X, y = shuffle(X, y, random_state=TRAINING_CONFIG['random_state'])
        
        # Convert labels to one-hot encoding
        from tensorflow.keras.utils import to_categorical
        y_categorical = to_categorical(y, num_classes=len(ISL_GESTURES))
        
        # Split into train, validation, and test
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y_categorical,
            test_size=0.1,
            random_state=TRAINING_CONFIG['random_state'],
            stratify=y
        )
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp,
            test_size=MODEL_CONFIG['validation_split'],
            random_state=TRAINING_CONFIG['random_state']
        )
        
        print(f"\nData split:")
        print(f"  Training: {len(self.X_train)} sequences")
        print(f"  Validation: {len(self.X_val)} sequences")
        print(f"  Test: {len(self.X_test)} sequences")
    
    def train(self):
        """Train the model"""
        if self.X_train is None:
            raise ValueError("Data not prepared. Call load_dataset() and prepare_data() first.")
        
        print("\n" + "="*60)
        print("Starting model training...")
        print("="*60)
        
        # Build model
        self.model.build_model()
        self.model.summary()
        
        # Checkpoint path
        checkpoint_path = os.path.join(
            DATA_PATHS['models'],
            f'isl_lstm_checkpoint_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5'
        )
        
        # Train model
        history = self.model.train(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            checkpoint_path
        )
        
        # Save final model
        final_model_path = os.path.join(DATA_PATHS['models'], 'isl_lstm_model.h5')
        self.model.save_model(final_model_path)
        
        # Save training history
        history_path = os.path.join(DATA_PATHS['models'], 'training_history.json')
        save_training_history(history, history_path)
        
        return history
    
    def evaluate(self):
        """Evaluate model on test set"""
        if self.X_test is None:
            raise ValueError("Test data not available.")
        
        print("\n" + "="*60)
        print("Evaluating model on test set...")
        print("="*60)
        
        # Evaluate
        test_loss, test_accuracy, test_top_k = self.model.model.evaluate(
            self.X_test, self.y_test,
            verbose=1
        )
        
        print(f"\nTest Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_accuracy*100:.2f}%")
        print(f"  Top-5 Accuracy: {test_top_k*100:.2f}%")
        
        # Detailed predictions
        predictions = self.model.model.predict(self.X_test)
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(self.y_test, axis=1)
        
        # Calculate per-class accuracy
        gesture_names = list(ISL_GESTURES.keys())
        print("\nPer-class accuracy:")
        for i, gesture in enumerate(gesture_names):
            mask = true_classes == i
            if np.sum(mask) > 0:
                class_acc = np.mean(pred_classes[mask] == i)
                print(f"  {gesture}: {class_acc*100:.2f}%")
        
        return test_accuracy
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Train Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Train Loss')
        ax2.plot(history.history['val_loss'], label='Val Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(DATA_PATHS['models'], 'training_history.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nTraining plot saved to {plot_path}")
        
        plt.show()

def create_sample_dataset():
    """
    Create sample dataset for testing
    (In production, replace with real video data)
    """
    print("Creating sample dataset...")
    
    processed_data_path = DATA_PATHS['processed_data']
    gesture_names = list(ISL_GESTURES.keys())
    
    for gesture_name in gesture_names:
        gesture_path = os.path.join(processed_data_path, gesture_name)
        os.makedirs(gesture_path, exist_ok=True)
        
        # Create 50 random sequences for each gesture
        for i in range(50):
            sequence = np.random.rand(30, 63)  # Random landmarks
            seq_file = os.path.join(gesture_path, f'sequence_{i:03d}.npy')
            np.save(seq_file, sequence)
        
        print(f"Created 50 sequences for '{gesture_name}'")
    
    print("Sample dataset created successfully!")

if __name__ == "__main__":
    print("="*60)
    print("ISL Recognition Model Training")
    print("="*60)
    
    # Check if dataset exists
    processed_data_path = DATA_PATHS['processed_data']
    
    if not os.path.exists(processed_data_path) or len(os.listdir(processed_data_path)) == 0:
        print("\nNo dataset found. Creating sample dataset...")
        create_sample_dataset()
    
    # Initialize trainer
    trainer = ISLTrainer()
    
    # Load dataset
    X, y = trainer.load_dataset(processed_data_path)
    
    # Prepare data
    trainer.prepare_data(X, y)
    
    # Train model
    history = trainer.train()
    
    # Plot training history
    trainer.plot_training_history(history)
    
    # Evaluate model
    trainer.evaluate()
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)
