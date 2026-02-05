# LSTM Model Architecture for ISL Recognition

import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
from config import MODEL_CONFIG, ISL_GESTURES

class ISLModel:
    """LSTM-based model for Indian Sign Language Recognition"""
    
    def __init__(self, num_classes=None):
        self.num_classes = num_classes or len(ISL_GESTURES)
        self.sequence_length = MODEL_CONFIG['sequence_length']
        self.num_features = MODEL_CONFIG['num_features']
        self.model = None
        
    def build_model(self):
        """
        Build LSTM model architecture
        
        Architecture:
        - Input: (sequence_length, num_features)
        - LSTM layers with dropout
        - Dense layers for classification
        - Output: Softmax probability distribution
        """
        model = models.Sequential([
            # Input layer
            layers.Input(shape=(self.sequence_length, self.num_features)),
            
            # First LSTM layer (return sequences for stacking)
            layers.LSTM(
                MODEL_CONFIG['lstm_units'][0],
                return_sequences=True,
                activation='tanh',
                recurrent_activation='sigmoid',
                name='lstm_1'
            ),
            layers.BatchNormalization(),
            layers.Dropout(MODEL_CONFIG['dropout_rate']),
            
            # Second LSTM layer
            layers.LSTM(
                MODEL_CONFIG['lstm_units'][1],
                return_sequences=True,
                activation='tanh',
                recurrent_activation='sigmoid',
                name='lstm_2'
            ),
            layers.BatchNormalization(),
            layers.Dropout(MODEL_CONFIG['dropout_rate']),
            
            # Third LSTM layer (no return sequences)
            layers.LSTM(
                MODEL_CONFIG['lstm_units'][2],
                return_sequences=False,
                activation='tanh',
                recurrent_activation='sigmoid',
                name='lstm_3'
            ),
            layers.BatchNormalization(),
            layers.Dropout(MODEL_CONFIG['dropout_rate']),
            
            # Dense layers
            layers.Dense(64, activation='relu', name='dense_1'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            layers.Dense(32, activation='relu', name='dense_2'),
            layers.Dropout(0.3),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax', name='output')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=MODEL_CONFIG['learning_rate']
            ),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        self.model = model
        return model
    
    def get_callbacks(self, checkpoint_path):
        """
        Get training callbacks
        
        Args:
            checkpoint_path: Path to save model checkpoints
            
        Returns:
            List of callbacks
        """
        callbacks = [
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Model checkpoint
            ModelCheckpoint(
                checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            
            # Learning rate reduction
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train(self, X_train, y_train, X_val, y_val, checkpoint_path):
        """
        Train the model
        
        Args:
            X_train: Training sequences
            y_train: Training labels (one-hot encoded)
            X_val: Validation sequences
            y_val: Validation labels
            checkpoint_path: Path to save checkpoints
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        callbacks = self.get_callbacks(checkpoint_path)
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=MODEL_CONFIG['batch_size'],
            epochs=MODEL_CONFIG['epochs'],
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, sequence):
        """
        Predict gesture from sequence
        
        Args:
            sequence: Input sequence of landmarks
            
        Returns:
            prediction: Predicted class
            confidence: Prediction confidence
            probabilities: All class probabilities
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call build_model() or load_model() first.")
        
        # Ensure correct shape
        if len(sequence.shape) == 2:
            sequence = np.expand_dims(sequence, axis=0)
        
        # Get predictions
        probabilities = self.model.predict(sequence, verbose=0)[0]
        
        # Get top prediction
        prediction_idx = np.argmax(probabilities)
        confidence = probabilities[prediction_idx]
        
        return prediction_idx, confidence, probabilities
    
    def save_model(self, filepath):
        """Save model to file"""
        if self.model is not None:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model from file"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        return self.model
    
    def summary(self):
        """Print model summary"""
        if self.model is not None:
            self.model.summary()
        else:
            print("Model not built yet. Call build_model() first.")

# Advanced Model with Attention Mechanism (Optional Enhancement)
class ISLModelWithAttention(ISLModel):
    """Enhanced LSTM model with attention mechanism"""
    
    def build_model(self):
        """Build LSTM model with attention layer"""
        
        # Input
        inputs = layers.Input(shape=(self.sequence_length, self.num_features))
        
        # LSTM layers
        lstm1 = layers.LSTM(128, return_sequences=True)(inputs)
        lstm1 = layers.BatchNormalization()(lstm1)
        lstm1 = layers.Dropout(0.3)(lstm1)
        
        lstm2 = layers.LSTM(64, return_sequences=True)(lstm1)
        lstm2 = layers.BatchNormalization()(lstm2)
        lstm2 = layers.Dropout(0.3)(lstm2)
        
        # Attention mechanism
        attention = layers.Dense(1, activation='tanh')(lstm2)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax')(attention)
        attention = layers.RepeatVector(64)(attention)
        attention = layers.Permute([2, 1])(attention)
        
        # Apply attention
        sent_representation = layers.Multiply()([lstm2, attention])
        sent_representation = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(sent_representation)
        
        # Dense layers
        dense1 = layers.Dense(64, activation='relu')(sent_representation)
        dense1 = layers.Dropout(0.4)(dense1)
        
        dense2 = layers.Dense(32, activation='relu')(dense1)
        dense2 = layers.Dropout(0.3)(dense2)
        
        # Output
        outputs = layers.Dense(self.num_classes, activation='softmax')(dense2)
        
        # Create model
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model

if __name__ == "__main__":
    # Test model creation
    print("Creating ISL LSTM Model...")
    model = ISLModel(num_classes=30)
    model.build_model()
    model.summary()
    
    print("\n" + "="*60)
    print("Model created successfully!")
    print(f"Input shape: ({MODEL_CONFIG['sequence_length']}, {MODEL_CONFIG['num_features']})")
    print(f"Output classes: 30 ISL gestures")
    print("="*60)
