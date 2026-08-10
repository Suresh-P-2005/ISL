import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam

def build_1d_cnn(n_features=126, num_classes=26, learning_rate=0.001):
    """
    Builds lightweight 1D CNN for static gesture landmark classification.
    """
    model = Sequential([
        Conv1D(64, 3, activation='relu', padding='same', input_shape=(n_features, 1)),
        MaxPooling1D(2),
        Conv1D(32, 3, activation='relu', padding='same'),
        GlobalAveragePooling1D(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
