import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Bidirectional, LSTM, LayerNormalization, Dropout, Dense
from tensorflow.keras.optimizers import Adam

def build_bidi_lstm(keyframes=30, n_features=126, num_classes=10, learning_rate=0.001):
    """
    Builds Bidirectional LSTM model for dynamic gesture sequence classification.
    """
    inp = Input(shape=(keyframes, n_features))
    x = Bidirectional(LSTM(64, return_sequences=False))(inp)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    out = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=Adam(learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
