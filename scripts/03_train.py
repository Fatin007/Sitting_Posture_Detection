import numpy as np
import tensorflow as tf
import os

# Import directly from standard standalone Keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, GlobalAveragePooling1D

# Path configurations
BASE_DIR = r'C:\Users\USERAS\Desktop\Sitting_posture_research'
X = np.load(os.path.join(BASE_DIR, 'data', 'X_final.npy'))
y = np.load(os.path.join(BASE_DIR, 'data', 'y_final.npy'))

# Neural Network Architecture
model = Sequential([
    Input(shape=(60, 5)),
    LSTM(64, return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(32, return_sequences=True),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(2, activation='softmax')
])

# Compile and Train
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=40, batch_size=32, validation_split=0.2)

# Save using your preferred .h5 format layout
model.save(os.path.join(BASE_DIR, 'models', 'custom_posture_model.h5'))
print("Model trained and safely saved in .h5 format!")