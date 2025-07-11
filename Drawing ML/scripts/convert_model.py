# convert_model.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
import joblib

# Load your binary sklearn model and scaler
sk_model = joblib.load('models/logistic_regression_binary.pkl')
scaler = joblib.load('models/logistic_regression_scaler.pkl')

# Build equivalent Keras model
IMAGE_SIZE = 256  # Match your training config
INPUT_SHAPE = IMAGE_SIZE * IMAGE_SIZE

model = Sequential([
    Input(shape=(INPUT_SHAPE,)),
    Dense(1, activation='sigmoid')  # Binary output
])

# Set weights manually
W, b = sk_model.coef_, sk_model.intercept_
model.layers[0].set_weights([W.T, b])

# Save Keras model
model.save("models/keras_logistic_binary.h5")
print("✅ Keras model saved.")
