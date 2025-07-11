# keras_to_tflite.py
# Converts a saved Keras model (.keras or .h5) into a TFLite model for mobile/embedded use.

import tensorflow as tf
import os

# === Step 1: Load the trained Keras model ===
# Make sure this matches your filename and path
keras_model_path = "Models/keras_voice_model.h5"
model = tf.keras.models.load_model(keras_model_path)

# === Step 2: Convert the Keras model to TensorFlow Lite format ===
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# === Step 3: Save the TFLite model ===
tflite_model_path = "Models/voice_model.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"✅ TFLite model saved successfully to: {tflite_model_path}")
