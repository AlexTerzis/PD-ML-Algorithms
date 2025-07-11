
import tensorflow as tf

# Load the Keras model
model = tf.keras.models.load_model("models/keras_logistic_binary.h5")

# Convert to TFLite with compatible opset
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]  # Default opset (use v1)
# OPTIONAL: reduce precision for size
# converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert and save
tflite_model = converter.convert()
with open("models/drawing_binary_classifier2.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ TFLite model converted and saved.")
