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

# Fold the scaler into the layer weights.
#
# The scaler was loaded above but never applied, so the exported model scored
# raw pixels using coefficients that had been fitted on standardized ones. The
# app's predictions were therefore unrelated to the training pipeline: a blank
# page came out at 0.30 instead of 0.004.
#
# Both the scaler and the classifier are linear, so the standardization folds
# into the weights exactly -- no approximation, and nothing for the app to
# apply at inference time:
#
#   w . (x - mean) / scale + b  ==  (w / scale) . x + (b - (w / scale) . mean)
mean = scaler.mean_ if scaler.with_mean else np.zeros(INPUT_SHAPE)
scale = scaler.scale_ if scaler.with_std else np.ones(INPUT_SHAPE)

W = sk_model.coef_ / scale          # (1, n_features)
b = sk_model.intercept_ - W @ mean  # (1,)

model.layers[0].set_weights([W.T, b])

# Prove the fold is exact rather than merely close.
#
# Two separate checks: the algebra is verified in float64 against the real
# sklearn pipeline, and the exported Keras graph is then checked against it in
# float32, which is all the precision a 65k-term dot product can carry.
rng = np.random.default_rng(0)
probe = rng.random((16, INPUT_SHAPE))

expected_logits = sk_model.decision_function(scaler.transform(probe))
folded_logits = (probe @ W.T).ravel() + b
np.testing.assert_allclose(folded_logits, expected_logits, rtol=1e-9, atol=1e-9)

expected = 1.0 / (1.0 + np.exp(-expected_logits))
actual = model.predict(probe.astype(np.float32), verbose=0).ravel()
np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)
# Plain ASCII: the default Windows console code page (cp1252) cannot encode
# emoji, and a UnicodeEncodeError here would abort the script before the save.
print(
    '[OK] Folded model matches the sklearn pipeline '
    f'(max diff {np.max(np.abs(actual - expected)):.2e}).'
)

# Save Keras model
model.save("models/keras_logistic_binary.h5")
print("[OK] Keras model saved.")
