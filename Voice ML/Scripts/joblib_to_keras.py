# joblib_to_keras.py
# Recreate a Keras model based on joblib-trained data

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import os

# Load your Parkinson's dataset
df = pd.read_csv("Data/parkinsons.data")

# Drop 'name' and extract features and labels
X = df.drop(['name', 'status'], axis=1)
y = df['status']

# Split into train/test and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define a simple Keras neural network
model = keras.Sequential([
    keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model to approximate the behavior of the joblib version
model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=30)

# Save to folder
os.makedirs("Models", exist_ok=True)
model.save("Models/keras_voice_model.h5")

print("✅ Saved Keras model to Models/keras_voice_model")
