import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from tensorflow import keras

# Load dataset
df = pd.read_csv("Data/parkinsons.data")

# Features that can be extracted offline
features = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
    'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
    'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
    'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR'
]

X = df[features]
y = df["status"]

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Gradient Boosting for validation
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.2, max_depth=3)
gb.fit(X_train, y_train)
print("✅ Accuracy:", accuracy_score(y_test, gb.predict(X_test)))

# Build and train Keras model
model = keras.Sequential([
    keras.layers.Input(shape=(len(features),)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30, batch_size=4, validation_split=0.2)

# Save to .h5 file (IMPORTANT: Must include .h5 extension)
model.save("Models/keras_voice_model.h5")
print("✅ Keras model saved to 'Models/keras_voice_model.h5'")

