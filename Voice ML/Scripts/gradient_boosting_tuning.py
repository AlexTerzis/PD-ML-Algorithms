from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.datasets import load_iris  # Placeholder; will replace with real data
from joblib import dump
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load real Parkinson's dataset
df = pd.read_csv("Data/parkinsons.data")

# Feature selection (full set + RPDE + DFA)
X = df.drop(['name', 'status'], axis=1).assign(RPDE=df['spread1'], DFA=df['D2'])
y = df['status']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define Gradient Boosting and parameter grid
gb = GradientBoostingClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [2, 3, 4, 5]
}

# GridSearchCV
grid_search = GridSearchCV(gb, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Best model
best_gb = grid_search.best_estimator_
print("🔍 Best Parameters:", grid_search.best_params_)
print("✅ Best Cross-Validation Accuracy:", grid_search.best_score_)

# Evaluate on test set
y_pred = best_gb.predict(X_test_scaled)
print("\n📋 Test Set Classification Report:")
print(classification_report(y_test, y_pred))
print(f"🎯 Test Set Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Feature importance
importances = best_gb.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
sns.barplot(x=np.array(X.columns)[sorted_idx][:15], y=importances[sorted_idx][:15], palette="viridis")
plt.title("Top 15 Feature Importances - Gradient Boosting")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Save the best model

dump(best_gb, "models/best_gradient_boosting_model.joblib")
