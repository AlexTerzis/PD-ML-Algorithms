import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib

# === CONFIG ===
IMAGE_SIZE = 256
BASE_DIR = os.path.join("data", "YOLODatasetFull")
IMAGE_DIR = os.path.join(BASE_DIR, "images", "train")
LABEL_DIR = os.path.join(BASE_DIR, "labels", "train")
CLASS_NAMES = ['Healthy', 'Parkinson']
CLASS_MAP = {0: 0, 1: 0, 2: 1, 3: 1}  # Map all to 0 or 1

# === Load Data ===
X, y = [], []

for label_file in os.listdir(LABEL_DIR):
    if not label_file.endswith('.txt'):
        continue

    with open(os.path.join(LABEL_DIR, label_file), 'r') as f:
        line = f.readline().strip()
        if line == '':
            continue
        original_id = int(line.split()[0])
        if original_id not in CLASS_MAP:
            continue
        class_id = CLASS_MAP[original_id]
        y.append(class_id)

    # Try to find corresponding image
    for ext in ['.jpg', '.JPG', '.png', '.jpeg']:
        candidate = os.path.join(IMAGE_DIR, label_file.replace('.txt', ext))
        if os.path.exists(candidate):
            image_path = candidate
            break
    else:
        print("⚠️ Image not found for:", label_file)
        continue

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img / 255.0
    X.append(img.flatten())

X = np.array(X)
y = np.array(y)
print(f"\n✅ Loaded {len(X)} samples")

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Feature Scaling ===
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === Train Model ===
model = LogisticRegression(
    penalty='l2', C=1.0, solver='lbfgs',
    max_iter=1000, class_weight='balanced'
)
model.fit(X_train, y_train)

# === Cross-Validation ===
cv = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
print("📊 Cross-Val F1 Scores:", cv)
print("📊 Mean CV F1 Score:", np.mean(cv))

# === Evaluation ===
y_pred = model.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.xticks(np.arange(2), CLASS_NAMES)
plt.yticks(np.arange(2), CLASS_NAMES)
plt.tight_layout()
plt.show()

# === ROC Curve ===
from sklearn.preprocessing import label_binarize
y_bin = label_binarize(y_test, classes=[0, 1])
y_prob = model.predict_proba(X_test)

fpr, tpr, _ = roc_curve(y_bin, y_prob[:, 1])
auc = roc_auc_score(y_bin, y_prob[:, 1])
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# === Misclassified Examples ===
print("\n❌ Misclassified examples:")
wrong = np.where(y_test != y_pred)[0]
for i in wrong[:5]:
    img = X_test[i].reshape(IMAGE_SIZE, IMAGE_SIZE)
    true_label = CLASS_NAMES[y_test[i]]
    pred_label = CLASS_NAMES[y_pred[i]]
    plt.imshow(img, cmap='gray')
    plt.title(f"True: {true_label} | Pred: {pred_label}")
    plt.axis('off')
    plt.show()

# === Visualize Learned Weights ===
print("🧠 Weight Visualization:")
weights = model.coef_[0].reshape(IMAGE_SIZE, IMAGE_SIZE)
plt.imshow(weights, cmap='seismic')
plt.title("Weight Heatmap (Parkinson class)")
plt.colorbar()
plt.tight_layout()
plt.show()

# === Save Model & Scaler ===
joblib.dump(model, "models/logistic_regression_binary.pkl")
joblib.dump(scaler, "models/logistic_regression_scaler.pkl")
print("\n💾 Model & Scaler saved in: models/")

