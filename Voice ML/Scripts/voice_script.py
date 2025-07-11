# voice_script_v2.py
# Parkinson's Voice Detection: Multi-Model Training + Full Evaluation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.manifold import TSNE

# Load dataset
df = pd.read_csv("Data/parkinsons.data")

# Define models with short comments
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),       # Linear boundary, interpretable
    "SVM": SVC(probability=True),                                    # High-dimensional margins
    "KNN": KNeighborsClassifier(),                                   # Lazy learning, based on neighbors
    "Random Forest": RandomForestClassifier(),                       # Ensemble of decision trees
    "Gradient Boosting": GradientBoostingClassifier(),               # Tree boosting for strong accuracy
    "Naive Bayes": GaussianNB(),                                     # Probabilistic, assumes feature independence
    "Decision Tree": DecisionTreeClassifier(),                       # Fast, interpretable splits
    "QDA (from paper)": QuadraticDiscriminantAnalysis()              # Assumes quadratic boundaries, used in original paper
}

# Define feature sets
feature_sets = {
    "All Features + RPDE + DFA": lambda df: df.drop(['name', 'status'], axis=1).assign(RPDE=df['spread1'], DFA=df['D2']),
    "Only RPDE + DFA": lambda df: df[['spread1', 'D2']]
}

# Loop through feature sets
for fs_name, feature_fn in feature_sets.items():
    print(f"\n==============================")
    print(f"▶️ FEATURE SET: {fs_name}")
    print(f"==============================")

    X = feature_fn(df)
    y = df['status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results, runtimes, probas, confusion_data = {}, {}, {}, {}

    for name, model in models.items():
        start = time.time()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        elapsed = time.time() - start

        y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        results[name] = acc
        runtimes[name] = elapsed
        probas[name] = (y_test, y_proba)
        confusion_data[name] = [tp, tn, fp, fn]

        print(f"\n🔍 {name}")
        print(f"✅ Accuracy: {acc:.4f}")
        print(f"⏱️ Runtime: {elapsed:.4f}s")
        print(f"Confusion matrix:\n[[{tn}, {fp}],\n [{fn}, {tp}]]")
        print(classification_report(y_test, y_pred))

    # Accuracy bar chart
    plt.figure(figsize=(10, 5))
    sorted_results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
    plt.bar(sorted_results.keys(), sorted_results.values(), color='skyblue')
    plt.ylabel("Accuracy")
    plt.title(f"Model Accuracy - {fs_name}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Runtime chart
    plt.figure(figsize=(10, 5))
    sorted_runtimes = dict(sorted(runtimes.items(), key=lambda item: item[1]))
    plt.bar(sorted_runtimes.keys(), sorted_runtimes.values(), color='orange')
    plt.ylabel("Seconds")
    plt.title(f"Model Runtime - {fs_name}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Stacked confusion chart
    confusion_df = pd.DataFrame.from_dict(confusion_data, orient='index', columns=["TP", "TN", "FP", "FN"])
    confusion_df.plot(kind='bar', stacked=True, colormap='tab20', figsize=(12, 6))
    plt.title(f"Confusion Breakdown - {fs_name}")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ROC curves
    plt.figure(figsize=(10, 6))
    for name, (true, score) in probas.items():
        fpr, tpr, _ = roc_curve(true, score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"ROC Curves - {fs_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    # t-SNE 2D visualization
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_embedded = tsne.fit_transform(X_test_scaled)

    # Use best model predictions for labeling
    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]
    best_preds = best_model.predict(X_test_scaled)

    df_tsne = pd.DataFrame()
    df_tsne["TSNE1"] = X_embedded[:, 0]
    df_tsne["TSNE2"] = X_embedded[:, 1]
    df_tsne["Actual"] = y_test.values
    df_tsne["Predicted"] = best_preds

    def label_type(row):
        if row["Actual"] == 1 and row["Predicted"] == 1:
            return "TP"
        elif row["Actual"] == 0 and row["Predicted"] == 0:
            return "TN"
        elif row["Actual"] == 0 and row["Predicted"] == 1:
            return "FP"
        else:
            return "FN"

    df_tsne["Outcome"] = df_tsne.apply(label_type, axis=1)

    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df_tsne, x="TSNE1", y="TSNE2", hue="Outcome", palette="Set1", style="Outcome", s=90)
    plt.title(f"t-SNE Visualization (Best: {best_model_name}) - {fs_name}")
    plt.tight_layout()
    plt.show()

    # Feature importance (only for full set)
    if fs_name == "All Features + RPDE + DFA":
        rf = RandomForestClassifier()
        rf.fit(X_train_scaled, y_train)
        importances = rf.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        plt.figure(figsize=(12, 6))
        sns.barplot(x=np.array(X.columns)[sorted_idx], y=importances[sorted_idx])
        plt.xticks(rotation=45)
        plt.title("Random Forest Feature Importance")
        plt.tight_layout()
        plt.show()

        # GridSearchCV on Random Forest
        print("\n🔧 GridSearchCV - Random Forest")
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5]
        }
        grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train_scaled, y_train)
        print("Best Params:", grid_search.best_params_)
        print("Best CV Accuracy:", grid_search.best_score_)
