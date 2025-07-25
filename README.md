# Parkinsons-ML-Algorithms
ML Algorithms for parkinson

Developed by Alex Terzis
Electrical & Computer Engineering student | AI Developer
🇸🇪 Based in Sweden

Parkinson's Disease Detection Using Drawing Analysis

This project uses machine learning models to detect **Parkinson’s Disease** by analyzing patients' **hand-drawn spirals and waves**.

It compares classical ML algorithms on feature data extracted from drawings, and prepares the best model for deployment.

Parkinson's Disease Detection Using Voice Analysis
| Model                 | Description                             |
|----------------------|-----------------------------------------|
| Logistic Regression  | Linear model for binary classification  |
| SVM                  | Finds optimal separating hyperplane     |
| KNN                  | Classifies based on neighbor majority   |
| Random Forest        | Ensemble of decision trees              |
| Gradient Boosting    | Iteratively improves weak learners      |
| Decision Tree        | Simple tree-based splitting             |
| QDA (from paper)     | Quadratic Discriminant Analysis         |

 **Best Accuracy** was achieved with: `GradientBoostingClassifier`

Features

- `draw_script.py` compares models and generates:
  - Confusion matrices
  - ROC curves
  - t-SNE visualization
  - Feature importance
- `tuning.py`: Performs GridSearchCV on top models
- `joblib_to_keras.py`: Converts model for mobile use
- `keras_to_tflite.py`: Exports model to `.tflite` format for Flutter

Results

Sample outputs include:

- Model Accuracy Comparison Bar Chart
- TP/TN/FP/FN Bar Breakdown
- Confusion Matrices
- ROC Curves
- Feature Importance Ranking
- t-SNE 2D Embedding of Feature Space

 Mobile Integration 

The exported `.tflite` model is ready for use in a **Flutter app** via `tflite_flutter`.  
Future versions may include:
- Real-time drawing input
- In-app model inference
- Patient test history and results sharing
#   P a r k i n s o n - M L - A l g o r i t h m s  
 #   P a r k i n s o n - M L - A l g o r i t h m s  
 #   P D - M L - A l g o r i t h m s  
 #   P D - M L - A l g o r i t h m s  
 #   P D - M L - A l g o r i t h m s  
 