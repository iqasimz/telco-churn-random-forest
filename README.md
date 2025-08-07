This project uses a Random Forest Classifier to predict whether a customer will churn (i.e., cancel their subscription) based on their service usage, billing history, contract type, and demographics.

Objectives:
- Predict customer churn (`Churn`: Yes/No) using historical telco data
- Handle categorical and numerical features in preprocessing
- Train and evaluate a Random Forest Classifier
- Measure performance using Accuracy, Precision, Recall, F1 Score, and ROC-AUC
- Visualize the most important features influencing churn
- Identify improvement opportunities for better customer retention

Methodology:

- Used the `WA_Fn-UseC_-Telco-Customer-Churn.csv` dataset from Kaggle
- Dropped `customerID` as it carries no predictive value
- Converted `TotalCharges` to numeric and filled missing values with the median
- Encoded binary columns like `Churn`, `Partner`, and `Dependents` as 0/1
- Label-encoded other categorical variables (e.g., `Contract`, `InternetService`)
- Split dataset into training (80%) and testing (20%) sets using stratified sampling
- Trained a `RandomForestClassifier(n_estimators=100, random_state=42)`
- Evaluated the model on the test set

Evaluation Metrics (Test Set):
Precision Recall F1 Score Accuracy ROC-AUC

Observations:
- The model performs well in identifying non-churning customers
- It struggles with predicting churners (Class 1), with a recall of only 50%
- This class imbalance could be addressed with `class_weight='balanced'` or SMOTE

Visualizations:
All visual outputs are saved in the `visuals/` folder:
- `confusion_matrix.png`: Breakdown of true positives, false positives, etc.
- `roc_curve.png`: AUC = 0.82 → model has good class separability
- `feature_importance.png`: Key drivers of churn (e.g., Contract, Tenure, MonthlyCharges)
