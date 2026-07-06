**Fraud Detection Pipeline — Project 2**

**Goal**

Build and tune a classification model to identify fraudulent credit card transactions in a highly imbalanced dataset, using strict, leak-free evaluation.

**Dataset**


**Source:** Kaggle — Credit Card Fraud Detection (ULB Machine Learning Group)
**Size:** 284,807 transactions, 31 features (Time, Amount, V1–V28 PCA components, Class)
**Imbalance:** 99.83% legitimate vs. 0.17% fraudulent


**Approach**


**Stratified train/test split** (80/20) performed before any scaling or resampling, to prevent data leakage.
**imblearn Pipeline** (not sklearn's) used so SMOTE is applied fresh inside every cross-validation fold, never on the test set.
Two models trained and tuned via **GridSearchCV** (scoring = ROC-AUC):

Logistic Regression (with StandardScaler + SMOTE)
Random Forest (with SMOTE only — tree splits are scale-invariant)



Evaluated on the untouched test set using **Precision, Recall, F1, and ROC-AUC** — accuracy was intentionally ignored, since it's misleading on a 99.83%-imbalanced dataset.


**Results**

Metric (Fraud class)   Logistic Regression   Random Forest
Precision               0.06                  **0.51**
Recall                  **0.92**                  0.89
F1-score                0.11                  **0.65**
ROC-AUC                 0.971                 **0.985**

Best CV ROC-AUC during tuning:


Logistic Regression: 0.9807 (C=0.01, smote__k_neighbors=5)
Random Forest: 0.9820 (max_depth=10, n_estimators=50, smote__k_neighbors=5)


**Precision–Recall Trade-off (Key Insight)**


**Logistic Regression** catches slightly more fraud (92% recall) but at a heavy cost: only 6% of its fraud flags are actually fraud. In production this would mean flooding a fraud team with false alarms.
**Random Forest** is far more balanced: it still catches 89% of fraud, but over half of its fraud flags (51%) are correct — a much more usable model for a real fraud-review workflow.
Random Forest wins overall on ROC-AUC, precision, and F1, while giving up only 3 percentage points of recall versus Logistic Regression.


**Key Learnings**


Accuracy is a trap on imbalanced data — a model predicting "legitimate" every time would score 99.83% accuracy while catching zero fraud.
Data leakage happens if SMOTE or scaling is applied before the train/test split — this makes the test set artificially easy and inflates scores.
imblearn.pipeline.Pipeline (not sklearn.pipeline.Pipeline) is required to safely combine resampling with cross-validation.
Model choice changes the precision/recall trade-off, not just the accuracy — this trade-off should drive model selection based on business needs (e.g., minimizing false alarms vs. missing fraud).


**Tech Stack**

Python, pandas, scikit-learn, imbalanced-learn (SMOTE), matplotlib, seaborn
