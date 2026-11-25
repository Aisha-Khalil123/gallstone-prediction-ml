# Gallstone Prediction with Machine Learning

Predicting whether a patient has gallstones from bioimpedance and laboratory data, comparing four tuned classifiers and explaining their decisions with SHAP and permutation importance. Built for the *Machine Learning* course at Al Hussein Technical University (Jan 2026).

## Problem

Gallstone disease is usually confirmed with ultrasound. This project asks a simpler question: can routine, non-invasive measurements predict gallstone status accurately enough to help flag patients for further testing? Examples of these measurements are body composition (fat, water, muscle), blood tests and patient history.

## Data

- **Source:** [Gallstone dataset, UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/1150/gallstone-1)
- 319 patients, 38 features plus the target `Gallstone Status` (0 = no, 1 = yes)
- The classes are balanced (about 50/50). There are no missing values or duplicates, and every feature is numeric.
- Download the CSV from the link above and save it as `dataset-uci.csv` next to the notebook.

## Method

1. **Exploration:** summary statistics, data types, missing values, duplicates, target distribution and a correlation heatmap
2. **Split and scaling:** stratified 80/20 train/test split (255 / 64 patients), with a `StandardScaler` fitted on the training set only
3. **Modelling:** four classifiers, each tuned with `GridSearchCV` (5-fold cross-validation)
4. **Evaluation:** accuracy, precision, recall, F1 and confusion matrix on the held-out test set
5. **Explainability:** SHAP summary plots (TreeExplainer for the tree models, LinearExplainer for the SVM) and permutation importance

## Results (test set, 64 patients)

| Model | Best parameters | CV accuracy | Test accuracy | Precision | Recall | F1 |
|-------|-----------------|-------------|---------------|-----------|--------|----|
| **SVM** | C = 10, linear kernel | 0.796 | **0.828** | 0.839 | 0.813 | **0.825** |
| Random Forest | 100 trees, max depth 5 | 0.788 | 0.813 | 0.813 | 0.813 | 0.813 |
| XGBoost | 100 trees, max depth 7 | 0.776 | 0.797 | 0.757 | 0.875 | 0.812 |
| Decision Tree | entropy, max depth 5 | 0.706 | 0.688 | 0.658 | 0.781 | 0.714 |

**The linear SVM performed best** (82.8% accuracy, F1 0.83). XGBoost had the highest recall (87.5%), which matters in a medical setting, where missing a true case is costly. The single Decision Tree overfit and did worst.

The SHAP and permutation-importance plots in the notebook show which body-composition and lab features drive each model's predictions.

## Limitations

- With only 319 patients, the test set has just 64 people, so one prediction changes accuracy by about 1.6 percentage points. Repeated cross-validation would give more stable estimates.
- Tree models reached about 100% training accuracy, which is a sign of overfitting. Stronger regularisation could help.
- This is a learning project, not a clinical tool.

## Tools

Python, Pandas, scikit-learn, XGBoost, SHAP, Matplotlib, Seaborn, Google Colab

## How to run

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn
```

Open `gallstone_prediction.ipynb` in Colab or Jupyter and run all cells.
