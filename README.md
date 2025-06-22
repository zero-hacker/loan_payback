# 🧠 Credit Default Prediction API (Classical ML - XGBoost + Flask)

This project demonstrates a classical machine learning pipeline to predict whether a loan applicant is likely to default, using the German Credit dataset. It includes preprocessing, model training, evaluation, export, and API deployment via Flask.

---

## 📁 Files

| File | Description |
|------|-------------|
| `GermanCredit.csv` | Raw dataset used for training |
| `loan_payback.ipynb` | Full notebook with preprocessing, training, and evaluation |
| `credit_default_pipeline.pkl` | Exported XGBoost + preprocessing pipeline |
| `app.py` | Flask API to serve predictions |
| `README.md` | This file |

---

## 📊 Dataset: German Credit Data

- Includes features like `age`, `credit_history`, `purpose`, `employment_duration`, and more
- Target variable: `credit_risk` (0 = pay back, 1 = default)

---

## 🔧 Model Pipeline

1. Loaded and cleaned dataset
2. Used `OneHotEncoder` for categorical features
3. Handled **class imbalance** with `scale_pos_weight`
4. Trained an `XGBoostClassifier` using `sklearn.pipeline`
5. Evaluated with:
   - Accuracy score
   - Confusion matrix
   - Cross-validation
6. Exported full pipeline as `.pkl`

---

## 🖥️ Running the Flask API

### ▶️ Step 1: Install dependencies

```bash
pip install flask pandas joblib scikit-learn xgboost