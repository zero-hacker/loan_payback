## 🧠 Credit Default Prediction API (Classical ML - XGBoost + Flask)

This project showcases a complete end-to-end classical machine learning pipeline that predicts whether a loan applicant is likely to default, using the **German Credit dataset**. It covers data preprocessing, model training, evaluation, pipeline export, and deployment via a RESTful Flask API. The deployed model is accessible publicly through a secured endpoint.

---

### Files

| File                          | Description                                                |
| ----------------------------- | ---------------------------------------------------------- |
| `GermanCredit.csv`            | Raw dataset used for training                              |
| `loan_payback.ipynb`          | Full notebook with preprocessing, training, and evaluation |
| `credit_default_pipeline.pkl` | Exported pipeline (preprocessing + XGBoost model)          |
| `app.py`                      | Flask API server code for model inference                  |
| `README.md`                   | This file                                                  |

---

### Dataset: German Credit Data

The dataset includes customer financial attributes with the goal of predicting credit default risk.

- **Size**: 1,000 records
- **Features**: 20 features including:
  - `age`
  - `credit_history`
  - `purpose`
  - `employment_duration`
  - `savings`
  - `housing`
- **Target Variable**: `credit_risk`
  - `0`: Will pay back
  - `1`: Likely to default

---

### Model Pipeline Design

#### 1. Preprocessing

- **Categorical Features**: Transformed via `OneHotEncoder`
- **Numerical Features**: Passed directly
- **Pipeline**: Used `ColumnTransformer` to unify preprocessing and `Pipeline` to chain preprocessing + model

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
```

#### 2. Handling Class Imbalance

- Found \~70:30 imbalance in class distribution
- Used `scale_pos_weight` in XGBoost to counteract bias toward majority class

```python
model = XGBClassifier(scale_pos_weight=2.2)
```

#### 3. Training & Evaluation

- Evaluated using:
  - Accuracy
  - Confusion matrix
  - Cross-validation
- Used stratified split for balanced testing

---

### Exporting the Model

The pipeline (preprocessing + XGBoost model) was exported as a `.pkl` file:

```python
import joblib
joblib.dump(pipeline, "credit_default_pipeline.pkl")
```

This allows for reusability and deployment without retraining.

---

### Running the Flask API

#### Step 1: Install dependencies

```bash
pip install flask pandas joblib scikit-learn xgboost
```

#### Step 2: Run the server

```bash
python app.py
```

---

### Public API Usage Guide

#### Endpoint

```
POST https://ai.kimjerry.com/predict
```

#### Content-Type

```
application/json
```

#### Sample single JSON payload

```json
{
  "status": "no_checking_account",
  "duration": 60,
  "credit_history": "critical_account_other_credits_existing",
  "purpose": "retraining",
  "amount": 10000,
  "savings": "unknown_no_savings_account",
  "employment_duration": "unemployed",
  "installment_rate": 6,
  "personal_status_sex": "male_single",
  "other_debtors": "guarantor",
  "present_residence": 1,
  "property": "unknown_no_property",
  "age": 25,
  "other_installment_plans": "bank",
  "housing": "rent",
  "number_credits": 5,
  "job": "unemployed_unskilled_non_resident",
  "people_liable": 6,
  "telephone": "yes",
  "foreign_worker": "yes"
}
```

#### Accepts multiple payloads in array

```json
[
  {
    "status": "no_checking_account",
    "duration": 60,
    ...
  },
  {
    "status": "critical_account",
    "duration": 24,
    ...
  }
]
```

#### Sample response

```json
[
  {
    "confidence": "83.44%",
    "outcome": "Likely to default (not pay back)"
  },
  {
    "confidence": "67.29%",
    "outcome": "Likely to pay back"
  }
]
```

---

Jerry Kim

- Repo: [https://github.com/zero-hacker/loan_payback](https://github.com/zero-hacker/loan_payback)
- Hosted at: [https://ai.kimjerry.com/predict](https://ai.kimjerry.com/predict)
