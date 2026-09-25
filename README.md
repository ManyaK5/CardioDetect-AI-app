# 🫀 CardioDetect AI — Multi-Model Cardiovascular Health Assessment Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**CardioDetect AI** is an interactive, clinical decision-support web application designed to evaluate 10-year cardiovascular disease (CVD) risk. Trained and cross-validated on **70,000 real-world patient records**, the platform allows clinicians and users to toggle between five distinct machine learning architectures, review real-time model telemetry, inspect automated vital classifications (ACC/AHA BP & WHO BMI), and explore Explainable AI (XAI) risk-factor contributions.

---

## 🌟 Key Features

* **Multi-Model Inference Engine**: Switch seamlessly between five trained and cross-validated classifiers:
  * **Gradient Boosting** (High-sensitivity primary screening)
  * **Random Forest (Bagging)** (Outlier-resilient ensemble)
  * **Decision Tree** (Transparent flowchart-style rules)
  * **AdaBoost (Boosting)** (High-precision confirmation)
  * **Logistic Regression** (Calibrated linear odds-ratio baseline)
* **Real-Time Dynamic Telemetry**: Active model badge displaying 5-fold CV accuracy, test accuracy, precision, and sensitivity.
* **Auto-Reset & Smooth Navigation**: Form input fields reset automatically when switching models, clearing outdated results and smoothly auto-scrolling to newly computed outcomes.
* **Evidence-Based Clinical Classifications**:
  * **ACC/AHA 2017 Blood Pressure Staging**: Normal, Elevated, Stage 1, Stage 2, and Hypertensive Crisis.
  * **WHO Body Mass Index (BMI) Categorization**: Underweight, Normal weight, Overweight, and Obesity.
* **Explainable AI (XAI) Waterfall Breakdown**: Visualizes patient-specific risk elevators (red) versus protective/baseline factors (green/slate).
* **Comprehensive Benchmarking & Decision Matrix**: Compares models across Accuracy, Precision, Recall, and F1-Score, alongside a clinical selection guide.
* **Population Analytics (70,000 Patients)**: Interactive epidemiological distributions examining age, systolic blood pressure, and cholesterol correlations.

---

## 🏗️ System Architecture & Workflow

```
Patient Inputs (Demographics, Vitals, Labs, Lifestyle)
                       │
                       ▼
       Pre-processing & Standard Scaling
                       │
                       ▼
        Active Clinical Model Engine
    (Gradient Boosting / Random Forest / ...)
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼
Cardiovascular Risk %          Explainable AI (XAI)
  & Severity Gauge              Risk Contributions
       │                               │
       └───────────────┬───────────────┘
                       ▼
    Clinical Actions & Tailored Recommendations
```

---

## 📊 Model Performance Summary (Kaggle 70k Cohort)

| Model Architecture | 5-Fold CV Mean Acc | Test Accuracy | Precision | Recall (Sensitivity) | F1-Score |
|---|---|---|---|---|---|
| **Gradient Boosting** | **73.36%** | **73.69%** | 75.64% | **69.03%** | **72.18%** |
| **Random Forest (Bagging)** | 73.26% | 73.37% | 76.12% | 67.25% | 71.41% |
| **Decision Tree** | 72.88% | 73.30% | 75.78% | 67.60% | 71.46% |
| **Logistic Regression** | 72.76% | 72.61% | 75.30% | 66.36% | 70.55% |
| **AdaBoost (Boosting)** | 72.51% | 72.56% | **76.44%** | 64.32% | 69.86% |

---

## 🛠️ Technology Stack

* **Frontend & Framework**: Streamlit, HTML5/CSS3 (Custom Medical Tech Design System)
* **Visualizations**: Plotly Express & Plotly Graph Objects
* **Machine Learning**: Scikit-Learn, Joblib, NumPy, Pandas
* **Deployment**: Streamlit Community Cloud / Hugging Face Spaces

---

## 🚀 Local Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/cardio-assessment.git
   cd cardio-assessment
   ```

2. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

---

## ⚠️ Medical Disclaimer

*CardioDetect AI is designed for educational and decision-support purposes only. Risk probabilities are statistical estimates derived from population data and must not be used as an independent clinical diagnosis. Always consult a licensed cardiologist or physician for clinical evaluation.*
