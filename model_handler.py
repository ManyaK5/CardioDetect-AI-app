import os
import joblib
import pandas as pd
import numpy as np

class CardioModelHandler:
    """
    Directly interfaces with trained clinical models, the fitted scaler,
    and cross-validation metrics saved to disk from ML_project.ipynb.
    """
    
    FEATURE_COLS = [
        'age', 'ap_hi', 'ap_lo', 'height', 'weight', 'BMI',
        'gluc', 'cholesterol', 'smoke', 'active', 'gender', 'alco'
    ]
    
    SCALED_COLS = ['age', 'ap_hi', 'ap_lo', 'height', 'weight', 'BMI']
    
    MODEL_FILES = {
        'Gradient Boosting': ['cardio_best_model.pkl', 'cardio_tuned_gradient_boosting_model.pkl', 'cardio_gradient_boosting_model.pkl'],
        'Random Forest (Bagging)': ['cardio_random_forest_model.pkl'],
        'Decision Tree': ['cardio_decision_tree_model.pkl'],
        'AdaBoost (Boosting)': ['cardio_adaboost_model.pkl'],
        'Logistic Regression': ['cardio_logistic_model.pkl']
    }
    
    def __init__(self, base_dir=None):
        self.models = {}
        self.metrics_summary = None
        self.cv_folds = None
        self.scaler = None
        
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(curr_dir)
        search_dirs = [curr_dir, parent_dir]
        if base_dir:
            search_dirs.insert(0, base_dir)
            
        # 1. Load Scaler
        scaler_loaded = False
        for d in search_dirs:
            p = os.path.join(d, "cardio_scaler.pkl")
            if os.path.exists(p):
                try:
                    self.scaler = joblib.load(p)
                    scaler_loaded = True
                    break
                except Exception:
                    pass
        if not scaler_loaded:
            raise FileNotFoundError("Could not locate 'cardio_scaler.pkl' on disk.")
            
        # 2. Load Models
        for display_name, filenames in self.MODEL_FILES.items():
            for fname in filenames:
                for d in search_dirs:
                    fpath = os.path.join(d, fname)
                    if os.path.exists(fpath):
                        try:
                            self.models[display_name] = joblib.load(fpath)
                            break
                        except Exception:
                            pass
                if display_name in self.models:
                    break
                    
        if not self.models:
            raise FileNotFoundError("No trained .pkl models could be loaded from disk.")
            
        # 3. Load Metrics Summary CSV
        for d in search_dirs:
            p = os.path.join(d, "cardio_models_summary.csv")
            if os.path.exists(p):
                try:
                    df_sum = pd.read_csv(p)
                    self.metrics_summary = df_sum
                    break
                except Exception:
                    pass
                    
        # 4. Load CV Fold-by-Fold Metrics CSV
        for d in search_dirs:
            p = os.path.join(d, "cardio_cv_fold_metrics.csv")
            if os.path.exists(p):
                try:
                    df_cv = pd.read_csv(p)
                    self.cv_folds = df_cv
                    break
                except Exception:
                    pass

    def get_available_models(self):
        """Returns ordered list of available models."""
        preferred_order = [
            'Gradient Boosting',
            'Random Forest (Bagging)',
            'Decision Tree',
            'AdaBoost (Boosting)',
            'Logistic Regression'
        ]
        return [m for m in preferred_order if m in self.models]

    def get_model_summary_df(self):
        """Returns unified summary DataFrame with single Gradient Boosting entry."""
        if self.metrics_summary is not None:
            df = self.metrics_summary.copy()
            # Normalize names to single 'Gradient Boosting'
            df['Model'] = df['Model'].replace({
                'Tuned Gradient Boosting': 'Gradient Boosting',
                'Gradient Boosting (Boosting)': 'Gradient Boosting'
            })
            df = df.drop_duplicates(subset=['Model'], keep='last').reset_index(drop=True)
            return df
            
        # Fallback if summary CSV unavailable
        rows = [
            {'Model': 'Gradient Boosting', 'Test Acc (%)': 73.69, 'Test Precision (%)': 75.64, 'Test Recall (%)': 69.03, 'Test F1-score (%)': 72.18, 'CV Acc Mean (%)': 73.36, 'Fit Status': 'Good fit'},
            {'Model': 'Random Forest (Bagging)', 'Test Acc (%)': 73.37, 'Test Precision (%)': 76.12, 'Test Recall (%)': 67.25, 'Test F1-score (%)': 71.41, 'CV Acc Mean (%)': 73.26, 'Fit Status': 'Good fit'},
            {'Model': 'Decision Tree', 'Test Acc (%)': 73.30, 'Test Precision (%)': 75.78, 'Test Recall (%)': 67.60, 'Test F1-score (%)': 71.46, 'CV Acc Mean (%)': 72.88, 'Fit Status': 'Good fit'},
            {'Model': 'Logistic Regression', 'Test Acc (%)': 72.61, 'Test Precision (%)': 75.30, 'Test Recall (%)': 66.36, 'Test F1-score (%)': 70.55, 'CV Acc Mean (%)': 72.76, 'Fit Status': 'Good fit'},
            {'Model': 'AdaBoost (Boosting)', 'Test Acc (%)': 72.56, 'Test Precision (%)': 76.44, 'Test Recall (%)': 64.32, 'Test F1-score (%)': 69.86, 'CV Acc Mean (%)': 72.51, 'Fit Status': 'Good fit'}
        ]
        return pd.DataFrame(rows)

    def get_model_metrics(self, model_name):
        """Returns specific metrics dictionary for selected model."""
        summary = self.get_model_summary_df()
        first_token = model_name.split()[0]
        row = summary[summary['Model'].str.contains(first_token, case=False, na=False)]
        if not row.empty:
            r = row.iloc[0]
            return {
                'test_acc': r.get('Test Acc (%)', 73.0),
                'precision': r.get('Test Precision (%)', 75.0),
                'recall': r.get('Test Recall (%)', 68.0),
                'f1': r.get('Test F1-score (%)', 71.5),
                'cv_mean': r.get('CV Acc Mean (%)', r.get('5-Fold CV Mean (%)', 73.3)),
                'cv_std': r.get('CV Acc Std (%)', r.get('CV Spread (std %)', 0.18))
            }
        return {'test_acc': 73.0, 'precision': 75.0, 'recall': 68.0, 'f1': 71.5, 'cv_mean': 73.3, 'cv_std': 0.18}

    def calculate_bmi(self, weight_kg, height_cm):
        """Calculates BMI with safety boundary checks."""
        height_m = height_cm / 100.0
        if height_m <= 0:
            return 0.0
        return round(weight_kg / (height_m ** 2), 2)

    def classify_bp(self, ap_hi, ap_lo):
        """Classify blood pressure into ACC/AHA categories."""
        if ap_hi < 120 and ap_lo < 80:
            return "Normal", "success", "Optimal BP (Systolic < 120 & Diastolic < 80 mmHg)"
        elif 120 <= ap_hi <= 129 and ap_lo < 80:
            return "Elevated", "info", "Elevated BP (Systolic 120-129 & Diastolic < 80 mmHg)"
        elif (130 <= ap_hi <= 139) or (80 <= ap_lo <= 89):
            return "Stage 1 Hypertension", "warning", "Stage 1 (Systolic 130-139 or Diastolic 80-89 mmHg)"
        elif (140 <= ap_hi <= 180) or (90 <= ap_lo <= 120):
            return "Stage 2 Hypertension", "error", "Stage 2 (Systolic 140+ or Diastolic 90+ mmHg)"
        else:
            return "Hypertensive Crisis", "error", "Hypertensive Crisis (Systolic > 180 or Diastolic > 120 mmHg)"

    def classify_bmi(self, bmi):
        """Classify BMI according to WHO standards."""
        if bmi < 18.5:
            return "Underweight", "info"
        elif 18.5 <= bmi < 25.0:
            return "Normal weight", "success"
        elif 25.0 <= bmi < 30.0:
            return "Overweight", "warning"
        else:
            return "Obesity", "error"

    def predict(self, input_dict, model_name='Gradient Boosting'):
        """
        Runs assessment on patient input using the chosen saved model,
        applying the saved StandardScaler on numerical features.
        """
        if model_name not in self.models:
            model_name = self.get_available_models()[0]
            
        model = self.models[model_name]
        
        bmi = input_dict.get('BMI')
        if bmi is None:
            bmi = self.calculate_bmi(input_dict['weight'], input_dict['height'])
            
        patient_data = {
            'age': [input_dict['age']],
            'ap_hi': [input_dict['ap_hi']],
            'ap_lo': [input_dict['ap_lo']],
            'height': [input_dict['height']],
            'weight': [input_dict['weight']],
            'BMI': [bmi],
            'gluc': [input_dict['gluc']],
            'cholesterol': [input_dict['cholesterol']],
            'smoke': [input_dict['smoke']],
            'active': [input_dict['active']],
            'gender': [input_dict['gender']],
            'alco': [input_dict['alco']]
        }
        
        df_raw = pd.DataFrame(patient_data)[self.FEATURE_COLS]
        
        # Scale numerical columns
        df_scaled = df_raw.copy()
        df_scaled[self.SCALED_COLS] = self.scaler.transform(df_raw[self.SCALED_COLS])
        
        # Predict probability
        if hasattr(model, 'predict_proba'):
            prob = float(model.predict_proba(df_scaled)[0][1])
        elif hasattr(model, 'decision_function'):
            dfunc = model.decision_function(df_scaled)[0]
            prob = float(1 / (1 + np.exp(-dfunc)))
        else:
            prob = float(model.predict(df_scaled)[0])
            
        prediction = int(model.predict(df_scaled)[0])
        risk_percentage = round(prob * 100, 1)
        
        # Feature impact calculation
        feature_contributions = {}
        if hasattr(model, 'feature_importances_'):
            importances = dict(zip(self.FEATURE_COLS, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            importances = dict(zip(self.FEATURE_COLS, np.abs(model.coef_[0])))
        else:
            importances = {col: 1.0 / len(self.FEATURE_COLS) for col in self.FEATURE_COLS}
            
        healthy_baselines = {
            'age': 0.0,
            'ap_hi': -0.7,
            'ap_lo': -0.6,
            'height': 0.0,
            'weight': -0.6,
            'BMI': -0.8,
            'gluc': 1.0,
            'cholesterol': 1.0,
            'smoke': 0.0,
            'active': 1.0,
            'gender': 1.0,
            'alco': 0.0
        }
        
        for col in self.FEATURE_COLS:
            val = float(df_scaled[col].iloc[0]) if col in self.SCALED_COLS else float(df_raw[col].iloc[0])
            base = healthy_baselines.get(col, 0.0)
            imp = importances.get(col, 0.05)
            
            if col == 'active':
                diff = (1.0 - val)
            else:
                diff = val - base
                
            impact = round(float(diff * imp * 8.0), 4)
            feature_contributions[col] = impact

        bp_category, bp_severity, bp_desc = self.classify_bp(input_dict['ap_hi'], input_dict['ap_lo'])
        bmi_category, bmi_severity = self.classify_bmi(bmi)
        
        return {
            'risk_prediction': prediction,
            'risk_probability': prob,
            'risk_percentage': risk_percentage,
            'model_used': model_name,
            'model_metrics': self.get_model_metrics(model_name),
            'bmi': bmi,
            'bmi_category': bmi_category,
            'bmi_severity': bmi_severity,
            'bp_category': bp_category,
            'bp_severity': bp_severity,
            'bp_desc': bp_desc,
            'feature_contributions': feature_contributions
        }
