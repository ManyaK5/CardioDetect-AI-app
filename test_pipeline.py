import os
import sys

# Ensure backend module can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_handler import CardioModelHandler

def test_model_pipeline():
    handler = CardioModelHandler()
    print("[OK] Successfully loaded model and scaler.")
    
    # Test case 1: Healthy sample profile
    healthy_profile = {
        'age': 45,
        'gender': 1, # Female
        'height': 165,
        'weight': 60,
        'ap_hi': 115,
        'ap_lo': 75,
        'cholesterol': 1, # Normal
        'gluc': 1,        # Normal
        'smoke': 0,
        'alco': 0,
        'active': 1
    }
    
    res1 = handler.predict(healthy_profile)
    print("Healthy Profile Test Result:", res1['risk_percentage'], "% | Prediction:", res1['risk_prediction'])
    assert 0.0 <= res1['risk_percentage'] <= 100.0, "Percentage must be between 0 and 100"
    
    # Test case 2: High risk sample profile
    high_risk_profile = {
        'age': 62,
        'gender': 2, # Male
        'height': 170,
        'weight': 95,
        'ap_hi': 160,
        'ap_lo': 100,
        'cholesterol': 3, # Well Above Normal
        'gluc': 2,        # Above Normal
        'smoke': 1,
        'alco': 1,
        'active': 0
    }
    
    res2 = handler.predict(high_risk_profile)
    print("High Risk Profile Test Result:", res2['risk_percentage'], "% | Prediction:", res2['risk_prediction'])
    assert res2['risk_percentage'] > res1['risk_percentage'], "High risk profile should have higher risk percentage"
    print("[OK] Pipeline test passed successfully!")

if __name__ == '__main__':
    test_model_pipeline()
