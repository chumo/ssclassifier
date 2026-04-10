import os
import joblib
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = "models/rf_model.joblib"
_model = None

def load_model():
    global _model
    if os.path.exists(MODEL_PATH):
        _model = joblib.load(MODEL_PATH)
    else:
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run scripts/train.py first.")

def predict(features) -> str:
    """
    Predicts a character given exactly 80 feature inputs (array of shape (1, 80)).
    features: list or numpy array of 80 elements.
    """
    global _model
    if _model is None:
        load_model()
    # Scikit-learn expects 2D array, we must ensure it is [features]
    prediction = _model.predict([features])
    return str(prediction[0])
