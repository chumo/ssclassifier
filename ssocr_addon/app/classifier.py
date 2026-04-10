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

def train_model(X, y):
    """
    Trains a new RandomForestClassifier on the provided features and labels.
    X: list of feature arrays (each length 80)
    y: list of string labels
    """
    global _model
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X, y)
    
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    
    # Reload the model into memory
    _model = clf
    return len(y)
