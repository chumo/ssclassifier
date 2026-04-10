import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

def generate_synthetic_data(num_samples=1000):
    """
    Generates synthetic 80-feature arrays mimicking digits.
    0: high activation in outer loop, low in middle
    1: high activation in right-side segments
    ... etc.
    This is extremely basic and used only for bootstrapping the pipeline.
    """
    np.random.seed(42)
    X = []
    y = []
    
    # We'll just create a few random patterns that are roughly stable per digit 0-9
    base_patterns = {}
    for digit in range(10):
        # A unique random pattern of 80 floats [0, 1] for each digit
        base_patterns[digit] = np.random.rand(80)
        
    for _ in range(num_samples):
        digit = np.random.randint(0, 10)
        # Add some noise
        pattern = base_patterns[digit] + np.random.normal(0, 0.1, 80)
        # Clip and normalize
        pattern = np.clip(pattern, 0, 1)
        X.append(pattern)
        y.append(str(digit))
        
    return np.array(X), np.array(y)

def train_bootstrap_model():
    print("Generating synthetic data...")
    X, y = generate_synthetic_data(num_samples=5000)
    
    print("Training Random Forest...")
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X, y)
    
    os.makedirs("models", exist_ok=True)
    model_path = "models/rf_model.joblib"
    print(f"Saving model to {model_path}...")
    joblib.dump(clf, model_path)
    print("Done. Ready for end-to-end inference testing.")

if __name__ == "__main__":
    train_bootstrap_model()
