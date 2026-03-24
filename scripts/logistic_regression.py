# logistic_regression.py
# Mark Antepenko

# Import libraries
import joblib
import pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import sys

def main():
    # Paths
    DATA_DIR = Path(__file__).resolve().parents[1] / "data"
    MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Verify data files exist before proceeding
    x_train_path = DATA_DIR / "processed_data/X_train_balanced.pickle"
    y_train_path = DATA_DIR / "processed_data/y_train_balanced.pickle"
    x_test_path = DATA_DIR / "processed_data/X_test.pickle"
    test_csv_path = DATA_DIR / "processed_data/test_data_preprocessed.csv"
    vectorizer_path = MODELS_DIR / "vectorizer.pickle"

    required_files = [x_train_path, y_train_path, x_test_path, test_csv_path, vectorizer_path]
    for path in required_files:
        if not path.exists():
            print(f"[ERROR] Missing file: {path}", file=sys.stderr)
            return

    # Load balanced training data
    print("Loading training data ...")  
    with open(x_train_path, 'rb') as file:
        X_train = pickle.load(file)
    with open(y_train_path, 'rb') as file:
        y_train = pickle.load(file)

    # Load fitted vectorizer
    print("Loading fitted vectorizer ...")
    with open(vectorizer_path, 'rb') as file:
        vectorizer = pickle.load(file)

    # Load test data
    print("Loading test data ...")
    with open(x_test_path, 'rb') as file:
        X_test = pickle.load(file)
    test_data = pd.read_csv(test_csv_path)

    if "sentiment" not in test_data.columns:
        print("[ERROR] 'sentiment' column not found in test_data_preprocessed.csv", file=sys.stderr)
        return
    y_test = test_data['sentiment']

    # Sanity check: ensure X_train and X_test have the same number of features.
    # Exit early if mismatch to prevent model errors.
    if not hasattr(X_train, "shape") or not hasattr(X_test, "shape"):
        print("[ERROR] X_train or X_test does not have a valid shape attribute.", file=sys.stderr)
        return
    
    if X_train.shape[1] != X_test.shape[1]:
        print(f"[ERROR] Feature mismatch: X_train has {X_train.shape[1]} features, X_test has {X_test.shape[1]} features.", file=sys.stderr)
        return

    # Train model
    print("Training Logistic Regression model ...")
    model = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
    model.fit(X_train, y_train)

    if len(y_test) != X_test.shape[0]:
        print(f"[ERROR] Row mismatch: X_test has {X_test.shape[0]} rows but y_test has {len(y_test)} labels.", file=sys.stderr)
        return
    
    # Evaluate on test
    y_pred = model.predict(X_test)

    print("[LR-Orig] Test accuracy:", accuracy_score(y_test, y_pred))
    print("[LR-Orig] Classification report:")
    print(classification_report(y_test, y_pred))

    assert len(y_test) == getattr(X_test, "shape", [len(y_test)])[0], "[INVARIANT] y_test length must match X_test rows"

    # Save the fitted vectorizer + model
    print("Saving Logistic Regression model and vectorizer bundle ...")
    bundle_path = MODELS_DIR / "logistic_regression_bundle.joblib"
    joblib.dump((model, vectorizer), bundle_path)
    print(f"[LR] Saved bundle -> {bundle_path}")

if __name__ == "__main__":
    main()