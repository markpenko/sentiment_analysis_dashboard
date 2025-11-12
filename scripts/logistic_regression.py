# Chapter 13 logistic_regression.py
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

    # Load balanced training data
    print("Loading training data ...")  
    with open(DATA_DIR / 'processed_data/X_train_balanced.pickle', 'rb') as file:
        X_train = pickle.load(file)
    with open(DATA_DIR / 'processed_data/y_train_balanced.pickle', 'rb') as file:
        y_train = pickle.load(file)

    # Load fitted vectorizer
    print("Loading fitted vectorizer ...")
    with open(MODELS_DIR / 'vectorizer.pickle', 'rb') as file:
        vectorizer = pickle.load(file)

    # Load test data
    print("Loading test data ...")
    with open(DATA_DIR / 'processed_data/X_test.pickle', 'rb') as file:
        X_test = pickle.load(file)
    test_data = pd.read_csv(DATA_DIR / 'processed_data/test_data_preprocessed.csv')
    y_test = test_data['sentiment']

    print(f"X_train type: {type(X_train)}")
    print(f"X_train shape: {getattr(X_train, 'shape', 'No shape')}")
    try:
        # Avoid printing a huge sparse structure
        if hasattr(X_train, "toarray"):
            preview = X_train[:2].toarray()
        else:
            preview = X_train[:2]
        print(f"X_train preview (first 2 rows): {preview}")
    except Exception as e:
        print(f"Preview skipped: {e}")

    # Sanity checks
    if hasattr(X_train, "shape") and hasattr(X_test, "shape"):
        if X_train.shape[1] != X_test.shape[1]:
            print(f"[ERROR] Feature mismatch: X_train has {X_train.shape[1]} features, X_test has {X_test.shape[1]} features.", file=sys.stderr)
            return

    # Train model
    print("Training Logistic Regression model ...")
    X_train_vec = X_train

    model = LogisticRegression(max_iter=1000, solver="liblinear", random_state=42, n_jobs=-1)
    model.fit(X_train_vec, y_train)

    # Evaluate on test
    X_test_vec = X_test
    y_pred = model.predict(X_test_vec)

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