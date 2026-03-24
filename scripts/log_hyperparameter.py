# log_hyperparameter.py
# Mark Antepenko

# Import Libraries
import json
import pickle
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import sys

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
        sys.exit(1)

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
    sys.exit(1)

y_test = test_data['sentiment']

# Sanity check: ensure X_train and X_test have the same number of features.
# Exit early if mismatch to prevent model errors.
if not hasattr(X_train, "shape") or not hasattr(X_test, "shape"):
    print("[ERROR] X_train or X_test does not have a valid shape attribute.", file=sys.stderr)
    sys.exit(1)

if X_train.shape[1] != X_test.shape[1]:
    print(f"[ERROR] Feature mismatch: X_train has {X_train.shape[1]} features, X_test has {X_test.shape[1]} features.", file=sys.stderr)
    sys.exit(1)

# Tune Logistic Regression classifier only
print("Tuning Logistic Regression classifier ...")
model = LogisticRegression(max_iter=1000, random_state=42)

param_grid = {
    "C": [0.1, 1, 10],
    "solver": ["lbfgs", "liblinear", "saga"],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=model, 
    param_grid=param_grid, 
    cv=cv, 
    scoring="accuracy", 
    n_jobs=-1, 
    verbose=1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("\n[LR] Test accuracy:", acc)
print("[LR] Classification report:\n", classification_report(y_test, y_pred, digits=4))

# Saving model and vectorizer as a bundle
print("Saving tuned model and vectorizer as a bundle...")
bundle_path = MODELS_DIR / "tuned_logistic_regression_bundle.joblib"
joblib.dump((best_model, vectorizer), bundle_path)

out_params = MODELS_DIR / "logreg_best_params.json"
with open(out_params, "w") as f:
    json.dump(
        {
            "best_params": grid.best_params_,
            "best_cv_score": float(grid.best_score_),
            "test_accuracy": float(acc)
        },
        f,
        indent=2
    )

# Display results
print(f"[LR] Saved bundle -> {bundle_path}")
print(f"[LR] Saved params -> {out_params}")
print(f"[LR] Best val score -> {grid.best_score_:.4f}")
print(f"[LR] Best params -> {grid.best_params_}")
print(f"[LR] Test accuracy -> {acc:.4f}")