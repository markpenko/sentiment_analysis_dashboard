# log_hyperparameter.py
# Mark Antepenko

# Import Libraries
import json
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import sys

from sklearn.pipeline import Pipeline

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = DATA_DIR / "processed_data"

# Verify data files exist before proceeding
train_path = PROCESSED_DIR / "train_data_preprocessed.csv"
test_path = PROCESSED_DIR / "test_data_preprocessed.csv"

required_files = [train_path, test_path]
for path in required_files:
    if not path.exists():
        print(f"[ERROR] Missing file: {path}", file=sys.stderr)
        sys.exit(1)

# Loading data
print("Loading training data ...")  
train_data = pd.read_csv(train_path)
print("Loading test data ...")
test_data = pd.read_csv(test_path)

if "review" not in train_data.columns or "sentiment" not in train_data.columns:
    print("[ERROR] train_data_preprocessed.csv must contain 'review' and 'sentiment' columns.", file=sys.stderr)
    sys.exit(1)

if "review" not in test_data.columns or "sentiment" not in test_data.columns:
    print("[ERROR] test_data_preprocessed.csv must contain 'review' and 'sentiment' columns.", file=sys.stderr)
    sys.exit(1)

# Extracting features and labels
X_train = train_data['review']
y_train = train_data['sentiment']
X_test = test_data['review']
y_test = test_data['sentiment']

if len(y_test) != X_test.shape[0]:
    print(f"[ERROR] Row mismatch: X_test has {X_test.shape[0]} rows but y_test has {len(y_test)} labels.", file=sys.stderr)
    sys.exit(1)

# Pipeline
print("Tuning TF-IDF + Logistic Regression pipeline ...")
pipe = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])

param_grid = {
    "tfidf__max_features": [2000, 5000, 10000],
    "tfidf__ngram_range": [(1,1), (1,2)],
    "tfidf__min_df": [1, 2, 5],
    "clf__C": [0.1, 1, 10],
    "clf__solver": ["lbfgs", "liblinear", "saga"],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=pipe, 
    param_grid=param_grid, 
    cv=cv, 
    scoring="accuracy", 
    n_jobs=-1, 
    verbose=1
)

grid.fit(X_train, y_train)

best_pipeline = grid.best_estimator_
y_pred = best_pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n[LR] Test accuracy:", acc)
print("[LR] Classification report:\n", classification_report(y_test, y_pred, digits=4))

# Saving full pipeline
print("Saving tuned pipeline ...")
bundle_path = MODELS_DIR / "logreg_model_tuned.joblib"
joblib.dump(best_pipeline, bundle_path)

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