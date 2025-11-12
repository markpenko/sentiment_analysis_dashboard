# Chapter 13 log_hyperparameter.py
# Mark Antepenko

# Import Libraries
import json
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Setting the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Load data
print("Loading data ...")
train_csv = DATA_DIR / "processed_data" / "train_data_preprocessed.csv"
test_csv = DATA_DIR / "processed_data" / "test_data_preprocessed.csv"
df_train = pd.read_csv(train_csv)
df_test = pd.read_csv(test_csv)
X_train = df_train["review"].astype(str)
y_train = df_train["sentiment"]
X_test = df_test["review"].astype(str)
y_test = df_test["sentiment"]

# Run tuning and save best pipeline
print("Tuning Logistic Regression...")
pipe = Pipeline(
    steps=[
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ]
)
param_grid = {
    "tfidf__max_features": [20000, 50000],
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [1, 2, 5],
    "clf__C": [0.1, 1, 10],
    "clf__solver": ["lbfgs", "liblinear", "saga"],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(pipe, param_grid=param_grid, cv=cv, scoring="accuracy", n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

best = grid.best_estimator_
y_pred = best.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("\n[LR] Test accuracy:", acc)
print("[LR] Classification report:\n", classification_report(y_test, y_pred, digits=4))

# Saving model and vectorizer as a bundle
print("Saving best model and vectorizer as a bundle...")
vectorizer = best.named_steps["tfidf"]
model = best.named_steps["clf"]
bundle_path = MODELS_DIR / "tuned_logistic_regression_bundle.joblib"
joblib.dump((model, vectorizer), bundle_path)

out_params = MODELS_DIR / "logreg_best_params.json"
with open(out_params, "w") as f:
    json.dump({"best_params": grid.best_params_, "test_accuracy": float(acc)}, f, indent=2)

# Display results
print(f"[LR] Saved bundle -> {bundle_path}")
print(f"[LR] Saved params -> {out_params}")
#print(f"[LR] Saved tokenizer -> (no value)")
print(f"[LR] Best val score -> {grid.best_score_:.4f}")
print(f"[LR] Best params -> {grid.best_params_}")
print(f"[LR] Test accuracy -> {acc:.4f}")