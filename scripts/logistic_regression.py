# logistic_regression.py
# Mark Antepenko

# Import libraries
import joblib
from pathlib import Path
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import sys

def main():
    # Setting the project root
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    PROCESSED_DIR = DATA_DIR / "processed_data"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Verify data files exist before proceeding
    train_path = PROCESSED_DIR / "train_data_preprocessed.csv"
    test_path = PROCESSED_DIR / "test_data_preprocessed.csv"

    required_files = [train_path, test_path]
    for path in required_files:
        if not path.exists():
            print(f"[ERROR] Missing file: {path}", file=sys.stderr)
            return

    # Loading data
    print("Loading training data ...")  
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    if "review" not in train_data.columns or "sentiment" not in train_data.columns:
        print("[ERROR] train_data_preprocessed.csv must contain 'review' and 'sentiment' columns.", file=sys.stderr)
        return

    if "review" not in test_data.columns or "sentiment" not in test_data.columns:
        print("[ERROR] test_data_preprocessed.csv must contain 'review' and 'sentiment' columns.", file=sys.stderr)
        return

    # Extracting features and labels
    X_train = train_data['review']
    y_train = train_data['sentiment']
    X_test = test_data['review']
    y_test = test_data['sentiment']

    if len(y_test) != X_test.shape[0]:
        print(f"[ERROR] Row mismatch: X_test has {X_test.shape[0]} rows but y_test has {len(y_test)} labels.", file=sys.stderr)
        return
    
     # Train model
    print("Training Logistic Regression model ...")
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000)),
        ("smote", SMOTE(random_state=42)),
        ("clf", LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42))
    ])
    
    pipe.fit(X_train, y_train)

    # Evaluate on test
    y_pred = pipe.predict(X_test)

    print("[LR-Orig] Test accuracy:", accuracy_score(y_test, y_pred))
    print("[LR-Orig] Classification report:")
    print(classification_report(y_test, y_pred))

    # Save the fitted vectorizer + model
    print("Saving Logistic Regression Pipeline ...")
    bundle_path = MODELS_DIR / "logistic_model_baseline.joblib"
    joblib.dump(pipe, bundle_path)
    print(f"[LR] Saved bundle -> {bundle_path}")

if __name__ == "__main__":
    main()