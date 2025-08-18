# Chapter 13 evaluate_model.py
# Mark Antepenko

# Import libraries
import os
import joblib
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import requests
from joblib import load
from pathlib import Path
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Runtime safety toggles 
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Set USE_TF_GPU=1 in your shell to re-enable GPU later
# export USE_TF_GPU=1
USE_GPU = os.environ.get("USE_TF_GPU", "0") == "1"

# Force CPU MacOS Issue
if not USE_GPU:
    try:
        tf.config.set_visible_devices([], "GPU")
        print("Using CPU (Metal GPU disabled). Set USE_TF_GPU=1 to enable.")
    except Exception as e:
        print("GPU disable skipped:", e)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Load test set
test_csv = DATA_DIR / "processed_data" / "test_data_preprocessed.csv"
df_test = pd.read_csv(test_csv)
X_test = df_test["review"].astype(str)

# Map sentiment to 0/1: negative->0, positive->1
y_test = (df_test["sentiment"].astype(str).str.lower() == "positive").astype(int)

# Evaluate Original Logistic Regression
print("\n" + "="*40)
print("[LR-Orig] Evaluating Original Logistic Regression")
orig_logreg_path = MODELS_DIR / "logistic_regression_bundle.joblib"
vectorizer_orig, model_orig = load(orig_logreg_path)
X_test_vec_orig = vectorizer_orig.transform(X_test)
y_pred_orig = model_orig.predict(X_test_vec_orig)

# Handle both string labels ("positive"/"negative") and numeric labels (0/1)
if isinstance(y_pred_orig[0], str):
    y_pred_orig = (y_pred_orig == "positive").astype(int)
else:
    y_pred_orig = y_pred_orig.astype(int)
accuracy_orig = accuracy_score(y_test, y_pred_orig)
print(f"[LR-Orig] Test accuracy -> {accuracy_orig:.4f}")
print("[LR-Orig] Classification report:\n", classification_report(y_test, y_pred_orig, digits=4))
cm_orig = confusion_matrix(y_test, y_pred_orig, labels=[0, 1])
disp_orig = ConfusionMatrixDisplay(confusion_matrix=cm_orig, display_labels=["Negative", "Positive"])
disp_orig.plot(cmap=plt.cm.Blues)
plt.title("Original Logistic Regression Confusion Matrix")
plt.show()


# Evaluate Tuned Logistic Regression
print("\n" + "="*40)
print("[LR-Tuned] Evaluating Tuned Logistic Regression")
tuned_logreg_path = MODELS_DIR / "tuned_logistic_regression_bundle.joblib"
vectorizer_tuned, model_tuned = load(tuned_logreg_path)
X_test_vec_tuned = vectorizer_tuned.transform(X_test)
y_pred_tuned = model_tuned.predict(X_test_vec_tuned)
y_pred_tuned = (y_pred_tuned == "positive").astype(int)
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
print(f"[LR-Tuned] Test accuracy -> {accuracy_tuned:.4f}")
print("[LR-Tuned] Classification report:\n", classification_report(y_test, y_pred_tuned, digits=4))
cm_tuned = confusion_matrix(y_test, y_pred_tuned, labels=[0, 1])
disp_tuned = ConfusionMatrixDisplay(confusion_matrix=cm_tuned, display_labels=["Negative", "Positive"])
disp_tuned.plot(cmap=plt.cm.Blues)
plt.title("Tuned Logistic Regression Confusion Matrix")
plt.show()

# Evaluate Original LSTM
print("\n" + "="*40)
print("[LSTM-Orig] Evaluating Original LSTM")
lstm_model_orig = tf.keras.models.load_model(MODELS_DIR / "lstm_model.keras", compile=False)
lstm_tok_orig = joblib.load(MODELS_DIR / "tokenizer.joblib")
try:
    MAX_LEN_ORIG = int(lstm_model_orig.input_shape[1])
except Exception:
    MAX_LEN_ORIG = 200
X_seq_orig = lstm_tok_orig.texts_to_sequences(X_test.tolist())
X_pad_orig = pad_sequences(X_seq_orig, maxlen=MAX_LEN_ORIG, padding="post", truncating="post")
probs_orig = lstm_model_orig.predict(X_pad_orig, verbose=0)
if probs_orig.shape[-1] == 1:
    y_pred_lstm_orig = (probs_orig.ravel() >= 0.5).astype(int)
else:
    y_pred_lstm_orig = probs_orig.argmax(axis=-1)
acc_lstm_orig = accuracy_score(y_test, y_pred_lstm_orig)
print(f"[LSTM-Orig] Test accuracy -> {acc_lstm_orig:.4f}")
print("[LSTM-Orig] Classification report:\n", classification_report(y_test, y_pred_lstm_orig, digits=4))
cm_lstm_orig = confusion_matrix(y_test, y_pred_lstm_orig, labels=[0, 1])
disp_lstm_orig = ConfusionMatrixDisplay(confusion_matrix=cm_lstm_orig, display_labels=["Negative", "Positive"])
disp_lstm_orig.plot(cmap=plt.cm.Blues)
plt.title("Original LSTM Confusion Matrix")
plt.show()

# Evaluate Tuned LSTM
print("\n" + "="*40)
print("[LSTM-Tuned] Evaluating Tuned LSTM")
lstm_model_tuned = tf.keras.models.load_model(MODELS_DIR / "lstm_best.keras", compile=False)
lstm_tok_tuned = joblib.load(MODELS_DIR / "lstm_best_tokenizer.joblib")
try:
    MAX_LEN_TUNED = int(lstm_model_tuned.input_shape[1])
except Exception:
    MAX_LEN_TUNED = 200
X_seq_tuned = lstm_tok_tuned.texts_to_sequences(X_test.tolist())
X_pad_tuned = pad_sequences(X_seq_tuned, maxlen=MAX_LEN_TUNED, padding="post", truncating="post")
probs_tuned = lstm_model_tuned.predict(X_pad_tuned, verbose=0)
if probs_tuned.shape[-1] == 1:
    y_pred_lstm_tuned = (probs_tuned.ravel() >= 0.5).astype(int)
else:
    y_pred_lstm_tuned = probs_tuned.argmax(axis=-1)
acc_lstm_tuned = accuracy_score(y_test, y_pred_lstm_tuned)
print(f"[LSTM-Tuned] Test accuracy -> {acc_lstm_tuned:.4f}")
print("[LSTM-Tuned] Classification report:\n", classification_report(y_test, y_pred_lstm_tuned, digits=4))
cm_lstm_tuned = confusion_matrix(y_test, y_pred_lstm_tuned, labels=[0, 1])
disp_lstm_tuned = ConfusionMatrixDisplay(confusion_matrix=cm_lstm_tuned, display_labels=["Negative", "Positive"])
disp_lstm_tuned.plot(cmap=plt.cm.Blues)
plt.title("Tuned LSTM Confusion Matrix")
plt.show()

# Measure response times
def measure_response_time(endpoint, data):
    start_time = time.time()
    response = requests.post(endpoint, data=data)
    end_time = time.time()
    response_time = end_time - start_time
    return response_time

# Measure response time for sentiment analysis
data = {'text': 'This is a great product!', 'model_type': 'logistic_regression'}
response_time = measure_response_time('http://localhost:5000/analyze', data)
print(f'Sentiment Analysis Response Time: {response_time} seconds')