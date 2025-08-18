# Chapter 13 lstm_hyperparameter.py
# Mark Antepenko

# Import Libraries
import os
import json
import joblib
from pathlib import Path
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models  # pyright: ignore[reportMissingImports]
from tensorflow.keras.preprocessing.text import Tokenizer  # pyright: ignore[reportMissingImports]
from tensorflow.keras.preprocessing.sequence import pad_sequences  # pyright: ignore[reportMissingImports]

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

# Data load (expects processed_data CSVs with 'review' and 'sentiment')
train_csv = DATA_DIR / "processed_data" / "train_data_preprocessed.csv"
test_csv = DATA_DIR / "processed_data" / "test_data_preprocessed.csv"
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)
X_train = train_df["review"].astype(str)
X_test = test_df["review"].astype(str)

# map sentiment to {0,1}; handles string ("positive"/"negative") or numeric
if train_df["sentiment"].dtype == "O":
    y_train = (train_df["sentiment"].str.lower() == "positive").astype(int).to_numpy()
    y_test = (test_df["sentiment"].str.lower() == "positive").astype(int).to_numpy()
else:
    y_train = train_df["sentiment"].astype(int).to_numpy()
    y_test = test_df["sentiment"].astype(int).to_numpy()
        
# LSTM sweep (no functions, no main guard)
num_words = 20000
maxlen = 200

tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

def vec(xs):
    return pad_sequences(
        tokenizer.texts_to_sequences(xs),
        maxlen=maxlen,
        padding="post",
        truncating="post",
    )

Xtr = vec(X_train)
Xte = vec(X_test)

combos = [
    {"embedding_dim": 64, "lstm_units": 64, "dropout": 0.2, "lr": 1e-3, "batch_size": 64, "epochs": 3},
    {"embedding_dim": 128, "lstm_units": 64, "dropout": 0.3, "lr": 5e-4, "batch_size": 64, "epochs": 3},
]

best_val = -1.0
best_model = None
best_cfg = None

for cfg in combos:
    vocab_size = min(num_words, len(tokenizer.word_index) + 1)
    model = models.Sequential(
        [
            layers.Embedding(input_dim=vocab_size, output_dim=cfg["embedding_dim"], input_length=maxlen),
            layers.Bidirectional(layers.LSTM(cfg["lstm_units"])),
            layers.Dropout(cfg["dropout"]),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=cfg["lr"])
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    hist = model.fit(
        Xtr,
        y_train,
        validation_split=0.1,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        verbose=2,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=1, restore_best_weights=True)],
    )
    val_acc = float(max(hist.history.get("val_accuracy", [0.0])))
    if val_acc > best_val:
        best_val = val_acc
        best_model = model
        best_cfg = cfg

test_loss, test_acc = best_model.evaluate(Xte, y_test, verbose=0)
print("\n[LSTM] Best val_accuracy:", best_val, "| Test accuracy:", float(test_acc))

# Persist artifacts
MODELS_DIR.mkdir(parents=True, exist_ok=True)

out_model = MODELS_DIR / "lstm_best.keras"
best_model.save(out_model.as_posix())

out_tok = MODELS_DIR / "lstm_best_tokenizer.joblib"
joblib.dump(tokenizer, out_tok)

out_params = MODELS_DIR / "lstm_best_params.json"
with open(out_params, "w") as f:
    json.dump(
        {"best_config": best_cfg, "val_accuracy": float(best_val), "test_accuracy": float(test_acc)},
        f,
        indent=2,
    )

# Display results
print(f"[LSTM] Saved model -> {out_model}")
print(f"[LSTM] Saved params -> {out_params}")
print(f"[LSTM] Saved tokenizer -> {out_tok}")
print(f"[LSTM] Best val score -> {best_val:.4f}")
print(f"[LSTM] Best params -> {best_cfg}")
print(f"[LSTM] Test accuracy -> {test_acc:.4f}")
