# lstm_hyperparameter.py
# Mark Antepenko

# Import Libraries
import os
import json
import joblib
import numpy as np
import random
from pathlib import Path
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models  # pyright: ignore[reportMissingImports]
from tensorflow.keras.preprocessing.text import Tokenizer  # pyright: ignore[reportMissingImports]
from tensorflow.keras.preprocessing.sequence import pad_sequences  # pyright: ignore[reportMissingImports]
from tensorflow.keras.optimizers.legacy import Adam as LegacyAdam  # pyright: ignore[reportMissingImports]
from tensorflow.keras.metrics import AUC  # pyright: ignore[reportMissingImports]
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # pyright: ignore[reportMissingImports]
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

# Runtime safety toggle
os.environ.setdefault("USE_TF_GPU", "0")  # default to CPU unless explicitly enabled
USE_GPU = os.environ.get("USE_TF_GPU", "0") == "1"
if not USE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Device configuration
if USE_GPU:
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
            print(f"GPU enabled: {len(gpus)} device(s) with memory growth set.")
        else:
            print("No GPU devices detected; running on CPU.")
    except Exception as e:
        print("GPU memory growth not set:", e)
else:
    try:
        # Hide GPUs from TF runtime explicitly
        tf.config.set_visible_devices([], 'GPU')
        print("GPU disabled explicitly. Running on CPU. Set USE_TF_GPU=1 to enable.")
    except Exception as e:
        print("GPU disable skipped:", e)

# Reproducibility
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

# Setting the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# Load preprocessed data
print("Loading preprocessed data...")
train_data = pd.read_csv(DATA_DIR / 'processed_data/train_data_preprocessed.csv')
test_data = pd.read_csv(DATA_DIR / 'processed_data/test_data_preprocessed.csv')

# Extract features and labels
print("Preparing data...")
X_train = train_data['review']
y_train = train_data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
X_test = test_data['review']
y_test = test_data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

VOCAB_SIZE = 5000
max_length = 200
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
vocab_size_effective = min(VOCAB_SIZE, len(tokenizer.word_index) + 1)

def vec(xs):
    return pad_sequences(
        tokenizer.texts_to_sequences(xs),
        maxlen=max_length,
        padding="post",
        truncating="post",
    )

Xtr = vec(X_train)
Xte = vec(X_test)

# Stratified train/validation split
print("Creating train/validation split...")
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
(tr_idx, val_idx), = sss.split(Xtr, y_train)
X_tr, X_val = Xtr[tr_idx], Xtr[val_idx]
y_tr, y_val = y_train[tr_idx], y_train[val_idx]

# Class weights for imbalance
print("Computing class weights...")
classes = np.array([0, 1])
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_tr)
class_weight_dict = {int(c): float(w) for c, w in zip(classes, class_weights)}

combos = [
    {"embedding_dim": 100, "lstm_units": 128, "dropout": 0.2, "lr": 1e-3, "batch_size": 32, "epochs": 10},
    {"embedding_dim": 128, "lstm_units": 128, "dropout": 0.3, "lr": 5e-4, "batch_size": 32, "epochs": 10},
]

best_val = -1.0
best_model = None
best_cfg = None

for cfg in combos:
    model = models.Sequential([
        layers.Embedding(input_dim=vocab_size_effective, output_dim=cfg["embedding_dim"], input_length=max_length, mask_zero=True),
        layers.LSTM(cfg["lstm_units"], return_sequences=True),
        layers.Dropout(cfg["dropout"]),
        layers.LSTM(cfg["lstm_units"] // 2),
        layers.Dropout(cfg["dropout"]),
        layers.Dense(1, activation="sigmoid")
    ])
    opt = LegacyAdam(learning_rate=cfg["lr"])
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy", AUC(name="auc")])
    hist = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        verbose=2,
        class_weight=class_weight_dict,
        callbacks=[
            EarlyStopping(monitor="val_auc", mode="max", patience=2, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, verbose=1)
        ],
    )
    val_auc = float(max(hist.history.get("val_auc", [0.0])))
    if val_auc > best_val:
        best_val = val_auc
        best_model = model
        best_cfg = cfg

metrics = best_model.evaluate(Xte, y_test, verbose=0, return_dict=True)
test_loss = float(metrics.get("loss", 0.0))
test_acc = float(metrics.get("accuracy", 0.0))
y_prob = best_model.predict(Xte, verbose=0).ravel()
try:
    test_auc = float(roc_auc_score(y_test, y_prob))
except Exception:
    test_auc = None
print("\n[LSTM] Best val_auc:", best_val, "| Test accuracy:", float(test_acc), "| Test AUC:", test_auc if test_auc is not None else "n/a")

# Persist artifacts
out_model = MODELS_DIR / "lstm_best.keras"
best_model.save(out_model.as_posix())

out_tok = MODELS_DIR / "lstm_best_tokenizer.joblib"
joblib.dump(tokenizer, out_tok)

out_params = MODELS_DIR / "lstm_best_params.json"
with open(out_params, "w") as f:
    json.dump(
        {"best_config": best_cfg, "val_auc": float(best_val), "test_accuracy": float(test_acc), "test_auc": test_auc},
        f,
        indent=2,
    )

# Save metadata
meta = {
    "vocab_size_requested": int(VOCAB_SIZE),
    "vocab_size_effective": int(vocab_size_effective),
    "max_length": int(max_length),
    "seed": int(SEED),
    "best_config": best_cfg
}
with open(MODELS_DIR / "lstm_hyper_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

# Display results
print(f"[LSTM] Saved model -> {out_model}")
print(f"[LSTM] Saved params -> {out_params}")
print(f"[LSTM] Saved tokenizer -> {out_tok}")
print(f"[LSTM] Best val score -> {best_val:.4f}")
print(f"[LSTM] Best params -> {best_cfg}")
print(f"[LSTM] Test accuracy -> {test_acc:.4f}")
