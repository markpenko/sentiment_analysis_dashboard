# lstm_hyperparameter.py
# Mark Antepenko

# Import Libraries
import os
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf 
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout # pyright: ignore[reportMissingImports]
from tensorflow.keras.metrics import AUC # pyright: ignore[reportMissingImports]
from tensorflow.keras.models import Sequential # pyright: ignore[reportMissingImports]
from tensorflow.keras.optimizers import Adam # pyright: ignore[reportMissingImports]
from tensorflow.keras.preprocessing.sequence import pad_sequences # pyright: ignore[reportMissingImports]
from tensorflow.keras.preprocessing.text import Tokenizer # pyright: ignore[reportMissingImports] 

# Runtime safety toggle
os.environ.setdefault("USE_TF_GPU", "0")  # default to CPU unless explicitly enabled
USE_GPU = os.environ.get("USE_TF_GPU", "0") == "1"

if USE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
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

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed_data"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Data Pipeline
print("Loading preprocessed data...")
train_data = pd.read_csv(DATA_DIR / 'train_data_preprocessed.csv')
test_data = pd.read_csv(DATA_DIR / 'test_data_preprocessed.csv')

# Extract features and labels
print("Preparing data...")
X_train = train_data['review']
y_train = train_data['sentiment'].map({"positive": 1, "negative": 0})

X_test = test_data['review']
y_test = test_data['sentiment'].map({"positive": 1, "negative": 0})

# Tokenize
print("Tokenizing text...")
VOCAB_SIZE = 5000
max_length = 200
OOV_TOKEN = '<OOV>'

meta_path = MODELS_DIR / "lstm_meta.json"
tokenizer_path = MODELS_DIR / "lstm_tokenizer.joblib"

if meta_path.exists():
    print("Loading existing metadata...")
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    VOCAB_SIZE = meta["vocab_size_requested"]
    max_length = meta["max_length"]

if tokenizer_path.exists():
    print("Loading existing tokenizer...")
    tokenizer = joblib.load(tokenizer_path)
else:
    print("No existing tokenizer found. Fitting a new tokenizer...")
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(X_train)

vocab_size_effective = min(VOCAB_SIZE, len(tokenizer.word_index) + 1)

# Padding sequences
print("Padding sequences...")
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

X_train_padded = pad_sequences(
    X_train_sequences,
    maxlen=max_length,
    padding='post',
    truncating='post'
)

X_test_padded = pad_sequences(
    X_test_sequences,
    maxlen=max_length,
    padding='post',
    truncating='post'
)

# Training and Validation Split
print("Creating train/validation split...")
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
(train_idx, val_idx), = sss.split(X_train_padded, y_train)

X_tr = X_train_padded[train_idx]
X_val = X_train_padded[val_idx]
y_tr = np.array(y_train)[train_idx]
y_val = np.array(y_train)[val_idx]

# Class weights
print("Computing class weights...")
classes = np.array([0, 1])
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes, 
    y=y_tr
)
class_weight_dict = {int(c): float(w) for c, w in zip(classes, class_weights)}

# Creating the hyperparameter search space
print("Creating hyperparameter search space...")
configs = [
    {
        "description": "baseline",
        "embedding_dim": 100,
        "lstm_units_1": 128,
        "lstm_units_2": 64,
        "dropout_rate": 0.2,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10,
    },
    {
        "description": "smaller_lstm",
        "embedding_dim": 100,
        "lstm_units_1": 64,
        "lstm_units_2": 32,
        "dropout_rate": 0.2,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10,
    },
    {
        "description": "higher_dropout",
        "embedding_dim": 100,
        "lstm_units_1": 128,
        "lstm_units_2": 64,
        "dropout_rate": 0.3,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10,
    },
    {
        "description": "lower_lr",
        "embedding_dim": 100,
        "lstm_units_1": 128,
        "lstm_units_2": 64,
        "dropout_rate": 0.2,
        "learning_rate": 0.0005,
        "batch_size": 32,
        "epochs": 10,
    },
    {
        "description": "smaller_batch",
        "embedding_dim": 100,
        "lstm_units_1": 128,
        "lstm_units_2": 64,
        "dropout_rate": 0.2,
        "learning_rate": 0.001,
        "batch_size": 16,
        "epochs": 10,
    },
]

best_val_auc = -1.0
best_config = None
best_model = None
results = []

print("Starting hyperparameter search...")

for i, config in enumerate(configs, start=1):
    print("\n" + "-" * 50)
    print(f"Config {i:2}/{len(configs)}: {config['description']}")
    print(config)
    print("-" * 50)

    # Clear state before building next model
    tf.keras.backend.clear_session()
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # Build the model
    model = Sequential([
        Embedding(
            input_dim=vocab_size_effective, 
            output_dim=config["embedding_dim"], 
            input_length=max_length, 
            mask_zero=True
        ),
        LSTM(config["lstm_units_1"], return_sequences=True),
        Dropout(config["dropout_rate"]),
        LSTM(config["lstm_units_2"]),
        Dropout(config["dropout_rate"]),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=config["learning_rate"]), 
        loss='binary_crossentropy',
        metrics=['accuracy', AUC(name='auc')]
    )

    # Train the model
    history = model.fit(
        X_tr, 
        y_tr,
        validation_data=(X_val, y_val),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        class_weight=class_weight_dict,
        callbacks=[
            EarlyStopping(
                monitor="val_auc", 
                mode="max", 
                patience=2, 
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor="val_loss", 
                factor=0.5, 
                patience=1, 
                verbose=1
            )
        ],
        verbose=1
    )

    val_auc = float(max(history.history.get('val_auc', [0.0])))
    val_acc = float(max(history.history.get('val_accuracy', [0.0])))

    result_row = {
        "description": config["description"],
        "val_auc": val_auc,
        "val_accuracy": val_acc,
        "embedding_dim": config["embedding_dim"],
        "lstm_units_1": config["lstm_units_1"],
        "lstm_units_2": config["lstm_units_2"],
        "dropout_rate": config["dropout_rate"],
        "learning_rate": config["learning_rate"],
        "batch_size": config["batch_size"],
        "epochs": config["epochs"],
    }
    results.append(result_row)

    print(f"Best val_auc for {config['description']}: {val_auc:.4f}")
    print(f"Best val_accuracy for {config['description']}: {val_acc:.4f}")

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_config = config.copy()
        best_model = model

# Best results summary
print("\n" + "=" * 50)
print("Hyperparameter search complete!")
print(f"Best config: {best_config['description']}")
print(f"Best validation AUC: {best_val_auc:.4f}")
print(best_config)

# Save all results
with open(MODELS_DIR / "lstm_tuning_results.json", "w") as f:
    json.dump(results, f, indent=2)

with open(MODELS_DIR / "lstm_best_params.json", "w") as f:
    json.dump(best_config, f, indent=2)

print(f"Saved tuning results to {MODELS_DIR / 'lstm_tuning_results.json'}")
print(f"Saved best params to {MODELS_DIR / 'lstm_best_params.json'}")

# Evaluate best model on test set
print("\nEvaluating best model on test set...")
y_pred_prob = best_model.predict(X_test_padded, verbose=0).ravel()
y_pred = (y_pred_prob > 0.5).astype(int)

test_accuracy = accuracy_score(y_test, y_pred)
test_auc = roc_auc_score(y_test, y_pred_prob)

print(f"Test Accuracy: {test_accuracy:.4f} | Test ROC-AUC: {test_auc:.4f}")
print(classification_report(y_test, y_pred, zero_division=0))

# Saving the best model
print("Saving best model...")
best_model.save(str(MODELS_DIR / "lstm_model_tuned.keras"))

tuned_meta = {
    "vocab_size_requested": int(VOCAB_SIZE),
    "vocab_size_effective": int(vocab_size_effective),
    "max_length": int(max_length),
    "seed": int(SEED),
    "best_config": best_config,
    "best_validation_auc": float(best_val_auc),
    "test_accuracy": float(test_accuracy),
    "test_roc_auc": float(test_auc),
}

with open(MODELS_DIR / "lstm_tuned_meta.json", "w") as f:
    json.dump(tuned_meta, f, indent=2)

print(f"Saved best model to {MODELS_DIR / 'lstm_model_tuned.keras'}")
print(f"Saved best model metadata to {MODELS_DIR / 'lstm_tuned_meta.json'}")
    