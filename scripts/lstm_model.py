# lstm_model.py
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

# Build the LSTM model
print("Building the LSTM model...")
EMBEDDING_DIM = 100
LSTM_UNITS_1 = 128
LSTM_UNITS_2 = 64
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001      

model = Sequential([
    Embedding(
        input_dim=vocab_size_effective, 
        output_dim=EMBEDDING_DIM, 
        input_length=max_length, 
        mask_zero=True
    ),
    LSTM(LSTM_UNITS_1, return_sequences=True),
    Dropout(DROPOUT_RATE),
    LSTM(LSTM_UNITS_2),
    Dropout(DROPOUT_RATE),
    Dense(1, activation='sigmoid')
])

# Compile the model
print("Compiling the model...")
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),  # CHANGED TO STANDARD ADAM
    loss='binary_crossentropy',
    metrics=['accuracy', AUC(name='auc')]
)

# Callbacks
callbacks = [
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
    ),
    ModelCheckpoint(
        filepath=str(MODELS_DIR / "lstm_model_checkpoint.weights.h5"),
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        save_weights_only=True
    )
]

# Train the model
print("Starting train" \
"ing...")
history = model.fit(
    X_tr, 
    y_tr,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, y_val),
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# Save the trained model and tokenizer
print("Saving the artifacts...")

model.save_weights(str(MODELS_DIR / "lstm_model_baseline.weights.h5"))

model_json = model.to_json()
with open(MODELS_DIR / "lstm_model_architecture.json", "w") as json_file:
    json_file.write(model_json)

model.save(str(MODELS_DIR / "lstm_model_baseline.keras"))
joblib.dump(tokenizer, MODELS_DIR / "lstm_tokenizer.joblib")

meta = {
    "vocab_size_requested": int(VOCAB_SIZE),
    "vocab_size_effective": int(vocab_size_effective),
    "max_length": int(max_length),
    "seed": int(SEED),
    "embedding_dim": int(EMBEDDING_DIM),
    "architecture": {
        "lstm1_units": int(LSTM_UNITS_1),
        "lstm2_units": int(LSTM_UNITS_2),
        "dropout_rate": float(DROPOUT_RATE),
        "optimizer": "Adam",
        "learning_rate": float(LEARNING_RATE),
        "loss": "binary_crossentropy"
    }
}

with open(MODELS_DIR / "lstm_baseline_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"Saved model weights -> {MODELS_DIR / 'lstm_model_baseline.weights.h5'}")
print(f"Saved model architecture -> {MODELS_DIR / 'lstm_model_architecture.json'}")
print(f"Saved full model -> {MODELS_DIR / 'lstm_model_baseline.keras'}")
print(f"Saved tokenizer -> {MODELS_DIR / 'lstm_tokenizer.joblib'}")
print(f"Saved metadata -> {MODELS_DIR / 'lstm_baseline_meta.json'}")

# Evaluate the model
print("Evaluating the model on the test set...")
y_pred_prob = model.predict(X_test_padded, verbose=0).ravel()
y_pred = (y_pred_prob > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)

print(f"Accuracy: {accuracy:.4f} | ROC-AUC: {auc:.4f}")
print(classification_report(y_test, y_pred, zero_division=0))