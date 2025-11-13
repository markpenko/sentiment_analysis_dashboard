# lstm_model.py
# Mark Antepenko

# Import Libraries
import pandas as pd
import joblib
from pathlib import Path
import os
import tensorflow as tf  
from tensorflow.keras.preprocessing.text import Tokenizer # pyright: ignore[reportMissingImports]
from tensorflow.keras.preprocessing.sequence import pad_sequences # pyright: ignore[reportMissingImports]
from tensorflow.keras.models import Sequential # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout # pyright: ignore[reportMissingImports]
from tensorflow.keras.optimizers.legacy import Adam as LegacyAdam  # pyright: ignore[reportMissingImports]
from tensorflow.keras.metrics import AUC # pyright: ignore[reportMissingImports]
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import random
import json
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # pyright: ignore[reportMissingImports]

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


# Tokenize and pad sequences
print("Tokenizing and padding sequences...")
VOCAB_SIZE = 5000
max_length = 200
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)
vocab_size_effective = min(VOCAB_SIZE, len(tokenizer.word_index) + 1)
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)
X_train_padded = pad_sequences(X_train_sequences, maxlen=max_length, padding='post',
truncating='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=max_length, padding='post',
truncating='post')

# Build the LSTM model
print("Building the LSTM model...")
embedding_dim = 100
model = Sequential([
Embedding(input_dim=vocab_size_effective, output_dim=embedding_dim, input_length=max_length, mask_zero=True),
LSTM(128, return_sequences=True),
Dropout(0.2),
LSTM(64),
Dropout(0.2),
Dense(1, activation='sigmoid')
])

# Compile the model
print("Compiling the model...")
model.compile(
    optimizer=LegacyAdam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', AUC(name='auc')]
)

# Stratified train/validation split
print("Creating train/validation split...")
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
(train_idx, val_idx), = sss.split(X_train_padded, y_train)
X_tr, X_val = X_train_padded[train_idx], X_train_padded[val_idx]
y_tr, y_val = np.array(y_train)[train_idx], np.array(y_train)[val_idx]

# Class weights for imbalance
print("Computing class weights...")
classes = np.array([0, 1])
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_tr)
class_weight_dict = {int(c): float(w) for c, w in zip(classes, class_weights)}

# Callbacks
callbacks = [
    EarlyStopping(monitor="val_auc", mode="max", patience=2, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, verbose=1),
    ModelCheckpoint(
        filepath=str(MODELS_DIR / "lstm_model_best.weights.h5"),
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        save_weights_only=True
    )
]

# Train the model
print("Starting training...")
history = model.fit(
    X_tr, y_tr,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, y_val),
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# Save the trained model and tokenizer
print("Saving the model and tokenizer...")
model.save(str(MODELS_DIR / "lstm_model.keras"))
joblib.dump(tokenizer, MODELS_DIR / "tokenizer.joblib")

# Save metadata
meta = {
    "vocab_size_requested": int(VOCAB_SIZE),
    "vocab_size_effective": int(vocab_size_effective),
    "max_length": int(max_length),
    "seed": int(SEED)
}
with open(MODELS_DIR / "lstm_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

# Evaluate the model on the test set
print("Evaluating the model on the test set...")
y_pred_prob = model.predict(X_test_padded).ravel()
y_pred = (y_pred_prob > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
try:
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_test, y_pred_prob)
    print(f"Accuracy: {accuracy:.4f} | ROC-AUC: {auc:.4f}")
except Exception:
    print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred, zero_division=0))