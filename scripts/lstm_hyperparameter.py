# lstm_hyperparameter.py
# Mark Antepenko

# Import Libraries
import pandas as pd
import joblib
from pathlib import Path
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer  # pyright: ignore[reportMissingImports]
from tensorflow.keras.preprocessing.sequence import pad_sequences  # pyright: ignore[reportMissingImports]
from tensorflow.keras.models import Sequential  # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout  # pyright: ignore[reportMissingImports]
from tensorflow.keras.optimizers.legacy import Adam as LegacyAdam  # pyright: ignore[reportMissingImports]
from tensorflow.keras.metrics import AUC  # pyright: ignore[reportMissingImports]
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # pyright: ignore[reportMissingImports]
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import random
import json
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

# Load existing model and tokenizer
print("Loading existing model and tokenizer...")
try:
    base_model = tf.keras.models.load_model(MODELS_DIR / "lstm_model.keras") 
    tokenizer = joblib.load(MODELS_DIR / "tokenizer.joblib")
    
    # Load metadata
    with open(MODELS_DIR / "lstm_meta.json", "r") as f:
        meta = json.load(f)
    
    VOCAB_SIZE = meta["vocab_size_requested"]
    max_length = meta["max_length"]
    vocab_size_effective = meta["vocab_size_effective"]
    
    print(f"Loaded existing model with vocab_size={vocab_size_effective}, max_length={max_length}")
    
except FileNotFoundError:
    print("ERROR: No existing model found. Please run lstm_model.py first!")
    exit(1)

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

# Evaluate baseline model first
print("Evaluating baseline model...")
baseline_metrics = base_model.evaluate(Xte, y_test, verbose=0, return_dict=True)
baseline_acc = float(baseline_metrics.get("accuracy", 0.0))
print(f"Baseline model accuracy: {baseline_acc:.4f}")

# Hyperparameter configurations to try (fine-tuning around existing parameters)
fine_tune_configs = [
    {"lr": 5e-4, "batch_size": 32, "epochs": 5, "description": "Lower LR"},
    {"lr": 2e-3, "batch_size": 32, "epochs": 5, "description": "Higher LR"},
    {"lr": 1e-3, "batch_size": 16, "epochs": 5, "description": "Smaller batch"},
    {"lr": 1e-3, "batch_size": 64, "epochs": 5, "description": "Larger batch"},
]

best_val = -1.0
best_model = None
best_cfg = None
best_description = ""

# Fine-tuning loop
print("Starting fine-tuning of existing model...")
for cfg in fine_tune_configs:
    print(f"Testing configuration: {cfg['description']} - {cfg}")
    
    # Load fresh copy of the base model for each experiment
    model = tf.keras.models.clone_model(base_model)
    model.set_weights(base_model.get_weights())  # Copy weights from original
    
    # Recompile with new optimizer settings
    opt = LegacyAdam(learning_rate=cfg["lr"])
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy", AUC(name="auc")])
    
    # Fine-tune the model
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
    print(f"Configuration '{cfg['description']}' achieved val_auc: {val_auc:.4f}")
    
    if val_auc > best_val:
        best_val = val_auc
        best_model = model
        best_cfg = cfg
        best_description = cfg['description']

# Evaluate best fine-tuned model
print("\nEvaluating best fine-tuned model...")
metrics = best_model.evaluate(Xte, y_test, verbose=0, return_dict=True)
test_loss = float(metrics.get("loss", 0.0))
test_acc = float(metrics.get("accuracy", 0.0))
y_prob = best_model.predict(Xte, verbose=0).ravel()