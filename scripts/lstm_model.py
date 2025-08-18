# Chapter 13 lstm_model.py
# Mark Antepenko

# Import Libraries
import pandas as pd
import joblib
from pathlib import Path
import os
import tensorflow as tf  # pyright: ignore[reportMissingImports]
from tensorflow.keras.preprocessing.text import Tokenizer # pyright: ignore[reportMissingImports]
from tensorflow.keras.preprocessing.sequence import pad_sequences # pyright: ignore[reportMissingImports]
from tensorflow.keras.models import Sequential # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout # pyright: ignore[reportMissingImports]
from tensorflow.keras.optimizers.legacy import Adam as LegacyAdam  # pyright: ignore[reportMissingImports]
from sklearn.metrics import accuracy_score, classification_report
import pickle

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

# Setting the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Load preprocessed data
train_data = pd.read_csv(DATA_DIR / 'processed_data/train_data_preprocessed.csv')
test_data = pd.read_csv(DATA_DIR / 'processed_data/test_data_preprocessed.csv')

# Extract features and labels
X_train = train_data['review']
y_train = train_data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
X_test = test_data['review']
y_test = test_data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)
max_length = 200
X_train_padded = pad_sequences(X_train_sequences, maxlen=max_length, padding='post',
truncating='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=max_length, padding='post',
truncating='post')

# Build the LSTM model
embedding_dim = 100
model = Sequential([
Embedding(input_dim=5000, output_dim=embedding_dim, input_length=max_length),
LSTM(128, return_sequences=True),
Dropout(0.2),
LSTM(64),
Dropout(0.2),
Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=LegacyAdam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_padded, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Save the trained model and tokenizer
model.save(str(MODELS_DIR / "lstm_model.keras"))
joblib.dump(tokenizer, MODELS_DIR / "tokenizer.joblib")

# Evaluate the model on the test set
y_pred_prob = model.predict(X_test_padded)
y_pred = (y_pred_prob.ravel() > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:4f}")
print(classification_report(y_test, y_pred, zero_division=0))