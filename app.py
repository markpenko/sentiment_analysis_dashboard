# Chapter 13 app.py
# Mark Antepenko

# Import libraries
import os
import joblib
import numpy as np
import tensorflow as tf
import plotly.express as px
import pandas as pd
import string
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from wordcloud import WordCloud
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')  
nltk.download('wordnet')

lemmatizer = nltk.WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

app = Flask(__name__)

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
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
TEMPLATE_DIR = PROJECT_ROOT / "templates"
UPLOAD_FOLDER = DATA_DIR / 'uploads'
STATIC_FOLDER = PROJECT_ROOT / "static"
ALLOWED_EXTENSIONS = {'csv', 'txt'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load the LSTM model and tokenizer
lstm_model = load_model(MODELS_DIR / 'lstm_best.keras', compile=False)
tokenizer = joblib.load(MODELS_DIR / 'lstm_best_tokenizer.joblib')

# Load the Logistic Regression model
vectorizer, logistic_regression_model = joblib.load(MODELS_DIR / 'tuned_logistic_regression_bundle.joblib')

MAX_SEQUENCE_LENGTH = 200


def preprocess_for_lstm(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in string.punctuation and word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    cleaned = ' '.join(tokens)
    sequences = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    return padded

def generate_wordcloud(text, output_path):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(str(output_path))
    plt.close()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    model_type = request.form['model_type']
    print("=== RAW input from form ===", repr(text))

    if model_type == 'lstm':
        padded = preprocess_for_lstm(text)
        prediction_prob = lstm_model.predict(padded)
        prediction = int(prediction_prob[0][0] > 0.5)
        sentiment = 'Positive' if prediction == 1 else 'Negative'

    elif model_type == 'logistic_regression':
        # Preprocess
        tokens = nltk.word_tokenize(text.lower())
        tokens = [word for word in tokens if word not in string.punctuation and word not in stop_words]
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        cleaned_text = ' '.join(tokens)

        prediction = logistic_regression_model.predict(vectorizer.transform([cleaned_text]))[0]

        if isinstance(prediction, str):
            sentiment = prediction.capitalize()  # handles "positive"/"negative"
        else:
            sentiment = 'Positive' if prediction == 1 else 'Negative'
    else:
        sentiment = 'Unknown model type'

    # Generate wordcloud image for the single text
    wordcloud_path = STATIC_FOLDER / 'wordcloud.png'
    generate_wordcloud(text, wordcloud_path)
    from time import time
    wordcloud_url = f"/static/wordcloud.png?cb={int(time())}"

    return jsonify({'sentiment': sentiment, 'wordcloud_url': wordcloud_url})



@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    file_text = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'message': 'No file part'})
    
        file = request.files['file']
        if file.filename == '':
            return jsonify({'message': 'No selected file'})
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)

            # Read file content (for text display)
            with open(save_path, 'r', encoding='utf-8') as f:
                file_text = f.read()

            ext = filename.rsplit('.', 1)[1].lower()

            if ext == 'txt':
                # Read contents as one observation
                text = file_text

                # Predict sentiment
                # For simplicity, use LSTM model for uploaded file sentiment
                preprocessed_text = preprocess_for_lstm(text)
                prediction_prob = lstm_model.predict(preprocessed_text)
                prediction = (prediction_prob > 0.5).astype(int)
                sentiment = 'Positive' if prediction[0] == 1 else 'Negative'

                # Create result_html as a simple paragraph with sentiment
                result_html = f"<p>Sentiment: <strong>{sentiment}</strong></p>"

                # Generate wordcloud
                wordcloud_path = STATIC_FOLDER / 'wordcloud.png'
                generate_wordcloud(text, wordcloud_path)
                from time import time
                wordcloud_url = f"/static/wordcloud.png?cb={int(time())}"

                return jsonify({'message': f'Text file processed', 'sentiment': sentiment, 'result_html': result_html, 'wordcloud_url': wordcloud_url})

            elif ext == 'csv':
                # Read csv into pandas
                df = pd.read_csv(save_path)

                # Check for a text column (assume first column)
                text_col = df.columns[0]

                # Predict sentiment for each row
                sentiments = []
                for text in df[text_col].astype(str):
                    # Use LSTM model for predictions
                    preprocessed_text = preprocess_for_lstm(text)
                    prediction_prob = lstm_model.predict(preprocessed_text)
                    prediction = (prediction_prob > 0.5).astype(int)
                    sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
                    sentiments.append(sentiment)

                df['Sentiment'] = sentiments

                # Create HTML table
                result_html = df.to_html(classes='table table-striped', index=False)

                # Generate wordcloud from all text combined
                combined_text = ' '.join(df[text_col].astype(str))
                wordcloud_path = STATIC_FOLDER / 'wordcloud.png'
                generate_wordcloud(combined_text, wordcloud_path)
                from time import time
                wordcloud_url = f"/static/wordcloud.png?cb={int(time())}"

                return jsonify({'message': f'CSV file processed', 'sentiment': None, 'result_html': result_html, 'wordcloud_url': wordcloud_url})

            else:
                return jsonify({'message': 'Invalid file type'})
    
    return render_template('upload.html', file_text=file_text)

if __name__ == '__main__':
    app.run(debug=True)