# Sentiment Analysis Dashboard

## Overview
This project is a web-based sentiment analysis application that classifies text as positive or negative using both traditional machine learning and deep learning models. The application allows users to input text or upload files and receive real-time predictions along with visualizations.

The goal of this project is to demonstrate an end-to-end machine learning pipeline, including data preprocessing, model training, evaluation, and deployment in an interactive web interface.

---

## Features
- Analyze sentiment from user input text
- Upload `.txt` or `.csv` files for batch sentiment analysis
- Choose between two models:
  - Logistic Regression (baseline ML model)
  - LSTM Neural Network (deep learning model)
- Automatic text preprocessing (tokenization, stopword removal, lemmatization)
- Word cloud visualization of input text
- Clean web interface built with Flask

---

## Models

### Logistic Regression
- TF-IDF vectorization
- SMOTE for class balancing
- Pipeline-based implementation
- Fast and efficient baseline model

### LSTM (Deep Learning)
- Tokenization + padding
- Embedding layer
- Two-layer LSTM architecture
- Hyperparameter tuning performed
- Final selected model: **baseline configuration**

---

## Model Performance

| Model                | Accuracy | ROC-AUC |
|---------------------|----------|--------|
| Logistic Regression | ~0.89    | ~0.89  |
| LSTM (baseline)     | ~0.88    | ~0.95  |

The LSTM model provides stronger probabilistic separation (ROC-AUC), while Logistic Regression offers competitive accuracy with faster inference.

---

## Dataset
- IMDB Movie Reviews Dataset  
- 50,000 labeled reviews (positive/negative)

Dataset source:
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

(Note: dataset is not stored in this repository due to size constraints)

---

## Project Structure
```
sentiment_analysis_dashboard/
│
├── app.py
├── scripts/
│   ├── data_preprocessing.py
│   ├── logistic_regression.py
│   ├── lstm_model.py
│   ├── lstm_hyperparameter.py
│
├── models/
│   ├── logistic_model_baseline.joblib
│   ├── logreg_model_tuned.joblib
│   ├── lstm_model_baseline.keras
│   ├── lstm_model_tuned.keras
│   ├── lstm_tokenizer.joblib
│
├── data/
│   ├── raw_data/
│   ├── processed_data/
│   ├── uploads/
│
├── nltk/
│   ├── corpora/
│   ├── tokenizers/
│
├── templates/
├── static/
└── requirements.txt
```
---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/sentiment_analysis_dashboard.git
cd sentiment_analysis_dashboard
```
### 2. Create environment
```bash
conda create -n ds python=3.11
conda activate ds
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Run the application
```bash
python app.py
```
### 5. Access the dashboard
Open your web browser and navigate to `http://127.0.0.1:5000/`

---
## Usage

### Text Input
	•	Enter text directly into the interface
	•	Select model (LSTM or Logistic Regression)
	•	View sentiment prediction and word cloud

### File Upload
	•	Upload .txt → single prediction
	•	Upload .csv → batch predictions
	•	First column is assumed to contain text

---

### NLP Pipeline

The following preprocessing steps are applied:
	1.	Lowercasing
	2.	Tokenization (NLTK)
	3.	Stopword removal
	4.	Punctuation removal
	5.	Lemmatization

---

### Key Learnings
	•	Built and compared classical ML vs deep learning models
	•	Implemented full ML pipeline from raw data to deployment
	•	Applied hyperparameter tuning for LSTM optimization
	•	Managed local NLP resources (NLTK) for reproducibility
	•	Designed a user-facing application using Flask

---

### Future Improvements
	•	Deploy to cloud (AWS / Render / Heroku alternative)
	•	Add model confidence scores to UI
	•	Improve UI/UX design
	•	Add API endpoint for external use
	•	Introduce transformer-based model (BERT)

---

## Author

Mark Antepenko
M.S. Data Science & Analytics