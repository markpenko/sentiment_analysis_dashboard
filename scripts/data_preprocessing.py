# data_preprocessing.py
# Mark Antepenko

# Importing libraries
import nltk
import string
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Setting the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = DATA_DIR / "processed_data"
NLTK_DIR = PROJECT_ROOT / "nltk"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
NLTK_DIR.mkdir(parents=True, exist_ok=True)

# Download NLTK resources
for pkg in ['punkt', 'stopwords', 'wordnet']:
    try:
        nltk.data.find(f'tokenizers/{pkg}' if pkg == 'punkt' else f'corpora/{pkg}')
    except LookupError:
        nltk.download(pkg, download_dir=str(NLTK_DIR))

## Reading and importing .csv for train and test datasets
# Load the IMDB Movie Reviews dataset
print("Loading CSV ...")
data = pd.read_csv(DATA_DIR / 'raw_data/IMDB_Dataset.csv')

# Visualize the dataset 
print(data.head())

# Split the dataset into training and test sets
print("Splitting dataset into training and test sets ...")
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Preprocessing: normalization, tokenization, stopword removal, lemmatization, and vectorization

# Initialize lemmatizer
print("Initializing lemmatizer ...")
lemmatizer = WordNetLemmatizer()
STOP = set(stopwords.words('english'))

# Define preprocessing function
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Tokenize text
    tokens = nltk.word_tokenize(text)
    # Remove punctuation and stop words
    tokens = [word for word in tokens if word not in string.punctuation and word not in STOP]
    # Lemmatize tokens
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Apply preprocessing to training and test sets
train_data['review'] = train_data['review'].apply(preprocess_text)
test_data['review'] = test_data['review'].apply(preprocess_text)

# Visualize the processed data
print("Training data:")
print(train_data.info())
print(train_data.head())
print("Test data:")
print(test_data.info())
print(test_data.head())

# Save the training and test sets
print("Saving preprocessed data ...")
train_data.to_csv(PROCESSED_DIR / 'train_data_preprocessed.csv', index=False)
test_data.to_csv(PROCESSED_DIR / 'test_data_preprocessed.csv', index=False)