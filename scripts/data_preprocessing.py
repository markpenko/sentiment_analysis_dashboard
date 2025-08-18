# Chapter 13 data_preprocessing.py
# Mark Antepenko

# Importing libraries
import nltk
import string
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Setting the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

## Reading and importing .csv for train and test datasets
# Load the IMDB Movie Reviews dataset
print("Loading CSV ...")
data = pd.read_csv(DATA_DIR / 'raw_data/IMDB_Dataset.csv')

# Visualize the dataset 
print(data.head())

# Split the dataset into training and test sets
print("Splitting dataset into training and test sets ...")
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Save the training and test sets
print("Saving processed data ...")
train_data.to_csv(DATA_DIR / 'processed_data/train_data.csv', index=False)
test_data.to_csv(DATA_DIR / 'processed_data/test_data.csv', index=False)

# Visualize the processed data
print("Training data:")
print(train_data.head())
print("Test data:")
print(test_data.head())

## Preprocessing: normalization, tokenization, stopword removal, lemmatization, and vectorization

# Initialize lemmatizer
print("Initializing lemmatizer ...")
lemmatizer = WordNetLemmatizer()


# Define preprocessing function
def preprocess_text(text):
       # Count how many times run.
    if not hasattr(preprocess_text, "i"):
        preprocess_text.i = 50001    # pyright: ignore[reportFunctionMemberAccess]
    preprocess_text.i += -1 # pyright: ignore[reportFunctionMemberAccess]
    print(f"Cycle: {preprocess_text.i}") # pyright: ignore[reportFunctionMemberAccess]
    
    # Convert text to lowercase
    text = text.lower()
    # Tokenize text
    tokens = nltk.word_tokenize(text)
    # Remove punctuation and stop words
    tokens = [word for word in tokens if word not in string.punctuation and word not in stopwords.words('english')]
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

# Save preprocessed data
print("Saving data ...")
train_data.to_csv(DATA_DIR / 'processed_data/train_data_preprocessed.csv', index=False)
test_data.to_csv(DATA_DIR / 'processed_data/test_data_preprocessed.csv', index=False)