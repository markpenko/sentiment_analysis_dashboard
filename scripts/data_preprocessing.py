# data_preprocessing.py
# Mark Antepenko

# Importing libraries
import nltk
import string
import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from imblearn.over_sampling import SMOTE

# Download NLTK resources
for pkg in ['punkt', 'stopwords', 'wordnet']:
    try:
        nltk.data.find(f'tokenizers/{pkg}' if pkg == 'punkt' else f'corpora/{pkg}')
    except LookupError:
        nltk.download(pkg) 

# Setting the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = DATA_DIR / "processed_data"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

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
train_data.to_csv(PROCESSED_DIR / 'train_data.csv', index=False)
test_data.to_csv(PROCESSED_DIR / 'test_data.csv', index=False)

###########################################################
## Preprocessing: normalization, tokenization, stopword removal, lemmatization, and vectorization

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

# Extract features and labels
print("Extracting features and labels ...")
X_train = train_data['review']
y_train = train_data['sentiment']
X_test = test_data['review']
y_test = test_data['sentiment']

# Vectorize the preprocessed text
print("Vectorizing text data ...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vectorizer = vectorizer.fit_transform(train_data['review']).toarray()  # pyright: ignore[reportAttributeAccessIssue] # 
X_test_vectorizer = vectorizer.transform(test_data['review']).toarray() # pyright: ignore[reportAttributeAccessIssue]

# Save the vectorizer and vectorized data
print("Saving vectorizer and vectorized data ...")
with open(PROJECT_ROOT / 'models/vectorizer.pickle', 'wb') as file:
    pickle.dump(vectorizer, file)
with open(DATA_DIR / 'processed_data/X_train.pickle', 'wb') as file:
    pickle.dump(X_train_vectorizer, file)
with open(DATA_DIR / 'processed_data/X_test.pickle', 'wb') as file:
    pickle.dump(X_test_vectorizer, file)
    
# Balance the dataset using SMOTE
print("Balancing dataset using SMOTE ...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_vectorizer, y_train)  # pyright: ignore[reportAssignmentType]

# Save the balanced data
print("Saving balanced training data ...")
with open(DATA_DIR / 'processed_data/X_train_balanced.pickle', 'wb') as file:
    pickle.dump(X_train_balanced, file)
with open(DATA_DIR / 'processed_data/y_train_balanced.pickle', 'wb') as file:
    pickle.dump(y_train_balanced, file)