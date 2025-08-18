# Chapter 13 data_preprocessing2.py
# Mark Antepenko

# Importing libraries
import pickle
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE

# Setting the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

print("Loading CSV ...")
train_data = pd.read_csv(DATA_DIR / 'processed_data/train_data.csv')
test_data = pd.read_csv(DATA_DIR / 'processed_data/test_data.csv')

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