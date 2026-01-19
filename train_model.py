import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import re
import string
import joblib
import os

# 1. Load Data
print("Loading datasets...")
current_dir = os.getcwd()
fake_path = os.path.join(current_dir, 'Fake.csv')
true_path = os.path.join(current_dir, 'True.csv')

def load_data(path, label):
    try:
        # Try reading with default utf-8, skipping bad lines
        df = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip')
        df['label'] = label
        return df
    except UnicodeDecodeError:
        print(f"Warning: Unicode error in {path}, trying 'latin1' encoding.")
        df = pd.read_csv(path, encoding='latin1', on_bad_lines='skip')
        df['label'] = label
        return df
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

fake_df = load_data(fake_path, 0)
true_df = load_data(true_path, 1)

if fake_df is None or true_df is None:
    print("Failed to load datasets. Exiting.")
    exit()

print(f"Loaded Fake: {len(fake_df)} rows")
print(f"Loaded True: {len(true_df)} rows")

# 2. Combine and Shuffle
df = pd.concat([fake_df, true_df], axis=0)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 3. Cleaning Function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

print("Cleaning text data (this may take a moment)...")
# Ensure 'text' column exists and treat as string
if 'text' not in df.columns:
    print("Error: 'text' column missing from CSVs.")
    exit()

df['cleaned_content'] = df['text'].astype(str).apply(clean_text)

# 4. Train/Test Split
X = df['cleaned_content']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Vectorization
print("Vectorizing text...")
try:
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)
    tfidf_test = tfidf_vectorizer.transform(X_test)
except Exception as e:
    print(f"Error during vectorization: {e}")
    exit()

# 6. Train Model
print("Training PassiveAggressiveClassifier...")
try:
    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(tfidf_train, y_train)
except Exception as e:
    print(f"Error during training: {e}")
    exit()

# 7. Evaluate
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'Training complete. Accuracy: {round(score*100,2)}%')

# 8. Save Model and Vectorizer
print("Saving model artifacts...")
try:
    joblib.dump(pac, 'model.pkl')
    joblib.dump(tfidf_vectorizer, 'vectorizer.pkl')
    print("Success! Model saved to 'model.pkl' and 'vectorizer.pkl'")
except Exception as e:
    print(f"Error saving model: {e}")
