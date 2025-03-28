import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
import joblib

# Project Objective:
# The goal of this project is to develop an SMS classification model that accurately identifies spam messages.
# The dataset consists of labeled SMS messages categorized as spam or ham (non-spam).
# Expected Outcome: A robust model capable of distinguishing between spam and legitimate messages with high accuracy.

# Load Dataset
df = pd.read_csv(r"C:\Users\Admin\OneDrive\Desktop\grothin\q1\spam.csv", encoding='latin-1')
df = df.iloc[:, [0, 1]]  # Keep only label and message columns
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # Convert labels to binary

# Data Visualization
plt.figure(figsize=(6, 4))
sns.countplot(x=df['label'], hue=df['label'], legend=False, palette='coolwarm')
plt.title("Distribution of Spam and Ham Messages")
plt.show()

# Word Cloud for Spam Messages
spam_words = " ".join(df[df['label'] == 1]['message'])
wordcloud = WordCloud(width=500, height=300, background_color='black').generate(spam_words)
plt.figure(figsize=(8, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud for Spam Messages")
plt.show()

# Text Preprocessing
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text.strip()

df['message'] = df['message'].apply(clean_text)

# Feature Extraction
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = vectorizer.fit_transform(df['message'])
y = df['label']

# Handle Class Imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train Models with Hyperparameter Tuning
param_grid = {
    "Logistic Regression": {'C': [0.1, 1, 10]},
    "Random Forest": {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
    "SVM": {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
}

models = {
    "Naïve Bayes": MultinomialNB(),
    "Logistic Regression": GridSearchCV(LogisticRegression(), param_grid["Logistic Regression"], cv=5, n_jobs=-1),
    "Random Forest": GridSearchCV(RandomForestClassifier(), param_grid["Random Forest"], cv=5, n_jobs=-1),
    "SVM": GridSearchCV(SVC(), param_grid["SVM"], cv=5, n_jobs=-1)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"\nModel: {name}")
    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Confusion Matrix Visualization
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='coolwarm', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Print Overall Model Performance
print("\nModel Performance Summary:")
for model, acc in results.items():
    print(f"{model}: {acc:.4f}")

# Save the vectorizer and best models
joblib.dump(vectorizer, "vectorizer.pkl")
for name, model in models.items():
    joblib.dump(model.best_estimator_ if isinstance(model, GridSearchCV) else model, f"{name.replace(' ', '_').lower()}_model.pkl")

# Close all plots to prevent runtime errors
plt.close('all')
