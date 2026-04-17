# ======================================================
# WEEK 08 - WEDNESDAY (FINAL SUBMISSION NOTEBOOK)
# ======================================================

# =========================
# 1. IMPORTS
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# =========================
# 2. LOAD REDDIT DATA
# =========================
df = pd.read_csv(r"C:\Users\avish\Desktop\iitgn AINPT\week-8\wednesday\Reddit_Data.csv")   # change path if needed

print("Sample Data:")
print(df.head())

print("\nColumns:", df.columns)

# =========================
# 3. EDA
# =========================
print("\nSentiment Distribution:")
print(df['category'].value_counts())

sns.countplot(x='category', data=df)
plt.title("Sentiment Distribution")
plt.show()

print("\nMissing Values:")
print(df.isnull().sum())

# =========================
# 4. CREATE HATE SPEECH LABEL
# =========================
def map_hate(x):
    if x == -1:
        return 1   # hate
    else:
        return 0   # not hate

df['hate_speech'] = df['category'].apply(map_hate)

print("\nHate Speech Distribution:")
print(df['hate_speech'].value_counts())

# =========================
# 5. TEXT CLEANING
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text

df['clean_text'] = df['clean_comment'].apply(clean_text)

# =========================
# 6. TF-IDF + CLASSIFIER
# =========================
X = df['clean_text']
y = df['hate_speech']

vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

model_lr = LogisticRegression(class_weight='balanced', max_iter=1000)
model_lr.fit(X_train, y_train)

y_pred = model_lr.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =========================
# 7. SEMANTIC SEARCH (SBERT)
# =========================
print("\nLoading Sentence-BERT model...")
model_emb = SentenceTransformer('all-MiniLM-L6-v2')

print("Encoding text...")
embeddings = model_emb.encode(df['clean_text'].tolist(), show_progress_bar=True)

def semantic_search(query, top_k=5):
    query_clean = clean_text(query)
    q_emb = model_emb.encode([query_clean])
    sims = cosine_similarity(q_emb, embeddings)[0]
    top_idx = sims.argsort()[-top_k:][::-1]
    return df.iloc[top_idx][['clean_comment', 'hate_speech']]

print("\nSemantic Search Example:")
print(semantic_search("I hate this leader"))

# =========================
# 8. MNIST CNN
# =========================
(x_train, y_train_mnist), (x_test, y_test_mnist) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

cnn = models.Sequential([
    layers.Input(shape=(28,28,1)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

cnn.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

print("\nTraining CNN...")
cnn.fit(x_train, y_train_mnist, epochs=3,
        validation_data=(x_test, y_test_mnist))

# =========================
# 9. FILTER VISUALIZATION
# =========================
# Find first Conv2D layer safely
for layer in cnn.layers:
    if isinstance(layer, tf.keras.layers.Conv2D):
        filters, biases = layer.get_weights()
        break

for i in range(6):
    plt.imshow(filters[:, :, 0, i])
    plt.title(f"Filter {i}")
    plt.show()

# =========================
# 10. TWO-STAGE PIPELINE
# =========================
def moderation_pipeline(text):
    clean = clean_text(text)
    vec = vectorizer.transform([clean])
    pred = model_lr.predict(vec)[0]

    if pred == 1:
        return "Flagged by classifier"

    similar = semantic_search(text, top_k=3)
    if similar['hate_speech'].sum() > 0:
        return "Flagged by semantic similarity"

    return "Safe"

print("\nPipeline Test:")
print(moderation_pipeline("This politician is terrible"))

# =========================
# 11. TF-IDF vs EMBEDDINGS
# =========================
def tfidf_search(query, top_k=5):
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, X_vec)[0]
    top_idx = sims.argsort()[-top_k:][::-1]
    return df.iloc[top_idx][['clean_comment']]

print("\nTF-IDF Results:")
print(tfidf_search("I hate them"))

print("\nEmbedding Results:")
print(semantic_search("I hate them"))

# =========================
# DONE
# =========================
print("\nNotebook execution complete.")