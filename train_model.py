import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Load your custom dataset
# df = pd.read_csv("dataset/fake_news_data.csv")  # Replace with your actual CSV file

df = pd.read_csv('dataset/fake_news_data.csv', encoding='latin1')

# 2. Combine title and text
df['content'] = df['title'] + " " + df['text']

# 3. Features and Labels
X = df['content']
y = df['label']

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# 5. Vectorize using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 6. Train Model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_vec, y_train)

# 7. Predict on test set
y_pred = model.predict(X_test_vec)

# 8. Evaluate
score = accuracy_score(y_test, y_pred)
print(f"Accuracy: {round(score*100, 2)}%")

conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
print("Confusion Matrix:\n", conf_matrix)


import pickle

# Save vectorizer and model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
