# 📰 Fake News Detection using Machine Learning

> A Flask-based web app that detects whether news is real or fake using machine learning and natural language processing.

---

## 📌 Table of Contents

- [About](#about)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Run the App](#run-the-app)
- [Project Structure](#project-structure)
- [Accuracy](#accuracy)
- [Screenshots](#screenshots)
- [Contributors](#contributors)
- [License](#license)

---

## 🧠 About

This project is designed to detect fake news based on the title and body text of news articles.  
It uses TF-IDF vectorization and a Passive Aggressive Classifier to predict whether news is **REAL** or **FAKE**.

---

## 🚀 Features

- ✅ Input news title and text  
- ✅ Detect fake or real news instantly  
- ✅ Built using machine learning and Flask  
- ✅ Simple and clean web interface  

---

## 🛠️ Tech Stack

- Python  
- Flask  
- scikit-learn  
- Pandas, NumPy  
- HTML/CSS (Bootstrap)  

---

## 📂 Dataset

- Source: `dataset/fake_news_data.csv`  
- Columns: `title`, `text`, `label`  
- Labels: `"FAKE"` or `"REAL"`  

---

## 🔧 Model Training

To train the model, run:

```bash
python train_model.py
