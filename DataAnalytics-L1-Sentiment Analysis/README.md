# Sentiment Analysis on Twitter Data (Hate-Speech Detection)

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A machine learning pipeline that classifies the sentiment/tone of tweets using **NLP preprocessing + TF-IDF + classical ML classifiers**, providing insight into public opinion and flagging potentially offensive content.

> 📌 **Internship Task 4 — Sentiment Analysis**
> Built as part of an internship project applying NLP and ML to real-world social media text data.

---

## 📖 Overview

This project builds an end-to-end sentiment classification pipeline over a dataset of **31,962 tweets**, each labeled as:

| Label | Meaning | Count | % of Data |
|:---:|---|---:|---:|
| `0` | Non-negative / normal tweet | 29,720 | 93.0% |
| `1` | Negative / hate-speech-flagged tweet | 2,242 | 7.0% |

Because the dataset is **heavily imbalanced**, model evaluation focuses on precision, recall, and F1-score per class — not accuracy alone.

---

## 🧠 Pipeline

```
Raw Tweets
   │
   ▼
Text Preprocessing  (lowercase → remove URLs/usernames/punctuation → tokenize → remove stopwords → lemmatize)
   │
   ▼
TF-IDF Vectorization (top 5,000 features)
   │
   ▼
Train/Test Split (80/20, stratified)
   │
   ▼
Model Training  →  Naive Bayes  |  Logistic Regression
   │
   ▼
Evaluation  →  Accuracy, Precision, Recall, F1, Confusion Matrix
   │
   ▼
Visualization  →  Class distribution, WordClouds, Model comparison
   │
   ▼
Error Analysis  →  Inspect misclassified tweets
```

---

## ⚙️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| Data handling | pandas, numpy |
| NLP | NLTK (stopwords, tokenization, WordNet lemmatizer) |
| Feature extraction | scikit-learn `TfidfVectorizer` |
| Models | scikit-learn `MultinomialNB`, `LogisticRegression` |
| Evaluation | scikit-learn metrics (accuracy, precision, recall, F1, confusion matrix) |
| Visualization | matplotlib, seaborn, WordCloud |
| Environment | Jupyter Notebook |

---

## 🧹 Text Preprocessing

Each tweet is cleaned with the following steps before vectorization:

- Convert text to lowercase
- Remove URLs
- Remove `@username` mentions
- Remove punctuation and numbers
- Tokenize into words
- Remove English stopwords
- Lemmatize each token (WordNet Lemmatizer)

## 🔢 Feature Extraction — TF-IDF

**TF-IDF (Term Frequency–Inverse Document Frequency)** converts cleaned tweets into numeric vectors, weighting words that are frequent *within* a tweet but rare *across* the dataset — surfacing the words that are most distinctive for classification, rather than generic filler words.

- Vocabulary capped at **top 5,000 features**
- Each row = one tweet, each column = one TF-IDF-weighted term

---

## 📊 Results

### Model Comparison

| Model | Accuracy |
|---|---:|
| Naive Bayes | **95.40%** |
| Logistic Regression | 95.17% |

### Classification Report — Naive Bayes

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| 0 (Non-negative) | 0.96 | 1.00 | 0.98 | 5,945 |
| 1 (Negative) | 0.91 | 0.38 | 0.54 | 448 |

### Classification Report — Logistic Regression

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| 0 (Non-negative) | 0.95 | 1.00 | 0.97 | 5,945 |
| 1 (Negative) | 0.91 | 0.35 | 0.50 | 448 |

> ⚠️ **Note on imbalance:** both models achieve high overall accuracy (~95%) largely because they perform well on the majority class. Recall on the minority "Negative" class is notably lower (35–38%), meaning many negative/hate-speech tweets are missed. This is a direct consequence of the 93/7 class imbalance and is discussed further in the notebook's error analysis.

### Visualizations included in the notebook
- Sentiment class distribution bar chart
- Model accuracy comparison chart
- Confusion matrices (Naive Bayes & Logistic Regression)
- WordClouds for each sentiment class

---

## 🔍 Error Analysis

Several tweets were misclassified by the models. Common causes identified:

- **Sarcasm or irony** — tone the model can't detect from word presence alone
- **Ambiguous wording** — borderline or context-dependent phrasing
- **Mixed sentiments** — a tweet containing both positive and negative cues
- **Lack of contextual understanding** — TF-IDF has no concept of word order or surrounding context, which is where classical bag-of-words models fall short compared to context-aware deep learning models

---

## 💡 Business Insights

- Organizations can monitor customer/public opinion in real time from social media text
- Businesses can identify dissatisfied or upset customers and respond proactively
- Platforms can use this as a first-pass filter to flag potentially offensive content for human moderation review
- Marketing and PR teams can track sentiment trends around campaigns, products, or public figures

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn wordcloud openpyxl
```

### Run the notebook
```bash
git clone https://github.com/lokeshadma/<repo-name>.git
cd <repo-name>
jupyter notebook Sentiment_Analysis.ipynb
```

On first run, the notebook downloads the required NLTK corpora (`punkt`, `stopwords`, `wordnet`, `omw-1.4`).

---

## 📁 Project Structure

```
.
├── Sentiment_Analysis.ipynb   # Full notebook: preprocessing → modeling → evaluation
├── twitter.xlsx                # Dataset (31,962 labeled tweets)
└── README.md
```

---

## 🏁 Conclusion

This project developed a complete sentiment analysis pipeline using NLP and classical machine learning. **Naive Bayes** achieved marginally higher overall accuracy (95.40% vs. 95.17%), though both models struggle with recall on the minority "negative" class due to the dataset's imbalance. Future improvements could include class-balancing techniques (SMOTE, class weighting), richer contextual features, or deep learning approaches (LSTM, BERT) to better capture sarcasm and implicit sentiment.

---

## 👤 Author

**Ishwarya** — [github.com/lokeshadma](https://github.com/lokeshadma)