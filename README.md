# 🛡️ SpamShield — Email & SMS Spam Classifier

A Streamlit machine-learning application that predicts whether a text message is **spam** or **legitimate (ham)** using TF-IDF and Multinomial Naive Bayes.

> **🚀 Live demo:** [Try SpamShield on Streamlit](https://email-spam-classifier-humva2iyr278fyi839ud9c.streamlit.app/)

## Features
- Clean, responsive dark-themed Streamlit interface
- Single-message classification with model-estimated spam probability
- Reproducible training script and held-out evaluation metrics
- TF-IDF text features, stopword removal, Porter stemming, Naive Bayes
- No paid APIs required

## Dataset
[SMS Spam Collection on Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset). Download `spam.csv` and place it at `data/spam.csv`. Columns `v1` and `v2` contain labels and messages respectively. This is an **SMS dataset**, so accuracy on real email is not established.

## Quick start
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py
streamlit run app.py
```

## Project structure
```
Email-Spam-Classifier/
├── app.py
├── train.py
├── src/
│   └── preprocessing.py
├── data/
│   └── spam.csv            # Add Kaggle CSV here
├── models/
│   ├── spam_pipeline.joblib # Created by training
│   └── metrics.json         # Created by training
├── .streamlit/config.toml
├── requirements.txt
└── README.md
```

## Deploy to Streamlit Community Cloud
1. Train locally or in Colab and commit `models/spam_pipeline.joblib` and `models/metrics.json` to GitHub.
2. Push project files to a public GitHub repository. `data/spam.csv` is not needed for inference after training.
3. Go to https://share.streamlit.io and create an app from your repository.
4. Choose branch `main`, entry point `app.py`, and deploy.

## How it works
1. Normalize text and tokenize words.
2. Remove English stopwords and stem terms.
3. Convert messages to TF-IDF vectors.
4. Train Multinomial Naive Bayes on a stratified 80/20 train/test split.
5. Predict spam/ham and show estimated spam probability.

## Evaluation
Run `python train.py` to generate **actual** accuracy, spam precision, spam recall, F1, and confusion matrix in `models/metrics.json`. No evaluation scores are claimed before training.

## Limitations
A short-message classifier is not a comprehensive phishing or malware detector. Predictions may be unreliable on long emails, unseen languages, and changing spam tactics. Probabilities may be uncalibrated. Never load untrusted `.joblib` model files.

## Attribution
Inspired by [CampusX's SMS Spam Classifier](https://github.com/campusx-official/sms-spam-classifier). Dataset credited to the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) via Kaggle.
