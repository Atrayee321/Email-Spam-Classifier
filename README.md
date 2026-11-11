<div align="center">

# 🛡️ SpamShield
### Email & SMS Spam Classifier

**An end-to-end NLP and machine learning application for classifying text messages as spam or legitimate.**

[![Live App](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge)](https://email-spam-classifier-humva2iyr278fyi839ud9c.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)

[**Try the App**](https://email-spam-classifier-humva2iyr278fyi839ud9c.streamlit.app/) · [**View Source**](https://github.com/Atrayee321/Email-Spam-Classifier) · [**Dataset**](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

</div>

---

## 📌 Overview

SpamShield is a text-classification web application that uses **Natural Language Processing (NLP)**, **TF-IDF vectorization**, and a **Multinomial Naive Bayes** classifier to predict whether an input message is **spam** or **ham** (legitimate).

The model was trained and evaluated using the **SMS Spam Collection** dataset. Although the interface accepts email text, **performance on real email has not been separately validated**.

### ✨ Key Features

- 🧠 **Machine learning inference** — classify user-submitted messages as spam or ham
- 📊 **Spam probability estimate** — view a model-estimated spam probability and progress indicator
- 🎨 **Modern dark-themed UI** — clean, responsive Streamlit interface
- 🔎 **Transparent evaluation** — accuracy, precision, recall, F1 and confusion matrix
- ⚙️ **Reproducible training** — retrain the complete scikit-learn pipeline with one command
- ☁️ **Cloud deployment** — accessible through Streamlit Community Cloud
- 🔑 **No external inference API key** — classification runs using a locally loaded model artifact

## 🌐 Live Application

<div align="center">

### [🚀 Launch SpamShield](https://email-spam-classifier-humva2iyr278fyi839ud9c.streamlit.app/)

**How to use:** Enter or paste a message → click **Analyze message** → review the predicted class and spam probability.

</div>

> **Note:** A classification result is a statistical estimate, not a security guarantee. Do not trust suspicious links or attachments based solely on a ham prediction.

## 📈 Model Performance

Results recorded in [`models/metrics.json`](models/metrics.json) from a held-out **20% stratified test split** after removing duplicate messages.

| Evaluation metric | Score |
|:---|---:|
| **Accuracy** | **98.16%** |
| Spam precision | 97.46% |
| Spam recall | 87.79% |
| Spam F1-score | 92.37% |
| Total unique labeled messages | 5,169 |
| Held-out test messages | 1,034 |

### Confusion Matrix

| Actual \ Predicted | Ham | Spam |
|:---|---:|---:|
| **Ham** | 900 | 3 |
| **Spam** | 16 | 115 |

**Interpretation:** On the held-out SMS test set, the model correctly classified 1,015 messages out of 1,034. It missed 16 spam messages and incorrectly flagged 3 legitimate messages as spam.

> Results are dataset-specific. The displayed spam probability is not guaranteed to be calibrated.

## 🧠 How It Works

```text
                  ┌─────────────────────┐
                  │ User enters message │
                  └──────────┬──────────┘
                             ▼
                  ┌─────────────────────┐
                  │ Text preprocessing  │
                  │ lowercase, tokenize │
                  │ stopwords, stemming │
                  └──────────┬──────────┘
                             ▼
                  ┌─────────────────────┐
                  │ TF-IDF vectorizer   │
                  │ max 3,000 features  │
                  └──────────┬──────────┘
                             ▼
                  ┌─────────────────────┐
                  │ MultinomialNB model │
                  └──────────┬──────────┘
                             ▼
                  ┌─────────────────────┐
                  │ Spam / Ham result   │
                  │ + probability       │
                  └─────────────────────┘
```

**Preprocessing:** Regex-based alphanumeric tokenization, lowercase conversion, English stopword removal, and Porter stemming.

**Feature extraction:** `TfidfVectorizer` converts the processed messages into numerical features.

**Classification:** `MultinomialNB(alpha=0.3)` learns patterns from labeled messages. The complete preprocessing/vectorizer/classifier pipeline is saved as a `joblib` artifact so inference uses the same transformations as training.

## 🛠️ Technology Stack

| Layer | Technology |
|:---|:---|
| Programming language | Python |
| Machine learning | scikit-learn |
| Text preprocessing | NLTK, regex |
| Feature extraction | TF-IDF |
| Classification | Multinomial Naive Bayes |
| Data handling | pandas |
| Model persistence | joblib |
| Web application | Streamlit |
| Hosting | Streamlit Community Cloud |
| Version control | Git & GitHub |
| Automated training | GitHub Actions |

## 📁 Project Structure

```text
Email-Spam-Classifier/
├── .github/
│   └── workflows/
│       └── train-model.yml       # Automated training workflow
├── .streamlit/
│   └── config.toml               # Streamlit theme
├── data/
│   ├── spam.csv                  # Labeled SMS dataset
│   └── README.txt
├── models/
│   ├── spam_pipeline.joblib      # Trained ML pipeline
│   ├── metrics.json              # Test-set evaluation
│   └── README.txt
├── src/
│   └── preprocessing.py          # Text preprocessing
├── app.py                        # Streamlit application
├── train.py                      # Model training and evaluation
├── requirements.txt              # Python dependencies
├── .gitignore
└── README.md
```

## ☁️ Streamlit Deployment

The project is deployed at:

**[email-spam-classifier-humva2iyr278fyi839ud9c.streamlit.app](https://email-spam-classifier-humva2iyr278fyi839ud9c.streamlit.app/)**

To deploy your own version:

1. Fork or clone this repository and push it to GitHub.
2. Confirm that `models/spam_pipeline.joblib` is committed.
3. Open [Streamlit Community Cloud](https://share.streamlit.io/).
4. Choose **Create app** and select your GitHub repository.
5. Set branch to `main` and entry point to `app.py`.
6. Deploy and open the generated public URL.

The included GitHub Actions workflow can regenerate the dataset and trained model when needed.

## 🗂️ Dataset

**Dataset:** [SMS Spam Collection (Kaggle)](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

**Original source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)

The dataset includes English SMS messages labeled:
- `ham` — legitimate messages
- `spam` — unsolicited or spam messages

The training script removes duplicate rows and uses the `v1` label and `v2` message columns.

## ⚠️ Limitations & Responsible Use

- Trained on **SMS**, not a representative corpus of full-length emails.
- Predictions may be less reliable for other languages, unusual formats, or new spam campaigns.
- False negatives occur: the held-out evaluation includes spam messages incorrectly labeled ham.
- Model-estimated probabilities may not be well calibrated.
- This application is **not** a malware scanner, phishing-link analyzer, or replacement for production email security tools.
- Only load model files from trusted sources; Python pickle-based artifacts can execute code when loaded.


## 🙏 Acknowledgements
- [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) — dataset
- [Streamlit](https://streamlit.io/) and [scikit-learn](https://scikit-learn.org/) — application and machine learning tools

---

<div align="center">

**Built with Python, NLP, and Machine Learning**

⭐ If you find this project useful, consider starring the repository.

</div>
