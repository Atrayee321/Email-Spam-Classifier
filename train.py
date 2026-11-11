"""Train a TF-IDF + Multinomial Naive Bayes classifier on Kaggle spam.csv."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from src.preprocessing import clean_text

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'spam.csv'
if not DATA.exists():
    raise SystemExit('Missing data/spam.csv. Download the Kaggle dataset and place spam.csv in data/.')
df = pd.read_csv(DATA, encoding='latin-1', usecols=['v1','v2']).dropna().drop_duplicates()
df = df[df['v1'].isin(['ham','spam'])]
X = df['v2'].astype(str)
y = (df['v1'] == 'spam').astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = Pipeline([
    ('tfidf', TfidfVectorizer(preprocessor=clean_text, lowercase=False, max_features=3000)),
    ('classifier', MultinomialNB(alpha=0.3))
])
model.fit(X_train, y_train)
pred = model.predict(X_test)
metrics = {'samples': len(df), 'test_samples': len(X_test), 'accuracy': accuracy_score(y_test,pred), 'spam_precision': precision_score(y_test,pred,zero_division=0), 'spam_recall': recall_score(y_test,pred,zero_division=0), 'spam_f1': f1_score(y_test,pred,zero_division=0), 'confusion_matrix': confusion_matrix(y_test,pred).tolist()}
(ROOT/'models').mkdir(exist_ok=True)
joblib.dump(model, ROOT/'models'/'spam_pipeline.joblib')
(ROOT/'models'/'metrics.json').write_text(json.dumps(metrics,indent=2))
print(json.dumps(metrics,indent=2))
print('Saved models/spam_pipeline.joblib')
