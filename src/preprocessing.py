import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem.porter import PorterStemmer

_stemmer = PorterStemmer()

def clean_text(text):
    words = re.findall(r'[a-z0-9]+', str(text).lower())
    return ' '.join(_stemmer.stem(w) for w in words if w not in ENGLISH_STOP_WORDS)
