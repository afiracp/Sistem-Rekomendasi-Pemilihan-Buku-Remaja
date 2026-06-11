import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# download sekali saja
nltk.download('stopwords')

# inisialisasi
stop_words = set(stopwords.words('indonesian'))
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# 1. Case Folding
def case_folding(text):
    if pd.isna(text):
        return ""
    return str(text).lower()

# 2. Tokenizing + Cleaning
def tokenizing(text):
    text = text.replace("/", " ")
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = text.split()
    return tokens

# 3. Stopword Removal
def filtering(tokens):
    return [word for word in tokens if word not in stop_words]

# 4. Stemming
def stemming(tokens):
    return [stemmer.stem(word) for word in tokens]

# Gabungan (opsional)
def preprocess(text):
    text = case_folding(text)
    tokens = tokenizing(text)
    tokens = filtering(tokens)
    tokens = stemming(tokens)
    return " ".join(tokens)