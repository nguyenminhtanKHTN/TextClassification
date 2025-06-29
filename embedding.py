from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd

def bow_vectorizer(texts: list[str]):
    vectorizer_bow = CountVectorizer(max_features=5000)
    X_bow = vectorizer_bow.fit_transform(texts)
    return X_bow, vectorizer_bow

