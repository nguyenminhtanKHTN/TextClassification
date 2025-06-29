
import pandas as pd
from data import data_preparation
from embedding import bow_vectorizer

if __name__ == "__main__":
    data_df = data_preparation('data/processed_imdb.csv', processed=True)
    print(data_df.head())
    print('*'*100)

    X_bow, vectorizer_bow = bow_vectorizer(data_df['processed_review'].to_list())
    labels = data_df['labels']

    print(X_bow.shape)
    print(labels.shape)


