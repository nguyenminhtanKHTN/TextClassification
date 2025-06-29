
import pandas as pd
from data import data_preparation
from embedding import bow_vectorizer
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    #Load data
    X_train, X_val, X_test, y_train, y_val, y_test = data_preparation('data/processed_imdb.csv', processed=True)
    print(X_train.shape, X_val.shape, X_test.shape)

