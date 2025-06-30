
import pandas as pd
from data import data_preparation
from embedding import bow_vectorizer
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    #Load data
    X_train_bow, X_val_bow, X_test_bow, y_train, y_val, y_test = data_preparation('data/processed_imdb.csv', 
                                                                                  processed=True)
    
    X_train_tfidf, X_val_tfidf, X_test_tfidf, _, _, _ = data_preparation('data/processed_imdb.csv', 
                                                                         processed=True, 
                                                                         _vectorizer = 'tfidf')
    # print(X_train.shape, X_val.shape, X_test.shape)

    print('BoW: ', X_train_bow[0])
    print('Tf-idf: ', X_train_tfidf[0])
