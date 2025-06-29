# import tensorflow as tf
# from tensorflow.keras.datasets import imdb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
tqdm.pandas()

from collections import Counter

import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from preprocess import preprocess_text
from eda import *

# def load_imdb_from_keras():
#     vocab_size = 10000
#     return imdb.load_data(num_words=vocab_size)
stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def load_imdb_from_csv(filepath):
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        print('File not found')
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    # data_df = load_imdb_from_csv('imdb.csv)
    # imdb_eda(data_df)
    # data_df['processed_review'] = data_df['review'].progress_apply(preprocess_text)

    # output_csv_dir = 'processed_imdb.csv'
    # data_df.to_csv(output_csv_dir, index=False, encoding='utf-8')

    processed_data_df = load_imdb_from_csv('data/processed_imdb.csv')
    print(processed_data_df.head())

    print('*'*100)

    # processed_imdb_eda(processed_data_df)

