import pandas as pd
from preprocess import preprocess_text
from embedding import bow_vectorizer
from sklearn.model_selection import train_test_split

def load_imdb_from_csv(filepath):
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        print('File not found')
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def data_preparation(data_filepath, processed = False, _vectorizer = 'bow', test_size = 0.2, val_size = 0.2):
    data_df = load_imdb_from_csv(data_filepath)
    if not processed:
        data_df['processed_review'] = data_df['review'].progress_apply(preprocess_text)
    label_mapping = {
        'negative': 0,
        'positive': 1
    }
    data_df['labels'] = data_df['sentiment'].map(label_mapping)

    #Vectorize text
    if _vectorizer == 'bow':
        X, vectorizer = bow_vectorizer(data_df['processed_review'].to_list())
    labels = data_df['labels']

    #Split data
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, labels,
                                                                test_size=test_size,
                                                                random_state=42,
                                                                stratify=labels)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                      test_size = val_size / (1.0 - test_size),
                                                      random_state=42,
                                                      stratify=y_train_val)
    return X_train, X_val, X_test, y_train, y_val, y_test