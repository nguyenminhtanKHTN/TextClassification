
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
tqdm.pandas()
from embedding import bow_vectorizer
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

    X_bow, vectorizer_bow = bow_vectorizer(processed_data_df['processed_review'].to_list())

    print(list(vectorizer_bow.vocabulary_.items())[:10])

    print(X_bow[0].toarray())



