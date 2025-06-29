import pandas as pd
from preprocess import preprocess_text
def load_imdb_from_csv(filepath):
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        print('File not found')
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def data_preparation(data_filepath, processed = False):
    data_df = load_imdb_from_csv(data_filepath)
    if not processed:
        data_df['processed_review'] = data_df['review'].progress_apply(preprocess_text)
    label_mapping = {
        'negative': 0,
        'positive': 1
    }
    data_df['labels'] = data_df['sentiment'].map(label_mapping)
    return data_df