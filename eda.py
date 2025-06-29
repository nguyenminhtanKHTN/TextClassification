import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
tqdm.pandas()

from collections import Counter

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

def imdb_eda(data_df):
    print('Samples')
    print(data_df.head())
    print('*'*100)
    data_df.info()
    print('*'*100)
    print("Total missing values")
    print(data_df.isnull().sum())
    print('*'*100)
    print('Labels frequency')
    print(data_df['sentiment'].value_counts())
    print('*'*100)
    data_df['txt_len'] = data_df['review'].apply(len)
    print('Text length')
    print(data_df['txt_len'].describe())
    print('*'*100)
    # Lấy 5 mẫu ngẫu nhiên
    for i, row in data_df.sample(5).iterrows():
        print(f"Nhãn: {row['sentiment']}")
        print(f"Văn bản: {row['review']}")
        print("-" * 50)

    # plt.figure(figsize=(10,5))
    # sns.histplot(data=data_df['txt_len'], bins=50, kde=True)
    # plt.show()

    
    # plt.figure(figsize=(7,5))
    # sns.countplot(data=data_df, x='sentiment')
    # plt.title('Sentiment labels dist')
    # plt.xlabel('Label')
    # plt.ylabel('Count')
    # plt.show()

def processed_imdb_eda(data_df):
    copied_data_df = data_df.copy()
    copied_data_df['processed_review_len'] = copied_data_df['processed_review'].apply(lambda x: len(str(x).split()))
    all_words = " ".join(copied_data_df['processed_review']).split()
    def text_length():
        print(copied_data_df['processed_review_len'].describe())

        plt.figure(figsize=(10,6))
        sns.histplot(data=copied_data_df['processed_review_len'], bins=50, kde=True)
        plt.show()
    def empty_text():
        empty_text = copied_data_df[copied_data_df['processed_review_len'] == 0]
        print('Number of empty texts : ', len(empty_text))
    def top_n_word(n):
        word_counts = Counter(all_words)

        most_common_words = word_counts.most_common(n)

        print(f'{n} most common words')
        for word, count in most_common_words:
            print(f'{word} - {count}')

    # text_length()
    # empty_text()
    top_n_word(30)
