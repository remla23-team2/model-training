"""
Module for preprocessing the fetched data
"""

import os
import pickle
import pandas as pd
import nltk
from lib_preproc import process_review
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

nltk.download('stopwords')

ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')


def _load_data():
    """
    Load data from data.tsv.
    """
    reviews = pd.read_csv("output/getdata/data.tsv", delimiter='\t',
                          quoting=3, dtype={'Review': 'str', 'Liked': 'int'})
    reviews = reviews[['Review', 'Liked']]
    return reviews

def pre_process(dataset=None, seed=42):
    """
    Preprocess the data.
    """
    if dataset is None:
        dataset = _load_data()
    corpus = [process_review(review) for review in dataset['Review']]

    vectorizer = CountVectorizer(max_features=100)
    data_x = vectorizer.fit_transform(corpus).toarray()
    data_y = dataset.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(
        data_x, data_y, test_size=0.20, random_state=seed)

    # Save the CountVectorizer
    with open('output/preprocess/vectorizer.pkl', "wb") as model_file:
        pickle.dump(vectorizer, model_file)

    # Save sets
    with open('output/preprocess/X_train.pkl', "wb") as x_train_file:
        pickle.dump(X_train, x_train_file)
    with open('output/preprocess/X_test.pkl', "wb") as x_test_file:
        pickle.dump(X_test, x_test_file)
    with open('output/preprocess/y_train.pkl', "wb") as y_train_file:
        pickle.dump(y_train, y_train_file)
    with open('output/preprocess/y_test.pkl', "wb") as y_test_file:
        pickle.dump(y_test, y_test_file)

    return X_train, X_test, y_train, y_test

def main():
    """
    Main function
    """
    pre_process()

if __name__ == '__main__':
    os.makedirs("output/preprocess", exist_ok=True)
    main()
