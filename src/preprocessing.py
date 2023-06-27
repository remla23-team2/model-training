"""
Module for preprocessing the fetched data
"""

import os
import re
import pickle
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')


def _load_data():
    reviews = pd.read_csv("output/getdata/data.tsv", delimiter='\t', quoting=3)
    return reviews

def process_review(review: str):
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    return review

'''
def pre_process(rs=42):
    dataset = _load_data()
    corpus = []
    for i in range(0, len(dataset)):
        processed_review = process_review(dataset['Review'][i])
        corpus.append(processed_review)
'''
def pre_process(seed, dataset=None):
    if dataset is None:
        dataset = _load_data()
    corpus = []
    for i in range(0, len(dataset)):
        processed_review = process_review(dataset['Review'][i])
        corpus.append(processed_review)

    cv = CountVectorizer(max_features=100)
    data_x = cv.fit_transform(corpus).toarray()
    data_y = dataset.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(
        data_x, data_y, test_size=0.20, random_state=seed)

    # Save the CountVectorizer
    pickle.dump(cv, open('output/preprocess/model.pkl', "wb"))

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
    seed = 42
    pre_process(seed)

if __name__ == '__main__':
    os.makedirs("output/preprocess", exist_ok=True)
    main()    