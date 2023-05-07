import argparse
import numpy as np
import pandas as pd
import os
import re
import pickle
import joblib

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

#Load data
def get_dataset(filename):
    dataset = pd.read_csv(filename, delimiter='\t', quoting=3)
    return dataset

# Pre-processing 
def pre_processing(dataset):
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')

    corpus = []
    for i in range(0, 900):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        corpus.append(review)

    return corpus

# Training
def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    return classifier, X_test, y_test

# Evaluation 
def evaluate_model(classifier, X_test, y_test):
    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(accuracy_score(y_test, y_pred))


def main():
    dataset = get_dataset('data/a1_RestaurantReviews_HistoricDump.tsv')
    corpus = pre_processing(dataset)

    cv = CountVectorizer(max_features=1420)
    X = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:, -1].values

    classifier, X_test, y_test = train(X, y)
    evaluate_model(classifier, X_test, y_test)

    # Save the trained model and the CountVectorizer
    joblib.dump(classifier, 'data/c2_Classifier_Sentiment_Model')
    pickle.dump(cv, open('data/c1_BoW_Sentiment_Model.pkl', "wb"))


if __name__ == '__main__':
    main()
