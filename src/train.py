"""
Module for training a sentiment analysis model with fetched and preprocessed data.
"""

import os
import pickle
from sklearn.naive_bayes import GaussianNB
import joblib


def _load_data():
    """
    Load the preprocessed training data.
    """ 

    with open("output/preprocess/X_train.pkl", "rb") as x_train_file:
        X_train = pickle.load(x_train_file)
    with open("output/preprocess/y_train.pkl", "rb") as y_train_file:
        y_train = pickle.load(y_train_file)
    return X_train, y_train


def train(X_train=None, y_train=None):
    """
    Train the model with the given training data.
    """
    
    if X_train is None or y_train is None:
        X_train, y_train = _load_data()
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # Save the trained model and the CountVectorizer
    with open("output/train/sentiment_model", "wb") as model_file:
        joblib.dump(classifier, model_file)

    return classifier


def main():
    train()


if __name__ == "__main__":
    os.makedirs("output/train", exist_ok=True)
    main()
