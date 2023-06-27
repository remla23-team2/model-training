import joblib
import pickle
import os
from sklearn.linear_model import LogisticRegression


def _load_data():
    X_train = pickle.load(open("output/preprocess/X_train.pkl", "rb"))
    y_train = pickle.load(open("output/preprocess/y_train.pkl", "rb"))
    return X_train, y_train


def train(X_train=None, y_train=None):
    if X_train is None or y_train is None:
        X_train, y_train = _load_data()
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    # Save the trained model and the CountVectorizer
    joblib.dump(classifier, open("output/train/sentiment_model", "wb"))

    return classifier


def main():
    train()


if __name__ == "__main__":
    os.makedirs("output/train", exist_ok=True)
    main()
