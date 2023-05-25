import joblib
import pickle

from sklearn.naive_bayes import GaussianNB

def _load_data():
    X_train = pickle.load(open("output/preprocess/X_train.pkl", 'rb'))
    y_train = pickle.load(open("output/preprocess/y_train.pkl", 'rb'))
    return X_train, y_train

def train():
    classifier = GaussianNB()
    X_train, y_train = _load_data()
    classifier.fit(X_train, y_train)

    # Save the trained model and the CountVectorizer
    joblib.dump(classifier, open('output/train/sentiment_model', 'wb'))
    
    return classifier

def main():
    train()

if __name__ =='__main__':
    main()