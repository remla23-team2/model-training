import json
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib
import pickle
import os

def _load_artifacts():
    classifier = joblib.load('output/train/sentiment_model')
    X_test = pickle.load(open('output/preprocess/X_test.pkl', 'rb'))
    y_test = pickle.load(open('output/preprocess/y_test.pkl', 'rb'))
    return classifier, X_test, y_test

def evaluate_model(classifier, X_test, y_test):
    #classifier, X_test, y_test = _load_artifacts()
    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)   
    acc = accuracy_score(y_test, y_pred)
    pickle.dump(cm, open('output/evaluate/cm.pkl', 'wb'))
    json.dump({"accuracy": acc}, open('output/metrics/metrics.json', 'w'))
    return acc, cm

def main():
    evaluate_model()

if __name__ == '__main__':
    os.makedirs("output/evaluate", exist_ok=True)
    os.makedirs("output/metrics", exist_ok=True)
    main()