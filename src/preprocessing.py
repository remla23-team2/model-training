import re
import pickle
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os
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
def pre_process(dataset: pd.DataFrame, seed):
    corpus = []
    for i in range(0, len(dataset)):
        processed_review = process_review(dataset['Review'][i])
        corpus.append(processed_review)
        
    cv = CountVectorizer(max_features=100)
    X = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:, -1].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)
    
    # Save the CountVectorizer
    pickle.dump(cv, open('output/preprocess/model.pkl', "wb"))
    # Save sets
    pickle.dump(X_train, open('output/preprocess/X_train.pkl', "wb"))
    pickle.dump(X_test, open('output/preprocess/X_test.pkl', "wb"))
    pickle.dump(y_train, open('output/preprocess/y_train.pkl', "wb"))
    pickle.dump(y_test, open('output/preprocess/y_test.pkl', "wb"))
    
    return X_train, X_test, y_train, y_test

def main():
    pre_process()

if __name__ == '__main__':
    os.makedirs("output/preprocess", exist_ok=True)
    main()

    