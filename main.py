from src.getdata import get_dataset
from src.preprocessing import pre_process
from src.train import train
from src.evaluate import evaluate

if __name__ == '__main__':
    dataset = get_dataset('data/input/a1_RestaurantReviews_HistoricDump.tsv')
    X_train, X_test, y_train, y_test = pre_process(dataset)

    classifier = train(X_train, y_train)
    acc, cm = evaluate(classifier, X_test, y_test)
