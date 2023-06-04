from src.getdata import get_dataset
from src.preprocessing import pre_process
from src.train import train
from src.evaluate import evaluate_model

def main(seed):
    dataset = get_dataset('data/input/a1_RestaurantReviews_HistoricDump.tsv')
    X_train, X_test, y_train, y_test = pre_process(dataset,seed)

    classifier = train(X_train, y_train)
    acc, cm = evaluate_model(classifier, X_test, y_test)
    
    
    # Show the cm in a nice format
    print(cm)

    return acc

if __name__ == '__main__':
    seed = 10
    main(seed)
