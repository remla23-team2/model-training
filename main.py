from src.getdata import get_dataset
from src.preprocessing import pre_process
from src.train import train
from src.evaluate import evaluate_model

<<<<<<< HEAD
def main(seed):
    dataset = get_dataset('data/input/a1_RestaurantReviews_HistoricDump.tsv')
    X_train, X_test, y_train, y_test = pre_process(dataset,seed)

    classifier = train(X_train, y_train)
    acc, cm = evaluate_model(classifier, X_test, y_test)
=======
if __name__ == '__main__':
    dataset = get_dataset()
    X_train, X_test, y_train, y_test = pre_process()
    classifier = train()
    acc, cm = evaluate_model()
>>>>>>> 5-a4-ml-config-management
    
    
    # Show the cm in a nice format
<<<<<<< HEAD
    print(cm)

    return acc

if __name__ == '__main__':
    seed = 10
    main(seed)
=======
    print(acc, cm)
>>>>>>> 5-a4-ml-config-management
