from src.getdata import get_dataset
from src.preprocessing import pre_process
from src.train import train
from src.evaluate import evaluate_model

if __name__ == '__main__':
    dataset = get_dataset()
    X_train, X_test, y_train, y_test = pre_process()
    classifier = train()
    acc, cm = evaluate_model()
    
    # Show the cm in a nice format
    print(acc, cm)
