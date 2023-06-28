"""
Module to run the ML Pipeline
"""
from src.getdata import get_dataset
from src.preprocessing import pre_process
from src.train import train
from src.evaluate import evaluate_model


def main():
    """
    Run the ML pipeline and return the model accuracy and confusion matrix

    Returns:
        float, np.array: accuracy and confusion matrix
    """
    get_dataset()
    pre_process()
    train()
    acc, confusion_matrix = evaluate_model()
    return acc, confusion_matrix

if __name__ == "__main__":
    acc, cm = main()
    print(acc, cm)
