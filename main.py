from src.getdata import get_dataset
from src.preprocessing import pre_process
from src.train import train
from src.evaluate import evaluate_model


def main(seed):
    get_dataset()
    pre_process(seed)
    train()
    acc, cm = evaluate_model()
    return acc, cm

    # Show the cm in a nice format


if __name__ == "__main__":
    seed = 10
    acc, cm = main(seed)
    print(acc, cm)
