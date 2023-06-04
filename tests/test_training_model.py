import pytest 
import random

from main import main



def test_model_rubustness():
    seed1 = random.randint(0, 100)
    seed2 = random.randint(0, 100)
    acc1 = main(seed1)
    acc2 = main(seed2)
    difference = abs(acc1 - acc2)
    print(difference)
    assert difference <= 0.1

