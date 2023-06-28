"""
Test the 'Monitoring' angle
"""
import time

from src.train import train

def test_training_time():
    """
    Test that training takes less than 20 seconds
    """
    current_time = time.time()

    # Train the model
    train()

    new_time = time.time()

    # Check whether training took less than 20 seconds
    assert new_time - current_time < 20
