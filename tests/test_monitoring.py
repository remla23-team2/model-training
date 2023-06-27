import time

from src.train import train

# Test how long it takes to train the model
def test_training_time():
    current_time = time.time()
    
    # Train the model
    train()
    
    new_time = time.time()
    
    # Check whether training took less than 20 seconds
    assert new_time - current_time < 20