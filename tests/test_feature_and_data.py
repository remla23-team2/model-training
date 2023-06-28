"""
Test the 'Feature and Data' angle""
"""
import os
import psutil
from src.getdata import get_dataset

def test_memory_usage():
    """
    Test the difference in memory usage before and after getting the data
    """
    current_memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2

    # Get the data
    get_dataset()

    new_memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2

    # Assert that the difference is less than 10 MB
    assert (new_memory_usage - current_memory_usage) < 10
