import numpy as np

def average(arr):
    unique, counts = np.unique(arr, return_counts=True)
    max_count_index = np.argmax(counts)
    return unique[max_count_index]

def format_number_and_round_numpy(number):
    if isinstance(number, np.int32) or isinstance(number, int):
        return np.int_(number)
    elif isinstance(number, float):
        return np.float_(round(number, 3))
    else:
        raise ValueError("Invalid number type")
    
def format_number_and_round(number):
    # Jika number merupakan bilangan bulat, hapus desimal
    if number.is_integer():
        return int(number)
    elif isinstance(number, float):
        return float(round(number, 3))
    else:
        raise ValueError("Invalid number type")