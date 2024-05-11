import numpy as np

def average(arr):
    unique, counts = np.unique(arr, return_counts=True)
    max_count_index = np.argmax(counts)
    return unique[max_count_index]

def format_number_and_round_numpy(number):
    # if isinstance(number, int) or isinstance(number, np.int32):
    #     return np.int_(number)
    # elif isinstance(number, float):
    #     return np.float_(round(number, 2))
    # else:
    #     print("oX:", number, "type:", type(number))
    #     raise ValueError("Invalid number type")
    if isinstance(number, np.int32) or isinstance(number, int):
        return np.int_(number)
    elif isinstance(number, float):
        return np.float_(round(number, 2))
    elif isinstance(number, str):
        # Coba untuk mengonversi string menjadi float
        try:
            float_value = float(number)
            rounded_value = round(float_value, 2)
            return np.float_(rounded_value)
        except ValueError:
            # Jika gagal mengonversi, kembalikan pesan kesalahan
            raise ValueError("Invalid string format")
    else:
        print("oX:", number, "type:", type(number))
        raise ValueError("Invalid number type")
    
def format_number_and_round(number):
    # Jika number merupakan bilangan bulat, hapus desimal
    if number.is_integer():
        return int(number)
    # Jika number memiliki desimal, bulatkan ke dua angka di belakang koma
    else:
        return float(round(number, 3))
    # else:
    #     raise ValueError("Invalid number type")