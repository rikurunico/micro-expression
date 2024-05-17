# # from itertools import combinations

# # # Data
# # data = [
# #     [1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]
# # ]

# # # Mencetak kombinasi
# # h = 0
# # for i in range(len(data)):
# #     for j in range(i + 1, len(data)):
# #         h = h+1
# #         print(f"kombinasi {h} {data[i]} {data[j]}")

# # Data
# data = [
#     [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]],
#     [[16, 17, 18], [19, 20, 21], [22, 23, 24], [25, 26, 27], [28, 29, 30]],
# ]

# # Mencetak kombinasi
# print("Kombinasi")
# combination_counter = 1

# # Mencetak kombinasi untuk setiap baris data
# for row_idx, row in enumerate(data):
#     print(f"---Row {row_idx}---")
#     for i, sublist1 in enumerate(row):
#         for j, sublist2 in enumerate(row):
#             if j > i:
#                 print(f"{combination_counter} Antara {chr(65 + i)} dan {chr(65 + j)}: {sublist1} dan {sublist2}")
#                 combination_counter += 1
# Function to format a value based on its data type
# def format_value(value):
#     if isinstance(value, int):
#         return int(value)
#     elif isinstance(value, float):
#         return f"{value:.1f}"  # Format to 1 decimal place
#     else:
#         return value  # Handle strings as is
    
# print(format_value(1.32323))
# print(format_value(16))
# print(format_value(0))
# print(format_value(4.5634))
# print(format_value(2.9723))
# print(format_value(2.9236))
import numpy as np

# def format_number_and_round_numpy(number):
#     if isinstance(number, np.int32):
#         return np.int_(number)
#     elif isinstance(number, float):
#         # Mengecek apakah notasi ilmiah, jika ya, format dengan dua angka di belakang koma
#         number = np.float_(round(number, 2))
#         print(number)
#         if len(str(number)) and 'e' in '{:e}'.format(number):
#             print("masuk sini")
#             # Konversi number menjadi string
#             num_str = str(number)
#             # Mengecek apakah angka negatif
#             if num_str[0] == '-':
#                 # Ambil 4 angka pertama setelah tanda minus
#                 first_four_digits = num_str[1:5]
#                 # Ubah ke dalam float dengan format satu angka di depan koma
#                 formatted_number = float('-' + first_four_digits[0] + '.' + first_four_digits[1:])
#             else:
#                 # Ambil 4 angka pertama dari depan
#                 first_four_digits = num_str[:4]
#                 # Ubah ke dalam float dengan format satu angka di depan koma
#                 formatted_number = float(first_four_digits[0] + '.' + first_four_digits[1:])
#                 # Bulatkan menjadi dua angka di belakang koma
#             number = np.float_(round(formatted_number, 2))
#         return np.float_(round(number, 2))
#     else:
#         raise ValueError("Invalid number type")

# # Contoh penggunaan dengan lima data sampel
# data_samples = [-2.14748e+09, 6.02214076123e+12, -12345.6789, 987654.321]
# for angka in data_samples:
#     print("Angka:", angka)
#     angka_hasil = format_number_and_round_numpy(angka)
#     print("Hasil konversi:", angka_hasil)
#     print()

from helper.helper import average

data = [
    -1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 
    -1, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    -1, 0, -1, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 
    0, 0, 0, 0, 0, -2, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 3, 
    0, 0, 0, 0, 0, 0, 0, 0, -1]

print(average(data))