# from itertools import combinations

# # Data
# data = [
#     [1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]
# ]

# # Mencetak kombinasi
# h = 0
# for i in range(len(data)):
#     for j in range(i + 1, len(data)):
#         h = h+1
#         print(f"kombinasi {h} {data[i]} {data[j]}")

# Data
data = [
    [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]],
    [[16, 17, 18], [19, 20, 21], [22, 23, 24], [25, 26, 27], [28, 29, 30]],
]

# Mencetak kombinasi
print("Kombinasi")
combination_counter = 1

# Mencetak kombinasi untuk setiap baris data
for row_idx, row in enumerate(data):
    print(f"---Row {row_idx}---")
    for i, sublist1 in enumerate(row):
        for j, sublist2 in enumerate(row):
            if j > i:
                print(f"{combination_counter} Antara {chr(65 + i)} dan {chr(65 + j)}: {sublist1} dan {sublist2}")
                combination_counter += 1