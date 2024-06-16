def generate_special_matrix(k):
    if k % 2 == 0:
        raise ValueError("k must be an odd number")

    matrix = [[0] * k for _ in range(k)]
    num = 1

    for i in range(k):
        if i % 2 == 0:
            for j in range(k):
                matrix[i][j] = num
                num += 1
        else:
            for j in range(k - 1, -1, -1):
                matrix[i][j] = num
                num += 1

    return matrix

# 示例
k = 3
special_matrix = generate_special_matrix(k)
for row in special_matrix:
    print(row)
