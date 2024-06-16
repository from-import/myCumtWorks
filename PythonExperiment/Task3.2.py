class Matrix:
    def __init__(self, matrix):
        self.matrix = matrix  # 初始化矩阵数据
        self.rows = len(matrix)  # 记录矩阵行数
        self.cols = len(matrix[0])  # 记录矩阵列数

    def __str__(self):
        result = ""
        for row in self.matrix:
            result += " ".join(map(str, row)) + "\n"  # 将矩阵转换为字符串形式
        return result.strip()

    def __add__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("矩阵维度必须相同才能相加")  # 检查矩阵维度是否相同

        result = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                row.append(self.matrix[i][j] + other.matrix[i][j])
            result.append(row)

        return Matrix(result)

    def __sub__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("矩阵维度必须相同才能相减")  # 检查矩阵维度是否相同

        result = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                row.append(self.matrix[i][j] - other.matrix[i][j])
            result.append(row)

        return Matrix(result)

    def inverse(self):
        if self.rows != self.cols:
            raise ValueError("矩阵必须是方阵才能求逆")  # 检查矩阵是否为方阵

        # 创建单位矩阵
        identity = [[1 if i == j else 0 for j in range(self.cols)] for i in range(self.rows)]
        augmented_matrix = [row[:] + identity[i][:] for i, row in enumerate(self.matrix)]

        # 高斯消元法求逆矩阵
        for col in range(self.cols):
            # 寻找主元素所在行
            max_row = max(range(col, self.rows), key=lambda i: abs(augmented_matrix[i][col]))
            augmented_matrix[col], augmented_matrix[max_row] = augmented_matrix[max_row], augmented_matrix[col]

            # 主元素归一化
            主元素 = augmented_matrix[col][col]
            if 主元素 == 0:
                raise ValueError("矩阵奇异，无法求逆")
            augmented_matrix[col] = [entry / 主元素 for entry in augmented_matrix[col]]

            # 消元操作，将当前列的其他行消成零
            for row in range(self.rows):
                if row != col:
                    因子 = augmented_matrix[row][col]
                    augmented_matrix[row] = [entry - 因子 * augmented_matrix[col][i] for i, entry in
                                             enumerate(augmented_matrix[row])]

        # 提取逆矩阵
        逆矩阵 = [row[self.cols:] for row in augmented_matrix]
        return Matrix(逆矩阵)


# 测试
if __name__ == "__main__":
    # 测试数据
    matrix1 = Matrix([[1, 2], [3, 4]])
    matrix2 = Matrix([[5, 6], [7, 8]])

    # 测试加法
    print("矩阵加法测试:")
    print(matrix1 + matrix2)

    # 测试减法
    print("\n矩阵减法测试:")
    print(matrix1 - matrix2)

    # 测试求逆
    matrix3 = Matrix([[1, 2], [3, 4]])
    print("\n原始矩阵:")
    print(matrix3)
    print("\n逆矩阵:")
    print(matrix3.inverse())
