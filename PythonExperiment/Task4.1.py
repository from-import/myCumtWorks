def create_and_write_integer(filename, value):
    with open(filename, 'w') as file:
        file.write(value)


create_and_write_integer("A.txt", "123456789012345678901234567890")
create_and_write_integer("B.txt", "762346264627233722346264242780")

print("A.txt 和 B.txt 文件已创建并写入大整数。")


def read_big_integer(filename):
    with open(filename, 'r') as file:
        big_integer = file.read().strip()
        return big_integer


def write_big_integer(filename, result):
    with open(filename, 'w') as file:
        file.write(result)


def add_big_integers(file_a, file_b, file_c):
    # 读取两个大整数
    num_a = read_big_integer(file_a)
    num_b = read_big_integer(file_b)

    # 将字符串形式的大整数转换为整数类型，然后相加
    sum_result = int(num_a) + int(num_b)

    # 将结果转换为字符串并写入文件
    write_big_integer(file_c, str(sum_result))
    print(f"大整数相加完成。结果已写入 {file_c} 文件。")


# 主程序入口
if __name__ == "__main__":
    add_big_integers("A.txt", "B.txt", "C.txt")
