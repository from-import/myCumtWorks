def is_armstrong_number(n, base):
    digits = [int(digit) for digit in str(n)]
    sum_of_powers = sum(digit ** base for digit in digits)
    return sum_of_powers == n

def find_armstrong_numbers(limit):
    armstrong_numbers = []
    for num in range(limit + 1):
        num_digits = len(str(num))
        if is_armstrong_number(num, num_digits):
            armstrong_numbers.append(num)
    return armstrong_numbers

def check_if_armstrong_number(k):
    num_digits = len(str(k))
    return is_armstrong_number(k, num_digits)

# 查找0到10000之间的自幂数
armstrong_numbers = find_armstrong_numbers(10000)
print("0到10000之间的自幂数有：", armstrong_numbers)

# 用户输入
k = int(input("请输入一个整数k："))
if check_if_armstrong_number(k):
    print(f"{k} 是一个自幂数。")
else:
    print(f"{k} 不是一个自幂数。")
