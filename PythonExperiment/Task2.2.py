def generate_fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [1]
    elif n == 2:
        return [1, 1]

    fibonacci_sequence = [1, 1]
    for i in range(2, n):
        next_value = fibonacci_sequence[-1] + fibonacci_sequence[-2]
        fibonacci_sequence.append(next_value)

    return fibonacci_sequence


n = 10
fibonacci_sequence = generate_fibonacci(n)
print(f"前 {n} 项斐波那契数列: {fibonacci_sequence}")