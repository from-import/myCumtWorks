def buy_chickens(A, B, C, X):
    solutions = []
    for cock in range(X + 1):
        for hen in range(X + 1 - cock):
            chicks = X - cock - hen
            if chicks % C == 0 and (cock * A + hen * B + chicks // C) == X:
                solutions.append((cock, hen, chicks))
    return solutions

# 测试函数
A = 1
B = 2
C = 3
X = 100
solutions = buy_chickens(A, B, C, X)
for solution in solutions:
    print(f"公鸡: {solution[0]} 只, 母鸡: {solution[1]} 只, 小鸡: {solution[2]} 只 , 共 {A*solution[0] + B*solution[1] + solution[2]/C} 元")