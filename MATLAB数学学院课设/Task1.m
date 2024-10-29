function main()
    % 矩阵A和向量b
    A = [1, 2, -1; 2, 1, -2; -3, 1, 1;];
    b = [3; 3; -6];

    try
        % 进行LU分解
        [L, U] = LU_factorization(A);

        % 显示L和U矩阵
        disp('矩阵L:');
        disp(L);
        disp('矩阵U:');
        disp(U);

        % 使用LU分解求解线性方程组
        x = back_substitution_two(L, U, b);

        % MATLAB直接求解
        x_direct = A \ b;

        % 输出结果
        disp('输入矩阵A:');
        disp(A);
        disp('输入向量b:');
        disp(b);

        disp('LU分解求解结果:');
        disp(x);

        disp('MATLAB直接求解结果:');
        disp(x_direct);

        % 比较结果
        disp('解向量x与MATLAB直接求解结果的差异:');
        disp(x - x_direct);
    catch ME
        % 输出错误信息
        fprintf('错误: %s\n', ME.message);
    end
end

function [L, U] = LU_factorization(A)
    % 检查A是否是方阵
    [m, n] = size(A);
    if m ~= n
        error('矩阵不是方阵');
    end

    % 初始化L和U
    L = eye(n);
    U = zeros(n);

    for i = 1:n
        % 检查主元是否为0
        if A(i, i) == 0
            error('主元为0，矩阵不可逆');
        end

        % 计算U的上三角部分
        for j = i:n
            U(i, j) = A(i, j) - L(i, 1:i-1) * U(1:i-1, j);
        end

        % 计算L的下三角部分
        for j = i+1:n
            L(j, i) = (A(j, i) - L(j, 1:i-1) * U(1:i-1, i)) / U(i, i);
        end
    end
end

function [X] = back_substitution_two(L, U, b)
    % Ly = b, Ux = y
    y = forward_substitution(L, b);
    X = back_substitution(U, y);
end

function y = forward_substitution(L, b)
    % 前向替换解决Ly = b
    n = length(b);
    y = zeros(n, 1);
    for i = 1:n
        y(i) = (b(i) - L(i, 1:i-1) * y(1:i-1)) / L(i, i);
    end
end

function x = back_substitution(U, y)
    % 回代解决Ux = y
    n = length(y);
    x = zeros(n, 1);
    for i = n:-1:1
        x(i) = (y(i) - U(i, i+1:end) * x(i+1:end)) / U(i, i);
    end
end

main();
