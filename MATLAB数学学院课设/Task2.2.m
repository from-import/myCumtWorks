function [L, U, P, Q] = full_pivot_LU(A)
% 全选主元 LU分解
% Input: 矩阵 A
% Output: L, U, P, Q
%   L: 下三角矩阵
%   U: 上三角矩阵
%   P: 行交换矩阵
%   Q: 列交换矩阵

[n, n2] = size(A);
if n ~= n2
    error('矩阵必须是方阵');
end

P = eye(n);
Q = eye(n);
for k = 1:n-1
    % 选择主元
    [~, idx] = max(abs(A(k:n, k:n)), [], 'all', 'linear');
    [i_max, j_max] = ind2sub([n-k+1, n-k+1], idx);
    i_max = i_max + k - 1;
    j_max = j_max + k - 1;

    % 交换行
    if i_max ~= k
        A([k, i_max], :) = A([i_max, k], :);
        P([k, i_max], :) = P([i_max, k], :);
    end

    % 交换列
    if j_max ~= k
        A(:, [k, j_max]) = A(:, [j_max, k]);
        Q(:, [k, j_max]) = Q(:, [j_max, k]);
    end

    % 消元
    if A(k, k) == 0
        error('矩阵是奇异的');
    end

    for i = k+1:n
        A(i, k) = A(i, k) / A(k, k);
        A(i, k+1:n) = A(i, k+1:n) - A(i, k) * A(k, k+1:n);
    end
end

L = tril(A, -1) + eye(n);
U = triu(A);

end

function x = back_substitution_two(L, U, b)
% Ly = b, Ux = y
% b : 列向量
% x : 解向量

% 前向替换解决Ly = b
n = length(b);
y = zeros(n, 1);
for i = 1:n
    y(i) = (b(i) - L(i, 1:i-1) * y(1:i-1)) / L(i, i);
end

% 回代解决Ux = y
x = zeros(n, 1);
for i = n:-1:1
    x(i) = (y(i) - U(i, i+1:end) * x(i+1:end)) / U(i, i);
end
end

% 测试代码
clc; clear all;
A = [1 2 -1; 2 1 -2; -3 1 1];
b1 = [3 3 -6]';
[L, U, P, Q] = full_pivot_LU(A);

disp('矩阵L:');
disp(L);
disp('矩阵U:');
disp(U);
disp('置换矩阵P:');
disp(P);
disp('置换矩阵Q:');
disp(Q);

X1 = back_substitution_two(L, U, P * b1);
disp('LU分解求解结果:');
disp(X1);

% MATLAB直接求解结果验证
X_direct = A \ b1;
disp('MATLAB直接求解结果:');
disp(X_direct);

% 计算增长因子
growth_factor = max(max(abs(U))) / max(max(abs(A)));
disp('全选主元增长因子:');
disp(growth_factor);
