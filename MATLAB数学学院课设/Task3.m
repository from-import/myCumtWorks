function [L, U, P] = random_pivot_LU(A)
% 随机列选主元 LU分解
% Input: 矩阵 A
% Output: L, U, P
%   L: 下三角矩阵
%   U: 上三角矩阵
%   P: 行交换矩阵

[n, n2] = size(A);
if n ~= n2
    error('矩阵必须是方阵');
end

P = eye(n);

for k = 1:n-1
    % 随机选择一个列
    col_indices = k:n;
    rand_col = col_indices(randi(length(col_indices)));

    % 选择该列中的最大元素作为主元
    [~, i_max] = max(abs(A(k:n, rand_col)));
    i_max = i_max + k - 1;

    % 交换行
    if i_max ~= k
        A([k, i_max], :) = A([i_max, k], :);
        P([k, i_max], :) = P([i_max, k], :);
    end

    % 高斯消去
    for i = k+1:n
        A(i, k) = A(i, k) / A(k, k);
        A(i, k+1:n) = A(i, k+1:n) - A(i, k) * A(k, k+1:n);
    end
end

L = tril(A, -1) + eye(n);
U = triu(A);

end

function [L, U, P, Q] = random_subset_pivot_LU(A, r)
% 随机列子集选主元 LU分解
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
    % 随机选择 r 列
    col_indices = k:n;
    rand_cols = col_indices(randperm(length(col_indices), r));

    % 选择这些列中的最大元素作为主元
    [~, idx] = max(abs(A(k:n, rand_cols)), [], 'all', 'linear');
    [i_max, j_max] = ind2sub([n-k+1, r], idx);
    i_max = i_max + k - 1;
    j_max = rand_cols(j_max);

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

    % 高斯消去
    for i = k+1:n
        A(i, k) = A(i, k) / A(k, k);
        A(i, k+1:n) = A(i, k+1:n) - A(i, k) * A(k, k+1:n);
    end
end

L = tril(A, -1) + eye(n);
U = triu(A);

end

% 测试代码
clc; clear all;

% 定义矩阵A和向量b
A = [1 2 -1; 2 1 -2; -3 1 1];
b = [3; 3; -6];

% 随机列选主元
[L_rand, U_rand, P_rand] = random_pivot_LU(A);
X_rand = back_substitution_two(L_rand, U_rand, P_rand * b);

disp('随机列选主元 LU分解结果:');
disp(X_rand);

% 随机列子集选主元
r = 2; % 子集大小
[L_sub, U_sub, P_sub, Q_sub] = random_subset_pivot_LU(A, r);
X_sub = back_substitution_two(L_sub, U_sub, P_sub * b);

disp(['随机列子集选主元 (r=', num2str(r), ') LU分解结果:']);
disp(X_sub);

% MATLAB 直接求解结果验证
X_direct = A \ b;
disp('MATLAB直接求解结果:');
disp(X_direct);

% 计算增长因子
growth_factor_rand = max(max(abs(U_rand))) / max(max(abs(A)));
growth_factor_sub = max(max(abs(U_sub))) / max(max(abs(A)));

disp('随机列选主元增长因子:');
disp(growth_factor_rand);

disp(['随机列子集选主元 (r=', num2str(r), ') 增长因子:']);
disp(growth_factor_sub);

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
