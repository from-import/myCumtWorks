function [L, U, P] = PLU_factorization(A)
% PA = LU分解
% Input: A
% Output: L, U, P
%   Version:            1.0
%   last modified:      09/27/2023

    n = length(A);
    % 第一次行交换
    [~, s] = max(A(1:n, 1)); % s 表示第一列最大元素的位置
    P = eye(n);
    P([1, s], :) = P([s, 1], :); 
    A = P * A; % 用初等矩阵左乘A 对 A 作行交换
    A([2:n], 1) = A([2:n], 1) * (1 / A(1, 1)); % 求第一层
    
    for r = 2:1:n
        % 先有行交换
        p = eye(n);  % 用 p 记录每一次的初等矩阵
        [~, s] = max(A(r:n, r));
        s = s + r - 1;
        p([r, s], :) = p([s, r], :);  
        A = p * A; % A的改变
        P = p * P; % 记录P的变化
        
        % 求第 r 层
        for k = r:1:n
            A(r, k) = A(r, k) - A(r, [1:r-1]) * A([1:r-1], k);
        end
        for m = r + 1:1:n
            A(m, r) = (A(m, r) - A(m, [1:r-1]) * A([1:r-1], r)) * (1 / A(r, r));
        end
    end
    
    L = tril(A, -1) + eye(n);
    U = triu(A, 0);
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
[L, U, P] = PLU_factorization(A);
disp('矩阵L:');
disp(L);
disp('矩阵U:');
disp(U);
disp('置换矩阵P:');
disp(P);

X1 = back_substitution_two(L, U, P * b1);
disp('LU分解求解结果:');
disp(X1);

% MATLAB直接求解结果验证
X_direct = A \ b1;
disp('MATLAB直接求解结果:');
disp(X_direct);
