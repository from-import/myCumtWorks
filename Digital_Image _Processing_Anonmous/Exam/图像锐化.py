import cv2
import numpy as np


"""
图像锐化滤波器的目的是增强图像中的边缘和细节。以下是一些常见的锐化滤波器及其算子：

1. 拉普拉斯算子（Laplacian Operator）
拉普拉斯算子是一种二阶微分算子，用于检测图像中的边缘。常用的拉普拉斯算子模板如下：

2. Sobel算子
Sobel算子是一种一阶微分算子，用于检测图像中的水平和垂直边缘。Sobel算子通常分为水平和垂直两个方向：
水平方向的Sobel算子：

3. Prewitt算子
Prewitt算子类似于Sobel算子，也是用于边缘检测的算子，但其权重不同：
"""



def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA):
    # 获取图像尺寸
    (h, w) = image.shape[:2]

    # 如果宽度和高度都为None，则返回原图
    if width is None and height is None:
        return image

    # 计算缩放比例
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    # 缩放图像
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

# 读取图像
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# 缩放图像以适应显示窗口
resized_image = resize_image(image, width=600)


# 1
# 使用拉普拉斯算子对图像进行二阶微分，检测图像中的快速变化区域，然后将其添加回原图像，达到锐化效果。

# 拉普拉斯滤波
laplacian = cv2.Laplacian(resized_image, cv2.CV_64F)
sharp_image = cv2.convertScaleAbs(resized_image - laplacian)

# 显示结果
cv2.imshow('Original Image', resized_image)
cv2.imshow('Laplacian Sharpened Image', sharp_image)






# 2
# 将图像的高频分量放大，再叠加回原图像，从而增强图像的细节和边缘。

# 读取图像
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# 缩放图像以适应显示窗口
resized_image = resize_image(image, width=600)

# 高斯模糊
blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)

# 高提升滤波
alpha = 1.5
high_boost_image = cv2.addWeighted(resized_image, alpha, blurred_image, -0.5, 0)

# 显示结果
cv2.imshow('High-Boost Filtered Image', high_boost_image)



# 未归一化的边缘检测（Unsharp Masking）
# 通过从原图像中减去平滑后的图像（即低通滤波后的图像），得到高频分量，然后将其放大后叠加回原图像。

# 读取图像
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# 缩放图像以适应显示窗口
resized_image = resize_image(image, width=600)

# 高斯模糊
blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)

# 计算高频分量
high_freq_image = resized_image - blurred_image

# 未归一化的边缘检测
unsharp_image = cv2.addWeighted(resized_image, 1.5, high_freq_image, -0.5, 0)

# 显示结果
cv2.imshow('Unsharp Masking Image', unsharp_image)

# 等待用户按键操作
cv2.waitKey(0)
cv2.destroyAllWindows()
