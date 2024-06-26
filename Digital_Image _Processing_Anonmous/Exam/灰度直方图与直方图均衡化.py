import cv2
import numpy as np

# 读取灰度图像
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# 计算灰度直方图
hist = cv2.calcHist([image], [0], None, [256], [0, 256])

# 创建一个黑色图像用于绘制直方图
hist_image = np.zeros((300, 256), dtype=np.uint8)

# 归一化直方图，使其值在0到300之间
cv2.normalize(hist, hist, 0, hist_image.shape[0], cv2.NORM_MINMAX)

# 绘制直方图
for x in range(256):
    cv2.line(hist_image, (x, hist_image.shape[0]), (x, hist_image.shape[0] - int(hist[x])), 255)



"""
直方图均衡化
"""
# 显示图像和直方图
cv2.imshow('Original Image', image)
cv2.imshow('Grayscale Histogram', hist_image)

import cv2
import numpy as np

# 定义缩放图像函数
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

# 读取灰度图像
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# 缩放图像以适应显示窗口
resized_image = resize_image(image, width=600)

# 直方图均衡化
equalized_image = cv2.equalizeHist(resized_image)

# 自适应直方图均衡化 (CLAHE)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_image = clahe.apply(resized_image)

# 显示结果
cv2.imshow('Original Image', resized_image)
cv2.imshow('Equalized Image', equalized_image)
cv2.imshow('CLAHE Image', clahe_image)

# 计算并显示直方图
def plot_histogram(image, title):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_image = np.zeros((300, 256), dtype=np.uint8)
    cv2.normalize(hist, hist, 0, hist_image.shape[0], cv2.NORM_MINMAX)
    for x in range(256):
        cv2.line(hist_image, (x, hist_image.shape[0]), (x, hist_image.shape[0] - int(hist[x])), 255)
    cv2.imshow(title, hist_image)

plot_histogram(resized_image, 'Original Histogram')
plot_histogram(equalized_image, 'Equalized Histogram')
plot_histogram(clahe_image, 'CLAHE Histogram')

# 等待用户按键操作
cv2.waitKey(0)
cv2.destroyAllWindows()
