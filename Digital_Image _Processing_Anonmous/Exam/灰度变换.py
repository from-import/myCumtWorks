import cv2
import numpy as np

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

# 缩放图像
resized_image = resize_image(image, width=600)

# 灰度拉伸
alpha = 1.5
beta = 0
stretched_image = cv2.convertScaleAbs(resized_image, alpha=alpha, beta=beta)

# 反转变换
negative_image = cv2.bitwise_not(resized_image)

# 伽马校正
gamma = 2.0
gamma_corrected_image = np.array(255 * (resized_image / 255) ** gamma, dtype='uint8')

# 对数变换，增加一个小常数以避免除零错误
c = 255 / np.log(1 + np.max(resized_image))
log_image = c * np.log(1 + resized_image + 1e-5)  # 加入小常数1e-5
log_image = np.array(log_image, dtype=np.uint8)

# 显示结果
cv2.imshow('Original Image', resized_image)
cv2.imshow('Contrast Stretched Image', stretched_image)
cv2.imshow('Negative Image', negative_image)
cv2.imshow('Gamma Corrected Image', gamma_corrected_image)
cv2.imshow('Log Transformed Image', log_image)


