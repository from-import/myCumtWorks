import cv2
import numpy as np

def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

# 读取灰度图像
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# 缩放图像
resized_image = resize_image(image, width=600)

# 对数变换
c = 255 / np.log(1 + np.max(resized_image))
log_image = c * np.log(1 + resized_image + 1e-5)  # 加入小常数1e-5
log_image = np.array(log_image, dtype=np.uint8)

# 伽马校正
gamma = 2.0
gamma_corrected_image = np.array(255 * (resized_image / 255) ** gamma, dtype='uint8')

# 幂律变换（与伽马校正相同）
power_law_image = np.array(255 * (resized_image / 255) ** gamma, dtype='uint8')

# 指数变换
k = 0.01
c = 255 / (np.exp(k * np.max(resized_image)) - 1)
exp_image = c * (np.exp(k * resized_image) - 1)
exp_image = np.array(exp_image, dtype=np.uint8)

# 显示结果
cv2.imshow('Original Image', resized_image)
cv2.imshow('Log Transformed Image', log_image)
cv2.imshow('Gamma Corrected Image', gamma_corrected_image)
cv2.imshow('Power-Law Transformed Image', power_law_image)
cv2.imshow('Exponential Transformed Image', exp_image)

# 等待用户按键操作
cv2.waitKey(0)
cv2.destroyAllWindows()
