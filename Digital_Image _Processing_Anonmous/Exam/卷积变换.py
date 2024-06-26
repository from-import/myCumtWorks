import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取灰度图像
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# 定义卷积核（如边缘检测核）
kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])

# 应用卷积
image_filtered = cv2.filter2D(image, -1, kernel)

# 显示原图像和卷积后的图像
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Filtered Image')
plt.imshow(image_filtered, cmap='gray')

plt.show()
