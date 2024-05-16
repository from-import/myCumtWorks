import cv2
import numpy as np

# 读取两个图像
image1 = cv2.imread('2.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('red.jpg', cv2.IMREAD_GRAYSCALE)

# 将red.jpg调整为和2.jpg相同的尺寸
image2_resized = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

# 进行傅里叶变换
f_transform1 = np.fft.fft2(image1)
f_transform2 = np.fft.fft2(image2_resized)

# 获取幅度谱和相位谱
magnitude1 = np.abs(f_transform1)
magnitude2 = np.abs(f_transform2)
phase1 = np.angle(f_transform1)
phase2 = np.angle(f_transform2)

# 将两个图像的幅度谱融合，保持其中一个图像的幅度谱不变，使用另一个图像的相位谱
f_transform_combined = magnitude2 * np.exp(1j * phase1)

# 进行逆傅里叶变换
image_combined = np.fft.ifft2(f_transform_combined)
image_combined = np.abs(image_combined)

# 将图像缩放到0-255之间
image_combined = cv2.normalize(image_combined, None, 0, 255, cv2.NORM_MINMAX)

# 显示融合后的图像
cv2.imshow('Combined Image', image_combined.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
