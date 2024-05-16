import cv2
import numpy as np

# 读取两个图像
image_night = cv2.imread('night.jpg')
image_light = cv2.imread('light.jpg')

# 转换为灰度图像
image_night_gray = cv2.cvtColor(image_night, cv2.COLOR_BGR2GRAY)
image_light_gray = cv2.cvtColor(image_light, cv2.COLOR_BGR2GRAY)

# 添加随机噪声
mean = 0
stddev = 15
noise_night = np.random.normal(mean, stddev, image_night_gray.shape).astype(np.uint8)
noise_light = np.random.normal(mean, stddev, image_light_gray.shape).astype(np.uint8)

image_night_noisy = cv2.add(image_night_gray, noise_night)
image_light_noisy = cv2.add(image_light_gray, noise_light)

# 进行傅里叶变换
fft_night = np.fft.fft2(image_night_noisy)
fft_light = np.fft.fft2(image_light_noisy)

# 添加频谱上的干扰
noise_fft_night = np.random.normal(mean, stddev, fft_night.shape).astype(np.complex64)
noise_fft_light = np.random.normal(mean, stddev, fft_light.shape).astype(np.complex64)

fft_night_noisy = fft_night + noise_fft_night
fft_light_noisy = fft_light + noise_fft_light

# 进行逆傅里叶变换
image_night_restored = np.fft.ifft2(fft_night_noisy).real.astype(np.uint8)
image_light_restored = np.fft.ifft2(fft_light_noisy).real.astype(np.uint8)

# 对图像进行对比度改变
alpha = 1.5  # 对比度增强因子
beta = 0  # 亮度调节值
image_night_contrast = cv2.convertScaleAbs(image_night_restored, alpha=alpha, beta=beta)
image_light_contrast = cv2.convertScaleAbs(image_light_restored, alpha=alpha, beta=beta)

# 对图像进行平滑
image_night_smooth = cv2.GaussianBlur(image_night_restored, (5, 5), 0)
image_light_smooth = cv2.GaussianBlur(image_light_restored, (5, 5), 0)

# 对图像进行锐化
kernel_sharpening = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
image_night_sharpened = cv2.filter2D(image_night_restored, -1, kernel_sharpening)
image_light_sharpened = cv2.filter2D(image_light_restored, -1, kernel_sharpening)

# 对图像进行滤波
image_night_filtered = cv2.medianBlur(image_night_restored, 5)
image_light_filtered = cv2.medianBlur(image_light_restored, 5)

# 显示增强后的图像
cv2.imshow('Enhanced Night Image - Contrast', image_night_contrast)
cv2.imshow('Enhanced Night Image - Smooth', image_night_smooth)
cv2.imshow('Enhanced Night Image - Sharpened', image_night_sharpened)
cv2.imshow('Enhanced Night Image - Filtered', image_night_filtered)

cv2.imshow('Enhanced Light Image - Contrast', image_light_contrast)
cv2.imshow('Enhanced Light Image - Smooth', image_light_smooth)
cv2.imshow('Enhanced Light Image - Sharpened', image_light_sharpened)
cv2.imshow('Enhanced Light Image - Filtered', image_light_filtered)

cv2.waitKey(0)
cv2.destroyAllWindows()
