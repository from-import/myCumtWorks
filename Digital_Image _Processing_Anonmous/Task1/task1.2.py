import cv2
import numpy as np

# 读取图像
image = cv2.imread('2.jpg', cv2.IMREAD_GRAYSCALE)

# 傅里叶变换
f_transform = np.fft.fft2(image)
f_shift = np.fft.fftshift(f_transform)
magnitude_spectrum = 20 * np.log(np.abs(f_shift))

# 显示原始图像和傅里叶变换结果
cv2.imshow('Original Image', image)
cv2.imshow('Magnitude Spectrum', magnitude_spectrum.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()

# 离散余弦变换
dct = cv2.dct(np.float32(image))
dct_shift = np.fft.fftshift(dct)
dct_spectrum = 20 * np.log(np.abs(dct_shift))

# 显示离散余弦变换结果
cv2.imshow('DCT Spectrum', dct_spectrum.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
