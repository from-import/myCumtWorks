"""
傅里叶变换后的低频和高频部分

在傅里叶变换中，图像从空间域转换到频域，得到频域图像。在频域图像中，不同的频率成分对应于图像的不同特征：

低频部分：
    位置：通常位于频域图像的中心位置（如果经过了频谱中心化处理）。
    对应的图像特征：低频部分表示图像中的大尺度结构和整体亮度。它包含了图像中缓慢变化的部分，例如图像的整体轮廓和大块区域的均匀亮度。
    影响：低频部分对图像的整体感知影响很大，去除低频部分会使图像失去大致的形状和结构。

高频部分：
    位置：通常位于频域图像的边缘位置（如果经过了频谱中心化处理）。
    对应的图像特征：高频部分表示图像中的细节和边缘。它包含了图像中变化迅速的部分，例如细小的纹理、边缘和噪声。
    影响：高频部分对图像的清晰度和细节有很大影响，去除高频部分会使图像变得模糊。


补充

二维离散傅里叶变换:
对图像的每一行进行一维傅里叶变换,再对结果的每一列进行一维傅里叶变换。

通过这种方式，我们可以减少计算复杂度，从直接计算二维DFT的 O(M2N2)O(M2N2) 次复数乘法，
降低到两次一维DFT的 O(MNlog⁡M+MNlog⁡N)O(MNlogM+MNlogN) 次复数乘法，这是快速傅里叶变换（FFT）的复杂度。
"""

"""
中心化傅里叶变换
"""

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

# 计算傅里叶变换
f = np.fft.fft2(image)

# 进行中心化处理
fshift = np.fft.fftshift(f)

# 计算频谱图
magnitude_spectrum = 20 * np.log(np.abs(fshift))

# 归一化频谱图以适应显示
magnitude_spectrum = np.asarray(magnitude_spectrum, dtype=np.uint8)

# 缩放图像和频谱图
resized_image = resize_image(image, width=600)
resized_magnitude_spectrum = resize_image(magnitude_spectrum, width=600)

# 显示原图像和中心化后的频谱图
cv2.imshow('Input Image', resized_image)
cv2.imshow('Centered Magnitude Spectrum', resized_magnitude_spectrum)






"""
高频和低频部分的可视化
"""
import cv2
import numpy as np



# 读取灰度图像
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# 缩放图像以适应显示窗口
resized_image = resize_image(image, width=600)

# 计算傅里叶变换
dft = cv2.dft(np.float32(resized_image), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# 计算频谱图
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))

# 创建低频和高频遮罩
rows, cols = resized_image.shape
crow, ccol = rows // 2 , cols // 2
mask_low = np.zeros((rows, cols, 2), np.uint8)
mask_high = np.ones((rows, cols, 2), np.uint8)

r = 30  # 半径
center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]
mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
mask_low[mask_area] = 1
mask_high[mask_area] = 0

# 应用遮罩分别获取低频和高频部分
fshift_low = dft_shift * mask_low
fshift_high = dft_shift * mask_high

# 傅里叶逆变换
img_back_low = cv2.idft(np.fft.ifftshift(fshift_low))
img_back_low = cv2.magnitude(img_back_low[:,:,0], img_back_low[:,:,1])

img_back_high = cv2.idft(np.fft.ifftshift(fshift_high))
img_back_high = cv2.magnitude(img_back_high[:,:,0], img_back_high[:,:,1])

# 归一化到0-255
img_back_low = cv2.normalize(img_back_low, None, 0, 255, cv2.NORM_MINMAX)
img_back_high = cv2.normalize(img_back_high, None, 0, 255, cv2.NORM_MINMAX)

# 将结果转换为uint8类型
img_back_low = np.uint8(img_back_low)
img_back_high = np.uint8(img_back_high)

# 显示结果
cv2.imshow('Original Image', resized_image)
cv2.imshow('Low Frequency Image', img_back_low)
cv2.imshow('High Frequency Image', img_back_high)
cv2.imshow('Magnitude Spectrum', np.uint8(magnitude_spectrum))

# 等待用户按键操作
cv2.waitKey(0)
cv2.destroyAllWindows()