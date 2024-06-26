"""
低通滤波器：用于平滑图像，去除噪声和细节，通常用于图像去噪。
高通滤波器：用于增强图像中的细节和边缘，通常用于图像锐化和增强。
带通滤波器：用于通过特定频率范围的信号，同时抑制高于和低于该范围的频率。一般在图像处理中的应用较少。
带阻滤波器：用于抑制特定频率范围的信号，同时通过高于和低于该范围的频率。在图像处理中应用较少。
"""





"""
同态滤波的基本思想是将图像的光照分量和反射分量分离，并对这两个分量分别进行处理。通常的步骤如下：
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

# 缩放图像以适应显示窗口
resized_image = resize_image(image, width=600)

# 1. 对数变换
log_image = np.log1p(np.array(resized_image, dtype="float"))

# 2. 傅里叶变换
dft = cv2.dft(log_image, flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# 3. 设计同态滤波器
rows, cols = resized_image.shape
crow, ccol = rows // 2 , cols // 2
radius = 30
H = np.ones((rows, cols, 2), dtype=np.float32)
for i in range(rows):
    for j in range(cols):
        d = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
        H[i, j] = 1 - np.exp(-((d ** 2) / (2 * (radius ** 2))))

# 4. 应用滤波器
filtered_dft = dft_shift * H

# 5. 傅里叶逆变换
idft_shift = np.fft.ifftshift(filtered_dft)
idft = cv2.idft(idft_shift)
idft_magnitude = cv2.magnitude(idft[:,:,0], idft[:,:,1])

# 进行适当的归一化和剪裁，以避免溢出问题
idft_magnitude = np.clip(idft_magnitude, 0, None)
idft_magnitude = cv2.normalize(idft_magnitude, None, 0, 255, cv2.NORM_MINMAX)

# 6. 指数变换
exp_image = np.expm1(idft_magnitude / 255.0) * 255.0

# 归一化到0-255
exp_image = cv2.normalize(exp_image, None, 0, 255, cv2.NORM_MINMAX)
exp_image = np.array(exp_image, dtype=np.uint8)

# 显示结果
cv2.imshow('Original Image', resized_image)
cv2.imshow('Homomorphic Filtered Image', exp_image)





"""
均值滤波
"""

# 读取图像
image = cv2.imread('image.jpg')

# 缩放图像以适应显示窗口
resized_image = resize_image(image, width=600)

# 应用均值滤波
mean_filtered_image = cv2.blur(resized_image, (5, 5))

# 显示原图像和均值滤波后的图像
cv2.imshow('Original Image', resized_image)
cv2.imshow('Mean Filtered Image', mean_filtered_image)






"""
中值滤波的基本步骤如下：

    为图像中的每一个像素，定义一个邻域窗口（通常为奇数大小，例如3x3, 5x5等）。
    将邻域窗口内的所有像素值排序。
    用排序后像素值的中值替换当前像素值。
"""

# 读取图像
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# 缩放图像以适应显示窗口
resized_image = resize_image(image, width=600)

# 应用中值滤波
median_filtered_image = cv2.medianBlur(resized_image, 5)  # 使用5x5窗口的中值滤波

# 显示结果
cv2.imshow('Original Image', resized_image)
cv2.imshow('Median Filtered Image', median_filtered_image)

# 等待用户按键操作
cv2.waitKey(0)
cv2.destroyAllWindows()