"""
高通滤波器：适用于需要增强细节和边缘的应用，如边缘检测和图像锐化。
低通滤波器：适用于需要平滑图像和去除噪声的应用，如图像去噪和模糊处理。
带通滤波器：适用于需要增强特定频率范围的应用，如特定纹理增强。
带阻滤波器：适用于需要抑制特定频率范围的应用，如去除周期性噪声。
"""

"""
增强图像细节和边缘

    使用高通滤波器（High-Pass Filter）
    目标：增强图像中的高频分量，突出图像的边缘和细节部分。
    适用场景：边缘检测、图像锐化、纹理增强。
"""

import cv2
import numpy as np


def high_pass_filter(image):
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols, 2), np.uint8)
    r = 30  # 半径
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
    mask[mask_area] = 0

    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# 读取图像
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
high_pass_image = high_pass_filter(image)

# 显示结果
cv2.imshow('Original Image', image)
cv2.imshow('High Pass Filtered Image', high_pass_image)
cv2.waitKey(0)
cv2.destroyAllWindows()



"""
平滑图像，去除噪声

    使用低通滤波器（Low-Pass Filter）
    目标：增强图像中的低频分量，去除图像中的高频噪声，使图像变得更加平滑。
    适用场景：图像去噪、模糊处理。
"""

import cv2
import numpy as np


def low_pass_filter(image):
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols, 2), np.uint8)
    r = 30  # 半径
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
    mask[mask_area] = 1

    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# 读取图像
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
low_pass_image = low_pass_filter(image)

# 显示结果
cv2.imshow('Original Image', image)
cv2.imshow('Low Pass Filtered Image', low_pass_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


"""
3. 保留特定频率范围，抑制其他频率

    使用带通滤波器（Band-Pass Filter）
    目标：通过特定频率范围的信号，抑制高于和低于该范围的频率。
    适用场景：当需要增强图像中特定频率范围的特征时。
"""
import cv2
import numpy as np


def band_pass_filter(image, low_radius, high_radius):
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols, 2), np.uint8)
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]

    # Low radius mask area
    mask_area_low = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= low_radius ** 2
    mask[mask_area_low] = 1

    # High radius mask area
    mask_area_high = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= high_radius ** 2
    mask[mask_area_high] = 0

    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# 读取图像
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
band_pass_image = band_pass_filter(image, low_radius=30, high_radius=100)

# 显示结果
cv2.imshow('Original Image', image)
cv2.imshow('Band Pass Filtered Image', band_pass_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


"""
抑制特定频率范围，保留其他频率

    使用带阻滤波器（Band-Stop Filter）
    目标：抑制特定频率范围的信号，通过高于和低于该范围的频率。
    适用场景：当需要去除图像中特定频率范围的噪声或干扰时。
"""

import cv2
import numpy as np


def band_stop_filter(image, low_radius, high_radius):
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols, 2), np.uint8)
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]

    # Low radius mask area
    mask_area_low = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= low_radius ** 2
    mask[mask_area_low] = 0

    # High radius mask area
    mask_area_high = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= high_radius ** 2
    mask[mask_area_high] = 1

    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# 读取图像
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
band_stop_image = band_stop_filter(image, low_radius=30, high_radius=100)

# 显示结果
cv2.imshow('Original Image', image)
cv2.imshow('Band Stop Filtered Image', band_stop_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
