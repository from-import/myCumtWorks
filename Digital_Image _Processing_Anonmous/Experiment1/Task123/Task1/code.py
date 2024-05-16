import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('peppers.jpg')

def pltImageShow(image1, name1, image2, name2):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    plt.title(name1)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    plt.title(name2)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# Task1

swapped_image = image.copy()
swapped_image[:, :, 2] = image[:, :, 1]  # 将绿色通道复制到蓝色通道
swapped_image[:, :, 1] = image[:, :, 2]  # 将红色通道复制到绿色通道

# 显示原图和处理后的图像
pltImageShow(image, 'Original Image', swapped_image, 'Swapped Red-Green Channels')

# Task2

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
pltImageShow(image,'Original Image',gray_image,'Gray Image')


# task3

# 图像旋转函数
def rotate_image(image, angle, interpolation=cv2.INTER_LINEAR):
    h, w = image.shape
    # 计算旋转中心
    center = (w // 2, h // 2)
    # 获取旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # 应用仿射变换
    rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return rotated_image

# 图像放大函数
def resize_image(image, scale_factor, interpolation=cv2.INTER_LINEAR):
    h, w = image.shape
    # 计算新的图像尺寸
    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)
    # 使用指定的插值方法进行图像放大
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    return resized_image

# 旋转灰度图像并显示
angle = 30
rotated_gray_image = rotate_image(gray_image, angle, interpolation=cv2.INTER_NEAREST)
plt.imshow(rotated_gray_image, cmap='gray')
plt.title(f'Rotated (Angle={angle}°, Interpolation=Nearest Neighbor)')
plt.axis('off')
plt.show()

# 缩放灰度图像并显示
scale_factor = 1.5
resized_gray_image = resize_image(gray_image, scale_factor, interpolation=cv2.INTER_CUBIC)
plt.imshow(resized_gray_image, cmap='gray')
plt.title(f'Resized (Scale Factor={scale_factor}, Interpolation=Cubic)')
plt.axis('off')
plt.show()


# Task4

# 读取图像1和图像2
image1 = cv2.imread('peppers.jpg')
image2 = cv2.imread('lotus.jpg')

# 确保两幅图像具有相同的尺寸
h, w, _ = image1.shape
image2 = cv2.resize(image2, (w, h))

# 图像拼接（横向）
concatenated_image = np.concatenate((image1, image2), axis=1)

# 图像加法（按通道）
added_image = cv2.add(image1, image2)

# 图像减法（按通道）
subtracted_image = cv2.subtract(image1, image2)

# 图像乘法（按通道）
multiplied_image = cv2.multiply(image1, image2)

# 图像除法（按通道）
# 避免溢出
image1_float = image1.astype(np.float32)
image2_float = image2.astype(np.float32)
divided_image = cv2.divide(image1_float, image2_float + np.finfo(float).eps)  # 避免除以0

# 显示结果
plt.figure(figsize=(16, 12))

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
plt.title('Image 1')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
plt.title('Image 2')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(cv2.cvtColor(concatenated_image, cv2.COLOR_BGR2RGB))
plt.title('Concatenated Images')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(cv2.cvtColor(added_image, cv2.COLOR_BGR2RGB))
plt.title('Added Images')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(cv2.cvtColor(subtracted_image, cv2.COLOR_BGR2RGB))
plt.title('Subtracted Images')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(cv2.cvtColor(multiplied_image, cv2.COLOR_BGR2RGB))
plt.title('Multiplied Images')
plt.axis('off')

plt.tight_layout()
plt.show()


# Task 6

def rgb_to_hsi(rgb_image):
    # 将 RGB 图像转换为浮点数类型
    rgb_image = rgb_image.astype(np.float32) / 255.0

    # 提取 RGB 通道
    r, g, b = rgb_image[:, :, 0], rgb_image[:, :, 1], rgb_image[:, :, 2]

    # 计算强度（Intensity）
    intensity = (r + g + b) / 3.0

    # 计算饱和度（Saturation）
    min_val = np.minimum.reduce([r, g, b])
    saturation = 1 - min_val / intensity
    saturation = np.nan_to_num(saturation)  # 处理除以零产生的 NaN

    # 计算色相（Hue）
    numerator = 0.5 * ((r - g) + (r - b))
    denominator = np.sqrt((r - g) ** 2 + (r - b) * (g - b))
    hue = np.arccos(np.clip(numerator / denominator, -1, 1))

    # 根据 B 分量大于 G 分量的情况调整色相值
    hue[b > g] = 2 * np.pi - hue[b > g]

    # 将强度、饱和度和色相值限制在合适的范围内
    intensity = np.clip(intensity, 0, 1)
    saturation = np.clip(saturation, 0, 1)
    hue = np.clip(hue / (2 * np.pi), 0, 1)

    # 将 HSI 通道合并为一个图像
    hsi_image = np.stack([hue, saturation, intensity], axis=-1)

    return hsi_image

def hsi_to_rgb(hsi_image):
    # 提取 HSI 通道
    hue, saturation, intensity = hsi_image[:, :, 0], hsi_image[:, :, 1], hsi_image[:, :, 2]

    # 将色相转换为弧度
    hue = hue * 2 * np.pi

    # 计算 RGB 分量
    r, g, b = np.zeros_like(hue), np.zeros_like(hue), np.zeros_like(hue)

    # 第一批计算
    for i in range(hue.shape[0]):
        for j in range(hue.shape[1]):
            if hue[i, j] < 2 * np.pi / 3:
                b[i, j] = intensity[i, j] * (1 - saturation[i, j])
                r[i, j] = intensity[i, j] * (1 + (saturation[i, j] * np.cos(hue[i, j]) / np.cos(np.pi / 3 - hue[i, j])))
                g[i, j] = 3 * intensity[i, j] - (r[i, j] + b[i, j])
            elif hue[i, j] < 4 * np.pi / 3:
                hue[i, j] -= 2 * np.pi / 3
                r[i, j] = intensity[i, j] * (1 - saturation[i, j])
                g[i, j] = intensity[i, j] * (1 + (saturation[i, j] * np.cos(hue[i, j]) / np.cos(np.pi / 3 - hue[i, j])))
                b[i, j] = 3 * intensity[i, j] - (r[i, j] + g[i, j])
            else:
                hue[i, j] -= 4 * np.pi / 3
                g[i, j] = intensity[i, j] * (1 - saturation[i, j])
                b[i, j] = intensity[i, j] * (1 + (saturation[i, j] * np.cos(hue[i, j]) / np.cos(np.pi / 3 - hue[i, j])))
                r[i, j] = 3 * intensity[i, j] - (g[i, j] + b[i, j])

    # 将 RGB 分量合并为一个图像
    rgb_image = np.stack([r, g, b], axis=-1)

    # 将浮点数值转换为 0-255 范围内的整数
    rgb_image = (rgb_image * 255).astype(np.uint8)

    return rgb_image

# 将 RGB 图像转换为 HSI 空间
hsi_image = rgb_to_hsi(image)

# 对 HSI 空间中的各通道进行变换
hsi_image[:, :, 0] += 0.1  # 调整色相
hsi_image[:, :, 1] *= 1.5  # 增加饱和度
hsi_image[:, :, 2] *= 0.8  # 减小强度

# 将变换后的 HSI 图像转换回 RGB 空间
transformed_image = hsi_to_rgb(hsi_image)
pltImageShow(image,"Original",transformed_image,"Transformed HSI Image")