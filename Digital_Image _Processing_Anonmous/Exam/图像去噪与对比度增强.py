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

# 读取图像
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# 缩放图像以适应显示窗口
resized_image = resize_image(image, width=600)

# 图像去噪（高斯模糊）
denoised_image = cv2.GaussianBlur(resized_image, (5, 5), 0)

# 对比度增强（CLAHE）
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
contrast_enhanced_image = clahe.apply(denoised_image)

# 显示结果
cv2.imshow('Original Image', resized_image)
cv2.imshow('Denoised Image', denoised_image)
cv2.imshow('Contrast Enhanced Image', contrast_enhanced_image)

# 等待用户按键操作
cv2.waitKey(0)
cv2.destroyAllWindows()