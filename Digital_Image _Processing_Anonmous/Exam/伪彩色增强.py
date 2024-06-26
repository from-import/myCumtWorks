"""
补充:伪彩色增强

伪彩色增强（Pseudocolor Enhancement）是一种图像处理技术，
通过将灰度图像转换为彩色图像，以增强视觉效果，便于分析和解释。
伪彩色增强技术在医学影像、地质调查、遥感图像以及其他科学和工程应用中广泛使用。

原理
伪彩色增强的基本原理是将灰度图像中的不同灰度值映射到不同的颜色，
以便突出图像中的某些特征。这种映射通常通过查找表（Look-Up Table, LUT）来实现。常见的映射方法包括：
"""

import cv2
import numpy as np

# 读取灰度图像
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# 创建一个伪彩色查找表（例如使用JET色图）
colormap = cv2.COLORMAP_JET

# 应用查找表
pseudo_color_image = cv2.applyColorMap(image, colormap)

# 显示结果
cv2.imshow('Original Image', image)
cv2.imshow('Pseudo Color Image', pseudo_color_image)

# 等待用户按键操作
cv2.waitKey(0)
cv2.destroyAllWindows()
