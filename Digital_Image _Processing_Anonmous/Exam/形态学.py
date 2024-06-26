"""
图像形态学

图像形态学（Image Morphology）是一种基于图像形状特征的图像处理技术，
主要用于分析和处理二值图像，但也可以扩展到灰度图像。
形态学操作主要依赖于结构元素（Structuring Element）对图像进行形状变换，
常用于图像预处理、形态特征提取、噪声消除和图像分割等领域。

基本概念

结构元素（Structuring Element）：
    结构元素是一个小的形状或模板，用于在图像上进行形态学操作。它可以是任何形状，如方形、圆形、十字形等。
    结构元素的每个元素可以取值为1或0，用于定义操作的邻域。

二值图像：
    图像的像素值只有0和1（或黑和白），常用于形态学操作。
    图像中的目标区域和背景区域通过像素值区分。


基本形态学操作

腐蚀（Erosion）：
    腐蚀操作会缩小图像中的白色区域（前景），主要用于消除噪声、分割物体和去掉小的目标。
    操作：用结构元素在图像中移动，如果结构元素完全包含在前景中，则中心像素保留为前景，否则变为背景。

膨胀（Dilation）：
    膨胀操作会扩大图像中的白色区域（前景），主要用于填充物体中的孔洞、连接相邻的物体和增强物体边界。
    操作：用结构元素在图像中移动，如果结构元素与前景有重叠，则中心像素变为前景。

开运算（Opening）：
    开运算是先腐蚀后膨胀的操作，用于去除小的物体、平滑物体边界和消除噪声。
    操作：对图像进行腐蚀操作，然后对腐蚀后的图像进行膨胀操作。

闭运算（Closing）：
    闭运算是先膨胀后腐蚀的操作，用于填补物体中的小孔、连接断裂的部分和平滑物体边界。
    操作：对图像进行膨胀操作，然后对膨胀后的图像进行腐蚀操作。

形态学梯度（Morphological Gradient）：
    形态学梯度是膨胀和腐蚀结果的差异，用于提取物体的边缘。
    操作：图像的膨胀结果减去腐蚀结果。
"""

import cv2
import numpy as np

def resize_image(image, size):
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

# 读取图像
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# 二值化图像
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# 定义结构元素
kernel = np.ones((5, 5), np.uint8)

# 腐蚀
eroded_image = cv2.erode(binary_image, kernel, iterations=1)

# 膨胀
dilated_image = cv2.dilate(binary_image, kernel, iterations=1)

# 开运算
opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

# 闭运算
closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

# 形态学梯度
gradient_image = cv2.morphologyEx(binary_image, cv2.MORPH_GRADIENT, kernel)

# 顶帽变换
top_hat_image = cv2.morphologyEx(binary_image, cv2.MORPH_TOPHAT, kernel)

# 黑帽变换
black_hat_image = cv2.morphologyEx(binary_image, cv2.MORPH_BLACKHAT, kernel)

# 缩放图像
size = (350, 350)
binary_image = resize_image(binary_image, size)
eroded_image = resize_image(eroded_image, size)
dilated_image = resize_image(dilated_image, size)
opened_image = resize_image(opened_image, size)
closed_image = resize_image(closed_image, size)
gradient_image = resize_image(gradient_image, size)
top_hat_image = resize_image(top_hat_image, size)
black_hat_image = resize_image(black_hat_image, size)

# 将图像组合成一张图
row1 = cv2.hconcat([binary_image, eroded_image, dilated_image])
row2 = cv2.hconcat([opened_image, closed_image, gradient_image])
row3 = cv2.hconcat([top_hat_image, black_hat_image, np.zeros_like(binary_image)])

combined_image = cv2.vconcat([row1, row2, row3])

# 添加图像标签
def add_text(image, text, position):
    return cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

combined_image = add_text(combined_image, 'Original Image', (10, 20))
combined_image = add_text(combined_image, 'Eroded Image', (10 + size[0], 20))
combined_image = add_text(combined_image, 'Dilated Image', (10 + 2 * size[0], 20))
combined_image = add_text(combined_image, 'Opened Image', (10, 20 + size[1]))
combined_image = add_text(combined_image, 'Closed Image', (10 + size[0], 20 + size[1]))
combined_image = add_text(combined_image, 'Gradient Image', (10 + 2 * size[0], 20 + size[1]))
combined_image = add_text(combined_image, 'Top-hat Image', (10, 20 + 2 * size[1]))
combined_image = add_text(combined_image, 'Black-hat Image', (10 + size[0], 20 + 2 * size[1]))

# 显示结果
cv2.imshow('Combined Image', combined_image)

# 等待用户按键操作
cv2.waitKey(0)
cv2.destroyAllWindows()