import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy import ndimage
image_url = "plane3.jpg"

# 1.将Image灰度化为gray对其进行阈值分割转换为BW；
image = cv2.imread(image_url)
ret,BWimage = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
cv2.namedWindow('result', cv2.WINDOW_NORMAL)
cv2.imshow('result', BWimage)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 2.形态学滤波
image = cv2.imread(image_url, 0)
ret, BWimage = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
LBimage1 = cv2.morphologyEx(BWimage, cv2.MORPH_OPEN, kernel)
LBimage2 = cv2.morphologyEx(BWimage, cv2.MORPH_CLOSE, kernel)
LBimage3 = cv2.morphologyEx(BWimage, cv2.MORPH_GRADIENT, kernel)
LBimage4 = cv2.morphologyEx(BWimage, cv2.MORPH_TOPHAT, kernel)
LBimage5 = cv2.morphologyEx(BWimage, cv2.MORPH_BLACKHAT, kernel)
A = cv2.hconcat([LBimage1, LBimage2, LBimage3 ])
B = cv2.hconcat([LBimage4, LBimage5,  BWimage])
C =cv2.vconcat([A, B])
cv2.namedWindow('result', cv2.WINDOW_NORMAL)
cv2.imshow('result', C)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 3. 绘制边缘线
image = cv2.imread(image_url)
# 将图像转换为灰度图像
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用Canny边缘检测
edges = cv2.Canny(gray_image, 300, 800)

# 寻找边缘的轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 创建一个新的图像副本用于绘制边缘线
image_with_edges = image.copy()

# 在原始图像上绘制红色边缘线
cv2.drawContours(image_with_edges, contours, -1, (0, 0, 255), 2)

# 显示带有边缘线的图像
cv2.imshow('Image with Red Edges', image_with_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()




# Task4 计算各区域边界点的傅里叶描绘子并用四分之一点重建边界;

def calculate_fourier_descriptors(contour, num_descriptors):
    # 提取边界点的复数表示
    complex_contour = contour[:, 0, 0] + 1j * contour[:, 0, 1]

    # 计算傅里叶描述子
    descriptors = np.fft.fft(complex_contour)

    # 仅保留指定数量的描述子
    descriptors = descriptors[:num_descriptors]

    return descriptors

def reconstruct_contour(descriptors, num_points):
    # 使用傅里叶描述子重建边界
    reconstructed_contour = np.fft.ifft(descriptors).real

    # 将复数表示转换为点坐标
    reconstructed_contour = np.array([[int(np.round(pt.real)), int(np.round(pt.imag))] for pt in reconstructed_contour])

    # 确保重建的边界点数量足够大，避免步长为零的情况
    if len(reconstructed_contour) < num_points:
        num_points = len(reconstructed_contour)  # 使用重建边界的全部点

    # 下采样边界点
    step = len(reconstructed_contour) // num_points
    reconstructed_contour = reconstructed_contour[::step]

    return reconstructed_contour

# 读取图像
image = cv2.imread(image_url)

# 转换为灰度图像并执行Canny边缘检测
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray_image, 300, 800)

# 寻找边缘的轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 提取第一个轮廓的边界点
contour = contours[0]

# 计算傅里叶描述子（取前10个描述子）
num_descriptors = 10
descriptors = calculate_fourier_descriptors(contour, num_descriptors)

# 重建边界（取50个重建点）
num_reconstructed_points = 50
reconstructed_contour = reconstruct_contour(descriptors, num_reconstructed_points)

# 在图像上绘制重建的边界
reconstructed_image = image.copy()
cv2.drawContours(reconstructed_image, [reconstructed_contour], -1, (0, 0, 255), 2)

# 显示带有重建边界的图像
cv2.imshow('Reconstructed Contour', reconstructed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()



# Task5 分割方法
# 读取图像并灰度化
image = cv2.imread(image_url, cv2.IMREAD_GRAYSCALE)

# 方法一：基于阈值的分割
_, thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# 方法二：自适应阈值分割
thresh2 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# 方法三：边缘检测分割
edges = cv2.Canny(gray_image, 300, 800)

# 使用Sobel算子计算梯度
gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

# 计算梯度幅值和方向
gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
gradient_direction = np.arctan2(gradient_y, gradient_x)

# 对梯度幅值进行阈值处理
_, sure_fg = cv2.threshold(gradient_magnitude, 0.7 * gradient_magnitude.max(), 255, 0)

# 膨胀操作
sure_fg = np.uint8(sure_fg)
kernel = np.ones((3, 3), np.uint8)
sure_fg = cv2.dilate(sure_fg, kernel, iterations=3)

# 背景区域的确定
unknown = cv2.subtract(sure_fg, thresh1)

# 标记初始化
markers, num_markers = ndimage.label(sure_fg)
markers[unknown == 255] = num_markers + 1

# 分水岭算法
markers = cv2.watershed(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), markers)

# 转换成CV_8UC3格式
image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

# 在分水岭结果上标记边界
image_rgb[markers == -1] = [255, 0, 0]

# 绘制四个子图
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(thresh1, cmap='gray')
plt.title('Threshold Binary')

plt.subplot(2, 2, 2)
plt.imshow(thresh2, cmap='gray')
plt.title('Adaptive Threshold')

plt.subplot(2, 2, 3)
plt.imshow(edges, cmap='gray')
plt.title('Edge Detection')

plt.subplot(2, 2, 4)
plt.imshow(image_rgb)
plt.title('Watershed Segmentation')

plt.show()




# Task 6
def segment_image(image, num_segments):
    # 获取图像形状
    height, width, _ = image.shape

    # 将图像数据重塑为 (num_pixels, num_features) 的数组
    image_data = image.reshape((-1, 3))  # 对于RGB图像，3代表三个通道

    # 初始化K均值模型
    kmeans = KMeans(n_clusters=num_segments, random_state=0)

    # 在图像数据上拟合K均值模型
    kmeans.fit(image_data)

    # 获取每个像素的分割标签
    segmentation = kmeans.labels_

    # 将分割结果重塑回图像形状
    segmented_image = segmentation.reshape((height, width))

    return segmented_image


def visualize_segmentation(segmented_image):
    plt.figure(figsize=(8, 8))
    plt.imshow(segmented_image, cmap='viridis')
    plt.axis('off')
    plt.title('Segmented Image')
    plt.show()

image = cv2.imread(image_url)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 设置分割数
num_segments = 2

# 对图像进行分割
segmented_image = segment_image(image, num_segments)

# 可视化分割结果
visualize_segmentation(segmented_image)