import cv2
import numpy as np
import matplotlib.pyplot as plt

image_url = "lotus.jpg"
image = cv2.imread(image_url)
grayImage = cv2.imread(image_url,0)


def pltImageShow(image1,name1,image2,name2):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image1, cmap='gray')
    plt.title(name1)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(image2, cmap='gray')
    plt.title(name2)
    plt.axis('off')
    plt.tight_layout()
    plt.show()



# 将图像转换为灰度图
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 显示灰度图像
plt.imshow(gray_image, cmap='gray')
plt.title('灰度图像',fontproperties="SimSun")
plt.axis('off')  # 不显示坐标轴
plt.show()

# 计算灰度直方图
histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

# 绘制灰度直方图
plt.figure()
plt.title('灰度直方图',fontproperties="SimSun")
plt.xlabel('像素强度',fontproperties="SimSun")
plt.ylabel('频数',fontproperties="SimSun")
plt.plot(histogram)
plt.xlim([0, 256])  # 设置 x 轴范围为 [0, 256]（256个灰度级）
plt.show()


# Task2

def piecewise_linear_transform(pixel_value):
    # 定义分段线性变换函数
    if pixel_value < 100:
        return pixel_value  # 小于100的部分保持不变
    else:
        return int(255 * (pixel_value - 100) / (255 - 100))

transformed_image = np.vectorize(piecewise_linear_transform)(grayImage) # 应用变换
pltImageShow(grayImage,'Original Image', transformed_image,'Transformed Image') # 显示结果


# Task3
equalizedImage = cv2.equalizeHist(grayImage) # 应用直方图均衡化
pltImageShow(grayImage,'Original Gray Image',equalizedImage,'Equalized Image')

# Task4
pseudocolorImage = cv2.applyColorMap(grayImage, cv2.COLORMAP_JET)
pltImageShow(grayImage,'Gray Image',pseudocolorImage,'Pseudocolor Image')

# Task5
mean = 0 # 添加高斯噪声
sigma = 25  # 调整噪声强度
noisyImage = grayImage + np.random.normal(mean, sigma, grayImage.shape)
noisyImage = np.clip(noisyImage, 0, 255)  # 将像素值限制在0到255之间
noisyImage = noisyImage.astype(np.uint8)

# 使用高斯模糊进行平滑处理
blurredImage = cv2.GaussianBlur(noisyImage, (5, 5), 0)  # 调整核大小和标准差

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(grayImage, cmap='gray')
plt.title('Original Gray Image')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(noisyImage, cmap='gray')
plt.title('Noisy Image')
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(blurredImage, cmap='gray')
plt.title('Blurred Image')
plt.axis('off')
plt.tight_layout()
plt.show()

# task6
sobel_x = cv2.Sobel(grayImage, cv2.CV_64F, 1, 0, ksize=3)  # 水平方向
sobel_y = cv2.Sobel(grayImage, cv2.CV_64F, 0, 1, ksize=3)  # 垂直方向
sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2) # 计算梯度幅值
sobel_magnitude = np.uint8(sobel_magnitude / np.max(sobel_magnitude) * 255) # 将结果转换为8位图像格式
alpha = 1.5  # 原图像的权重
beta = -0.5  # 边缘增强部分的权重
sharpened_image = cv2.addWeighted(grayImage, alpha, sobel_magnitude, beta, 0)
pltImageShow(grayImage,'Original Image',sharpened_image,'Sharpened Image')


# Task7
low_threshold = 50 # 设定低阈值和高阈值
high_threshold = 150

edges = cv2.Canny(grayImage, low_threshold, high_threshold) # 使用Canny算子检测边缘
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) # 定义一个3x3的内核
morph_gradient = cv2.morphologyEx(grayImage, cv2.MORPH_GRADIENT, kernel) # 计算形态学梯度

# 进行图像锐化，alpha和beta是调节锐化强度的参数
sharpened_image = cv2.addWeighted(grayImage, 1.0, edges, 1.0, 0)
sharpened_image = cv2.addWeighted(sharpened_image, 1.0, morph_gradient, 1.0, 0)

# 使用matplotlib显示结果
plt.figure(figsize=(10, 7))

plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(grayImage, cmap='gray')

plt.subplot(2, 2, 2)
plt.title('Canny Edges')
plt.imshow(edges, cmap='gray')

plt.subplot(2, 2, 3)
plt.title('Morphological Gradient')
plt.imshow(morph_gradient, cmap='gray')

plt.subplot(2, 2, 4)
plt.title('Sharpened Image')
plt.imshow(sharpened_image, cmap='gray')

plt.show()

# Task8
# 对图像进行傅里叶变换
dft = cv2.dft(np.float32(grayImage), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# 获取图像的大小
rows, cols = grayImage.shape
crow, ccol = rows // 2 , cols // 2

# 创建低通滤波器掩码
mask = np.zeros((rows, cols, 2), np.uint8)
r = 30  # 半径
center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]
mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
mask[mask_area] = 1

# 创建高通滤波器掩码
mask_high = 1 - mask

# 应用低通滤波器
fshift_low = dft_shift * mask

# 应用高通滤波器
fshift_high = dft_shift * mask_high

# 低通滤波结果的逆傅里叶变换
f_ishift_low = np.fft.ifftshift(fshift_low)
img_back_low = cv2.idft(f_ishift_low)
img_back_low = cv2.magnitude(img_back_low[:, :, 0], img_back_low[:, :, 1])

# 高通滤波结果的逆傅里叶变换
f_ishift_high = np.fft.ifftshift(fshift_high)
img_back_high = cv2.idft(f_ishift_high)
img_back_high = cv2.magnitude(img_back_high[:, :, 0], img_back_high[:, :, 1])

# 使用matplotlib显示结果
plt.figure(figsize=(10, 7))

plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(grayImage, cmap='gray')

plt.subplot(2, 2, 2)
plt.title('Low Pass Filtered Image')
plt.imshow(img_back_low, cmap='gray')

plt.subplot(2, 2, 3)
plt.title('High Pass Filtered Image')
plt.imshow(img_back_high, cmap='gray')

plt.show()