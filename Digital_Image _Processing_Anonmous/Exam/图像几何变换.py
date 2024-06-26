import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('image.jpg')
rows, cols = image.shape[:2]

# 平移
M_translation = np.float32([[1, 0, 50], [0, 1, 30]])
image_translation = cv2.warpAffine(image, M_translation, (cols, rows))

# 旋转
M_rotation = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
image_rotation = cv2.warpAffine(image, M_rotation, (cols, rows))

# 缩放
image_scaling = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

# 镜像
image_flip_horizontal = cv2.flip(image, 1)
image_flip_vertical = cv2.flip(image, 0)

# 剪切
M_shear = np.float32([[1, 0.5, 0], [0, 1, 0]])
image_shear = cv2.warpAffine(image, M_shear, (int(cols*1.5), rows))

# 仿射变换
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
M_affine = cv2.getAffineTransform(pts1, pts2)
image_affine = cv2.warpAffine(image, M_affine, (cols, rows))

# 投影变换
pts1 = np.float32([[50, 50], [200, 50], [50, 200], [200, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250], [300, 200]])
M_perspective = cv2.getPerspectiveTransform(pts1, pts2)
image_perspective = cv2.warpPerspective(image, M_perspective, (cols, rows))

# 显示变换后的图像
plt.figure(figsize=(12, 6))

plt.subplot(2, 4, 1)
plt.title('Original')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(2, 4, 2)
plt.title('Translation')
plt.imshow(cv2.cvtColor(image_translation, cv2.COLOR_BGR2RGB))

plt.subplot(2, 4, 3)
plt.title('Rotation')
plt.imshow(cv2.cvtColor(image_rotation, cv2.COLOR_BGR2RGB))

plt.subplot(2, 4, 4)
plt.title('Scaling')
plt.imshow(cv2.cvtColor(image_scaling, cv2.COLOR_BGR2RGB))

plt.subplot(2, 4, 5)
plt.title('Horizontal Flip')
plt.imshow(cv2.cvtColor(image_flip_horizontal, cv2.COLOR_BGR2RGB))

plt.subplot(2, 4, 6)
plt.title('Vertical Flip')
plt.imshow(cv2.cvtColor(image_flip_vertical, cv2.COLOR_BGR2RGB))

plt.subplot(2, 4, 7)
plt.title('Shear')
plt.imshow(cv2.cvtColor(image_shear, cv2.COLOR_BGR2RGB))

plt.subplot(2, 4, 8)
plt.title('Affine')
plt.imshow(cv2.cvtColor(image_affine, cv2.COLOR_BGR2RGB))

plt.subplot(2, 4, 9)
plt.title('Perspective')
plt.imshow(cv2.cvtColor(image_perspective, cv2.COLOR_BGR2RGB))

plt.tight_layout()
plt.show()
