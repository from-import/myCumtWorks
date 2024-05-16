import cv2

# 读取灰度直方图图片
image_gray = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)

# 对灰度图进行二值化处理
_, binary_image = cv2.threshold(image_gray, 128, 255, cv2.THRESH_BINARY)

# 显示原始灰度图和二值化图像
cv2.imshow('Gray Image', image_gray)
cv2.imshow('Binary Image', binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
