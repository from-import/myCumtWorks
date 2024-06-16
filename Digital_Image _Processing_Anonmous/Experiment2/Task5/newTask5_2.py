import cv2

# 加载图像
image_path = 'image.png'
image = cv2.imread(image_path, 0)  # 读取为灰度图像

# 进行二值化处理
_, imageBinary = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)

# 进行轮廓查找
contours, hierarchy = cv2.findContours(imageBinary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

MinArea = 4000
Nine = []
for index, contour in enumerate(contours):
    rect = cv2.minAreaRect(contour)
    area = cv2.contourArea(contour)

    # 筛选出面积大于 MinArea 的九宫格轮廓
    if area > MinArea:
        boxPoint = cv2.boxPoints(rect)
        x = [boxPoint[0][0], boxPoint[1][0], boxPoint[2][0], boxPoint[3][0]]
        y = [boxPoint[0][1], boxPoint[1][1], boxPoint[2][1], boxPoint[3][1]]
        x_min = int(min(x))
        x_max = int(max(x))
        y_min = int(min(y))
        y_max = int(max(y))
        Nine.append([x_min, x_max, y_min, y_max])

# 使用内部轮廓数量进行筛选
minNum = 10000
minIndex = -1

for i in range(len(Nine)):
    x1, x2, y1, y2 = Nine[i][0], Nine[i][1], Nine[i][2], Nine[i][3]
    roi = imageBinary[y1:y2, x1:x2]
    contoursCut, _ = cv2.findContours(roi, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # 选择包含最少轮廓数量的九宫格
    if len(contoursCut) < minNum:
        minNum = len(contoursCut)
        minIndex = i

print("MinNum: {}\tMinIndex: {}".format(minNum, minIndex))

# 提取最小轮廓数量的九宫格区域图像
resultImage = imageBinary[Nine[minIndex][2]:Nine[minIndex][3], Nine[minIndex][0]:Nine[minIndex][1]]

# 保存结果图像到指定目录
output_path = 'cut_region.png'
cv2.imwrite(output_path, resultImage)

print(f"结果已保存至 {output_path}")
