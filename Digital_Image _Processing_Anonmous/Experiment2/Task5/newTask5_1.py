import cv2
import matplotlib.pyplot as plt



def NineCut(image, flag=0):
    imageNum = 0  # 初始化图像计数器
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 将输入图像转换为灰度图
    _, imageBinary = cv2.threshold(imageGray, 200, 255, cv2.THRESH_BINARY)  # 将灰度图像进行二值化

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 创建一个3x3的矩形结构元素

    imageNineErode = cv2.morphologyEx(imageBinary, cv2.MORPH_ERODE, kernel)  # 对二值化图像进行腐蚀操作
    contoursNine, hierarchyNine = cv2.findContours(imageNineErode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 找到图像中的外部轮廓

    imageROIs = []  # 存储分割后的图像块
    cornerBox = []  # 存储图像块的边界框
    for contour in contoursNine:  # 遍历每一个轮廓
        rect = cv2.minAreaRect(contour)  # 计算包围轮廓的最小面积矩形
        area = cv2.contourArea(contour)  # 计算轮廓的面积

        if area > 4000:  # 过滤掉面积小于4000的轮廓
            cornerPoint = cv2.boxPoints(rect)  # 获取矩形的四个角点
            x = [cornerPoint[0][0], cornerPoint[1][0], cornerPoint[2][0], cornerPoint[3][0]]  # 提取角点的x坐标
            y = [cornerPoint[0][1], cornerPoint[1][1], cornerPoint[2][1], cornerPoint[3][1]]  # 提取角点的y坐标

            x_min = int(min(x))  # 计算x坐标的最小值
            x_max = int(max(x))  # 计算x坐标的最大值
            y_min = int(min(y))  # 计算y坐标的最小值
            y_max = int(max(y))  # 计算y坐标的最大值

            cornerBox.append([x_min, x_max, y_min, y_max])  # 将边界框加入列表
            imageROI = image[y_min:y_max, x_min:x_max]  # 截取原图中的ROI（感兴趣区域）
            imageNum = imageNum + 1  # 更新图像计数器

            if flag == 1:  # 如果flag等于1，则将分割后的图像保存为文件
                cv2.imwrite(str(imageNum) + '.jpg', imageROI)
            imageROIs.append(imageROI)  # 将ROI加入列表

    return imageROIs, cornerBox  # 返回分割后的图像块和边界框


# 加载图像和模板
image_path = 'image.png'
template_path = 'target.jpg'
oriimage = cv2.imread(image_path)  # 读取为彩色图像
template = cv2.imread(template_path, 0)  # 读取为灰度图像

# 获取九宫格的矩形区域
useless, cornerBox = NineCut(oriimage, 0)

# 准备模板匹配
matching_results = []
threshold = 0.9  # 设定匹配阈值

for (x_min, x_max, y_min, y_max) in cornerBox:
    roi = oriimage[y_min:y_max, x_min:x_max]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) # 灰度图

    # 模板匹配
    res = cv2.matchTemplate(roi_gray, template, cv2.TM_CCOEFF_NORMED)

    # 找到匹配位置
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    if max_val >= threshold:
        top_left = max_loc  # 左上角的匹配位置

        # 记录匹配结果
        matching_results.append({
            "box": (x_min, x_max, y_min, y_max),
            "match_value": max_val,
            "match_location": top_left
        })

# 打印所有匹配结果
for result in matching_results:
    print(f"Box: {result['box']}, Match Value: {result['match_value']}, Match Location: {result['match_location']}")

# 可视化所有匹配结果
result_image = oriimage.copy()

for result in matching_results:
    (x_min, x_max, y_min, y_max) = result['box']
    (best_x, best_y) = result['match_location']

    cv2.rectangle(result_image, (x_min + best_x, y_min + best_y),
                  (x_min + best_x + template.shape[1], y_min + best_y + template.shape[0]), (0, 255, 0), 2)

# 保存结果图像
cv2.imwrite('result.png', result_image)
print("结果已保存为 'result.png'")