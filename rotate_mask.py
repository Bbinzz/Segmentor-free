import cv2
import numpy as np
import random
# 随机生成旋转角度
# angle = random.uniform(0, 360)
# angle = random.choice([180])
# angle = 180

def rotate_image(image,angle):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    return rotated_image

def rotate_mask(maskpath,angle=0):
    mask = cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask,(512,512))
    # 获取mask的中心点坐标
    center = tuple(np.array(mask.shape[1::-1]) // 2)

    # 生成旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)

    # 对mask进行仿射变换
    rotated_mask = cv2.warpAffine(mask, rotation_matrix, mask.shape[1::-1])
    
    # 使用阈值处理将图像转换为二值图像
    _, binary_mask = cv2.threshold(rotated_mask, 128, 255, cv2.THRESH_BINARY)
    # 显示原始mask和旋转后的mask
    # output_path = 'output_mask.png'
    # cv2.imwrite(output_path, binary_mask)

    return binary_mask

# 读取mask图片
# mask_path = '/data1/wzb/dataset/polyp_dataset/TrainDataset/masks/1.png'

# 随机生成旋转角度

# rotate_mask(maskpath=mask_path)