# -*- coding: utf-8 -*-
# Author : kaswary
# Time   : 2020/5/16 10:44


# 调整亮度和饱和度
# import numpy as np
# import cv2
# import os
# # 调整最大值
# MAX_VALUE = 100
#
# def update(input_img_path, output_img_path, lightness, saturation):
#     """
#     用于修改图片的亮度和饱和度
#     :param input_img_path: 图片路径
#     :param output_img_path: 输出图片路径
#     :param lightness: 亮度
#     :param saturation: 饱和度
#     """
#
#     # 加载图片 读取彩色图像归一化且转换为浮点型
#     image = cv2.imread(input_img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
#
#     # 颜色空间转换 BGR转为HLS
#     hlsImg = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
#
#     # 1.调整亮度（线性变换)
#     hlsImg[:, :, 1] = (1.0 + lightness / float(MAX_VALUE)) * hlsImg[:, :, 1]
#     hlsImg[:, :, 1][hlsImg[:, :, 1] > 1] = 1
#     # 饱和度
#     hlsImg[:, :, 2] = (1.0 + saturation / float(MAX_VALUE)) * hlsImg[:, :, 2]
#     hlsImg[:, :, 2][hlsImg[:, :, 2] > 1] = 1
#     # HLS2BGR
#     lsImg = cv2.cvtColor(hlsImg, cv2.COLOR_HLS2BGR) * 255
#     lsImg = lsImg.astype(np.uint8)
#     cv2.imwrite(output_img_path, lsImg)
#
#
# dataset_dir = "E:/machine_learning/Resnet/resnet50_new/data_green/valid/1"
# output_dir = "E:/machine_learning/Resnet/resnet50_new/data_green/valid_final/1"
#
# #这里调参！！！1
# lightness = int(input("lightness(亮度-100~+100):")) # 亮度
# saturation = int(input("saturation(饱和度-100~+100):")) # 饱和度
#
# # 获得需要转化的图片路径并生成目标路径
# image_filenames = [(os.path.join(dataset_dir, x), os.path.join(output_dir, x))
#                     for x in os.listdir(dataset_dir)]
# # 转化所有图片
# for path in image_filenames:
#     update(path[0], path[1], lightness, saturation)

# 旋转
# import cv2
# import numpy as np
# import os
#
# def XRotate(path, new1, new2, new3):
# # def XRotate(path, new2):
#     src = cv2.imread(path)
#     print("read")
#     #原图的高、宽 以及通道数
#     rows, cols, channel = src.shape
#
#     #绕图像的中心旋转
#     #参数：旋转中心 旋转度数 scale
#     M1 = cv2.getRotationMatrix2D((cols/2, cols/2), 90, 1)
#     M2 = cv2.getRotationMatrix2D((cols/2, rows/2), 180, 1)
#     M3 = cv2.getRotationMatrix2D((rows/2, cols/2), 180, 1)
#     #参数：原始图像 旋转参数 元素图像宽高
#     rotated1 = cv2.warpAffine(src, M1, (rows, cols))
#     rotated2 = cv2.warpAffine(src, M2, (cols, rows))
#     rotated3 = cv2.warpAffine(rotated1, M3, (rows, cols))
#
#     #显示图像
#     # cv2.imshow("src", src)
#     # cv2.imshow("rotated", rotated1)
#
#     #等待显示
#     # cv2.waitKey(0)
#     print("start write")
#     cv2.imwrite(new1, rotated1)
#     cv2.imwrite(new2, rotated2)
#     cv2.imwrite(new3, rotated3)
#     print("finish write")
#     # cv2.destroyAllWindows()
#
# data_base_dir = "E:/machine_learning/Resnet/resnet50_new/data_new/valid/1"#输入文件夹的路径
# outfile_dir = "E:/machine_learning/Resnet/resnet50_new/data_new/valid/1"#输出文件夹的路径
#
# # 获得需要转化的图片路径并生成目标路径
# image_filenames = [(os.path.join(data_base_dir, x), outfile_dir+'/'+x.split('.')[0]+'_new1.jpg',
#                     outfile_dir+'/'+x.split('.')[0]+'_new2.jpg', outfile_dir+'/'+x.split('.')[0]+'_new3.jpg',)
#                     for x in os.listdir(data_base_dir)]
# # 转化所有图片
# for path in image_filenames:
#     print(path[0], path[1])
#     XRotate(path[0], path[1], path[2], path[3])
#
#
# # image_filenames = [(os.path.join(data_base_dir, x), outfile_dir+'/'+x.split('.')[0]+'_new1.jpg')
# #                     for x in os.listdir(data_base_dir)]
# #
# # for path in image_filenames:
# #     print(path[0], path[1])
# #     XRotate(path[0], path[1])


# 分离三通道 转化为[R,R,R]/[G,G,G]
import numpy as np
import cv2  # 导入opencv模块
from PIL import Image
import os
import shutil

data_root = "E:/machine_learning/myOUBAO"  # get data root path
image_path = data_root + "/"  # image data set path
test_dir = image_path + "test/"
valid_dir = image_path + "valid/"
train_dir = image_path + "train/"
target_root = "E:/machine_learning/myOUBAO/rgb/"
target_test = target_root + "test/"
target_valid = target_root + "valid/"
target_train = target_root + "train/"

def rgb(path, new_path):
    image = cv2.imread(path)     # 读取要处理的图片
    B, G, R = cv2.split(image)  # 分离出图片的B，R，G颜色通道
    new = cv2.merge([R, R, R])  # 合并
    print("start write")
    cv2.imwrite(new_path, new)
    print("finish write")

cate = [valid_dir + x for x in os.listdir(valid_dir) if os.path.isdir(valid_dir + x)]

for name in cate:
    class_path = name + "/"
    for file in os.listdir(class_path):
        path = class_path + file
        temp = name.split('/')
        new_path = target_valid + temp[-1] + "/" +file
        print("new path: ", new_path)
        rgb(path, new_path)

print("finish")




