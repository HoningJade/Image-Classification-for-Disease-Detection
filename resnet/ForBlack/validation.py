# -*- coding: utf-8 -*-
# Author : kaswary
# Time   : 2020/5/24 10:12

# validation 并将fp fn tp tn分别放入对应文件夹
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import resnet101
from model import resnet50
import tensorflow as tf
from keras import backend as K
import shutil

data_root = "D:/machine_learning/Resnet/resnet50_new"  # get data root path
image_path = data_root + "/data_new/"  # image data set path
test_dir = image_path + "valid/"
test_pos = test_dir + "0"
test_neg = test_dir + "1"

save_dir_t0 = "D:/machine_learning/Resnet/resnet50_new/result/difference/valid/hblc/tp/"
save_dir_t1 = "D:/machine_learning/Resnet/resnet50_new/result/difference/valid/hblc/tn/"
save_dir_f0 = "D:/machine_learning/Resnet/resnet50_new/result/difference/valid/hblc/fp/"
save_dir_f1 = "D:/machine_learning/Resnet/resnet50_new/result/difference/valid/hblc/fn/"

im_height = 224
im_width = 224
batch_size = 16
epochs = 10

from PIL import Image
import numpy as np
from PIL import Image
from keras.preprocessing import image
import matplotlib.pyplot as plt
import os
import cv2
# from scipy.misc import toimage
import matplotlib
# 生成图片地址和对应标签
image_list = []
label_list = []
add_list = []
name_list = []
cate = [test_dir + x for x in os.listdir(test_dir) if os.path.isdir(test_dir + x)]

for name in cate:
    class_path = name + "/"
    for file in os.listdir(class_path):
        print(file)
        temp = name.split('/')
        if temp[-1] == '0':
            label_list.append(0)
        else:
            label_list.append(1)
        add_list.append(class_path + file)
        name_list.append(file)

        img_obj = Image.open(class_path + file)  # 读取图片
        img_array = np.array(img_obj)
        resized = cv2.resize(img_array, (im_height, im_width))  # 裁剪
        resized = resized.astype('float32')
        resized /= 255.
        resized = (resized - 0.5) * 2.0
        resized = np.array([resized])
        if(len(resized.shape)==3):
            resized = resized[:, :, :, np.newaxis]
            resized = np.concatenate((resized, resized, resized), axis=-1)
            print("   ---------------------   ")
        print(resized.shape)
        image_list.append(resized)

print("finished")
print(label_list)
print(add_list)

model = resnet50(num_classes=2, include_top=True)
model.load_weights(filepath="D:/machine_learning/Resnet/resnet50_new/save_weights/resNet_50_0.25+2_new1.ckpt")

tp = tn = fp = fn = 0
# test
for i in range(len(label_list)):
    test_pred = model(image_list[i])
    test_pred = tf.argmax(tf.nn.softmax(logits=test_pred), 1)
    pred = K.eval(test_pred)
    if label_list[i] == 0 and pred == 0:
        tp = tp+1
        shutil.copy(add_list[i], save_dir_t0+name_list[i])
    elif label_list[i] == 1 and pred == 0:
        fp = fp+1
        shutil.copy(add_list[i], save_dir_f0+name_list[i])
    elif label_list[i] == 1 and pred == 1:
        tn = tn+1
        shutil.copy(add_list[i], save_dir_t1+name_list[i])
    else:
        fn = fn+1
        shutil.copy(add_list[i], save_dir_f1+name_list[i])

    print(i, " finish and continue")

print("\n", "tp: ", tp, "fp: ", fp, "tn: ", tn, "fn: ", fn)



# # focal loss validation
# def pre_function(img):
#     img = img / 255.
#     img = (img - 0.5) * 2.0
#     return img
#
# test_image_generator = ImageDataGenerator(preprocessing_function=pre_function)
#
# test_data_gen = test_image_generator.flow_from_directory(directory=test_dir,
#                                                               batch_size=batch_size,
#                                                               shuffle=False,
#                                                               target_size=(im_height, im_width),
#                                                               class_mode='categorical')
# total_val = test_data_gen.n
#
# model = resnet50(num_classes=2, include_top=True)
# model.load_weights(filepath="D:/machine_learning/Resnet/resnet50_new/save_weights/resNet50_OUBAO_01.ckpt")
#
#
# def loss_function(true, logits):
#     y_ = tf.nn.softmax(logits=logits)
#     loss = tf.reduce_mean(- tf.reduce_sum(0.25 * true * tf.pow(1-y_, 2.) * tf.math.log(y_)))
#     return loss
#
# test_loss = tf.keras.metrics.Mean(name='test_loss')
# test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
#
# @tf.function
# def test_step(images, labels):
#     output = model(images)
#     t_loss = loss_function(labels, output)
#     pred = tf.argmax(tf.nn.softmax(logits=output), 1)
#     test_accuracy(labels, output)
#     return t_loss, pred
#
# total_acc = 0.
# total_loss = 0.
# time = 0
# best_test_loss = float('inf')
# tp = tn = fp = fn = 0
# for epoch in range(1, epochs + 1):
#     test_accuracy.reset_states()  # clear history info
#     # test
#     print("\n", "epoch: ", epoch)
#     for step in range(total_val // batch_size):
#         test_images, test_labels = next(test_data_gen)
#         test_loss, test_pred = test_step(test_images, test_labels)
#         test_labels = tf.argmax(test_labels, 1)
#         pred = K.eval(test_pred)
#         true = K.eval(test_labels)
#         loss = K.eval(test_loss)
#         acc = K.eval(test_accuracy.result()*100)
#
#         total_acc = total_acc + acc
#         total_loss = total_loss + loss
#         time = time + 1
#
#         # tensorflow中输入为tensor，if和for语句失效！！
#         for i in range(len(true)):
#             if true[i] == 0 and pred[i] == 0:
#                 tp = tp+1
#                 # shutil.copy(test_pos+test_images[i], save_dir+name)
#             elif true[i] == 1 and pred[i] == 0:
#                 fp = fp+1
#             elif true[i] == 1 and pred[i] == 1:
#                 tn = tn+1
#             else:
#                 fn = fn+1
#
#         print("test true: ", true)
#         print("test pred: ", pred)
#         print("test loss: ", loss, " test acc: ", acc)
#
#     print("\n", "tp: ", tp, "fp: ", fp, "tn: ", tn, "fn: ", fn)
#
# print("\n", "average loss:", total_loss/time, "average acc:", total_acc/time)


