# -*- coding: utf-8 -*-
# Author : kaswary
# Time   : 2020/6/5 20:01

# loss function: focal foss
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import matplotlib.pyplot as plt
# from model import mynet
# import tensorflow as tf
# import json
# import numpy as np
# from keras.optimizers import SGD
# from sklearn.preprocessing import LabelBinarizer
# from sklearn.metrics import classification_report
# from keras import backend as K
#
# data_root = "D:/machine_learning/Resnet/resnet50_new"  # get data root path
# image_path = data_root + "/data_new/"  # image data set path
# train_dir = image_path + "train"
# validation_dir = image_path + "valid"
# test_dir = image_path + "test"
# #train_dir = image_path + "train_black"
# #validation_dir = image_path + "val_black"
#
# im_height = 224
# im_width = 224
# batch_size = 16
# epochs = 3
#
#
# def pre_function(img):
#     # img = im.open('test.jpg')
#     # img = np.array(img).astype(np.float32)
#     img = img / 255.
#     img = (img - 0.5) * 2.0
#     return img
#
# # data generator with data augmentation
# train_image_generator = ImageDataGenerator(horizontal_flip=True,
#                                            preprocessing_function=pre_function)
#
# validation_image_generator = ImageDataGenerator(preprocessing_function=pre_function)
#
# test_image_generator = ImageDataGenerator(preprocessing_function=pre_function)
#
# train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
#                                                            batch_size=batch_size,
#                                                            shuffle=True,
#                                                            target_size=(im_height, im_width),
#                                                            class_mode='categorical')
# total_train = train_data_gen.n
#
# # get class dict
# class_indices = train_data_gen.class_indices
#
# # transform value and key of dict
# inverse_dict = dict((val, key) for key, val in class_indices.items())
# # write dict into json file
# json_str = json.dumps(inverse_dict, indent=4)
# with open('class_indices.json', 'w') as json_file:
#     json_file.write(json_str)
#
# # shuffle false->true
# val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,
#                                                               batch_size=batch_size,
#                                                               shuffle=True,
#                                                               target_size=(im_height, im_width),
#                                                               class_mode='categorical')
# # img, _ = next(train_data_gen)
# total_val = val_data_gen.n
#
# test_data_gen = test_image_generator.flow_from_directory(directory=test_dir,
#                                                               batch_size=batch_size,
#                                                               shuffle=True,
#                                                               target_size=(im_height, im_width),
#                                                               class_mode='categorical')
# total_test = test_data_gen.n
#
# def loss_function(true, logits):
#     y_ = tf.nn.softmax(logits=logits)
#     loss = tf.reduce_mean(- tf.reduce_sum(0.25 * true * tf.pow(1-y_, 2.) * tf.math.log(y_)))
#     # pred = K.eval(tf.argmax(y_, axis=1))
#     # label = K.eval(tf.argmax(true, axis=1))
#     print("true: ", true)
#     # print("pred: ", pred)
#     return loss
# # 'binary_crossentropy'
# mynet.compile(loss=loss_function,
#               optimizer=tf.keras.optimizers.SGD(lr=1e-3),
#               metrics=['accuracy'])
#
# print("[INFO] training w/ generator...")
# history = mynet.fit_generator(train_data_gen,
#                             steps_per_epoch=total_train // batch_size,
#                             epochs=epochs,
#                             validation_data=val_data_gen,
#                             validation_steps=total_val // batch_size)
# predIdxs1 = mynet.predict_generator(train_data_gen,
# 	steps=(total_train // batch_size))
# predIdxs1 = np.argmax(predIdxs1, axis=1)
# print("train_pred", predIdxs1)
#
# predIdxs2 = mynet.predict_generator(val_data_gen,
# 	steps=(total_val // batch_size))
# predIdxs2 = np.argmax(predIdxs2, axis=1)
# print("valid_pred", predIdxs2)
#
# mynet.save_weights("D:/machine_learning/DenseNet/save_weights/DenseNet.ckpt", save_format="tf")
#
# # plot the training loss and accuracy
# N = epochs
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(np.arange(1, N+1), history.history["loss"], label="train_loss")
# plt.plot(np.arange(1, N+1), history.history["val_loss"], label="val_loss")
# plt.plot(np.arange(1, N+1), history.history["accuracy"], label="train_acc")
# plt.plot(np.arange(1, N+1), history.history["val_accuracy"], label="val_acc")
# plt.title("Training Loss and Accuracy on Dataset")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend(loc="lower left")
# plt.savefig("D:/machine_learning/DenseNet/result/train_loss+acc.jpg")
# plt.show()


# loss function: cross entropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from model import mynet, net_test
import tensorflow as tf
import json
from keras import backend as K
import os
import PIL.Image as im
import numpy as np


data_root = "D:/machine_learning/Resnet/resnet50_new"  # get data root path
image_path = data_root + "/data_new/"  # image data set path
train_dir = image_path + "train"
validation_dir = image_path + "valid"
test_dir = image_path + "test"
#train_dir = image_path + "train_black"
#validation_dir = image_path + "val_black"



im_height = 224
im_width = 224
batch_size = 32
epochs = 15


def pre_function(img):
    # img = im.open('test.jpg')
    # img = np.array(img).astype(np.float32)
    img = img / 255.
    img = (img - 0.5) * 2.0
    return img

# data generator with data augmentation
train_image_generator = ImageDataGenerator(horizontal_flip=True,
                                           preprocessing_function=pre_function)

validation_image_generator = ImageDataGenerator(preprocessing_function=pre_function)

train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                           batch_size=batch_size,
                                                           shuffle=True,
                                                           target_size=(im_height, im_width),
                                                           class_mode='categorical')
total_train = train_data_gen.n

# get class dict
class_indices = train_data_gen.class_indices

# transform value and key of dict
inverse_dict = dict((val, key) for key, val in class_indices.items())
# write dict into json file
json_str = json.dumps(inverse_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

# shuffle false->true
val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,
                                                              batch_size=batch_size,
                                                              shuffle=True,
                                                              target_size=(im_height, im_width),
                                                              class_mode='categorical')
# img, _ = next(train_data_gen)
total_val = val_data_gen.n


# using keras low level api for training
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
# train_auc = tf.keras.metrics.Mean(name='train_auc')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
# test_auc = tf.keras.metrics.Mean(name='test_tp')

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        output = mynet(images, training=True)
        loss = loss_object(labels, output)
    gradients = tape.gradient(loss, mynet.trainable_variables)
    optimizer.apply_gradients(zip(gradients, mynet.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, output)
    # aucvalue = auc(labels, output, weights=None, num_thresholds=200, name=None, summation_method='trapezoidal')
    # train_auc(aucvalue)
    pred = tf.argmax(tf.nn.softmax(logits=output), 1)
    return pred

@tf.function
def test_step(images, labels):
    output = net_test(images, training=False)
    t_loss = loss_object(labels, output)

    test_loss(t_loss)
    test_accuracy(labels, output)
    # aucvalue = auc(labels, output, weights=None, num_thresholds=200, name=None, summation_method='trapezoidal')
    # test_auc(aucvalue)
    pred = tf.argmax(tf.nn.softmax(logits=output), 1)
    return pred

loss_train = []
loss_test = []
acc_train = []
acc_test = []


best_test_loss = float('inf')
for epoch in range(1, epochs + 1):
    train_loss.reset_states()  # clear history info
    train_accuracy.reset_states()  # clear history info
    test_loss.reset_states()  # clear history info
    test_accuracy.reset_states()  # clear history info
    #test_auc.reset_states()         # clear history info
    #train_auc.reset_states()         # clear history info

    # train
    for step in range(total_train // batch_size):
        images, labels = next(train_data_gen)
        train_pred = K.eval(train_step(images, labels))
        labels = K.eval(tf.argmax(labels, 1))
        print("true: ", labels)
        print("pred: ", train_pred)

        # print train process
        #rate = (step + 1) / (total_train // batch_size)
        #a = "*" * int(rate * 50)
        #b = "." * int((1 - rate) * 50)
        #acc = train_accuracy.result().numpy()
        #print("\r[{}]train acc: {:^3.0f}%[{}->{}]{:.4f}".format(epoch, int(rate * 100), a, b, acc), end="")
    #print()
    mynet.save_weights("D:/machine_learning/DenseNet/save_weights/DenseNet1.ckpt", save_format="tf")

    # validate
    net_test.load_weights(filepath="D:/machine_learning/DenseNet/save_weights/DenseNet1.ckpt")
    for step in range(5):
        test_images, test_labels = next(val_data_gen)
        test_pred = K.eval(test_step(test_images, test_labels))
        test_labels = K.eval(tf.argmax(test_labels, 1))
        print("valid true: ", test_labels)
        print("valid pred: ", test_pred)

    loss_train.append(K.eval(train_loss.result()))
    loss_test.append(K.eval(test_loss.result()))
    acc_train.append(K.eval(train_accuracy.result()))
    acc_test.append(K.eval(test_accuracy.result()))

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Valid Loss: {}, Valid Accuracy: {}'
    print(template.format(epoch,
                          K.eval(train_loss.result()),
                          K.eval(train_accuracy.result() * 100),
                          K.eval(test_loss.result()),
                          K.eval(test_accuracy.result() * 100,)))
    if test_loss.result() < best_test_loss:
        best_test_loss = test_loss.result()

# plot the training loss and accuracy
N = epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(1, N+1), loss_train, label="train_loss")
plt.plot(np.arange(1, N+1), loss_test, label="val_loss")
plt.plot(np.arange(1, N+1), acc_train, label="train_acc")
plt.plot(np.arange(1, N+1), acc_test, label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("D:/machine_learning/DenseNet/result/train_loss+acc.jpg")
plt.show()
