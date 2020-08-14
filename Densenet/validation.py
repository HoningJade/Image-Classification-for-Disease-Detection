# -*- coding: utf-8 -*-
# Author : kaswary
# Time   : 2020/6/6 17:53


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import mynet, net_test
import tensorflow as tf
from keras import backend as K
import numpy as np

data_root = "D:/machine_learning/Resnet/resnet50_new"  # get data root path
image_path = data_root + "/data_new/"  # image data set path
test_dir = image_path + "train"

im_height = 224
im_width = 224
batch_size = 16
epochs = 10

def pre_function(img):
    img = img / 255.
    img = (img - 0.5) * 2.0
    return img

test_image_generator = ImageDataGenerator(preprocessing_function=pre_function)

test_data_gen = test_image_generator.flow_from_directory(directory=test_dir,
                                                              batch_size=batch_size,
                                                              shuffle=True,
                                                              target_size=(im_height, im_width),
                                                              class_mode='categorical')
total_val = test_data_gen.n

mynet.load_weights(filepath="D:/machine_learning/DenseNet/save_weights/DenseNet1.ckpt")


# using keras low level api for training
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')


@tf.function
def test_step(images, labels):
    output = net_test(images, training=False)
    t_loss = loss_object(labels, output)

    test_loss(t_loss)
    test_accuracy(labels, output)
    pred = tf.argmax(tf.nn.softmax(logits=output), 1)
    return pred

total_acc = 0.
total_loss = 0.
time = 0
best_test_loss = float('inf')
tp = tn = fp = fn = 0
# for epoch in range(1, epochs + 1):
test_accuracy.reset_states()  # clear history info
test_loss.reset_states()  # clear history info
# test
# print("\n", "epoch: ", epoch)
for step in range(20):
    test_images, test_labels = next(test_data_gen)
    pred = K.eval(test_step(test_images, test_labels))
    true = K.eval(tf.argmax(test_labels, 1))
    loss = K.eval(test_loss.result())
    acc = K.eval(test_accuracy.result()*100)
    total_acc = total_acc + acc
    total_loss = total_loss + loss
    time = time + 1

    for i in range(len(true)):
        if true[i] == 0 and pred[i] == 0:
            tp = tp+1
        elif true[i] == 1 and pred[i] == 0:
            fp = fp+1
        elif true[i] == 1 and pred[i] == 1:
            tn = tn+1
        else:
            fn = fn+1

    print("test true: ", true)
    print("test pred: ", pred)
    print("test loss: ", loss, " test acc: ", acc)

print("\n", "tp: ", tp, "fp: ", fp, "tn: ", tn, "fn: ", fn)

print("average loss:", total_loss/time, "average acc:", total_acc/time)
