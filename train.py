# -*- coding: utf-8 -*-
# Author : kaswary
# Time   : 2020/5/7 21:10


# loss function: cross entropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from model import resnet50
from model import resnet101
import tensorflow as tf
import json
from keras import backend as K
import os
import PIL.Image as im
import numpy as np


data_root = "D:/machine_learning/myOUBAO/"  # get data root path
image_path = data_root   # image data set path
train_dir = image_path + "train"
validation_dir = image_path + "valid"
test_dir = image_path + "test"
#train_dir = image_path + "train_black"
#validation_dir = image_path + "val_black"



im_height = 224
im_width = 224
batch_size = 16
epochs = 20


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

test_image_generator = ImageDataGenerator(preprocessing_function=pre_function)

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

# shuffle false->true
test_data_gen = test_image_generator.flow_from_directory(directory=test_dir,
                                                              batch_size=batch_size,
                                                              shuffle=True,
                                                              target_size=(im_height, im_width),
                                                              class_mode='categorical')
# img, _ = next(train_data_gen)
total_test = test_data_gen.n

model = resnet50(num_classes=2, include_top=True)
# transfer时加载已训练参数
# model.load_weights(filepath="D:/machine_learning/Resnet/resnet50_new/save_weights/resNet50_OUBAO_01.ckpt")
model.summary()


# using keras low level api for training
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
# train_auc = tf.keras.metrics.Mean(name='train_auc')

valid_loss = tf.keras.metrics.Mean(name='valid_loss')
valid_accuracy = tf.keras.metrics.CategoricalAccuracy(name='valid_accuracy')
# test_auc = tf.keras.metrics.Mean(name='test_tp')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
# test_auc = tf.keras.metrics.Mean(name='test_tp')

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        output = model(images, training=True)
        loss = loss_object(labels, output)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, output)
    pred = tf.argmax(tf.nn.softmax(logits=output), 1)
    return pred

@tf.function
def valid_step(images, labels):
    output = model(images, training=False)
    t_loss = loss_object(labels, output)

    valid_loss(t_loss)
    valid_accuracy(labels, output)
    pred = tf.argmax(tf.nn.softmax(logits=output), 1)
    return pred

@tf.function
def test_step(images, labels):
    output = model(images, training=False)
    t_loss = loss_object(labels, output)

    test_loss(t_loss)
    test_accuracy(labels, output)
    pred = tf.argmax(tf.nn.softmax(logits=output), 1)
    return pred

loss_train = []
loss_test = []
loss_valid = []

acc_train = []
acc_test = []
acc_valid = []


best_test_loss = float('inf')
for epoch in range(1, epochs + 1):
    train_loss.reset_states()  # clear history info
    train_accuracy.reset_states()  # clear history info
    test_loss.reset_states()  # clear history info
    test_accuracy.reset_states()  # clear history info
    valid_loss.reset_states()  # clear history info
    valid_accuracy.reset_states()  # clear history info

    # train
    for step in range(total_train // batch_size):
        images, labels = next(train_data_gen)
        train_pred = K.eval(train_step(images, labels))
        labels = K.eval(tf.argmax(labels, 1))
        print("true: ", labels)
        print("pred: ", train_pred)

    # validate
    for step in range(5):
        valid_images, valid_labels = next(val_data_gen)
        valid_pred = K.eval(valid_step(valid_images, valid_labels))
        valid_labels = K.eval(tf.argmax(valid_labels, 1))
        print("valid true: ", valid_labels)
        print("valid pred: ", valid_pred)

    # test
    for step in range(5):
        test_images, test_labels = next(test_data_gen)
        test_pred = K.eval(test_step(test_images, test_labels))
        test_labels = K.eval(tf.argmax(test_labels, 1))
        print("test true: ", test_labels)
        print("test pred: ", test_pred)

    loss_train.append(K.eval(train_loss.result()))
    loss_valid.append(K.eval(valid_loss.result()))
    loss_test.append(K.eval(test_loss.result()))
    acc_train.append(K.eval(train_accuracy.result()))
    acc_valid.append(K.eval(valid_accuracy.result()))
    acc_test.append(K.eval(test_accuracy.result()))

    template = 'Epoch {}, Loss: {}, Accuracy: {}'
    print(template.format(epoch,
                          K.eval(train_loss.result()),
                          K.eval(train_accuracy.result() * 100)))
    template = 'Valid Loss: {}, Valid Accuracy: {}, test Loss: {}, test Accuracy: {}'
    print(template.format(K.eval(valid_loss.result()),
                          K.eval(valid_accuracy.result() * 100,),
                          K.eval(test_loss.result()),
                          K.eval(test_accuracy.result() * 100,)))

    if test_loss.result() < best_test_loss:
        best_test_loss = test_loss.result()

    model.save_weights("D:/machine_learning/Resnet/resnet50_new/save_weights/resNet50_OUBAO_01_3lines.ckpt", save_format="tf")

x = [i+1 for i in range(epochs)]
fig_loss = plt.figure()
ax = fig_loss.add_subplot(111)
ax.plot(x, loss_train, label="train")
ax.plot(x, loss_valid, label="valid")
ax.plot(x, loss_test, label="test")
ax.legend(loc=0)
plt.savefig('D:/machine_learning/Resnet/resnet50_new/result/loss_OUBAO_01_3lines.jpg')
plt.show()
plt.close(fig_loss)

fig_acc = plt.figure()
ax1 = fig_acc.add_subplot(111)
ax1.plot(x, acc_train, label="train")
ax1.plot(x, acc_valid, label="valid")
ax1.plot(x, acc_test, label="test")
ax1.legend(loc=0)
plt.savefig('D:/machine_learning/Resnet/resnet50_new/result/acc_OUBAO_01_3lines.jpg')
plt.show()

