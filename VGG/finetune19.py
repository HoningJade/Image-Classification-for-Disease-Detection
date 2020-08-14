# -*- coding: utf-8 -*-
# Author : kaswary
# Time   : 2020/3/20 11:20


import tensorflow as tf
from vgg19 import Vgg19
from create_batch_images import ImageDataGenerator
from datetime import datetime
import glob
import os

learning_rate = 1e-3
num_epochs = 20  # 代的个数
num_iter = 25
batch_size = 16
num_classes = 2  # 类别标签
train_layer = ['fc8', 'fc7', 'fc6', 'conv5_4', 'conv5_3', 'conv5_2', 'conv5_1', 'conv4_4', 'conv4_3', 'conv4_2', 'conv4_1',
               'conv3_4', 'conv3_3', 'conv3_2', 'conv3_1', 'conv2_2', 'conv2_1', 'conv1_2', 'conv1_1']

filewriter_path = "D:/machine_learning/vgg/tensorboard19"  # 存储tensorBoard文件
checkpoint_path = "D:/machine_learning/vgg/checkpoints19"  # 训练好的模型和参数存放目录

training_0_image_path = 'D:/machine_learning/vgg/data/train_data/0/'
training_1_image_path = 'D:/machine_learning/vgg/data/train_data/1/'
testing_0_image_path = 'D:/machine_learning/vgg/data/validation_data/0/'
testing_1_image_path = 'D:/machine_learning/vgg/data/validation_data/1/'

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True

training_labels = []
testing_labels = []

# 加载训练数据
training_images = glob.glob(training_0_image_path + '*.jpg')
training_images[len(training_images):] = glob.glob(training_1_image_path + '*.jpg')
for i in range(len(training_images)):
    if i < 262:
        training_labels.append(0)
    else:
        training_labels.append(1)

training = ImageDataGenerator(
    images=training_images,
    labels=training_labels,
    batch_size=batch_size,
    num_classes=num_classes)

# 加载测试数据
testing_images = glob.glob(testing_0_image_path + '*.jpg')
testing_images[len(testing_images):] = glob.glob(testing_1_image_path + '*.jpg')
for i in range(len(testing_images)):
    if i < 61:
        testing_labels.append(0)
    else:
        testing_labels.append(1)

testing = ImageDataGenerator(
    images=testing_images,
    labels=testing_labels,
    batch_size=batch_size,
    num_classes=num_classes)

x = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
y = tf.placeholder(tf.float32, [batch_size, num_classes])
keep_prob = tf.placeholder(tf.float32)

# 图片数据通过VGG16网络处理
model = Vgg19(bgr_image=x, num_class=num_classes, vgg19_npy_path='D:/machine_learning/vgg/vgg19.npy')
score = model.fc8
y_ = model.prob

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layer]

# focal loss
with tf.name_scope('loss'):
    loss = tf.reduce_mean(- tf.reduce_sum(0.15 * y * tf.pow(1-y_, 5.) * tf.log(y_), reduction_indices=[1]))

gradients = tf.gradients(loss, var_list)
gradients = list(zip(gradients, var_list))

with tf.name_scope('optimizer'):
    # 优化器，采用梯度下降算法进行优化
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#    train_op = optimizer.apply_gradients(grads_and_vars=gradients)
    train_op = optimizer.minimize(loss)

with tf.name_scope("accuracy"):
    # 定义网络精确度
    correct_pred = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 把精确度加入到TensorBoard
tf.summary.scalar('loss', loss)

merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter(filewriter_path)
saver = tf.train.Saver()

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
# with tf.compat.v1.Session(config=config) as sess:
#    with tf.device("/gpu:0"):
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(sess=sess)

    # 把模型图加入TensorBoard
    writer.add_graph(sess.graph)

    # 总共训练20代
    for epoch in range(num_epochs):

        print("{} Epoch number: {} start".format(datetime.now(), epoch + 1))
        # 开始训练每一代

        for step in range(num_iter):
            img_batch = sess.run(training.image_batch)
            label_batch = sess.run(training.label_batch)
            _, train_loss = sess.run([train_op, loss], feed_dict={x: img_batch, y: label_batch})
            print('step ', step+1, ': ', train_loss)

        # 测试模型精确度
        print("{} Start testing".format(datetime.now()))

        test_accuracy = 0.
        for step in range(5):
            img_batch = sess.run(testing.image_batch)
            label_batch = sess.run(testing.label_batch)
            acc = sess.run(accuracy, feed_dict={x: img_batch, y: label_batch})
            test_accuracy += acc
        test_accuracy /= 5

        # # valid
        # tp = tn = fn = fp = 0
        #
        # for _ in range(5):
        #     img_batch = sess.run(testing.image_batch)
        #     label_batch = sess.run(testing.label_batch)
        #     softmax_prediction = sess.run(score, feed_dict={x: img_batch, y: label_batch})
        #     prediction_label = sess.run(tf.argmax(softmax_prediction, 1))
        #     actual_label = sess.run(tf.argmax(label_batch, 1))
        #
        #     print('prediction :', prediction_label)
        #     print('actual :', actual_label)
        #
        #     for i in range(len(prediction_label)):
        #         if prediction_label[i] == actual_label[i] == 1:
        #             tp += 1
        #         elif prediction_label[i] == actual_label[i] == 0:
        #             tn += 1
        #         elif prediction_label[i] == 1 and actual_label[i] == 0:
        #             fp += 1
        #         else:
        #             fn += 1
        #
        # precision = (tp + tn) / (tp + fp + tn + fn)
#        recall = tp / (tp + fn)
#        f1 = (2 * tp) / (2 * tp + fp + fn)  # f1为精确率precision和召回率recall的调和平均
        print("{} Testing accuracy = {:.4f}".format(datetime.now(), test_accuracy))
#        print("{} Testing Recall = {:.4f}".format(datetime.now(), recall))
#        print("{} Testing F1 = {:.4f}".format(datetime.now(), f1))

        # 把训练好的模型存储起来
        print("{} Saving checkpoint of model...".format(datetime.now()))

        checkpoint_name = os.path.join(checkpoint_path, 'model_epoch' + str(epoch + 1) + '.ckpt')
        save_path = saver.save(sess, checkpoint_name)

        print("{} Epoch number: {} end".format(datetime.now(), epoch + 1))
