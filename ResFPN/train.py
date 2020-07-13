from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from FPN import resnet50
import tensorflow as tf
import json
import os
import PIL.Image as im
import numpy as np
from auc import auc

CUDA_VISIBLE_DEVICES = 1

data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
image_path = data_root + "/data/"  # image data set path
if not os.path.exists("save_weights"):
    os.makedirs("save_weights")

# train_dir = image_path + "train_black"
# validation_dir = image_path + "val_black"
# train_dir = image_path + "train_green"
# validation_dir = image_path + "val_green"
# train_dir = image_path + "train_oubao"
# validation_dir = image_path + "val_oubao"

im_height = 224
im_width = 224
# batch_size = 73   for black
# batch_size = 46  for green
# batch_size = 57  # for oubao
epochs = 25
k = 1

def pre_function(img):
    # img = im.open('test.jpg')
    # img = np.array(img).astype(np.float32)
    img = img / 255.
    # img = (img - 0.5) * 2.0
    return img


# focal loss, 0.25 for 正负例; 2 for 难训练的样本  0.15 for black without argu and 0.3 with argu; 0.3 for tnb
def loss_function(true, logits):
    # logits = tf.nn.softmax(logits=logits)
    # pred = tf.argmax(logits, 1)
    real1 = tf.argmax(true, 1)
    real2 = tf.abs(real1 - 1)
    true_len = logits.get_shape()
    cat =tf.zeros([true_len[0], 1], dtype=tf.int64)
    real1 = tf.reshape(real1, [true_len[0], 1])
    real2 = tf.reshape(real2, [true_len[0], 1])
    true1 = tf.cast(tf.concat([cat, real1], -1), tf.float32)
    true2 = tf.cast(tf.concat([cat, real2], -1), tf.float32)
    alpha = 0.2
    los1 = -alpha * true1 * tf.pow(1-logits, 2) * tf.math.log(logits)
    los2 = -(1-alpha) * true2 * tf.pow(logits, 2) * tf.math.log(tf.clip_by_value(1-logits, 1e-10, tf.reduce_max(1-logits)))  # tf.math.log(1-logits)
    los = tf.reduce_sum(los1 + los2) / 3.5
    return los

# data generator with data augmentation
train_image_generator = ImageDataGenerator(horizontal_flip=True,
                                           preprocessing_function=pre_function)

validation_image_generator = ImageDataGenerator(preprocessing_function=pre_function)


model = resnet50(num_classes=2)
# model = resnet50(num_classes=2, include_top=True)
model.summary()


# using keras low level api for training
# loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
train_auc = tf.keras.metrics.Mean(name='train_auc')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
test_auc = tf.keras.metrics.Mean(name='test_tp')

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        output = model(images, training=True)
        loss = loss_function(labels, output)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, output)
    aucvalue = auc(labels, output, weights=None, num_thresholds=200, name=None, summation_method='trapezoidal')
    train_auc(aucvalue)

@tf.function
def test_step(images, labels, tp, tn, fn, fp):
    output = model(images, training=False)
    t_loss = loss_function(labels, output)
    test_loss(t_loss)
    test_accuracy(labels, output)
    aucvalue = auc(labels, output, weights=None, num_thresholds=200, name=None, summation_method='trapezoidal')
    test_auc(aucvalue)
    # print("output: ", output)
    # print("labels: ", labels)
    y_pred = tf.argmax(output, 1)
    y_true = tf.argmax(labels, 1)
    # print("y_pred： ", y_pred)
    # print("y_true： ", y_true)
    pred = tf.cast(y_pred, tf.int32)
    true = tf.cast(y_true, tf.int32)
    # print("pred: ", pred)
    # print("true: ", true)
    for i in range(len(pred)):
        if pred[i] == true[i] and true[i] == 1:
            tp += 1
        elif pred[i] == true[i] and true[i] == 0:
            tn += 1
        elif pred[i] == 1 and true[i] == 0:
            fp += 1
        else:
            fn += 1
    return tp, tn, fp, fn, output


@tf.function
def print_result(mtp, mtn, mfp, mfn):
    ftp = ftn = ffp = ffn = 0
    ftp += mtp
    ftn += mtn
    ffp += mfp
    ffn += mfn
    ptemplate = 'TP: {}, TN: {}, FP: {}, FN: {}'
    print(ptemplate.format(ftp, ftn, ffp, ffn))


y1 = []
y2 = []
y3 = []
y4 = []
best_test_loss = float('inf')
if k == 1:
    # batch_size for green is 32 for black is 21 for oubao is 30
    dirlist = [1, 2, 3, 4, 5]
    dirlist.remove(dirlist[k-1])
    dir1 = image_path + "black/" + str(dirlist[0])
    dir2 = image_path + "black/" + str(dirlist[1])
    dir3 = image_path + "black/" + str(dirlist[2])
    dir4 = image_path + "black/" + str(dirlist[3])
    dir5 = image_path + "black/" + str(k)
    # batch = [61, 62, 61, 62, 62]   green
    # batch = [50, 52, 52, 52, 52]  # tnb
    batch = [53, 67, 67, 67, 67]   # black
    val_data_gen = validation_image_generator.flow_from_directory(directory=dir5, batch_size=batch[k-1], shuffle=True,
                                                        target_size=(im_height, im_width), class_mode='categorical')
    total_val = val_data_gen.n
    batch.remove(batch[k - 1])
    data_gen1 = train_image_generator.flow_from_directory(directory=dir1, batch_size=batch[0], shuffle=True,
                                                          target_size=(im_height, im_width), class_mode='categorical')
    data_gen2 = train_image_generator.flow_from_directory(directory=dir2, batch_size=batch[1], shuffle=True,
                                                          target_size=(im_height, im_width), class_mode='categorical')
    data_gen3 = train_image_generator.flow_from_directory(directory=dir3, batch_size=batch[2], shuffle=True,
                                                          target_size=(im_height, im_width), class_mode='categorical')
    data_gen4 = train_image_generator.flow_from_directory(directory=dir4, batch_size=batch[3], shuffle=True,
                                                          target_size=(im_height, im_width), class_mode='categorical')

    total_train = data_gen1.n + data_gen2.n + data_gen3.n + data_gen4.n

    for epoch in range(1, epochs+1):

        train_loss.reset_states()        # clear history info
        train_accuracy.reset_states()    # clear history info
        train_auc.reset_states()    # clear history info
        test_auc.reset_states()         # clear history info
        test_loss.reset_states()         # clear history info
        test_accuracy.reset_states()     # clear history info
        tp = tn = fp = fn = 0
        images1, labels1 = next(data_gen1)
        images2, labels2 = next(data_gen2)
        images3, labels3 = next(data_gen3)
        images4, labels4 = next(data_gen4)
        images5, labels5 = next(val_data_gen)
        # train_auc = auc(labels, output, weights=None, num_thresholds=200, name=None, summation_method='trapezoidal')

        #  output1 = model(images1, training=True)
        #  loss = loss_function(labels1, output1)
        train_step(images1, labels1)
        ptp, ptn, pfp, pfn, my_out = test_step(images5, labels5, 0, 0, 0, 0)
        tp += ptp
        tn += ptn
        fp += pfp
        fn += pfn
        train_step(images2, labels2)
        ptp, ptn, pfp, pfn, my_out = test_step(images5, labels5, 0, 0, 0, 0)
        tp += ptp
        tn += ptn
        fp += pfp
        fn += pfn
        train_step(images3, labels3)
        ptp, ptn, pfp, pfn, my_out = test_step(images5, labels5, 0, 0, 0, 0)
        tp += ptp
        tn += ptn
        fp += pfp
        fn += pfn

        train_step(images4, labels4)
        # for step in range(total_val // 21):
        # for step in range(total_val // 30):
        # for step in range(total_val // 32):
        # test_images, test_labels = next(val_data_gen)
        ptp, ptn, pfp, pfn, my_out = test_step(images5, labels5, 0, 0, 0, 0)
        tp += ptp
        tn += ptn
        fp += pfp
        fn += pfn
        # global_step.assign_add(1)
        y1.append(train_loss.result())
        y2.append(test_loss.result())
        y3.append(train_accuracy.result() * 100)
        y4.append(test_accuracy.result() * 100)
        template = 'Epoch {}, Loss: {}, Accuracy: {}, AUC: {}, Test Loss: {}, Test Accuracy: {}, Test AUC: {}, TP: {}, TN: {}, FP: {}, FN: {}'
        print(template.format(epoch,
                            train_loss.result(),
                            train_accuracy.result() * 100,
                            train_auc.result(),
                            test_loss.result(),
                            test_accuracy.result() * 100,
                            test_auc.result(),
                            tp, tn, fp, fn))
        if test_loss.result() < best_test_loss:
            best_test_loss = test_loss.result()
            #  model.save_weights("./save_weights/myGoogLeNet.h5")
            model.save_weights("./save_weights/myResNet.ckpt".format(epoch), save_format='tf')

x = [i+1 for i in range(epochs)]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y1, label="train")
ax.plot(x, y2, label="test")
ax.legend(loc=0)
plt.savefig('./black_loss_k1.jpg')

plt.clf()
pic = plt.figure()
ax = pic.add_subplot(111)
ax.plot(x, y3, label="train")
ax.plot(x, y4, label="test")
ax.legend(loc=0)
plt.savefig('./black_accuracy_k1.jpg')