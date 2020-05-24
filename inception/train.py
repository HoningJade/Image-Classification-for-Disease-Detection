from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import matplotlib.pyplot as plt
from model import GoogLeNet
import tensorflow as tf
import json
import os
from auc import auc
# from sklearn.metrics import roc_auc_score

CUDA_VISIBLE_DEVICES = 1
data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
image_path = data_root + "/data/"  # image data set path
# train_dir = image_path + "train_black"
# validation_dir = image_path + "val_black"
# train_dir = image_path + "train_green"
# validation_dir = image_path + "val_green"
train_dir = image_path + "train_oubao"
validation_dir = image_path + "val_oubao"

# create direction for saving weights
if not os.path.exists("save_weights"):
    os.makedirs("save_weights")

im_height = 224
im_width = 224
# batch_size = 61 # for black
# batch_size = 46  # for green
batch_size = 57  # for oubao
epochs = 30

# (1,0) means ill; (0,1) means ok

def pre_function(img):
    # img = im.open('test.jpg')
    # img = np.array(img).astype(np.float32)
    img = img / 255.
    # img = (img - 0.5) * 2.0
    return img


# data generator with data augmentation
train_image_generator = ImageDataGenerator(preprocessing_function=pre_function,
                                           horizontal_flip=True, brightness_range=[0.7, 2.5], zoom_range=[0.7, 1.0])

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

# batch_size for green is 32 for black is 21 for oubao is 30
val_data_gen = train_image_generator.flow_from_directory(directory=validation_dir,
                                                         batch_size=30,
                                                         shuffle=True,
                                                         target_size=(im_height, im_width),
                                                         class_mode='categorical')
total_val = val_data_gen.n

model = GoogLeNet(im_height=im_height, im_width=im_width, class_num=2, aux_logits=True)
# model.build((batch_size, 224, 224, 3))  # when using subclass model
model.summary()

# using keras low level api for training
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
# 10^-3 - 10^-6 自动调学习率

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
train_auc = tf.keras.metrics.Mean(name='train_auc')
# train_auc = auc(labels, output, weights=None, num_thresholds=200, name=None, summation_method='trapezoidal')
# train_auc = auc(name='train_auc')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
test_auc = tf.keras.metrics.Mean(name='test_tp')
# test_tp = tf.keras.metrics.Sum(name='test_tp')
# test_fp = tf.keras.metrics.Sum(name='test_fp')
# test_tn = tf.keras.metrics.Sum(name='test_tn')
# test_fn = tf.keras.metrics.Sum(name='test_fn')


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        aux1, aux2, output = model(images, training=True)
        loss1 = loss_object(labels, aux1)
        loss2 = loss_object(labels, aux2)
        loss3 = loss_object(labels, output)
        loss = loss1*0.3 + loss2*0.3 + loss3
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)  # average loss
    train_accuracy(labels, output)
    aucvalue = auc(labels, output, weights=None, num_thresholds=200, name=None, summation_method='trapezoidal')
    train_auc(aucvalue)

@tf.function
def test_step(images, labels, tp, tn, fn, fp):
    _, _, output = model(images, training=False)
    t_loss = loss_object(labels, output)
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
    #tp = tf.keras.metrics.TruePositives(labels, output)
    #fp = tf.keras.metrics.FalsePositives(labels, output)
    #tn = tf.keras.metrics.TrueNegatives(labels, output)
    #fn = tf.keras.metrics.FalseNegatives(labels, output)
    #test_tp(tp)
    #test_fp(fp)
    #test_tn(tn)
    #test_fn(fn)


best_test_loss = float('inf')
for epoch in range(1, epochs+1):
    train_loss.reset_states()        # clear history info
    train_accuracy.reset_states()    # clear history info
    train_auc.reset_states()    # clear history info
    test_auc.reset_states()         # clear history info
    test_loss.reset_states()         # clear history info
    test_accuracy.reset_states()     # clear history info
    # test_tp.reset_states()  # clear history info
    # test_fp.reset_states()  # clear history info
    # test_tn.reset_states()  # clear history info
    # test_fn.reset_states()  # clear history info
    tp = tn = fp = fn = 0

    for step in range(total_train // batch_size):
    # for step in range(total_train // 276):
        images, labels = next(train_data_gen)
        train_step(images, labels)
        # train_auc = auc(labels, output, weights=None, num_thresholds=200, name=None, summation_method='trapezoidal')

    # for step in range(total_val // 21):
    for step in range(total_val // 30):
    # for step in range(total_val // 32):
        test_images, test_labels = next(val_data_gen)
        ptp, ptn, pfp, pfn, my_out = test_step(test_images, test_labels, 0, 0, 0, 0)
        tp += ptp
        tn += ptn
        fp += pfp
        fn += pfn

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
        model.save_weights("./save_weights/myGoogLeNet.ckpt".format(epoch), save_format='tf')