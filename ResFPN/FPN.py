import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential


class BasicBlock(Model):

    def __init__(self, in_channels, out_channels, strides=1):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(out_channels, kernel_size=3, strides=strides,
                                   padding="same", use_bias=False)
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2D(out_channels, kernel_size=3, strides=1,
                                   padding="same", use_bias=False)
        self.bn2 = layers.BatchNormalization()

        if strides != 1 or in_channels != out_channels:
            self.shortcut = Sequential([
                layers.Conv2D(out_channels, kernel_size=1,
                              strides=strides, use_bias=False),
                layers.BatchNormalization()]
            )
        else:
            self.shortcut = lambda x, _: x

    def call(self, x, training=False):
        # if training: print("=> training network ... ")
        out = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        out = self.bn2(self.conv2(out), training=training)
        out += self.shortcut(x, training)
        return tf.nn.relu(out)


class FPN(tf.keras.Model):
    def __init__(self, block, num_blocks):
        super(FPN, self).__init__()
        self.in_channels = 64

        self.conv1 = tf.keras.layers.Conv2D(64, 7, 2, padding="same", use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()

        # Bottom --> up layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Smooth layers
        self.smooth1 = layers.Conv2D(256, 3, 1, padding="same")
        self.smooth2 = layers.Conv2D(256, 3, 1, padding="same")
        self.smooth3 = layers.Conv2D(256, 3, 1, padding="same")
        self.smooth4 = layers.Conv2D(256, 3, 1, padding="same")

        # Lateral layers
        self.lateral_layer1 = layers.Conv2D(256, 1, 1, padding="valid")
        self.lateral_layer2 = layers.Conv2D(256, 1, 1, padding="valid")
        self.lateral_layer3 = layers.Conv2D(256, 1, 1, padding="valid")
        self.lateral_layer4 = layers.Conv2D(256, 1, 1, padding="valid")

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return tf.keras.Sequential(layers)

    def _upsample_add(self, x, y):
        _, H, W, C = y.shape
        return tf.image.resize(x, size=(H, W), method="bilinear") + y

    def call(self, x, training=False):
        C1 = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        C1 = tf.nn.max_pool2d(C1, ksize=3, strides=2, padding="SAME")

        # Bottom --> up
        C2 = self.layer1(C1, training=training)
        C3 = self.layer2(C2, training=training)
        C4 = self.layer3(C3, training=training)
        C5 = self.layer4(C4, training=training)

        # Top-down
        M5 = self.lateral_layer1(C5)
        M4 = self._upsample_add(M5, self.lateral_layer2(C4))
        M3 = self._upsample_add(M4, self.lateral_layer3(C3))
        M2 = self._upsample_add(M3, self.lateral_layer4(C2))

        # Smooth
        P5 = self.smooth1(M5)
        P4 = self.smooth2(M4)
        P3 = self.smooth3(M3)
        P2 = self.smooth4(M2)
        _, mm, nn, tt = P2.shape
        P5 = tf.image.resize(P5, size=(mm, nn), method="bilinear")
        output = tf.concat([P2, P5], 3)
        return output


def _resnet(block, blocks_num, im_width=224, im_height=224, num_classes=1000, training = True):
    # tensorflow中的tensor通道排序是NHWC
    # (None, 224, 224, 3)
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")
    resfpn = FPN(block, blocks_num)

    # predict = layers.GlobalAvgPool2D()(resfpn(input_image, training=training))  # pool + flatten
    predict = layers.AveragePooling2D()(resfpn(input_image, training=training))
    predict = layers.Flatten()(predict)
    predict = layers.Dense(num_classes, name="logits")(predict)
    predict = layers.Softmax()(predict)

    model = Model(inputs=input_image, outputs=predict)
    return model


def resnet50(im_width=224, im_height=224, num_classes=2, training = True):
    return _resnet(BasicBlock, [3, 4, 6, 3], im_width, im_height, num_classes, training)
