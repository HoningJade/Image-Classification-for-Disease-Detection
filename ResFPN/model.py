from tensorflow.keras import layers, Model, Sequential

# 34层对应的残差结构
class BasicBlock(layers.Layer):
    expansion = 1   #  50，101：4

    def __init__(self, out_channel, strides=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(out_channel, kernel_size=3, strides=strides,
                                   padding="SAME", use_bias=False)   #  残差结构
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        # -----------------------------------------
        self.conv2 = layers.Conv2D(out_channel, kernel_size=3, strides=1,
                                   padding="SAME", use_bias=False)   #  第二卷积层，outchannel是卷积核个数
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        # -----------------------------------------
        self.downsample = downsample   # 下采样操作
        self.relu = layers.ReLU()   # 激活函数
        self.add = layers.Add()    # 加法运算

    def call(self, inputs, training=False):
        identity = inputs  # 传入的特征矩阵
        if self.downsample is not None:
            identity = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        x = self.add([identity, x])   #  主分支输出和捷径分支输出相加
        x = self.relu(x)

        return x

# 50层 101层对应的残差结构
class Bottleneck(layers.Layer):
    expansion = 4  #扩展系数

    def __init__(self, out_channel, strides=1, downsample=None, **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(out_channel, kernel_size=1, use_bias=False, name="conv1")
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv1/BatchNorm")
        # -----------------------------------------
        self.conv2 = layers.Conv2D(out_channel, kernel_size=3, use_bias=False,
                                   strides=strides, padding="SAME", name="conv2")
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv2/BatchNorm")
        # -----------------------------------------
        self.conv3 = layers.Conv2D(out_channel * self.expansion, kernel_size=1, use_bias=False, name="conv3")
        self.bn3 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv3/BatchNorm")
        # -----------------------------------------
        self.relu = layers.ReLU()
        self.downsample = downsample
        self.add = layers.Add()

    def call(self, inputs, training=False):
        identity = inputs
        if self.downsample is not None:
            identity = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)

        x = self.add([x, identity])
        x = self.relu(x)  # 残差结构最终输出

        return x


def _make_layer(block, in_channel, channel, block_num, name, strides=1):
    downsample = None
    if strides != 1 or in_channel != channel * block.expansion:  # 2种残差结构
        downsample = Sequential([
            layers.Conv2D(channel * block.expansion, kernel_size=1, strides=strides,
                          use_bias=False, name="conv1"),
            layers.BatchNormalization(momentum=0.9, epsilon=1.001e-5, name="BatchNorm")
        ], name="shortcut")     # 下采样

    layers_list = []   #  所使用的残差结构列表
    layers_list.append(block(channel, downsample=downsample, strides=strides, name="unit_1"))

    for index in range(1, block_num):
        layers_list.append(block(channel, name="unit_" + str(index + 1)))

    return Sequential(layers_list, name=name)


def _resnet(block, blocks_num, im_width=224, im_height=224, num_classes=1000, include_top=True):
    # tensorflow中的tensor通道排序是NHWC，blocks_num是残差结构个数
    # (None, 224, 224, 3)

    input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")
    x = layers.Conv2D(filters=64, kernel_size=7, strides=2,
                      padding="SAME", use_bias=False, name="conv1")(input_image)  #  图像缩小为原来一半，C1
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv1/BatchNorm")(x)
    x = layers.ReLU()(x)  # C1

    x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME")(x)

    x = _make_layer(block, x.shape[-1], 64, blocks_num[0], name="block1")(x)   # C2
    x = _make_layer(block, x.shape[-1], 128, blocks_num[1], strides=2, name="block2")(x)  # C3
    x = _make_layer(block, x.shape[-1], 256, blocks_num[2], strides=2, name="block3")(x)  # C4
    x = _make_layer(block, x.shape[-1], 512, blocks_num[3], strides=2, name="block4")(x)  # C5

    if include_top:
        x = layers.GlobalAvgPool2D()(x)  # pool + flatten 全局平均池化下采样层  展平处理
        x = layers.Dense(num_classes, name="logits")(x)   # 全连接层
        predict = layers.Softmax()(x)  # 转换成概率分布
    else:
        predict = x

    model = Model(inputs=input_image, outputs=predict)

    return model


def resnet34(im_width=224, im_height=224, num_classes=1000):
    return _resnet(BasicBlock, [3, 4, 6, 3], im_width, im_height, num_classes)


def resnet50(im_width=224, im_height=224, num_classes=1000, include_top=True):
    return _resnet(Bottleneck, [3, 4, 6, 3], im_width, im_height, num_classes, include_top)


def resnet101(im_width=224, im_height=224, num_classes=1000, include_top=True):
    return _resnet(Bottleneck, [3, 4, 23, 3], im_width, im_height, num_classes, include_top)

