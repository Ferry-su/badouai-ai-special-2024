# -------------------------------------------------------------#
#   MobileNet 的网络部分（适配 TensorFlow 2.x）
# -------------------------------------------------------------#
import warnings
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    DepthwiseConv2D, Input, Activation, Dropout, Reshape,
    BatchNormalization, GlobalAveragePooling2D, Conv2D
)
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras import backend as K


def relu6(x):
    """ReLU 激活函数，最大值为 6。"""
    return K.relu(x, max_value=6)


def _conv_block(inputs, filters, kernel=(3, 3), strides=(1, 1)):
    """普通卷积块：Conv2D + BatchNormalization + ReLU6"""
    x = Conv2D(filters, kernel, padding='same', use_bias=False, strides=strides, name='conv1')(inputs)
    x = BatchNormalization(name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)


def _depthwise_conv_block(inputs, pointwise_conv_filters, depth_multiplier=1, strides=(1, 1), block_id=1):
    """深度可分离卷积块：DepthwiseConv2D + Conv2D"""
    x = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=depth_multiplier,
                        strides=strides, use_bias=False, name=f'conv_dw_{block_id}')(inputs)
    x = BatchNormalization(name=f'conv_dw_{block_id}_bn')(x)
    x = Activation(relu6, name=f'conv_dw_{block_id}_relu')(x)

    x = Conv2D(pointwise_conv_filters, (1, 1), padding='same', use_bias=False,
               strides=(1, 1), name=f'conv_pw_{block_id}')(x)
    x = BatchNormalization(name=f'conv_pw_{block_id}_bn')(x)
    return Activation(relu6, name=f'conv_pw_{block_id}_relu')(x)


def MobileNet(input_shape=(224, 224, 3), depth_multiplier=1, dropout=1e-3, classes=1000):
    """MobileNet 模型结构"""
    img_input = Input(shape=input_shape)

    # 网络结构
    x = _conv_block(img_input, 32, strides=(2, 2))  # 224x224x3 -> 112x112x32
    x = _depthwise_conv_block(x, 64, depth_multiplier, block_id=1)  # 112x112x32 -> 112x112x64
    x = _depthwise_conv_block(x, 128, depth_multiplier, strides=(2, 2), block_id=2)  # 112x112x64 -> 56x56x128
    x = _depthwise_conv_block(x, 128, depth_multiplier, block_id=3)  # 56x56x128 -> 56x56x128
    x = _depthwise_conv_block(x, 256, depth_multiplier, strides=(2, 2), block_id=4)  # 56x56x128 -> 28x28x256
    x = _depthwise_conv_block(x, 256, depth_multiplier, block_id=5)  # 28x28x256 -> 28x28x256
    x = _depthwise_conv_block(x, 512, depth_multiplier, strides=(2, 2), block_id=6)  # 28x28x256 -> 14x14x512

    # 14x14x512 -> 14x14x512 (5 次重复)
    for i in range(5):
        x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=7 + i)

    x = _depthwise_conv_block(x, 1024, depth_multiplier, strides=(2, 2), block_id=12)  # 14x14x512 -> 7x7x1024
    x = _depthwise_conv_block(x, 1024, depth_multiplier, block_id=13)  # 7x7x1024 -> 7x7x1024

    # 全局平均池化和输出
    x = GlobalAveragePooling2D()(x)  # 7x7x1024 -> 1x1x1024
    x = Reshape((1, 1, 1024), name='reshape_1')(x)
    x = Dropout(dropout, name='dropout')(x)
    x = Conv2D(classes, (1, 1), padding='same', name='conv_preds')(x)
    x = Activation('softmax', name='act_softmax')(x)
    x = Reshape((classes,), name='reshape_2')(x)

    # 创建模型
    model = Model(img_input, x, name='mobilenet_1_0_224_tf')
    return model


def preprocess_input(x):
    """对图片进行归一化"""
    x /= 255.0
    x -= 0.5
    x *= 2.0
    return x


if __name__ == '__main__':
    # 加载模型
    model = MobileNet(input_shape=(224, 224, 3))

    # 加载图片
    img_path = 'elephant.jpg'  # 确保路径下有图片文件
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    print('Input image shape:', x.shape)

    # 进行预测
    preds = model.predict(x)
    print(np.argmax(preds))  # 输出分类索引
    print('Predicted:', decode_predictions(preds, top=1))  # 输出 Top-1 分类