from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization


def AlexNet(input_shape=(224, 224, 3), output_shape=2):
    # AlexNet 原始结构，不减少 filter 和全连接层大小
    model = Sequential()

    # 第一层卷积：步长为4x4，大小为11的卷积核，输出的特征层为96层，输出的shape为(55, 55, 96)
    model.add(
        Conv2D(
            filters=96,
            kernel_size=(11, 11),
            strides=(4, 4),
            padding='valid',
            input_shape=input_shape,
            activation='relu'
        )
    )
    model.add(BatchNormalization())

    # 第一层最大池化：步长为2，池化窗口为3x3，输出的shape为(27, 27, 96)
    model.add(
        MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='valid'
        )
    )

    # 第二层卷积：步长为1x1，大小为5的卷积核，输出的特征层为256层，输出的shape为(27, 27, 256)
    model.add(
        Conv2D(
            filters=256,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )
    model.add(BatchNormalization())

    # 第二层最大池化：步长为2，池化窗口为3x3，输出的shape为(13, 13, 256)
    model.add(
        MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='valid'
        )
    )

    # 第三层卷积：步长为1x1，大小为3的卷积核，输出的特征层为384层，输出的shape为(13, 13, 384)
    model.add(
        Conv2D(
            filters=384,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )

    # 第四层卷积：步长为1x1，大小为3的卷积核，输出的特征层为384层，输出的shape为(13, 13, 384)
    model.add(
        Conv2D(
            filters=384,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )

    # 第五层卷积：步长为1x1，大小为3的卷积核，输出的特征层为256层，输出的shape为(13, 13, 256)
    model.add(
        Conv2D(
            filters=256,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )

    # 第三层最大池化：步长为2，池化窗口为3x3，输出的shape为(6, 6, 256)
    model.add(
        MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='valid'
        )
    )

    # 全连接层
    model.add(Flatten())

    # 第一全连接层：输出为4096
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    # 第二全连接层：输出为4096
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    # 输出层：类别数为2（猫和狗）
    model.add(Dense(output_shape, activation='softmax'))

    return model