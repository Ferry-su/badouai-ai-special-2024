from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import np_utils
from keras.optimizers import Adam
from model.AlexNet import AlexNet
import numpy as np
import cv2
import utils

def generate_arrays_from_file(lines, batch_size):
    """
    数据生成器，从文件中加载数据
    """
    n = len(lines)
    i = 0
    while True:
        X_train = []
        Y_train = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            label = lines[i].split(';')[1].strip()

            # 读取图片
            img = cv2.imread(r"./data/image/train/" + name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            X_train.append(img)
            Y_train.append(label)

            i = (i + 1) % n

        X_train = utils.resize_image(X_train, (224, 224))
        X_train = np.array(X_train).reshape(-1, 224, 224, 3)
        Y_train = np_utils.to_categorical(np.array(Y_train), num_classes=2)
        yield (X_train, Y_train)

if __name__ == "__main__":
    # 模型保存路径
    log_dir = "./logs/"

    # 读取数据集描述文件
    with open(r"./data/dataset.txt", "r") as f:
        lines = f.readlines()

    # 随机打乱数据
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    # 划分训练集和验证集
    num_val = int(len(lines) * 0.1)
    num_train = len(lines) - num_val

    # 构建 AlexNet 模型
    model = AlexNet(input_shape=(224, 224, 3), output_shape=2)

    # 定义回调函数
    checkpoint = ModelCheckpoint(
        log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_accuracy',
        save_weights_only=False,
        save_best_only=True,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=3,
        verbose=1
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=10,
        verbose=1
    )

    # 编译模型
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(lr=1e-3),
        metrics=['accuracy']
    )

    # 批次大小
    batch_size = 128

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

    # 开始训练
    model.fit_generator(
        generate_arrays_from_file(lines[:num_train], batch_size),
        steps_per_epoch=max(1, num_train // batch_size),
        validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
        validation_steps=max(1, num_val // batch_size),
        epochs=50,
        initial_epoch=0,
        callbacks=[checkpoint, reduce_lr, early_stopping]
    )

    # 保存最终模型
    model.save_weights(log_dir + 'last1.h5')