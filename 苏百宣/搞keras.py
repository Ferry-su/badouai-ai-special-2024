# 实现keras author：苏百宣

# 1. 加载 MNIST 数据集
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# 加载数据
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 打印形状
print('训练集图片 shape:', train_images.shape)
print('训练集标签 shape:', train_labels.shape)
print('测试集图片 shape:', test_images.shape)
print('测试集标签 shape:', test_labels.shape)

# 可视化数据
digit = test_images[6].reshape(28, 28)  # 恢复为二维图片
plt.imshow(digit, cmap=plt.cm.binary)
plt.title("Test Image")
plt.show()

# 2.数据预处理
# 将二维数据图片转化为一维数据，并且归一化到[0,1]
train_images = train_images.reshape((60000, 28 * 28)).astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28)).astype('float32') / 255

# 将标签转换为 one-hot 编码
print("before change:", test_labels[6])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print("after change:", test_labels[6])

# 3. 构建神经网络
from tensorflow.keras import models, layers

# 搭建神经网络
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

# 编译模型数据
network.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4.训练模型
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# 5.测试模型
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

# 6.让模型对测试集中的某一张图片进行测试
predictions = network.predict(test_images)
for i in range(len(test_images)):
    predicted_label = predictions[i].argmax()
    print(f"Image {i} is predicted as: {predicted_label}")