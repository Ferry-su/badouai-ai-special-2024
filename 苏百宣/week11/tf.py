import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 使用 numpy 生成 200 个随机点
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

# 定义神经网络模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='tanh', input_shape=(1,)),  # 中间层
    tf.keras.layers.Dense(1, activation='tanh')                     # 输出层
])

# 定义损失函数和优化器
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
              loss='mean_squared_error')

# 训练模型
model.fit(x_data, y_data, epochs=2000, verbose=0)

# 获取预测值
prediction_value = model.predict(x_data)

# 画图
plt.figure()
plt.scatter(x_data, y_data)  # 散点是真实值
plt.plot(x_data, prediction_value, 'r-', lw=5)  # 曲线是预测值
plt.show()
