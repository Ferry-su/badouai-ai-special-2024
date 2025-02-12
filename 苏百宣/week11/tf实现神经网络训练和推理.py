import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 生产训练数据
# 使用numpy生成随机数据：y=x^2 + noise
x_data = np.linspace(-1,1,500)[:,np.newaxis]
noise = np.random.normal(1,0.05,x_data.shape)
y_data = np.square(x_data) + noise

# 定义神经元网络模型
model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='tanh', input_shape=(x_data.shape[1],)),   # 隐藏层 1，10 个节点
        tf.keras.layers.Dense(10, activation='tanh'),                                    # 隐藏层 2，10 个节点
        tf.keras.layers.Dense(1,activation='linear')                                       # 输出层，1 个节点，线性激活
])

# 编译模型
model.compile(optimizer=tf. keras .optimizers.Adam(learning_rate=0.01),  # 优化器：Adam
              loss='mean_squared_error',                               # 损失函数：均方误差
              metrics=['mean_absolute_error'])

# 训练模型
history = model.fit(x_data, y_data, epochs=100, verbose=1,batch_size=32)

# 预测（推理）
x_test = np.linspace(-1, 1, 100)[:, np.newaxis]  # 测试数据
y_pred = model.predict(x_test)

# 绘制图像
# 可视化结果
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, label="True Data", color='blue', alpha=0.5)  # 真实数据
plt.plot(x_test, y_pred, label="Predicted Curve", color='red', lw=2)     # 预测曲线
plt.legend()
plt.title("Neural Network Regression Example")
plt.show()