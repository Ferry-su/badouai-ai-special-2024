import tensorflow as tf
import numpy as np
import time
import math
import Cifar10_data
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

# 设置最大训练步数
max_steps = 4000  # max_steps表示训练过程中最大的迭代次数。这里设置为4000次。

# 设置批次大小
batch_size = 100  # batch_size表示每次训练时使用的图像样本数。这里设置为每次训练使用100张图片。

# 设置评估用的样本数量
num_examples_for_eval = 10000  # num_examples_for_eval表示在测试时用于计算准确率的样本数量。这里设置为10000张测试图片。

# 设置CIFAR-10数据集的路径
data_dir = "Cifar_data/cifar-10-batches-bin"  # data_dir指定存放CIFAR-10数据集文件的目录路径。这个路径包含数据集的二进制文件。


def variable_with_weight_loss(shape, stddev, w1):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if w1 is not None:
        wights_loss = tf.multiply(tf.nn.l2_loss(var), w1, name="weight_loss")
        tf.add_to_collection("losses", wights_loss)
    return var


image_train, labels_train = Cifar10_data.inputs(data_dir=data_dir, batch_size=batch_size, distorted=True)
image_test, labels_test = Cifar10_data.inputs(data_dir=data_dir, batch_size=batch_size, distorted=None)

x = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
y = tf.placeholder(tf.int32, [batch_size])
# 创建第一个卷积层
kernel1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, w1=0.0)
conv1 = tf.nn.conv2d(x, kernel1, strides=[1, 1, 1, 1], padding='SAME')
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias1))
pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
# 创建第二个卷积层
kernel2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, w1=0.0)
conv2 = tf.nn.conv2d(pool1, kernel2, strides=[1, 1, 1, 1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.0, shape=[64]))
relu2 = tf.nn.relu(tf.nn.bias_add(conv1, bias2))
pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

reshaped = tf.reshape(pool2, [batch_size, -1])
dim = reshaped.get_shape()[1].value
# 建立第一个全连接层
weight1 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, w1=0.004)
fc_bias1 = tf.Variable(tf.constant(0.1, shape=[384]))
fc_1 = tf.nn.relu(tf.matmul(reshaped, weight1) + fc_bias1)

# 建立第二个全连接层
weight2 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, w1=0.004)
fc_bias2 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(fc_1, weight2) + fc_bias2)

# 建立第三个全连接层
weight3 = variable_with_weight_loss(shape=[192, 10], stddev=1 / 192, w1=0.0)
fc_bias3 = tf.Variable(tf.constant(0.1, shape=[10]))
result = tf.add(tf.matmul(local4, weight3) + fc_bias3)

# 计算损失，包括权重参数的正则化损失和交叉熵损失
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=result, logits=tf.cast(y, tf.float64))
weights_with_12_loss = tf.add_n(tf.get_collection("losses"))
loss = tf.reduce_mean(cross_entropy) + weights_with_12_loss
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

top_k_op = tf.nn.in_top_k(result, y, 1)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    tf.train.start_queue_runners()

    for step in range(max_steps):
        start_time = time.time()
        images_batch, label_batch = sess.run([image_train, labels_train])
        _, loss_value = sess.run([train_op, loss], feed_dict={x: images_batch, y: label_batch})
        duration = time.time() - start_time

        if step % 100 == 0:
            examples_per_sec = batch_size / duration
            sec_per_batch = float(duration)
            print("step %d,loss=%.2f(%.1f example/sec;%.3f sec/batch)" % (
            step, loss_value, examples_per_sec, sec_per_batch))

        num_batch = int(math.ceil(num_examples_for_eval / batch_size))
        true_count = 0
        total_sample_count = num_batch * batch_size

        for i in range(num_batch):
            image_batch, label_batch = sess.run([image_test, labels_test])
            predictions = sess.run([top_k_op], feed_dict={x: image_batch, y: label_batch})
            true_count += np.sum(np.argmax(predictions, 1) == label_batch)

        print("accuracy = %.3f%%" % ((true_count / total_sample_count) * 100))