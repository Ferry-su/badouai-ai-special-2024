# 数据处理和模型训练的多线程支持
num_parallel_calls = 4  # 数据加载时的并行处理线程数

# 模型输入图片的大小
input_shape = 416  # 输入图片的宽和高（416x416）

# 每张图片中最多允许的检测框数量
max_boxes = 20  # 一张图片中最多保留的目标框数量

# 数据增强相关参数
jitter = 0.3  # 随机抖动的幅度，用于数据增强
hue = 0.1  # 色调变化范围，用于数据增强
sat = 1.0  # 饱和度调整范围，用于数据增强
cont = 0.8  # 对比度调整范围，用于数据增强
bri = 0.1  # 亮度调整范围，用于数据增强

# Batch Normalization 的参数
norm_decay = 0.99  # BN 权重的衰减系数
norm_epsilon = 1e-3  # 防止 BN 中除以零的极小值

# 是否加载预训练权重
pre_train = True  # 如果为 True，加载预训练模型

# Anchor Box 和类别数量
num_anchors = 9  # 使用的 Anchor Boxes 数量
num_classes = 80  # 数据集中类别的数量（COCO 数据集为 80 类）

# 模型是否处于训练模式
training = True  # 如果为 True，则进入训练模式

# 忽略阈值，用于筛选低置信度的预测框
ignore_thresh = 0.5  # 小于此值的预测框会被忽略

# 优化器的学习率设置
learning_rate = 0.001  # 学习率

# Batch 大小设置
train_batch_size = 10  # 训练时每个 batch 包含的样本数量
val_batch_size = 10  # 验证时每个 batch 包含的样本数量

# 训练和验证集的样本总数
train_num = 2800  # 训练样本数量
val_num = 5000  # 验证样本数量

# 训练过程中总的 epoch 数量
Epoch = 50  # 训练的总 epoch 数

# 推理相关阈值
obj_threshold = 0.5  # 目标检测的置信度阈值
nms_threshold = 0.5  # 非极大值抑制的阈值

# 指定 GPU 设备
gpu_index = "0"  # 使用的 GPU 编号

# 日志和模型存储目录
log_dir = './logs'  # 日志保存路径（如 TensorBoard 日志）
data_dir = './model_data'  # 模型相关数据（如权重和类别文件）存储路径
model_dir = './test_model/model.ckpt-192192'  # 训练后模型的保存路径

# YOLOv3 的预训练权重
pre_train_yolo3 = True  # 是否加载 YOLOv3 的预训练权重
yolo3_weights_path = './model_data/yolov3.weights'  # YOLOv3 预训练权重路径

# Darknet-53 的预训练权重
darknet53_weights_path = './model_data/darknet53.weights'  # Darknet-53 预训练权重路径

# Anchor Box 和类别文件的路径
anchors_path = './model_data/yolo_anchors.txt'  # Anchor Box 的定义文件路径
classes_path = './model_data/coco_classes.txt'  # 数据集的类别文件路径

# 测试图片路径
image_file = "./img/img2.jpg"  # 用于推理的图片路径