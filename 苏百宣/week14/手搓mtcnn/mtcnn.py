from keras.layers import Conv2D, Input, MaxPool2D, Reshape, Activation, Flatten, Dense, Permute  # 导入Keras层
from keras.layer.advanced_activations import PReLU
from keras.models import Model, Sequential
import tensorflow as tf
import numpy as np
import utils
import cv2

#-----------------------------#
#   创建P-Net网络，用于粗略计算人脸框
#   输出人脸框的位置和置信度
#-----------------------------#
def create_Pnet(weight_path):
    input = Input(shape=[None, None, 3]) # 定义输入层，输入图像的尺寸不固定，但有3个通道（RGB）

    # 第一卷积层，10个3x3卷积核，步长为1，valid padding（没有填充）
    x = Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1, 2], name='PReLU1')(x)  # 激活函数PReLU，shared_axes参数表示共享激活函数的维度
    x = MaxPool2D(pool_size=2)(x)  # 使用2x2最大池化层减少尺寸

    # 第二卷积层，16个3x3卷积核
    x = Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='PReLU2')(x)

    # 第三卷积层，32个3x3卷积核
    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='PReLU3')(x)

    # 输出两部分，分类器（softmax）和bbox回归
    classifier = Conv2D(2, (1, 1), activation='softmax', name='conv4-1')(x)  # 分类器，判断是否有人脸
    bbox_regress = Conv2D(4, (1, 1), name='conv4-2')(x)  # 边框回归，输出4个值用于人脸框的调整
    # -----------------------------#
    #   创建R-Net网络，用于精细修正人脸框
    # -----------------------------#
def create_Rnet(weight_path):
    input = Input(shape=[24, 24, 3])  # 输入层，图像尺寸固定为24x24，3个通道（RGB）

    # 第一卷积层，28个3x3卷积核，步长为1，valid padding
    x = Conv2D(28, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1, 2], name='prelu1')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)  # 池化层，3x3，步长为2


    # 第二卷积层，48个3x3卷积核
    x = Conv2D(48, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu2')(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)  # 池化层，3x3，步长为2

    # 第三卷积层，64个2x2卷积核
    x = Conv2D(64, (2, 2), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu3')(x)

    # 将数据形状调整为64x3x3
    x = Permute((3, 2, 1))(x)
    x = Flatten()(x)  # 扁平化

    # 全连接层，输入为576，输出为128
    x = Dense(128, name='conv4')(x)
    x = PReLU(name='prelu4')(x)

    # 全连接层，输入为576，输出为128
    x = Dense(128, name='conv4')(x)
    x = PReLU(name='prelu4')(x)

    # 分类器：判断是否为人脸，边框回归和关键点回归
    classifier = Dense(2, activation='softmax', name='conv5-1')(x)  # 分类器
    bbox_regress = Dense(4, name='conv5-2')(x)  # 边框回归
    model = Model([input], [classifier, bbox_regress])  # 创建模型
    model.load_weights(weight_path, by_name=True)  # 加载预训练权重
    return model  # 返回R-Net模型
#-----------------------------#
#   创建O-Net网络，用于更精细的人脸框回归和面部关键点预测
#-----------------------------#
def create_Onet(weight_path):
    input = Input(shape=[48, 48, 3])  # 输入层，图像尺寸为48x48，3个通道（RGB）

    # 第一卷积层，32个3x3卷积核
    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1, 2], name='prelu1')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    # 第二卷积层，64个3x3卷积核
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu2')(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)

    # 第三卷积层，64个3x3卷积核
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu3')(x)
    x = MaxPool2D(pool_size=2)(x)

    # 第四卷积层，128个2x2卷积核
    x = Conv2D(128, (2, 2), strides=1, padding='valid', name='conv4')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu4')(x)

    # 将数据形状调整为128x12x12
    x = Permute((3, 2, 1))(x)

    # 全连接层，输入为1152，输出为256
    x = Flatten()(x)
    x = Dense(256, name='conv5')(x)
    x = PReLU(name='prelu5')(x)

    # 分类器，判断是否为人脸，边框回归和面部关键点回归
    classifier = Dense(2, activation='softmax', name='conv6-1')(x)
    bbox_regress = Dense(4, name='conv6-2')(x)
    landmark_regress = Dense(10, name='conv6-3')(x)  # 面部关键点回归，输出10个关键点

    model = Model([input], [classifier, bbox_regress, landmark_regress])  # 创建模型
    model.load_weights(weight_path, by_name=True)  # 加载预训练权重
    return model  # 返回O-Net模型

#-----------------------------#
#   MTCNN人脸检测类
#-----------------------------#
class mtcnn():
    def __init__(self):
        self.Pnet = create_Pnet('model_data/pnet.h5')  # 创建P-Net模型
        self.Rnet = create_Rnet('model_data/rnet.h5')  # 创建R-Net模型
        self.Onet = create_Onet('model_data/onet.h5')  # 创建O-Net模型

    def detectFace(self, img, threshold):
        #-----------------------------#
        #   归一化处理
        #-----------------------------#
        copy_img = (img.copy() - 127.5) / 127.5  # 图像归一化处理，将像素值从[0, 255]映射到[-1, 1]
        origin_h, origin_w, _ = copy_img.shape  # 获取图像的原始尺寸
        #-----------------------------#
        #   计算图像金字塔的尺度
        #-----------------------------#
        scales = utils.calculateScales(img)  # 计算不同尺度下的图像尺寸

        out = []  # 用于存储P-Net输出结果

        #-----------------------------#
        #   P-Net：粗略计算人脸框
        #-----------------------------#
        for scale in scales:
            hs = int(origin_h * scale)  # 计算缩放后的高度
            ws = int(origin_w * scale)  # 计算缩放后的宽度
            scale_img = cv2.resize(copy_img, (ws, hs))  # 缩放图像
            inputs = scale_img.reshape(1, *scale_img.shape)  # 调整图像为网络输入格式
            output = self.Pnet.predict(inputs)  # P-Net预测结果
            out.append(output)  # 将结果保存

 #-----------------------------#
        #   通过P-Net得到人脸框
        #-----------------------------#
        rectangles = []
        for i in range(len(scales)):
            cls_prob = out[i][0][0][:,:,1]  # 获取人脸的概率
            roi = out[i][1][0]  # 获取回归的偏移量
            out_h, out_w = cls_prob.shape  # 获取缩放后的输出尺寸
            out_side = max(out_h, out_w)  # 取最大边长作为输出尺寸
            rectangle = utils.detect_face_12net(cls_prob, roi, out_side, 1 / scales[i], origin_w, origin_h, threshold[0])  # 计算矩形框
            rectangles.extend(rectangle)  # 将结果添加到矩形框列表

        rectangles = utils.NMS(rectangles, 0.7)  # 使用NMS去重

        if len(rectangles) == 0:
            return rectangles  # 如果没有检测到人脸框，返回空列表

       #-----------------------------#
        #   R-Net：稍微精确计算人脸框
        #-----------------------------#
        predict_24_batch = []
        for rectangle in rectangles:
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]  # 裁剪人脸区域
            scale_img = cv2.resize(crop_img, (24, 24))  # 缩放图像为24x24
            predict_24_batch.append(scale_img)

        predict_24_batch = np.array(predict_24_batch)  # 批量处理
        out = self.Rnet.predict(predict_24_batch)  # 使用R-Net进行预测

        cls_prob = out[0]  # 获取分类结果
        roi_prob = out[1]  # 获取边框回归结果
        rectangles = utils.filter_face_24net(cls_prob, roi_prob, rectangles, origin_w, origin_h, threshold[1])  # 过滤有效框

        if len(rectangles) == 0:
            return rectangles  # 如果没有检测到人脸框，返回空列表

        #-----------------------------#
        #   O-Net：精确计算人脸框并获得五个面部关键点
        #-----------------------------#
        predict_batch = []
        for rectangle in rectangles:
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]  # 裁剪人脸区域
            scale_img = cv2.resize(crop_img, (48, 48))  # 缩放图像为48x48
            predict_batch.append(scale_img)

        predict_batch = np.array(predict_batch)  # 批量处理
        output = self.Onet.predict(predict_batch)  # 使用O-Net进行预测
        cls_prob = output[0]  # 获取分类结果
        roi_prob = output[1]  # 获取边框回归结果
        pts_prob = output[2]  # 获取面部关键点预测

        rectangles = utils.filter_face_48net(cls_prob, roi_prob, pts_prob, rectangles, origin_w, origin_h, threshold[2])  # 最终过滤框

        return rectangles  # 返回最终的人脸框