# -------------------------------------------------------------#
#   使用 TensorFlow/Keras 提供的预定义 InceptionV3 模型
#   自动加载 ImageNet 预训练权重，进行图片分类
# -------------------------------------------------------------#
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

def load_and_preprocess_image(img_path, target_size=(299, 299)):
    """
    加载并预处理图片。

    参数:
    - img_path: 图片路径。
    - target_size: 图片调整到的尺寸，默认 (299, 299)。

    返回:
    - 预处理后的图片张量。
    """
    # 加载图片并调整大小
    img = image.load_img(img_path, target_size=target_size)
    # 转换为数组
    x = image.img_to_array(img)
    # 增加批量维度
    x = np.expand_dims(x, axis=0)
    # 应用 InceptionV3 的预处理
    x = preprocess_input(x)
    return x

def main():
    # 加载 InceptionV3 模型，包含 ImageNet 的预训练权重
    model = InceptionV3(weights='imagenet')

    # 图片路径
    img_path = 'elephant.jpg'  # 请确保路径下有该图片
    try:
        # 加载并预处理图片
        x = load_and_preprocess_image(img_path)

        # 使用模型进行预测
        preds = model.predict(x)

        # 解码预测结果
        print('Predicted:', decode_predictions(preds, top=3))  # 显示前3个预测结果
    except FileNotFoundError:
        print(f"图片文件未找到，请检查路径是否正确: {img_path}")

if __name__ == '__main__':
    main()