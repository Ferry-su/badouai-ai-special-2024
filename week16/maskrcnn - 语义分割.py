import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import cv2

# 加载预训练模型并设置设备
def load_model():
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    return model, device

# 预处理图像：转换为Tensor并添加batch维度
def preprocess_image(image):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    return transform(image).unsqueeze(0)  # 增加batch维度

# 执行推理并返回预测结果
def infer(image_path, model, device):
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess_image(image).to(device)

    with torch.no_grad():
        prediction = model(image_tensor)

    return prediction

# 显示结果：将mask覆盖到原图上并显示
def show_result(image_path, predictions):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    color_mapping = {
        1: (255, 0, 0),  # 蓝色表示人
        2: (0, 255, 0),  # 绿色表示自行车
        3: (0, 0, 255),  # 红色表示汽车
    }

    for pred in predictions[0]:  # predictions是一个列表，包含单个元素（因为batch size为1）
        masks = pred['masks'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()

        for mask, label, score in zip(masks, labels, scores):
            if score > 0.5:  # 只显示置信度大于0.5的预测
                mask = (mask[0] > 0.5).astype(np.uint8)  # 将mask转换为二值图
                color = color_mapping.get(label.item(), (255, 255, 255))  # 默认白色
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image, contours, -1, color, 2)  # 在原图上绘制mask轮廓

    image = cv2.resize(image, (700, 700))  # 缩放图像
    cv2.imshow('Result', image)  # 显示图像
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 主程序：加载模型、推理并显示结果
def main(image_path):
    model, device = load_model()  # 加载模型
    predictions = infer(image_path, model, device)  # 执行推理
    show_result(image_path, predictions)  # 显示推理结果

# 使用示例
image_path = 'street.jpg'  # 替换为你的图像路径
main(image_path)