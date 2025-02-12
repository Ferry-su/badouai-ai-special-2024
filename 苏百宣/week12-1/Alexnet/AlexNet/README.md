# AlexNet in Keras

## 项目简介
这是一个使用 Keras 实现的 AlexNet 模型项目，包含训练、验证和预测功能。

## 文件结构
- `model/AlexNet.py`：AlexNet 模型定义
- `train.py`：训练脚本
- `predict.py`：预测脚本
- `utils.py`：辅助工具函数
- `data/`：数据文件夹
- `logs/`：保存训练好的模型

## 使用方法
### 1. 准备数据
将训练集和验证集分别放入 `data/image/train` 和 `data/image/val` 中。

### 2. 训练模型
运行以下命令开始训练：
```bash
python train.py