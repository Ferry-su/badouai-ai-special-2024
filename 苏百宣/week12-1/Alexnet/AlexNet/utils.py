import cv2
import numpy as np

def resize_image(image, size):
    """
    调整图片大小到指定尺寸。
    Args:
        image: 图片列表
        size: 目标尺寸 (宽, 高)
    Returns:
        调整大小后的图片列表 (numpy 数组)
    """
    resized_images = []
    for img in image:
        img_resized = cv2.resize(img, size)
        resized_images.append(img_resized)
    return np.array(resized_images)