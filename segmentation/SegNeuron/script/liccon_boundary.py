import numpy as np
import tifffile as tiff
import cv2
from scipy.ndimage import sobel

# Load the 3D TIFF file
input_path = "/data/wangfeiran/code/brainbow/segmentation/SegNeuron/data/czi/1_4_pro/104.tif"
output_path = "/data/wangfeiran/code/brainbow/segmentation/SegNeuron/data/czi/1_4_pro/104_edge.tif"

# 读取 3D 图像
image_3d = tiff.imread(input_path)
print(f"Original Shape: {image_3d.shape}")

# 归一化图像到 0-255
image_3d_normalized = ((image_3d - image_3d.min()) / (image_3d.max() - image_3d.min()) * 255).astype(np.uint8)

# 创建空白 3D 数组来存储边缘检测结果
edges_3d = np.zeros_like(image_3d_normalized, dtype=np.uint8)

# 2D Canny Edge Detection（对每一张切片进行边缘检测）
for i in range(image_3d_normalized.shape[0]):
    img = image_3d_normalized[i]

    # 高斯模糊去噪
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)

    # Canny 边缘检测（调整阈值以适应归一化数据）
    edges = cv2.Canny(img_blur, 50, 150)
    # 取反边缘图像
    edges_inverted = cv2.bitwise_not(edges)

    edges_3d[i] = edges_inverted


# 保存边缘检测后的 3D 图像
tiff.imwrite(output_path, edges_3d.astype(np.uint8))
print(f"Edge-detected 3D volume saved at: {output_path}")
