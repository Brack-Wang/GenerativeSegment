import numpy as np
import tifffile as tiff
import cv2

# 读取两张 3D TIFF 图像
input_path1 = "/data/wangfeiran/code/brainbow/segmentation/SegNeuron/data/czi/1_4_pro/neuron.tif"
input_path2 = "/data/wangfeiran/code/brainbow/segmentation/SegNeuron/data/czi/1_4_pro/104_edge.tif"
output_path = "/data/wangfeiran/code/brainbow/segmentation/SegNeuron/data/czi/1_4_pro/combined_and_edges.tif"

# 加载 3D 图像
image_3d_1 = tiff.imread(input_path1).astype(np.uint16)  # 假设原始强度值已归一化到 0-255
image_3d_2 = tiff.imread(input_path2).astype(np.uint16)

image_3d_1 = np.where(image_3d_1 > 0, 255, 0).astype(np.uint16)

# 检查尺寸是否一致
if image_3d_1.shape != image_3d_2.shape:
    raise ValueError("The two 3D volumes must have the same dimensions.")

# 执行按位与 (AND) 操作
combined_edges = cv2.bitwise_or(image_3d_1, image_3d_2)

# 保存合成后的 3D 图像
tiff.imwrite(output_path, image_3d_1.astype(np.uint16))
print(f"Combined edge-detected 3D volume saved at: {output_path}")
