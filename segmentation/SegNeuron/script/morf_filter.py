import numpy as np
import tifffile as tiff
import cv2
# File paths

data_root = "/data/wangfeiran/code/brainbow/segmentation/SegNeuron/data/morf3/small/inference/"
mask_path = data_root +  "neuron_ndl.tif"  # 替换为实际路径（左图）
image_path = data_root + "neuron.tif"  # 替换为实际路径（右图）
output_path = data_root + "segmentation.tif"  # 输出路径

# Load the 3D mask and raw image
mask = tiff.imread(mask_path)  # Mask in uint16
raw = tiff.imread(image_path) # Raw image in uint32

# kernel = np.ones((5, 5), np.uint16)  # Structuring element for dilation
# mask = cv2.dilate(mask, kernel, iterations=1)  # Adjust iterations for more expansion

mask_binary = (mask > 0).astype(np.uint16)
raw_min, raw_max = raw.min(), raw.max()



raw_scaled = ((raw - raw_min) / (raw_max - raw_min) * 65535).astype(np.uint16)


masked_raw = raw_scaled * mask_binary
print(mask_binary.max(), raw_scaled.max())
print(mask_binary.min(), raw_scaled.min())
print(masked_raw.max(), masked_raw.min())



tiff.imwrite(output_path, masked_raw)


# import numpy as np
# import tifffile as tiff
# import cv2

# # --------------- File Paths ---------------
# data_root = "/data/wangfeiran/code/brainbow/segmentation/SegNeuron/data/morf3/large/"
# mask_path = data_root + "basson_ndl.tif"  # Mask path
# image_path = data_root + "basson/neuron.tif"  # Raw image path
# output_path = data_root + "filtered.tif"  # Output path

# # --------------- Load Mask and Raw Image ---------------
# mask = tiff.imread(mask_path).astype(np.uint16)  # Mask in uint16
# raw = tiff.imread(image_path).astype(np.uint32)  # Raw image in uint32

# # Convert mask to binary (0 or 1)
# mask_binary = (mask > 0).astype(np.uint8)  # Convert to uint8 for OpenCV processing

# # --------------- Remove Small Objects using Connected Components ---------------
# min_area = 500  # Set the minimum area threshold (adjust as needed)

# # Find connected components
# num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)

# # Create an empty mask
# filtered_mask = np.zeros_like(mask_binary, dtype=np.uint8)

# # Keep only large components
# for i in range(1, num_labels):  # Skip background (0)
#     if stats[i, cv2.CC_STAT_AREA] >= min_area:
#         filtered_mask[labels == i] = 1  # Retain large components

# # --------------- Morphological Dilation (Expand Remaining Regions) ---------------
# kernel = np.ones((5, 5), np.uint8)  # Structuring element for dilation
# expanded_mask = cv2.dilate(filtered_mask, kernel, iterations=1)  # Adjust iterations for more expansion

# # --------------- Normalize and Apply Mask ---------------
# raw_min, raw_max = raw.min(), raw.max()
# if raw_max > raw_min:
#     raw_scaled = ((raw - raw_min) / (raw_max - raw_min) * 65535).astype(np.uint16)
# else:
#     raw_scaled = np.zeros_like(raw, dtype=np.uint16)  # If constant value, return all zero

# masked_raw = raw_scaled * expanded_mask.astype(np.uint16)

# # --------------- Save Processed Image ---------------
# tiff.imwrite(output_path, masked_raw)

# print("Filtered mask saved to:", output_path)

