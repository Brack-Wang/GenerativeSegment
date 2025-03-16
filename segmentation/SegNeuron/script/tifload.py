import tifffile as tiff
import numpy as np
import os
from skimage.transform import resize

# Load the TIFF file
tif_path = "/data/wangfeiran/code/brainbow/segmentation/SegNeuron/data/0207_czi/intensity_result/intensity_z1-22.tif"
tif_data = tiff.imread(tif_path)

# Print original shape
print("Original shape:", tif_data.shape)

# Resize to 1/4 of the original size
new_shape = (tif_data.shape[0], tif_data.shape[1] // 4, tif_data.shape[2] // 4)
resized_data = resize(tif_data, new_shape, anti_aliasing=True, preserve_range=True).astype(tif_data.dtype)

print("Resized shape:", resized_data.shape)
# Save the resized image to the same folder
save_path = os.path.join(os.path.dirname(tif_path), "resized_" + os.path.basename(tif_path))
tiff.imwrite(save_path, resized_data)

print("Resized image saved to:", save_path)
