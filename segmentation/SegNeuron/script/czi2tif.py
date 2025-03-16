
from aicspylibczi import CziFile
import numpy as np
import tifffile as tiff
import skimage.transform

# Load the .czi file
file_path = "/data/wangfeiran/code/brainbow/segmentation/SegNeuron/data/0207_czi/z1-10.czi"
czi = CziFile(file_path)

# Get the shape and dimension order
dims_shape = czi.get_dims_shape()  # Returns a dictionary with shape info
print(f"Image dimensions: {dims_shape}")

# Read the image as a NumPy array
image_array, shp = czi.read_image()

# Check the type and shape of the loaded array
print(f"Original Image array shape: {image_array.shape}")
print(f"Image dtype: {image_array.dtype}")
image_array = image_array[0, 0, 0, 1, :, :, :]  # Extract the first channel
print(f"MORF3 Image array shape: {image_array.shape}")

# Remove unnecessary dimensions and reshape to (Z, Y, X)
image_array = image_array.squeeze()  # Removes singleton dimensions

# Downscale the image to 1/4 of its original size
scale_factor = 0.25  # Reduce to 1/4 of original size
new_shape = (
    image_array.shape[0],  # Keep the number of Z slices the same
    int(image_array.shape[1] * scale_factor),  # Scale Y
    int(image_array.shape[2] * scale_factor)   # Scale X
)

image_array_resized = skimage.transform.resize(
    image_array,
    new_shape,
    anti_aliasing=True,
    preserve_range=True
).astype(image_array.dtype)  # Convert back to original dtype

print(f"Resized Image array shape: {image_array_resized.shape}")

# Save as a .tif file
save_path = "/data/wangfeiran/code/brainbow/segmentation/SegNeuron/data/0207_czi/resize_result/resized_z1-10.tif"
tiff.imwrite(save_path, image_array_resized, dtype=image_array.dtype)

print(f"Saved resized 3D TIFF: {save_path}")

