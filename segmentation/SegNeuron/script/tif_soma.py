import tifffile as tiff
import numpy as np
import cv2
import os

# Load the TIFF file
tif_path = "/data/wangfeiran/code/brainbow/segmentation/SegNeuron/data/0207_czi/intensity_result/intensity_z1-22.tif"
tif_data = tiff.imread(tif_path)

# Ensure it's a 3D volume (Z, Y, X)
assert len(tif_data.shape) == 3, "Input TIFF should be a 3D volume."

# Create an empty mask for the filled soma
filled_soma_mask = np.zeros_like(tif_data, dtype=np.uint8)

# Process each slice along the Z dimension
for z in range(tif_data.shape[0]):
    # Convert to binary mask (assuming soma is the bright region)
    binary_mask = (tif_data[z] > 0).astype(np.uint8) * 255

    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank mask to fill the soma
    filled_mask = np.zeros_like(binary_mask)

    # Fill contours (assuming soma is a large, roughly circular shape)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 3000 and area <12000 :  # Adjust threshold based on expected soma size
            cv2.drawContours(filled_mask, [contour], -1, 255, thickness=cv2.FILLED)

    # Store the filled soma mask
    filled_soma_mask[z] = filled_mask

# Define output path
output_path = "/data/wangfeiran/code/brainbow/segmentation/SegNeuron/data/0207_czi/intensity_result/filled_soma_mask.tif"

# Save the filled soma mask as a new TIFF file
tiff.imwrite(output_path, filled_soma_mask.astype(np.uint8))

print(f"Filled soma mask saved at: {output_path}")
