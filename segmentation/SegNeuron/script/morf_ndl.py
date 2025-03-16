import numpy as np
import tifffile as tiff
import cv2
from skimage.morphology import remove_small_objects, closing, square
from skimage.util import img_as_ubyte

# Load the 3D TIFF file
input_path = "/data/wangfeiran/code/brainbow/segmentation/SegNeuron/data/morf3/small/basson.tif"
output_path = "/data/wangfeiran/code/brainbow/segmentation/SegNeuron/data/morf3/small/inference/neuron_ndl.tif"

image_3d = tiff.imread(input_path)  # Shape: [Z, Y, X]

# Ensure the image is in grayscale (Z, Y, X)
print(f"Original Shape: {image_3d.shape}")

# Create an empty array for storing the segmented results
segmented_3d = np.zeros_like(image_3d, dtype=np.uint16)

# Process each slice in the Z direction
for i in range(image_3d.shape[0]):
    img = image_3d[i].astype(np.uint16)  # Ensure it is a NumPy uint16 array

    # Apply Gaussian blur to remove noise
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)

    # Apply Otsu’s thresholding to segment neurons
    _, img_thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 计算 Otsu 阈值
    otsu_threshold, img_thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 降低 Otsu 阈值，例如降低 10%
    lower_threshold = max(0, int(otsu_threshold * 0.6))  

    # 重新应用手动阈值
    _, img_thresh = cv2.threshold(img_blur, lower_threshold, 255, cv2.THRESH_BINARY)


    # Convert to a proper NumPy array and apply morphological closing
    img_morph = closing(img_thresh.astype(bool), square(3))  # Convert to binary for skimage operations

    # Convert to NumPy uint16 explicitly
    img_morph = np.array(img_morph, dtype=np.uint16) * 255

    # Store the processed slice
    segmented_3d[i] = img_morph

# Save the segmented 3D volume as a TIFF file
tiff.imwrite(output_path, segmented_3d.astype(np.uint16))

print(f"Segmented 3D neuron structures saved at: {output_path}")
