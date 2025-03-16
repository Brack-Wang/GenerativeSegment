import numpy as np
from tifffile import TiffFile, imwrite
import os
import cv2
from scipy.ndimage import gaussian_filter
from natsort import natsorted

def locate_bounding_box(mask):
    """
    Locate the smallest x, y, z coordinates where the mask is non-zero.

    Args:
        mask (np.array): The loaded mask with dimensions (Z, 1, Y, X).

    Returns:
        tuple: Smallest z, y, x indices of the bounding box.
    """
    z_indices, y_indices, x_indices = np.where(mask > 0)
    if len(z_indices) == 0:  # No non-zero elements in the mask
        return None
    min_z, min_y, min_x = np.min(z_indices), np.min(y_indices), np.min(x_indices)
    return min_z, min_y, min_x

def crop_and_pad(raw_image, mask, min_z, min_y, min_x, bbox_size):
    """
    Crop and pad the raw image and mask based on the bounding box.

    Args:
        raw_image (np.array): The raw image with dimensions (Z, 4, Y, X).
        mask (np.array): The mask with dimensions (Z, 1, Y, X).
        min_z, min_y, min_x (int): Bounding box starting coordinates.
        bbox_size (tuple): Desired bounding box size (Z, Y, X).

    Returns:
        tuple: Cropped and padded raw image and mask.
    """
    bbox_z, bbox_y, bbox_x = bbox_size
    max_z, max_y, max_x = raw_image.shape[0], raw_image.shape[2], raw_image.shape[3]

    # Calculate end coordinates for cropping
    end_z = min(min_z + bbox_z, max_z)
    end_y = min(min_y + bbox_y, max_y)
    end_x = min(min_x + bbox_x, max_x)

    # Crop the raw image and mask
    cropped_raw = raw_image[min_z:end_z, :, min_y:end_y, min_x:end_x]
    cropped_mask = mask[min_z:end_z, min_y:end_y, min_x:end_x]

    # Calculate padding required to reach desired size
    pad_z_before = max(0, -min_z)
    pad_y_before = max(0, -min_y)
    pad_x_before = max(0, -min_x)

    pad_z_after = max(0, bbox_z - cropped_raw.shape[0] - pad_z_before)
    pad_y_after = max(0, bbox_y - cropped_raw.shape[2] - pad_y_before)
    pad_x_after = max(0, bbox_x - cropped_raw.shape[3] - pad_x_before)

    # Apply padding
    padded_raw = np.pad(cropped_raw, ((pad_z_before, pad_z_after), (0, 0), (pad_y_before, pad_y_after), (pad_x_before, pad_x_after)), mode='constant', constant_values=0)
    padded_mask = np.pad(cropped_mask, ((pad_z_before, pad_z_after), (pad_y_before, pad_y_after), (pad_x_before, pad_x_after)), mode='constant', constant_values=0)

    return padded_raw, padded_mask

def spatial_segmentation_smoothing(generated_image, sigma=2, target_dtype=np.uint16):
    """
    Apply spatial segmentation and smoothing to refine the sharp edges of a 3D mask while retaining the desired data type.

    Args:
        generated_image (np.array): The generated 3D mask (Z, Y, X).
        sigma (float): Standard deviation for Gaussian smoothing.
        target_dtype: Target data type for the mask (e.g., np.uint16).

    Returns:
        np.array: Smoothed 3D mask with refined edges and the desired data type.
    """
    # Initialize an empty array to store smoothed slices
    smoothed_3d = np.zeros_like(generated_image, dtype=target_dtype)

    # Iterate over each slice along the Z-axis
    for z in range(generated_image.shape[0]):
        slice_2d = generated_image[z]

        # Normalize the 2D slice to 8-bit for processing
        if slice_2d.max() > 0:  # Avoid dividing by zero
            temp_image = ((slice_2d / slice_2d.max()) * 255).astype(np.uint8)
        else:
            temp_image = slice_2d.astype(np.uint8)

        # Detect edges using Canny
        edges = cv2.Canny(temp_image, 50, 150)

        # Dilate the edges to create a mask for sharp regions
        kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)

        # Convert edge mask to binary
        edge_mask = (dilated_edges > 0).astype(np.uint8)

        # Smooth the 2D slice
        smoothed_image = gaussian_filter(temp_image.astype(np.float32), sigma=sigma)

        # Blend smoothed regions with the original slice
        refined_image = temp_image * (1 - edge_mask) + smoothed_image * edge_mask

        # Normalize back to the original range and cast to the target dtype
        refined_slice = (refined_image / 255 * slice_2d.max()).astype(target_dtype)

        # Store the refined slice in the 3D array
        smoothed_3d[z] = refined_slice

    return smoothed_3d


def spatial_segmentation_smoothing_v2(generated_image, sigma=2, target_dtype=np.uint16):
    """
    Apply spatial segmentation and smoothing with refined edge handling.

    Args:
        generated_image (np.array): The generated 3D mask (Z, Y, X).
        sigma (float): Standard deviation for Gaussian smoothing.
        target_dtype: Target data type for the mask (e.g., np.uint16).

    Returns:
        np.array: Smoothed 3D mask with refined edges and the desired data type.
    """
    smoothed_3d = np.zeros_like(generated_image, dtype=target_dtype)

    for z in range(generated_image.shape[0]):
        slice_2d = generated_image[z]

        # Normalize the 2D slice
        if slice_2d.max() > 0:
            temp_image = ((slice_2d / slice_2d.max()) * 255).astype(np.uint8)
        else:
            temp_image = slice_2d.astype(np.uint8)

        # Detect edges
        edges = cv2.Canny(temp_image, 50, 150)

        # Create a soft edge mask
        edge_mask = gaussian_filter(edges.astype(np.float32) / 255.0, sigma=1)

        # Smooth the image
        smoothed_image = gaussian_filter(temp_image.astype(np.float32), sigma=sigma)

        # Blend original and smoothed regions
        refined_image = (1 - edge_mask) * temp_image + edge_mask * smoothed_image

        # Normalize and convert back to original dtype
        refined_slice = (refined_image / 255 * slice_2d.max()).astype(target_dtype)
        smoothed_3d[z] = refined_slice

    return smoothed_3d


def apply_gaussian_to_mask(mask, sigma=2, target_dtype=np.uint16):
    """
    Apply Gaussian smoothing to the entire mask and filter the raw image.

    Args:
        image (np.array): The raw input image.
        mask (np.array): The binary mask (same shape as image).
        sigma (float): Standard deviation for Gaussian smoothing.

    Returns:
        np.array: The image with the smoothed mask applied.
    """
    # Normalize the mask to [0, 1]
    normalized_mask = mask / mask.max()

    # Apply Gaussian filter to smooth the mask
    smoothed_mask = gaussian_filter(normalized_mask.astype(np.float32), sigma=sigma)

    smoothed_mask = smoothed_mask * 255.0

    return smoothed_mask.astype(target_dtype)


def apply_mask_to_raw(cropped_raw, cropped_mask):
    """
    Retain all pixel values within the segmentation mask in the raw image and set others to zero.

    Args:
        cropped_raw (np.array): The cropped raw image with dimensions (Z, C, Y, X).
        cropped_mask (np.array): The segmentation mask with dimensions (Z, Y, X).

    Returns:
        np.array: Raw image with all pixels outside the mask set to zero.
    """
    # Expand the mask dimensions to match the raw image (Z, C, Y, X)
    expanded_mask = np.expand_dims(cropped_mask, axis=1)  # Shape: (Z, 1, Y, X)
    expanded_mask = np.repeat(expanded_mask, cropped_raw.shape[1], axis=1)
    
    # Ensure the mask values are binary (0 or 1)
    binary_mask = (expanded_mask > 0).astype(cropped_raw.dtype)

    # Apply the mask to the raw image
    masked_raw = cropped_raw * binary_mask  # Broadcasting ensures correct masking

    return masked_raw.astype(cropped_raw.dtype)


if __name__ == "__main__":
    # Define paths
    raw_folder_path = "/data/wangfeiran/code/brainbow/datasets/compress/raw"
    mask_folder_base_path = "/data/wangfeiran/code/brainbow/datasets/segmentation_data/masks"
    output_dir = "/data/wangfeiran/code/brainbow/datasets/segmentation_data"
    raw_output_folder_path = output_dir + "/cropped_raw_80"
    mask_output_folder_path = output_dir + "/cropped_masks_80_smooth_v3"
    bbox_size = (10, 80, 80)  # Desired bounding box size (Z, Y, X)

    # Create output folders if they don't exist
    os.makedirs(raw_output_folder_path, exist_ok=True)
    os.makedirs(mask_output_folder_path, exist_ok=True)

    global_id = 0

    # Iterate over each raw image in the raw folder
    for raw_file in sorted(os.listdir(raw_folder_path)):
        if raw_file.endswith(".tif"):
            raw_image_path = os.path.join(raw_folder_path, raw_file)

            # Load raw image
            with TiffFile(raw_image_path) as tif:
                raw_image = tif.asarray()
            print(f"Raw image {raw_file} loaded with shape: {raw_image.shape}")

            # Determine corresponding mask folder
            mask_folder_name = raw_file[:-4]  # Remove '.tif' from raw file name
            mask_folder_path = os.path.join(mask_folder_base_path, mask_folder_name)

            if not os.path.exists(mask_folder_path):
                print(f"Mask folder not found for {raw_file}, skipping.")
                continue

            # Iterate over each mask in the corresponding mask folder
            for mask_file in natsorted(os.listdir(mask_folder_path)):
                if mask_file.endswith(".tif"):
                    mask_path = os.path.join(mask_folder_path, mask_file)

                    # Load mask
                    with TiffFile(mask_path) as tif:
                        mask = tif.asarray()
                    print(f"Mask {mask_file} loaded with shape: {mask.shape}")

                    # Locate bounding box
                    bbox_start = locate_bounding_box(mask)
                    if bbox_start is None:
                        print(f"Skipping {mask_file}, no non-zero elements.")
                        continue

                    min_z, min_y, min_x = bbox_start
                    print(f"Bounding box start for {mask_file}: z={min_z}, y={min_y}, x={min_x}")

                    # Crop and pad raw image and mask
                    cropped_raw, cropped_mask = crop_and_pad(raw_image, mask, min_z, min_y, min_x, bbox_size)
                    # cropped_mask = spatial_segmentation_smoothing_v2(cropped_mask, sigma=2, target_dtype=cropped_raw.dtype)
                    cropped_mask = apply_gaussian_to_mask(cropped_mask, sigma=1, target_dtype=cropped_raw.dtype)

                    # Save cropped raw image and mask
                    cropped_raw_path = os.path.join(raw_output_folder_path, f"neuron_{global_id}.tif")
                    cropped_mask_path = os.path.join(mask_output_folder_path, f"neuron_{global_id}.tif")

                    imwrite(cropped_raw_path, cropped_raw, dtype=cropped_raw.dtype, imagej=True)
                    imwrite(cropped_mask_path, cropped_mask, dtype=cropped_mask.dtype, imagej=True)

                    print(f"Cropped and padded raw image saved to {cropped_raw_path}")
                    print(f"Cropped and padded mask saved to {cropped_mask_path}")

                    # Apply mask to raw image
                    # threshold_output_folder_path  = output_dir + "/cropped_raw_clean_80"
                    # os.makedirs(threshold_output_folder_path, exist_ok=True)
                    # threshold_raw = apply_mask_to_raw(cropped_raw, cropped_mask)
                    # cropped_raw_threshold_path = os.path.join(threshold_output_folder_path, f"neuron_{global_id}.tif")
                    # imwrite(cropped_raw_threshold_path, threshold_raw, dtype=threshold_raw.dtype, imagej=True)
                     # print(f"Threshold raw image saved to {cropped_raw_threshold_path}")
                     
                    global_id += 1
