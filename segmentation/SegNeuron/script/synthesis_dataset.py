import tifffile as tiff
import numpy as np
import random
import os
import scipy.ndimage as ndi
import colorsys

# Paths to folders
generated_folder = '/data/wangfeiran/code/brainbow/output/segsp_pro_11/inference_slices/generated'
mask_folder = '/data/wangfeiran/code/brainbow/output/segsp_pro_11/inference_slices/mask'

# Output directory (we'll save multiple outputs here)
output_dir = '/data/wangfeiran/code/brainbow/output/segsp_pro_11/inference_slices/synthesis_dataset'
os.makedirs(output_dir, exist_ok=True)
output_mask_dir = output_dir + '/mask'
os.makedirs(output_mask_dir, exist_ok=True)
output_generated_dir = output_dir + '/generated'
os.makedirs(output_generated_dir, exist_ok=True)    

# We'll create 400 combined images, each with 8 neurons
NUM_COMBINED = 1000
NEURON_NUMBER = 12
mask_expantion = 5

# Dimensions of each single mask/image
combined_shape = (50, 1, 250, 250)  # Z, C, Y, X

def get_bounding_box(mask, margin=5):
    """
    Returns a bounding box for the non-zero region of the mask, expanded by 'margin' voxels/pixels
    in all directions but clamped so as not to exceed the mask boundaries.
    """
    zsize, csize, ysize, xsize = mask.shape
    non_zero = np.argwhere(mask > 0)

    z_min_raw, y_min_raw, x_min_raw = non_zero.min(axis=0)[[0, 2, 3]]
    z_max_raw, y_max_raw, x_max_raw = non_zero.max(axis=0)[[0, 2, 3]]

    z_min = max(0, z_min_raw - margin)
    z_max = min(zsize, z_max_raw + 1 + margin)
    y_min = max(0, y_min_raw - margin)
    y_max = min(ysize, y_max_raw + 1 + margin)
    x_min = max(0, x_min_raw - margin)
    x_max = min(xsize, x_max_raw + 1 + margin)

    return z_min, z_max, y_min, y_max, x_min, x_max

def place_mask_randomly(cropped_mask, cropped_generated_image):
    """
    Places a cropped mask/generated pair into a new array of shape combined_shape at random positions.
    """
    new_mask = np.zeros(combined_shape, dtype=np.float32)
    new_generated = np.zeros(combined_shape, dtype=np.float32)
    
    z, c, y, x = combined_shape
    z_crop, c_crop, y_crop, x_crop = cropped_mask.shape
    
    max_z = z - z_crop
    max_y = y - y_crop
    max_x = x - x_crop

    start_z = random.randint(0, max_z)
    start_y = random.randint(0, max_y)
    start_x = random.randint(0, max_x)

    new_mask[start_z:start_z + z_crop, :, start_y:start_y + y_crop, start_x:start_x + x_crop] += cropped_mask
    new_generated[start_z:start_z + z_crop, :, start_y:start_y + y_crop, start_x:start_x + x_crop] += cropped_generated_image

    return new_mask, new_generated


def generate_unique_colors(num_colors):
    colors = []
    for i in range(num_colors):
        # Evenly space hues between 0 and 1
        hue = i / num_colors
        # Convert HSV to RGB (using full saturation and value)
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        # Scale RGB values to 0-255 and convert to integers
        rgb = [int(c * 255) for c in rgb]
        colors.append(rgb)
    return colors

def colorize_masks(images):
    """
    Colorize up to 8 mask images and merge them into a single RGB volume of shape (50, 3, 250, 250).
    """
    unique_colors = generate_unique_colors(NEURON_NUMBER)

    # Create an empty array for colorized output
    colorized = np.zeros((50, 3, 250, 250), dtype=np.uint16)

    for idx, img in enumerate(images):
        # img shape: (50, 1, 250, 250)
        mask_3d = img[:, 0, :, :]  # => (50, 250, 250)
        color = unique_colors[idx]

        for c in range(3):
            colorized[:, c, :, :] += (mask_3d * color[c]).astype(np.uint16)

    # Clip to [0, 255]
    colorized = np.clip(colorized, 0, 255).astype(np.uint8)
    return colorized

def expand_and_denoise(mask, generated_image, expansion_size=5, bg_weight=0.2):
    """
    Morphologically dilate 'mask' and then weight the 'generated_image' to reduce background noise.
    """
    binary_mask = (mask > 0).astype(np.uint8)  # shape: (Z, 1, Y, X)
    z, c, y, x = binary_mask.shape
    mask_3d = binary_mask.reshape(z, y, x)  
    structure = np.ones((1, expansion_size, expansion_size), dtype=np.uint8)
    expanded_mask_3d = ndi.binary_dilation(mask_3d, structure=structure)
    expanded_mask = expanded_mask_3d.reshape(z, 1, y, x).astype(np.float32)
    denoised_image = generated_image * (expanded_mask + bg_weight * (1.0 - expanded_mask))
    return denoised_image

def create_combined_image(mask_files, generated_files, output_mask_path, output_generated_path):
    """
    Given 8 randomly selected files, combine them into a single mask and single image,
    then save to the specified output paths.
    """
    # Each run creates one combined image from the 8 chosen pairs.
    mask_images = []
    generated_images = []

    for file in mask_files:
        # Read mask
        mask = tiff.imread(os.path.join(mask_folder, file))  # shape: (50, 1, 250, 250)
        
        # Derive the corresponding generated filename
        idx = file.split('_')[-1].split('.')[0]
        synth_file = f'synth_image_{idx}.tif'
        generated_image = tiff.imread(os.path.join(generated_folder, synth_file))

        # (Optional) expand & denoise
        generated_image = expand_and_denoise(mask, generated_image, expansion_size=mask_expantion, bg_weight=0.5)

        # Get bounding box and crop
        z_min, z_max, y_min, y_max, x_min, x_max = get_bounding_box(mask, margin=10)
        cropped_mask = mask[z_min:z_max, :, y_min:y_max, x_min:x_max]
        cropped_generated = generated_image[z_min:z_max, :, y_min:y_max, x_min:x_max]

        # Place randomly into an empty combined_shape
        new_mask, new_generated = place_mask_randomly(cropped_mask, cropped_generated)

        mask_images.append(new_mask)
        generated_images.append(new_generated)

    # Now colorize the combined mask images
    combined_mask = colorize_masks(mask_images)

    # For the combined generated image, sum them up
    combined_synthesis = np.zeros_like(generated_images[0])
    for img in generated_images:
        combined_synthesis += img
    
    # Expand to 3 channels
    combined_synthesis = np.repeat(combined_synthesis, 3, axis=1)

    # Save results
    tiff.imwrite(output_mask_path, combined_mask.astype(np.float32))
    tiff.imwrite(output_generated_path, combined_synthesis.astype(np.float32))

def main():
    # 1. List all mask files (we assume 800 .tif in mask_folder)
    all_mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith('.tif')])
    total_masks = len(all_mask_files)
    print(f"Found {total_masks} mask files.")

    # We want to create 400 combined images, each with 8 neurons
    for i in range(NUM_COMBINED):
        # Randomly pick 8 distinct files
        chosen_mask_files = random.sample(all_mask_files, NEURON_NUMBER)
        
        # Output filenames
        out_mask = os.path.join(output_mask_dir, f"combined_mask_{i}.tif")
        out_generated = os.path.join(output_generated_dir, f"combined_generated_{i}.tif")

        # Create and save
        create_combined_image(chosen_mask_files, None, out_mask, out_generated)
        print(f"Created {out_mask} and {out_generated}")

if __name__ == "__main__":
    main()
