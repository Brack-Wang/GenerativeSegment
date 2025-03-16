import os
import random
import tifffile as tiff 
import numpy as np
# Define source directories
raw_dir = "output/segsp_pro_11/inference_slices/synthesis_dataset/generated"
mask_dir = "output/segsp_pro_11/inference_slices/synthesis_dataset/mask"

# Define new dataset directories
train_dir = "segmentation/SegNeuron/data/brainbow/train2"
valid_dir = "segmentation/SegNeuron/data/brainbow/valid"

ratio = 0.002

# Create new dataset directories if they do not exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)

# Get and sort file lists
raw_files = sorted([f for f in os.listdir(raw_dir) if f.endswith(".tif")])
mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".tif")])

# Ensure raw and mask files match
assert len(raw_files) == len(mask_files), "Mismatch in raw and mask file counts!"

# Generate indices for splitting (80% train, 20% valid)
num_files = len(raw_files)
indices = list(range(num_files))
random.shuffle(indices)  # Shuffle for randomness

split_index = int(num_files * ratio)  # 80% cutoff
train_indices = indices[:split_index]
valid_indices = indices[split_index:]

# # Function to load, reshape using mean, and save .tif images while preserving the original type
# def process_and_save_tif(file_path, save_path):
#     img = tiff.imread(file_path)  # Load .tif file
#     if img.shape[1] == 3:  # Ensure shape is (50,3,250,250)
#         imgseg = img.mean(axis=1)  # Compute mean along the channel dimension (3), reducing shape to (50,250,250)
    
#     tiff.imwrite(save_path, imgseg.astype(img.dtype))  # Preserve the original data type

# Function to load, apply weighted averaging, and save .tif images
def process_and_save_tif(file_path, save_path):
    img = tiff.imread(file_path)  # Load .tif file
    if img.shape[1] == 3:  # Ensure shape is (50,3,250,250)
        weights = np.array([0.25, 0.50, 0.25])  # Assign more weight to the center channel
        imgseg = np.average(img, axis=1, weights=weights)  # Weighted averaging
    
    tiff.imwrite(save_path, imgseg.astype(img.dtype))  

# Function to copy and rename files sequentially with reshaping
def copy_and_rename_files(indices, target_folder):
    for new_idx, old_idx in enumerate(indices):  # Assign new sequential index
        raw_file = raw_files[old_idx]
        mask_file = mask_files[old_idx]

        # Define new names with sequential numbering
        new_raw_name = f"{new_idx}.tif"
        new_mask_name = f"{new_idx}_MaskIns.tif"

        # Process and save files with correct shape
        process_and_save_tif(os.path.join(raw_dir, raw_file), os.path.join(target_folder, new_raw_name))
        process_and_save_tif(os.path.join(mask_dir, mask_file), os.path.join(target_folder, new_mask_name))

        print(f"Processed and saved: {raw_file} -> {new_raw_name}, {mask_file} -> {new_mask_name}")

# Copy 80% to train and 20% to valid
copy_and_rename_files(train_indices, train_dir)
# copy_and_rename_files(valid_indices, valid_dir)

# Print success message
print(f"Successfully copied and renamed {len(train_indices)} files to {train_dir} and {len(valid_indices)} files to {valid_dir}")