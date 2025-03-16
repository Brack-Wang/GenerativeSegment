import os
import sys
import numpy as np
import torch
import tifffile as tiff
from sklearn.model_selection import train_test_split
monai_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "MONAI")
sys.path.append(monai_path)
from monai.transforms import Compose, Lambdad, EnsureTyped
from monai.data import Dataset, DataLoader
from .constants import brainbow_dir, mask_dir, train_number, batch_size, set_determinism_seed
from monai.utils import set_determinism

def load_tiff_with_tifffile(file_path):
    """
    Load TIFF files and ensure they have 4 dimensions.
    """
    image = tiff.imread(file_path)
    if image.ndim == 3:  # Add channel axis if missing
        image = image[:, np.newaxis, :, :]
    if image.ndim != 4:
        raise ValueError(f"Unexpected image shape {image.shape}. Expected 4D (depth, channels, height, width).")
    return image.astype(np.float32)

def get_transforms():
    """
    Define transformations for both images and masks.
    """
    return Compose([
        # Load the image and mask using your custom TIFF loader with Lambdad
        Lambdad(keys=["image"], func=lambda path: load_tiff_with_tifffile(path)),
        Lambdad(keys=["mask"], func=lambda path: load_tiff_with_tifffile(path)),
        
        # Ensure the data types are compatible with PyTorch tensors
        EnsureTyped(keys=["image", "mask"]),

        # Merge 4 channels into 1 by computing the mean
        Lambdad(keys=["image"], func=lambda x: torch.mean(torch.tensor(x), dim=1, keepdim=True)),

        # Normalize image intensities to [0, 1]
        Lambdad(keys=["image"], func=lambda x: (x - x.min()) / (x.max() - x.min())), 
        Lambdad(keys=["mask"], func=lambda x: (x - x.min()) / (x.max() - x.min())),  

        # Permute axes for compatibility with PyTorch (depth, height, width, channel -> channel, depth, height, width)
        Lambdad(keys=["image"], func=lambda x: torch.tensor(x).permute(1, 3, 2, 0)),
        Lambdad(keys=["mask"], func=lambda x: torch.tensor(x).permute(1, 3, 2, 0)),

        # Train the model with only 3 channels
        # Lambdad(keys=["image"], func=lambda x: torch.tensor(x)[:3, :, :, :]),
        # Lambdad(keys=["mask"], func=lambda x: torch.tensor(x).repeat(3, 1, 1, 1)),
    ])

def prepare_datasets():
    """
    Prepare training and validation datasets and DataLoaders.
    """
    set_determinism(set_determinism_seed)

    # Load image and mask file paths
    image_files = [os.path.join(brainbow_dir, f"neuron_{i}.tif") for i in range(train_number)]
    mask_files = [os.path.join(mask_dir, f"neuron_{i}.tif") for i in range(train_number)]
    if len(image_files) != len(mask_files):
        raise ValueError("The number of images and masks do not match.")
    
    data_files = [{"image": img, "mask": msk} for img, msk in zip(image_files, mask_files)]
    train_files, val_files = train_test_split(data_files, test_size=0.2, random_state=42)

    # Create MONAI datasets
    train_ds = Dataset(data=train_files, transform=get_transforms())
    val_ds = Dataset(data=val_files, transform=get_transforms())

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True, drop_last=True)


    first_train_sample = train_ds[0]
    print(f'Train Image shape: {first_train_sample["image"].shape}')
    print(f'Train Mask shape: {first_train_sample["mask"].shape}')
    print(f"Total data: {len(train_ds) + len(val_ds)}")
    print(f"Training data: {len(train_loader.dataset)}")
    print(f"Validation data: {len(val_loader.dataset)}")

    return train_loader, val_loader
