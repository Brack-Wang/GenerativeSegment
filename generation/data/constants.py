import os

# ### Setup a data directory and download dataset
# Directory to read raw images and masks
data_dir = "/data/wangfeiran/code/brainbow/generation/datasets/segmentation_data"
brainbow_dir = data_dir + "/cropped_raw"
mask_dir = data_dir + "/cropped_masks"
model_dir = "/data/wangfeiran/code/brainbow/output/test"

# Training Parameters
batch_size = 10
train_number = 884
set_determinism_seed = 42

vae_epochs = 800
val_interval = 50

ddpm_epochs = 3000
ddpm_val_interval = 200



# Directories for saving models, images, and other outputs
ddpm_dir = os.path.join(model_dir, "ddpm")
ddpm_checkpoint_dir = os.path.join(ddpm_dir, "checkpoints")
controlnet_dir = os.path.join(model_dir, "controlnet")
controlnet_checkpoint_dir = os.path.join(controlnet_dir, "checkpoints")
autoencoder_dir = os.path.join(model_dir, "autoencoder")
autoencoder_checkpoint_dir = os.path.join(autoencoder_dir, "checkpoints")
original_image_dir = os.path.join(autoencoder_dir, "images")
reconstruction_dir = os.path.join(autoencoder_dir, "reconstructions")
vae_masks_dir = os.path.join(autoencoder_dir, "masks")
ddpm_original_image_dir = os.path.join(ddpm_dir, "images")
ddpm_synth_image_dir = os.path.join(ddpm_dir, "reconstructions")
ddpm_mask_image_dir = os.path.join(ddpm_dir, "masks")
output_dir = os.path.join(model_dir, "inference_slices")
generated_dir = os.path.join(output_dir, "generated")
raw_dir = os.path.join(output_dir, "raw")
maskout_dir = os.path.join(output_dir, "mask")

# Create all required directories
def create_directories():
    """
    Ensure all directories are created.
    """
    os.makedirs(brainbow_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(ddpm_dir, exist_ok=True)
    os.makedirs(ddpm_checkpoint_dir, exist_ok=True)
    os.makedirs(controlnet_dir, exist_ok=True)
    os.makedirs(controlnet_checkpoint_dir, exist_ok=True)
    os.makedirs(autoencoder_dir, exist_ok=True)
    os.makedirs(autoencoder_checkpoint_dir, exist_ok=True)
    os.makedirs(original_image_dir, exist_ok=True)
    os.makedirs(vae_masks_dir, exist_ok=True)
    os.makedirs(reconstruction_dir, exist_ok=True)
    os.makedirs(ddpm_original_image_dir, exist_ok=True)
    os.makedirs(ddpm_synth_image_dir, exist_ok=True)
    os.makedirs(ddpm_mask_image_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(generated_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(maskout_dir, exist_ok=True)

# Call the function to create directories
create_directories()


