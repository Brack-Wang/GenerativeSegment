import os
import sys
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn import L1Loss
from torch.cuda.amp import GradScaler

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from generative.networks.nets import VQVAE, PatchDiscriminator
from generative.losses import PerceptualLoss, PatchAdversarialLoss

# Import constants and data loader
from data.constants import autoencoder_checkpoint_dir, vae_epochs, val_interval, original_image_dir, reconstruction_dir, vae_masks_dir

def save_learning_curves(recon_loss_list, gen_loss_list, disc_loss_list, val_recon_loss_list=None, output_dir=autoencoder_checkpoint_dir):
    """
    Plot and save the learning curves for training and validation.

    Args:
        recon_loss_list (list): Reconstruction losses over training epochs.
        gen_loss_list (list): Generator losses over training epochs.
        disc_loss_list (list): Discriminator losses over training epochs.
        val_recon_loss_list (list, optional): Validation reconstruction losses. Default is None.
        output_dir (str): Directory where the plots will be saved. Default is the current directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Plot and save reconstruction loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(
        np.linspace(1, len(recon_loss_list), len(recon_loss_list)),
        recon_loss_list,
        label="Train Reconstruction Loss",
        color="C0",
        linewidth=2,
    )
    if val_recon_loss_list:
        plt.plot(
            np.linspace(1, len(val_recon_loss_list), len(val_recon_loss_list)),
            val_recon_loss_list,
            label="Validation Reconstruction Loss",
            color="C1",
            linewidth=2,
        )
    plt.title("Reconstruction Loss Curve", fontsize=18)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "reconstruction_loss_curve.png"))
    plt.close()

    # Plot and save adversarial loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(
        np.linspace(1, len(gen_loss_list), len(gen_loss_list)),
        gen_loss_list,
        label="Generator Loss",
        color="C0",
        linewidth=2,
    )
    plt.plot(
        np.linspace(1, len(disc_loss_list), len(disc_loss_list)),
        disc_loss_list,
        label="Discriminator Loss",
        color="C1",
        linewidth=2,
    )
    plt.title("Adversarial Training Loss Curve", fontsize=18)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "adversarial_loss_curve.png"))
    plt.close()

    print(f"Learning curves saved in {output_dir}")




class VQGANTrainer:
    def __init__(self, device, train_loader, val_loader):
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.n_example_images = 4

        # Initialize VQGAN models
        self.autoencoder = self.initialize_autoencoder()
        self.discriminator = self.initialize_discriminator()

        # Initialize loss functions
        self.l1_loss = L1Loss()
        self.adv_loss = PatchAdversarialLoss(criterion="least_squares")
        self.perceptual_loss = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2)
        self.perceptual_loss.to(self.device)

        # Initialize optimizers
        self.optimizer_g = torch.optim.Adam(params=self.autoencoder.parameters(), lr=5e-5)
        self.optimizer_d = torch.optim.Adam(params=self.discriminator.parameters(), lr=3e-4)

        # Initialize gradient scalers
        self.scaler_g = GradScaler()
        self.scaler_d = GradScaler()

        # Training history
        self.epoch_recon_loss_list = []
        self.epoch_gen_loss_list = []
        self.epoch_disc_loss_list = []
        self.val_recon_epoch_loss_list = []
        self.intermediary_images = []

    def initialize_autoencoder(self):
        """
        Initialize the VQVAE model.
        """
        downsample_params = [
            ((2, 2, 2), (4, 4, 4), (1, 1, 1), (1, 1, 1)),
            ((2, 2, 2), (4, 4, 4), (1, 1, 1), (1, 1, 1)),
        ]
        upsample_params = [
            ((2, 2, 2), (4, 4, 4), (1, 1, 1), (1, 1, 1), (0, 0, 0)),
            ((2, 2, 2), (5, 5, 5), (1, 1, 1), (1, 1, 1), (1, 1, 1)),
        ]
        model = VQVAE(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            downsample_parameters=downsample_params,
            upsample_parameters=upsample_params,
            num_res_layers=2,
            num_channels=(64, 128),
            num_res_channels=32,
            num_embeddings=512,
            embedding_dim=64,
            commitment_cost=0.25,
            decay=0.99,
            epsilon=1e-5,
        )
        return model.to(self.device)

    def initialize_discriminator(self):
        """
        Initialize the PatchDiscriminator model.
        """
        return PatchDiscriminator(spatial_dims=3, in_channels=1, num_layers_d=3, num_channels=64).to(self.device)

    def load_checkpoint(self):
        """
        Load the latest checkpoint if available.
        """
        checkpoint_files = glob.glob(os.path.join(autoencoder_checkpoint_dir, "checkpoint_epoch_*.pth"))
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
            checkpoint = torch.load(latest_checkpoint)
            self.autoencoder.load_state_dict(checkpoint["autoencoder_state_dict"])
            self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
            self.optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
            self.optimizer_d.load_state_dict(checkpoint["optimizer_d_state_dict"])
            self.epoch_recon_loss_list = checkpoint["epoch_recon_loss_list"]
            self.epoch_gen_loss_list = checkpoint["epoch_gen_loss_list"]
            self.epoch_disc_loss_list = checkpoint["epoch_disc_loss_list"]
            self.val_recon_epoch_loss_list = checkpoint["val_recon_epoch_loss_list"]
            self.intermediary_images = checkpoint.get("intermediary_images", [])
            print(f"Checkpoint loaded from {latest_checkpoint}")
            return checkpoint["epoch"] + 1
        return 0

    def valiate_and_save(self, epoch):
        """
        Perform validation, calculate loss, and save reconstructed and reference images.
        
        Args:
            epoch (int): Current epoch number.
        """
        self.autoencoder.eval()
        val_loss = 0
        images_val = None  # Initialize variable to avoid UnboundLocalError
        reconstruction_val = None
        masks_val = None
        with torch.no_grad():
            for val_step, batch in enumerate(self.val_loader, start=1):
                images_val = batch["image"].to(self.device)
                masks_val = batch["mask"].to(self.device)
                # # TODO: Segmentation based
                # images_val = images_val * masks_val
                # # Normalize to [0, 1]
                # min_val = images_val.amin(dim=(2, 3, 4), keepdim=True)  # Min over spatial dimensions
                # max_val = images_val.amax(dim=(2, 3, 4), keepdim=True)  # Max over spatial dimensions
                # images_val = (images_val - min_val) / (max_val - min_val + 1e-8)  # Avoid division by zero
                

                # Forward pass
                reconstruction_val, quantization_loss = self.autoencoder(images=images_val)

                # Save first batch for visualization
                if val_step == 1:
                    self.intermediary_images.append(reconstruction_val[:self.n_example_images, 0])

                # Compute reconstruction loss
                recons_loss = self.l1_loss(reconstruction_val.float(), images_val.float())
                val_loss += recons_loss.item()
                val_loss /= val_step
                print(f"Validation Loss: {val_loss}")

                for i in range(3):
                    image_sample = images_val[i][0].cpu().detach().numpy()
                    masks_sample = masks_val[i][0].cpu().detach().numpy()
                    recon_sample = reconstruction_val[i][0].cpu().detach().numpy()

                    if image_sample.shape == (250, 250, 50) and recon_sample.shape == (250, 250, 50):
                        recon_sample = np.repeat(recon_sample[..., np.newaxis], 1, axis=-1)
                        recon_sample = np.transpose(recon_sample, (2, 3, 1, 0))
                        image_sample = np.repeat(image_sample[..., np.newaxis], 1, axis=-1)
                        image_sample = np.transpose(image_sample, (2, 3, 1, 0))
                        masks_sample = np.repeat(masks_sample[..., np.newaxis], 1, axis=-1)
                        masks_sample = np.transpose(masks_sample, (2, 3, 1, 0))

                        # Save original image and reconstruction with epoch number in filenames
                        image_path = os.path.join(original_image_dir, f"image_epoch_{epoch+1}_sample{i}.tif")  # Change to .tif
                        recon_path = os.path.join(reconstruction_dir, f"reconstruction_epoch_{epoch+1}_sample{i}.tif")  # Change to .tif
                        masks_path = os.path.join(vae_masks_dir, f"masks_epoch_{epoch+1}_sample{i}.tif")  # Change to .tif

                        tiff.imwrite(image_path, image_sample)  # Save original image as .tif
                        tiff.imwrite(recon_path, recon_sample)  # Save 4-channel reconstruction as .tif
                        tiff.imwrite(masks_path, masks_sample)  # Save 4-channel masks as .tif
        
        self.val_recon_epoch_loss_list.append(val_loss)


        checkpoint = {
            "epoch": epoch,
            "autoencoder_state_dict": self.autoencoder.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "optimizer_g_state_dict": self.optimizer_g.state_dict(),
            "optimizer_d_state_dict": self.optimizer_d.state_dict(),
            "epoch_recon_loss_list": self.epoch_recon_loss_list,
            "epoch_gen_loss_list": self.epoch_gen_loss_list,
            "epoch_disc_loss_list": self.epoch_disc_loss_list,
            "val_recon_epoch_loss_list": self.val_recon_epoch_loss_list,
            "intermediary_images": self.intermediary_images,
        }
        checkpoint_path = os.path.join(autoencoder_checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
        torch.save(checkpoint, checkpoint_path)

    def train_epoch(self, epoch):
        """
        Train the model for one epoch.
        """
        self.autoencoder.train()
        self.discriminator.train()
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), ncols=110)
        progress_bar.set_description(f"Epoch {epoch}")

        epoch_loss, gen_epoch_loss, disc_epoch_loss = 0, 0, 0

        for step, batch in progress_bar:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            # TODO: Segmentation based
            # images = images * masks
            # # Normalize to [0, 1]
            # min_val = images.amin(dim=(2, 3, 4), keepdim=True)  # Min over spatial dimensions
            # max_val = images.amax(dim=(2, 3, 4), keepdim=True)  # Max over spatial dimensions
            # images = (images - min_val) / (max_val - min_val + 1e-8)  # Avoid division by zero

            # Generator update
            self.optimizer_g.zero_grad(set_to_none=True)
            reconstruction, quantization_loss = self.autoencoder(images=images)
            logits_fake = self.discriminator(reconstruction.contiguous())[-1]
            recons_loss = self.l1_loss(reconstruction, images)
            p_loss = self.perceptual_loss(reconstruction, images)
            generator_loss = self.adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            loss_g = recons_loss + quantization_loss + 0.001 * p_loss + 0.01 * generator_loss
            loss_g.backward()
            self.optimizer_g.step()

            # Discriminator update
            self.optimizer_d.zero_grad(set_to_none=True)
            logits_fake = self.discriminator(reconstruction.detach())[-1]
            logits_real = self.discriminator(images.detach())[-1]
            loss_d_fake = self.adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
            loss_d_real = self.adv_loss(logits_real, target_is_real=True, for_discriminator=True)
            loss_d = 0.5 * (loss_d_fake + loss_d_real)
            loss_d.backward()
            self.optimizer_d.step()

            epoch_loss += recons_loss.item()
            gen_epoch_loss += generator_loss.item()
            disc_epoch_loss += loss_d.item()

        self.epoch_recon_loss_list.append(epoch_loss / len(self.train_loader))
        self.epoch_gen_loss_list.append(gen_epoch_loss / len(self.train_loader))
        self.epoch_disc_loss_list.append(disc_epoch_loss / len(self.train_loader))

    def train(self):
        """
        Train the VQGAN for all epochs.
        """
        start_epoch = self.load_checkpoint()
        for epoch in range(start_epoch, vae_epochs):
            self.train_epoch(epoch)
            if (epoch + 1) % val_interval == 0:
                self.valiate_and_save(epoch)

        # Save final learning curves
        save_learning_curves(self.epoch_recon_loss_list, self.epoch_gen_loss_list, self.epoch_disc_loss_list)
