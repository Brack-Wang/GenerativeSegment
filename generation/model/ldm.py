import os
import glob
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler
from generative.inferers import LatentDiffusionInferer
from generative.metrics import SSIMMetric, MultiScaleSSIMMetric, MMDMetric
import tifffile as tiff
from data.constants import (
    ddpm_checkpoint_dir,
    ddpm_original_image_dir,
    ddpm_synth_image_dir,
    ddpm_mask_image_dir,
    ddpm_val_interval,
    ddpm_epochs,
    generated_dir,
    raw_dir,
    maskout_dir
)


class LDMTrainer:
    def __init__(self, device, train_loader, val_loader, autoencoder):
        self.timesteps = 1000
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.autoencoder = autoencoder.eval()

        # Initialize DDPM components
        self.unet = self.initialize_unet()
        self.scheduler = self.initialize_scheduler()
        self.inferer = LatentDiffusionInferer(scheduler=self.scheduler, scale_factor=1)

        # Optimizer and scaler
        self.optimizer = torch.optim.Adam(params=self.unet.parameters(), lr=1e-4)
        self.scaler = GradScaler()

        # Metrics
        self.ssim = SSIMMetric(spatial_dims=3, data_range=1.0, kernel_size=2)
        self.ms_ssim = MultiScaleSSIMMetric(spatial_dims=3, data_range=1.0, kernel_size=2)
        self.mmd = MMDMetric()

        # Loss history
        self.epoch_loss_list = []

        # Load checkpoint
        self.start_epoch = self.load_checkpoint()

    def initialize_unet(self):
        """
        Initialize the UNet model for DDPM.
        """
        return DiffusionModelUNet(
            spatial_dims=3,
            in_channels=64,  # Match VQGAN embedding_dim
            out_channels=64,  # Match VQGAN embedding_dim
            num_res_blocks=2,
            num_channels=(64, 128),
            attention_levels=(False, True),
            num_head_channels=(64, 64),
        ).to(self.device)

    def initialize_scheduler(self):
        """
        Initialize the DDPM scheduler.
        """
        return DDPMScheduler(num_train_timesteps=self.timesteps, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0195)

    def load_checkpoint(self):
        """
        Load the latest checkpoint if available.
        """
        checkpoint_files = glob.glob(os.path.join(ddpm_checkpoint_dir, "checkpoint_epoch_*.pth"))
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            checkpoint = torch.load(latest_checkpoint)
            self.unet.load_state_dict(checkpoint["unet_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_diff_state_dict"])
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            self.epoch_loss_list = checkpoint["epoch_loss_list"]
            print(f"Resuming training from epoch {checkpoint['epoch']} using checkpoint {latest_checkpoint}")
            return checkpoint["epoch"]
        else:
            print("No checkpoints found, starting from scratch.")
            return 0

    def train_epoch(self, epoch):
        """
        Train the DDPM model for one epoch.
        """
        self.unet.train()
        epoch_loss = 0
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), ncols=80)
        progress_bar.set_description(f"Epoch {epoch}")
        

        for step, batch in progress_bar:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            images = images * masks

            # Generate conditioning mask
            condition_mask = self.autoencoder.encode_stage_2_inputs(masks)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=True):
                noise = torch.randn_like(condition_mask).to(self.device)
                timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (images.size(0),), device=self.device).long()
                noise_pred = self.inferer(
                    inputs=images,
                    autoencoder_model=self.autoencoder,
                    diffusion_model=self.unet,
                    noise=noise,
                    timesteps=timesteps,
                    condition=condition_mask,
                    mode="concat",
                )
                loss = F.mse_loss(noise_pred, noise)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})

        self.epoch_loss_list.append(epoch_loss / len(self.train_loader))

        # Function to calculate PSNR
    def calculate_3d_psnr(self, original, generated):
        mse = F.mse_loss(generated, original, reduction='mean')
        max_pixel_value = original.max()
        psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
        return psnr.item()

    def calculate_metrics(self, synthetic_images, images):
        """
        Calculate SSIM, MS-SSIM, PSNR, and MMD for synthetic and original images.
        """
        batch_size = synthetic_images.shape[0]

        ssim_list = []
        ms_ssim_list = []
        mmd_list = []
        psnr_list = []

        for i in range(batch_size):
            original_image = images[i].unsqueeze(0)  # Keep batch dimension for metrics
            generated_image = synthetic_images[i].unsqueeze(0)


            # Calculate metrics
            ssim_now = self.ssim(generated_image.float().cpu(), original_image.float().cpu())
            ms_ssim_now = self.ms_ssim(generated_image.float().cpu(), original_image.float().cpu())
            mmd_now = self.mmd(generated_image.float().cpu(), original_image.float().cpu())
            psnr_now = self.calculate_3d_psnr(original_image, generated_image)

            # Append metrics
            ssim_list.append(ssim_now.mean().item())
            ms_ssim_list.append(ms_ssim_now.mean().item())
            mmd_list.append(mmd_now.mean().item())
            psnr_list.append(psnr_now)

        # Calculate mean values
        mean_ssim = torch.tensor(ssim_list).mean().item()
        mean_ms_ssim = torch.tensor(ms_ssim_list).mean().item()
        mean_mmd = torch.tensor(mmd_list).mean().item()
        mean_psnr = torch.tensor(psnr_list).mean().item()

        # Log metrics
        print(f"Mean SSIM: {mean_ssim}")
        print(f"Mean MS-SSIM: {mean_ms_ssim}")
        print(f"Mean MMD: {mean_mmd}")
        print(f"Mean PSNR: {mean_psnr}")

        return mean_ssim, mean_ms_ssim, mean_mmd, mean_psnr

    def validate(self, epoch):
        """
        Perform validation and save reconstructed and original images.
        """
        self.unet.eval()
        with torch.no_grad():
            for val_step, batch in enumerate(self.val_loader):
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                images = images * masks
                condition_mask = self.autoencoder.encode_stage_2_inputs(masks)

                noise = torch.randn_like(condition_mask).to(self.device)
                self.scheduler.set_timesteps(num_inference_steps=self.timesteps)
                synthetic_images = self.inferer.sample(
                    input_noise=noise,
                    autoencoder_model=self.autoencoder,
                    diffusion_model=self.unet,
                    scheduler=self.scheduler,
                    conditioning=condition_mask,
                    mode="concat",
                )

                self.calculate_metrics(synthetic_images, images)
                # Save the first sample
                if val_step == 0:
                    self.save_images(images, synthetic_images, masks, epoch)
                    break

        


    def save_images(self, images, synthetic_images, masks, epoch):
        """
        Save original, synthesized, and mask images in shape (Z, C, Y, X).
        """
        for idx in range(min(images.size(0), 5)):  # Save up to 5 samples
            # Permute dimensions to (Z, C, Y, X)
            original_sample = images[idx].permute(3, 0, 2, 1).cpu().numpy()  # (Z, C, Y, X)
            synthetic_sample = synthetic_images[idx].permute(3, 0, 2, 1).cpu().numpy()  # (Z, C, Y, X)
            mask_sample = masks[idx].permute(3, 0, 2, 1).cpu().numpy()  # (Z, C, Y, X)

            # Save images
            tiff.imwrite( os.path.join(ddpm_original_image_dir, f"original_epoch_{epoch+1}_sample_{idx}.tif"), original_sample )
            tiff.imwrite( os.path.join(ddpm_synth_image_dir, f"synth_epoch_{epoch+1}_sample_{idx}.tif"), synthetic_sample )
            tiff.imwrite( os.path.join(ddpm_mask_image_dir, f"mask_epoch_{epoch+1}_sample_{idx}.tif"), mask_sample )

    def infer(self):
        """
        Perform inference and save reconstructed and original images.
        """
        self.unet.eval()
        idx = 0  # Initialize index counter
        with torch.no_grad():
            for val_step, batch in enumerate(self.val_loader):
            # for val_step, batch in enumerate(self.train_loader):
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                images = images * masks
                condition_mask = self.autoencoder.encode_stage_2_inputs(masks)

                noise = torch.randn_like(condition_mask).to(self.device)
                self.scheduler.set_timesteps(num_inference_steps=self.timesteps)
                synthetic_images = self.inferer.sample(
                    input_noise=noise,
                    autoencoder_model=self.autoencoder,
                    diffusion_model=self.unet,
                    scheduler=self.scheduler,
                    conditioning=condition_mask,
                    mode="concat",
                )

                # Save each sample with consistent indexing
                self.save_inferred_images(images, synthetic_images, masks, idx)
                idx += images.size(0)  # Increment index by batch size

            for val_step, batch in enumerate(self.train_loader):
            # for val_step, batch in enumerate(self.train_loader):
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                images = images * masks
                condition_mask = self.autoencoder.encode_stage_2_inputs(masks)

                noise = torch.randn_like(condition_mask).to(self.device)
                self.scheduler.set_timesteps(num_inference_steps=self.timesteps)
                synthetic_images = self.inferer.sample(
                    input_noise=noise,
                    autoencoder_model=self.autoencoder,
                    diffusion_model=self.unet,
                    scheduler=self.scheduler,
                    conditioning=condition_mask,
                    mode="concat",
                )

                # Save each sample with consistent indexing
                self.save_inferred_images(images, synthetic_images, masks, idx)
                idx += images.size(0)  # Increment index by batch size


    def save_inferred_images(self, images, synthetic_images, masks, start_idx):
        """
        Save original, synthesized, and mask images in shape (Z, C, Y, X).
        """
        for i in range(images.size(0)):  # Iterate over the batch
            idx = start_idx + i  # Compute the global index
            # Permute dimensions to (Z, C, Y, X) for saving as .tif
            original_sample = images[i].permute(3, 0, 2, 1).cpu().numpy()
            synthetic_sample = synthetic_images[i].permute(3, 0, 2, 1).cpu().numpy()
            mask_sample = masks[i].permute(3, 0, 2, 1).cpu().numpy()

            # Save images
            tiff.imwrite(os.path.join(raw_dir, f"raw_image_{idx}.tif"), original_sample)
            tiff.imwrite(os.path.join(generated_dir, f"synth_image_{idx}.tif"), synthetic_sample)
            tiff.imwrite(os.path.join(maskout_dir, f"mask_image_{idx}.tif"), mask_sample)

    def train(self):
        """
        Train the DDPM for all epochs.
        """
        for epoch in range(self.start_epoch, ddpm_epochs):
            self.train_epoch(epoch)
            if (epoch + 1) % ddpm_val_interval == 0:
                self.save_checkpoint(epoch)
                self.validate(epoch)

    def save_checkpoint(self, epoch):
        """
        Save the model checkpoint.
        """
        checkpoint_path = os.path.join(ddpm_checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
        torch.save({
            "epoch": epoch + 1,
            "unet_state_dict": self.unet.state_dict(),
            "optimizer_diff_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "epoch_loss_list": self.epoch_loss_list,
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
