import torch
from data.data_loader import prepare_datasets
from model.vqgan import VQGANTrainer
from model.ldm import LDMTrainer
from model.ldm_spade import LDMSpadeTrainer
from model.ddpm import DDPMTrainer

def main():
    """
    Main function to run the data preparation and visualization.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare DataLoaders
    train_loader, val_loader = prepare_datasets()

    print("Starting VAGAN training...")
    vqgan_trainer = VQGANTrainer(device=device, train_loader=train_loader, val_loader=val_loader)
    vqgan_trainer.train()

    print("Starting LDM training...")
    ddpm_trainer = LDMTrainer(device, train_loader, val_loader, vqgan_trainer.autoencoder)
    # ddpm_trainer.train()
    ddpm_trainer.infer()

    # print("Starting LDM Spade training...")
    # ddpm_trainer = LDMSpadeTrainer(device, train_loader, val_loader, vqgan_trainer.autoencoder)
    # ddpm_trainer.validate()
    # ddpm_trainer.train()

    # print("Starting DDPM training...")
    # ddpm_trainer = DDPMTrainer(device, train_loader, val_loader)
    # ddpm_trainer.train()


if __name__ == "__main__":
    main()
