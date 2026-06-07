"""
DCGAN Training Script
Train Deep Convolutional GAN on MNIST dataset
"""

import os
import argparse
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import numpy as np
from tqdm import tqdm
import pandas as pd

from model import Generator, Discriminator, weights_init


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Create directories
        self.setup_directories()

        # Initialize models
        self.setup_models()

        # Setup data
        self.setup_data()

        # Training metrics
        self.metrics = {
            'epoch': [],
            'd_loss': [],
            'g_loss': [],
            'd_real_acc': [],
            'd_fake_acc': [],
            'time': []
        }

    def setup_directories(self):
        """Create necessary directories for saving results"""
        self.output_dir = self.args.output_dir
        self.samples_dir = os.path.join(self.output_dir, 'samples')
        self.models_dir = os.path.join(self.output_dir, 'models')
        self.metrics_dir = os.path.join(self.output_dir, 'metrics')

        os.makedirs(self.samples_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)

    def setup_models(self):
        """Initialize Generator and Discriminator"""
        self.G = Generator(
            latent_dim=self.args.latent_dim,
            channels=self.args.channels,
            feature_maps=self.args.feature_maps
        ).to(self.device)

        self.D = Discriminator(
            channels=self.args.channels,
            feature_maps=self.args.feature_maps
        ).to(self.device)

        # Apply weight initialization
        self.G.apply(weights_init)
        self.D.apply(weights_init)

        # Loss function
        self.criterion = nn.BCELoss()

        # Optimizers
        self.optimizer_G = optim.Adam(
            self.G.parameters(),
            lr=self.args.lr,
            betas=(self.args.beta1, 0.999)
        )
        self.optimizer_D = optim.Adam(
            self.D.parameters(),
            lr=self.args.lr,
            betas=(self.args.beta1, 0.999)
        )

        # Print model info
        g_params = sum(p.numel() for p in self.G.parameters())
        d_params = sum(p.numel() for p in self.D.parameters())
        print(f"Generator parameters: {g_params:,}")
        print(f"Discriminator parameters: {d_params:,}")

    def setup_data(self):
        """Setup data loader"""
        transform = transforms.Compose([
            transforms.Resize(self.args.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        dataset = datasets.MNIST(
            root=self.args.data_root,
            train=True,
            download=True,
            transform=transform
        )

        self.dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True
        )

        print(f"Dataset size: {len(dataset)}")
        print(f"Number of batches: {len(self.dataloader)}")

    def train_discriminator(self, real_images):
        """Train Discriminator for one step"""
        self.D.zero_grad()
        batch_size = real_images.size(0)

        # Real labels = 1, Fake labels = 0
        real_labels = torch.ones(batch_size).to(self.device)
        fake_labels = torch.zeros(batch_size).to(self.device)

        # Train on real images
        real_output = self.D(real_images)
        d_loss_real = self.criterion(real_output, real_labels)
        d_real_acc = (real_output > 0.5).float().mean().item()

        # Train on fake images
        z = torch.randn(batch_size, self.args.latent_dim).to(self.device)
        fake_images = self.G(z)
        fake_output = self.D(fake_images.detach())
        d_loss_fake = self.criterion(fake_output, fake_labels)
        d_fake_acc = (fake_output < 0.5).float().mean().item()

        # Total loss and backward
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.optimizer_D.step()

        return d_loss.item(), d_real_acc, d_fake_acc

    def train_generator(self, batch_size):
        """Train Generator for one step"""
        self.G.zero_grad()

        # We want fake images to be classified as real (label = 1)
        real_labels = torch.ones(batch_size).to(self.device)

        # Generate fake images
        z = torch.randn(batch_size, self.args.latent_dim).to(self.device)
        fake_images = self.G(z)
        fake_output = self.D(fake_images)

        # Generator loss
        g_loss = self.criterion(fake_output, real_labels)
        g_loss.backward()
        self.optimizer_G.step()

        return g_loss.item()

    def save_samples(self, epoch, fixed_z=None):
        """Save generated samples"""
        self.G.eval()

        if fixed_z is None:
            fixed_z = torch.randn(64, self.args.latent_dim).to(self.device)

        with torch.no_grad():
            fake_images = self.G(fixed_z).cpu()

        # Denormalize
        fake_images = fake_images * 0.5 + 0.5

        # Save grid
        grid = make_grid(fake_images, nrow=8, padding=2, normalize=False)
        save_image(
            grid,
            os.path.join(self.samples_dir, f'epoch_{epoch:03d}.png')
        )

        self.G.train()
        return fixed_z

    def save_metrics(self):
        """Save training metrics to CSV"""
        df = pd.DataFrame(self.metrics)
        df.to_csv(
            os.path.join(self.metrics_dir, 'training_metrics.csv'),
            index=False
        )

    def save_checkpoint(self, epoch):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'G_state_dict': self.G.state_dict(),
            'D_state_dict': self.D.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict(),
            'metrics': self.metrics
        }, os.path.join(self.models_dir, f'checkpoint_epoch_{epoch:03d}.pt'))

    def train(self):
        """Main training loop"""
        print("\n" + "="*50)
        print("Starting Training")
        print("="*50 + "\n")

        # Fixed noise for consistent sample generation
        fixed_z = torch.randn(64, self.args.latent_dim).to(self.device)

        # Save initial samples
        self.save_samples(0, fixed_z)

        start_time = time.time()

        for epoch in range(1, self.args.epochs + 1):
            epoch_start = time.time()

            d_losses = []
            g_losses = []
            d_real_accs = []
            d_fake_accs = []

            # Training loop with progress bar
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}/{self.args.epochs}")
            for i, (real_images, _) in enumerate(pbar):
                real_images = real_images.to(self.device)
                batch_size = real_images.size(0)

                # Train Discriminator
                d_loss, d_real_acc, d_fake_acc = self.train_discriminator(real_images)
                d_losses.append(d_loss)
                d_real_accs.append(d_real_acc)
                d_fake_accs.append(d_fake_acc)

                # Train Generator
                g_loss = self.train_generator(batch_size)
                g_losses.append(g_loss)

                # Update progress bar
                pbar.set_postfix({
                    'D_loss': f'{d_loss:.4f}',
                    'G_loss': f'{g_loss:.4f}',
                    'D_real_acc': f'{d_real_acc:.3f}',
                    'D_fake_acc': f'{d_fake_acc:.3f}'
                })

            # Epoch statistics
            epoch_time = time.time() - epoch_start
            avg_d_loss = np.mean(d_losses)
            avg_g_loss = np.mean(g_losses)
            avg_d_real_acc = np.mean(d_real_accs)
            avg_d_fake_acc = np.mean(d_fake_accs)

            # Save metrics
            self.metrics['epoch'].append(epoch)
            self.metrics['d_loss'].append(avg_d_loss)
            self.metrics['g_loss'].append(avg_g_loss)
            self.metrics['d_real_acc'].append(avg_d_real_acc)
            self.metrics['d_fake_acc'].append(avg_d_fake_acc)
            self.metrics['time'].append(epoch_time)

            # Print epoch summary
            print(f"\nEpoch {epoch}/{self.args.epochs} Summary:")
            print(f"  D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f}")
            print(f"  D Real Acc: {avg_d_real_acc:.3f} | D Fake Acc: {avg_d_fake_acc:.3f}")
            print(f"  Time: {epoch_time:.2f}s")

            # Save samples every N epochs
            if epoch % self.args.sample_interval == 0:
                self.save_samples(epoch, fixed_z)

            # Save checkpoint every N epochs
            if epoch % self.args.save_interval == 0:
                self.save_checkpoint(epoch)

            # Save metrics
            self.save_metrics()

        # Save final models
        torch.save(self.G.state_dict(),
                   os.path.join(self.models_dir, 'generator_final.pt'))
        torch.save(self.D.state_dict(),
                   os.path.join(self.models_dir, 'discriminator_final.pt'))

        total_time = time.time() - start_time
        print("\n" + "="*50)
        print("Training Complete!")
        print(f"Total Time: {total_time:.2f}s ({total_time/60:.2f} min)")
        print("="*50 + "\n")


def main():
    parser = argparse.ArgumentParser(description='DCGAN Training on MNIST')

    # Model parameters
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='Size of latent vector (default: 100)')
    parser.add_argument('--channels', type=int, default=1,
                        help='Number of image channels (default: 1 for MNIST)')
    parser.add_argument('--feature_maps', type=int, default=64,
                        help='Base number of feature maps (default: 64)')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Beta1 for Adam optimizer (default: 0.5)')

    # Data parameters
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory for dataset')
    parser.add_argument('--image_size', type=int, default=28,
                        help='Image size (default: 28 for MNIST)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory for results')
    parser.add_argument('--sample_interval', type=int, default=5,
                        help='Save samples every N epochs')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save checkpoint every N epochs')

    args = parser.parse_args()

    # Print configuration
    print("\nConfiguration:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    print()

    # Train
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
