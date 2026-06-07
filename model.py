"""
DCGAN Model Implementation
Deep Convolutional Generative Adversarial Network
Reference: Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (Radford et al., 2015)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """
    DCGAN Generator
    Transforms a latent vector into an image
    Architecture: FC -> ConvTranspose2d -> BatchNorm -> ReLU -> Tanh
    """
    def __init__(self, latent_dim=100, channels=1, feature_maps=64):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim

        # For MNIST: 28x28 images
        # Start from 1x1, upscale to 28x28
        self.main = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, feature_maps * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),

            # State: (feature_maps*4) x 4 x 4
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),

            # State: (feature_maps*2) x 8 x 8
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),

            # State: feature_maps x 16 x 16
            nn.ConvTranspose2d(feature_maps, channels, 4, 2, 3, bias=False),
            nn.Tanh()
            # Output: channels x 28 x 28
        )

    def forward(self, z):
        return self.main(z.view(-1, self.latent_dim, 1, 1))


class Discriminator(nn.Module):
    """
    DCGAN Discriminator
    Classifies images as real or fake
    Architecture: Conv2d -> BatchNorm -> LeakyReLU -> FC -> Sigmoid
    """
    def __init__(self, channels=1, feature_maps=64):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            # Input: 1 x 28 x 28
            nn.Conv2d(channels, feature_maps, 4, 2, 1, bias=False),  # 64 x 14 x 14
            nn.LeakyReLU(0.2, inplace=True),

            # State: feature_maps x 14 x 14
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),  # 128 x 7 x 7
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # State: (feature_maps*2) x 7 x 7
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),  # 256 x 3 x 3
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # State: (feature_maps*4) x 3 x 3
            nn.Conv2d(feature_maps * 4, 1, 3, 1, 0, bias=False),  # 1 x 1 x 1
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.main(x)
        return output.view(output.size(0), -1).squeeze(1)


def weights_init(m):
    """
    Custom weights initialization called on Generator and Discriminator
    As described in the DCGAN paper:
    - Weights initialized from normal distribution with mean=0, std=0.02
    - BatchNorm weights initialized from normal distribution with mean=1, std=0.02
    - BatchNorm bias initialized to 0
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


if __name__ == "__main__":
    # Test model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    G = Generator().to(device)
    D = Discriminator().to(device)

    # Apply weight initialization
    G.apply(weights_init)
    D.apply(weights_init)

    # Test forward pass
    z = torch.randn(64, 100).to(device)
    fake_images = G(z)
    print(f"Generator output shape: {fake_images.shape}")

    output = D(fake_images)
    print(f"Discriminator output shape: {output.shape}")

    # Count parameters
    g_params = sum(p.numel() for p in G.parameters())
    d_params = sum(p.numel() for p in D.parameters())
    print(f"Generator parameters: {g_params:,}")
    print(f"Discriminator parameters: {d_params:,}")
