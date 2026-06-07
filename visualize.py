"""
Visualization Script for DCGAN Results
Generate comprehensive visualizations including:
- Loss curves
- Accuracy curves
- Generated samples comparison
- Training metrics table
- Sample quality analysis
"""

import os
import argparse
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from PIL import Image
import torch
from torchvision.utils import make_grid

from model import Generator


def set_style():
    """Set matplotlib style for better visualizations"""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['legend.fontsize'] = 12


def plot_loss_curves(metrics_df, save_dir):
    """Plot Generator and Discriminator loss curves"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax = axes[0]
    ax.plot(metrics_df['epoch'], metrics_df['d_loss'],
            label='Discriminator Loss', linewidth=2, marker='o', markersize=4)
    ax.plot(metrics_df['epoch'], metrics_df['g_loss'],
            label='Generator Loss', linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Loss difference
    ax = axes[1]
    loss_diff = metrics_df['d_loss'] - metrics_df['g_loss']
    ax.plot(metrics_df['epoch'], loss_diff,
            label='D Loss - G Loss', linewidth=2, color='purple', marker='^', markersize=4)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Equilibrium')
    ax.fill_between(metrics_df['epoch'], loss_diff, 0,
                    where=(loss_diff > 0), alpha=0.3, color='blue', label='D stronger')
    ax.fill_between(metrics_df['epoch'], loss_diff, 0,
                    where=(loss_diff < 0), alpha=0.3, color='orange', label='G stronger')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Difference')
    ax.set_title('Loss Balance Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: loss_curves.png")


def plot_accuracy_curves(metrics_df, save_dir):
    """Plot Discriminator accuracy curves"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy curves
    ax = axes[0]
    ax.plot(metrics_df['epoch'], metrics_df['d_real_acc'],
            label='Real Image Accuracy', linewidth=2, marker='o', markersize=4)
    ax.plot(metrics_df['epoch'], metrics_df['d_fake_acc'],
            label='Fake Image Accuracy', linewidth=2, marker='s', markersize=4)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random Guess')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Discriminator Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    # Combined accuracy
    ax = axes[1]
    combined_acc = (metrics_df['d_real_acc'] + (1 - metrics_df['d_fake_acc'])) / 2
    ax.plot(metrics_df['epoch'], combined_acc,
            label='Combined Accuracy', linewidth=2, color='green', marker='^', markersize=4)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Equilibrium')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Discriminator Combined Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'accuracy_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: accuracy_curves.png")


def plot_training_time(metrics_df, save_dir):
    """Plot training time per epoch"""
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(metrics_df['epoch'], metrics_df['time'],
           color='steelblue', alpha=0.7, edgecolor='navy')
    ax.axhline(y=metrics_df['time'].mean(), color='red',
               linestyle='--', linewidth=2, label=f'Mean: {metrics_df["time"].mean():.2f}s')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Training Time per Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_time.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: training_time.png")


def plot_samples_comparison(samples_dir, save_dir, epochs=None):
    """Plot generated samples from different epochs side by side"""
    sample_files = sorted(glob.glob(os.path.join(samples_dir, 'epoch_*.png')))

    if not sample_files:
        print("No sample files found!")
        return

    # Select specific epochs if provided
    if epochs:
        selected_files = []
        for epoch in epochs:
            file_path = os.path.join(samples_dir, f'epoch_{epoch:03d}.png')
            if os.path.exists(file_path):
                selected_files.append(file_path)
        sample_files = selected_files if selected_files else sample_files[::max(1, len(sample_files)//5)]

    # Limit to 6 samples for visualization
    if len(sample_files) > 6:
        indices = np.linspace(0, len(sample_files)-1, 6, dtype=int)
        sample_files = [sample_files[i] for i in indices]

    n_samples = len(sample_files)
    fig, axes = plt.subplots(1, n_samples, figsize=(4*n_samples, 4))

    if n_samples == 1:
        axes = [axes]

    for i, file_path in enumerate(sample_files):
        img = Image.open(file_path)
        axes[i].imshow(img, cmap='gray')
        epoch = os.path.basename(file_path).split('_')[1].split('.')[0]
        axes[i].set_title(f'Epoch {int(epoch)}')
        axes[i].axis('off')

    plt.suptitle('Generated Samples Progress', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'samples_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: samples_comparison.png")


def create_metrics_table(metrics_df, save_dir):
    """Create a formatted table of training metrics"""
    # Summary statistics
    summary = {
        'Metric': [
            'Total Epochs',
            'Final D Loss',
            'Final G Loss',
            'Final D Real Acc',
            'Final D Fake Acc',
            'Mean D Loss',
            'Mean G Loss',
            'Mean Training Time',
            'Total Training Time'
        ],
        'Value': [
            len(metrics_df),
            f"{metrics_df['d_loss'].iloc[-1]:.4f}",
            f"{metrics_df['g_loss'].iloc[-1]:.4f}",
            f"{metrics_df['d_real_acc'].iloc[-1]:.4f}",
            f"{metrics_df['d_fake_acc'].iloc[-1]:.4f}",
            f"{metrics_df['d_loss'].mean():.4f}",
            f"{metrics_df['g_loss'].mean():.4f}",
            f"{metrics_df['time'].mean():.2f} s",
            f"{metrics_df['time'].sum():.2f} s ({metrics_df['time'].sum()/60:.2f} min)"
        ]
    }

    summary_df = pd.DataFrame(summary)

    # Save as CSV
    summary_df.to_csv(os.path.join(save_dir, 'summary_metrics.csv'), index=False)
    print("Saved: summary_metrics.csv")

    # Create table visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    table = ax.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.4, 0.4]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)

    # Style header
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style cells
    for i in range(1, len(summary_df) + 1):
        for j in range(len(summary_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    plt.title('Training Summary', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_table.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: metrics_table.png")

    return summary_df


def plot_comprehensive_dashboard(metrics_df, save_dir):
    """Create a comprehensive dashboard with all metrics"""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Loss curves
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(metrics_df['epoch'], metrics_df['d_loss'],
             label='D Loss', linewidth=2, marker='o', markersize=3)
    ax1.plot(metrics_df['epoch'], metrics_df['g_loss'],
             label='G Loss', linewidth=2, marker='s', markersize=3)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Accuracy curves
    ax2 = fig.add_subplot(gs[1, :2])
    ax2.plot(metrics_df['epoch'], metrics_df['d_real_acc'],
             label='D Real Acc', linewidth=2, marker='o', markersize=3)
    ax2.plot(metrics_df['epoch'], metrics_df['d_fake_acc'],
             label='D Fake Acc', linewidth=2, marker='s', markersize=3)
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Discriminator Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    # 3. Training time
    ax3 = fig.add_subplot(gs[2, :2])
    ax3.bar(metrics_df['epoch'], metrics_df['time'],
            color='steelblue', alpha=0.7)
    ax3.axhline(y=metrics_df['time'].mean(), color='red',
                linestyle='--', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Time (s)')
    ax3.set_title('Training Time per Epoch')
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Loss distribution
    ax4 = fig.add_subplot(gs[0, 2])
    ax4.hist(metrics_df['d_loss'], bins=20, alpha=0.7, label='D Loss')
    ax4.hist(metrics_df['g_loss'], bins=20, alpha=0.7, label='G Loss')
    ax4.set_xlabel('Loss')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Loss Distribution')
    ax4.legend()

    # 5. Accuracy distribution
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.hist(metrics_df['d_real_acc'], bins=20, alpha=0.7, label='Real Acc')
    ax5.hist(metrics_df['d_fake_acc'], bins=20, alpha=0.7, label='Fake Acc')
    ax5.set_xlabel('Accuracy')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Accuracy Distribution')
    ax5.legend()

    # 6. Statistics text
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    stats_text = f"""
    Training Statistics
    ─────────────────────
    Total Epochs: {len(metrics_df)}
    Total Time: {metrics_df['time'].sum():.1f}s ({metrics_df['time'].sum()/60:.1f} min)

    Final D Loss: {metrics_df['d_loss'].iloc[-1]:.4f}
    Final G Loss: {metrics_df['g_loss'].iloc[-1]:.4f}

    Final D Real Acc: {metrics_df['d_real_acc'].iloc[-1]:.4f}
    Final D Fake Acc: {metrics_df['d_fake_acc'].iloc[-1]:.4f}

    Mean D Loss: {metrics_df['d_loss'].mean():.4f}
    Mean G Loss: {metrics_df['g_loss'].mean():.4f}
    """
    ax6.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center')

    plt.suptitle('DCGAN Training Dashboard', fontsize=18, y=0.98)
    plt.savefig(os.path.join(save_dir, 'dashboard.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: dashboard.png")


def generate_final_samples(model_path, save_dir, latent_dim=100, num_samples=100):
    """Generate final samples using trained generator"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load generator
    G = Generator(latent_dim=latent_dim).to(device)
    G.load_state_dict(torch.load(model_path, map_location=device))
    G.eval()

    # Generate samples
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        samples = G(z).cpu()

    # Denormalize
    samples = samples * 0.5 + 0.5

    # Save grid
    fig, axes = plt.subplots(10, 10, figsize=(15, 15))
    for i in range(10):
        for j in range(10):
            idx = i * 10 + j
            axes[i, j].imshow(samples[idx].squeeze(), cmap='gray')
            axes[i, j].axis('off')

    plt.suptitle('Generated Samples', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'final_samples.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: final_samples.png")


def main():
    parser = argparse.ArgumentParser(description='Visualize DCGAN Training Results')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory containing training results')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained generator model')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='Latent dimension')

    args = parser.parse_args()

    # Set style
    set_style()

    # Create visualization directory
    vis_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    # Load metrics
    metrics_path = os.path.join(args.output_dir, 'metrics', 'training_metrics.csv')
    if not os.path.exists(metrics_path):
        print(f"Metrics file not found: {metrics_path}")
        return

    metrics_df = pd.read_csv(metrics_path)
    print(f"Loaded metrics for {len(metrics_df)} epochs\n")

    # Generate all visualizations
    print("Generating visualizations...")
    print("-" * 50)

    plot_loss_curves(metrics_df, vis_dir)
    plot_accuracy_curves(metrics_df, vis_dir)
    plot_training_time(metrics_df, vis_dir)
    create_metrics_table(metrics_df, vis_dir)
    plot_comprehensive_dashboard(metrics_df, vis_dir)

    # Plot samples comparison
    samples_dir = os.path.join(args.output_dir, 'samples')
    if os.path.exists(samples_dir):
        plot_samples_comparison(samples_dir, vis_dir)

    # Generate final samples if model provided
    if args.model_path and os.path.exists(args.model_path):
        generate_final_samples(
            args.model_path,
            vis_dir,
            latent_dim=args.latent_dim
        )

    print("-" * 50)
    print(f"\nAll visualizations saved to: {vis_dir}")


if __name__ == '__main__':
    main()
