"""
Main script to run DCGAN training and visualization
"""

import os
import sys
import subprocess


def run_command(cmd, description):
    """Run a command and print status"""
    print("\n" + "="*60)
    print(f" {description}")
    print("="*60)
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Error: {description} failed!")
        sys.exit(1)
    return result


def main():
    # Configuration
    EPOCHS = 50
    BATCH_SIZE = 128
    OUTPUT_DIR = "./output"

    print("\n" + "="*60)
    print(" DCGAN Training Pipeline")
    print(" MNIST Dataset")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Output Directory: {OUTPUT_DIR}")

    # Step 1: Train DCGAN
    train_cmd = f"python train.py --epochs {EPOCHS} --batch_size {BATCH_SIZE} --output_dir {OUTPUT_DIR}"
    run_command(train_cmd, "Step 1: Training DCGAN")

    # Step 2: Generate Visualizations
    model_path = os.path.join(OUTPUT_DIR, "models", "generator_final.pt")
    vis_cmd = f"python visualize.py --output_dir {OUTPUT_DIR} --model_path {model_path}"
    run_command(vis_cmd, "Step 2: Generating Visualizations")

    print("\n" + "="*60)
    print(" Training Complete!")
    print("="*60)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"  - Models: {OUTPUT_DIR}/models/")
    print(f"  - Samples: {OUTPUT_DIR}/samples/")
    print(f"  - Metrics: {OUTPUT_DIR}/metrics/")
    print(f"  - Visualizations: {OUTPUT_DIR}/visualizations/")
    print("\n")


if __name__ == '__main__':
    main()
