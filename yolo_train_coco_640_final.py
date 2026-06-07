import os
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import json
import time
from pathlib import Path
import pandas as pd

print("="*70)
print("YOLOv8s Object Detection - COCO 640 Dataset")
print("="*70)

# 检查GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU Memory: {gpu_memory:.1f} GB")

# 创建输出目录
os.makedirs('output_yolov8s_coco_640', exist_ok=True)
os.makedirs('output_yolov8s_coco_640/charts', exist_ok=True)
os.makedirs('output_yolov8s_coco_640/samples', exist_ok=True)

# 加载YOLOv8s模型
print("\nLoading YOLOv8s model...")
model = YOLO('yolov8s.pt')

# 配置信息
print("\n" + "="*70)
print("Training Configuration:")
print("="*70)
print("Model: YOLOv8s (Small - 11.1M params)")
print("Dataset: COCO 640 (640 images)")
print("Epochs: 150")
print("Image size: 640")
print("Batch size: 8")
print("Device: CUDA GPU")
print("Expected training time: ~3 hours")
print("="*70)

start_time = time.time()

# 训练模型
results = model.train(
    data='datasets/coco_640.yaml',
    epochs=150,
    imgsz=640,
    batch=8,
    device=device,
    project='output_yolov8s_coco_640',
    name='yolo_train',
    exist_ok=True,
    plots=True,
    save=True,
    verbose=True,
    workers=2,
    amp=True,
    cache='disk',
    patience=30,
    cos_lr=True,
    warmup_epochs=5,
)

training_time = time.time() - start_time
training_hours = training_time / 3600
training_minutes = (training_hours % 1) * 60

print(f"\n{'='*70}")
print(f"Training completed in {int(training_hours)} hours {int(training_minutes)} minutes")
print(f"{'='*70}")

# 生成详细报告
print("\nGenerating comprehensive report...")

# 读取训练结果
metrics_path = Path('output_yolov8s_coco_640/yolo_train/results.csv')
if metrics_path.exists():
    df = pd.read_csv(metrics_path)
    
    # 创建训练曲线
    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    
    # mAP曲线
    if 'metrics/mAP50(B)' in df.columns:
        axes[0, 0].plot(df['epoch'], df['metrics/mAP50(B)'], 'b-', linewidth=2, label='mAP@0.5')
        axes[0, 0].plot(df['epoch'], df['metrics/mAP50-95(B)'], 'r-', linewidth=2, label='mAP@0.5:0.95')
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('mAP', fontsize=12)
        axes[0, 0].set_title('Mean Average Precision', fontsize=14)
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 1)
    
    # 训练损失曲线
    if 'train/box_loss' in df.columns:
        axes[0, 1].plot(df['epoch'], df['train/box_loss'], 'b-', linewidth=2, label='Box Loss')
        axes[0, 1].plot(df['epoch'], df['train/cls_loss'], 'r-', linewidth=2, label='Cls Loss')
        axes[0, 1].plot(df['epoch'], df['train/dfl_loss'], 'g-', linewidth=2, label='DFL Loss')
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Loss', fontsize=12)
        axes[0, 1].set_title('Training Loss', fontsize=14)
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
    
    # 验证损失
    if 'val/box_loss' in df.columns:
        axes[1, 0].plot(df['epoch'], df['val/box_loss'], 'b-', linewidth=2, label='Box Loss')
        axes[1, 0].plot(df['epoch'], df['val/cls_loss'], 'r-', linewidth=2, label='Cls Loss')
        axes[1, 0].plot(df['epoch'], df['val/dfl_loss'], 'g-', linewidth=2, label='DFL Loss')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Loss', fontsize=12)
        axes[1, 0].set_title('Validation Loss', fontsize=14)
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
    
    # 精确率和召回率
    if 'metrics/precision(B)' in df.columns:
        axes[1, 1].plot(df['epoch'], df['metrics/precision(B)'], 'b-', linewidth=2, label='Precision')
        axes[1, 1].plot(df['epoch'], df['metrics/recall(B)'], 'r-', linewidth=2, label='Recall')
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Score', fontsize=12)
        axes[1, 1].set_title('Precision & Recall', fontsize=14)
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(0, 1)
    
    plt.suptitle('YOLOv8s Training Metrics (COCO 640 Images)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output_yolov8s_coco_640/charts/training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generated: training_curves.png")
    
    # 保存CSV数据
    df.to_csv('output_yolov8s_coco_640/training_metrics.csv', index=False)
    print("✓ Generated: training_metrics.csv")

# 验证模型
print("\nRunning validation...")
val_results = model.val()

# 生成检测样本
print("\nGenerating detection samples...")
sample_images = [
    'https://ultralytics.com/images/bus.jpg',
    'https://ultralytics.com/images/zidane.jpg',
    'https://ultralytics.com/images/image2.jpg'
]

for i, img_path in enumerate(sample_images):
    try:
        results = model(img_path)
        for r in results:
            im_array = r.plot()
            im = Image.fromarray(im_array[..., ::-1])
            im.save(f'output_yolov8s_coco_640/samples/detection_{i+1}.png')
            print(f"✓ Generated: detection_{i+1}.png")
    except Exception as e:
        print(f"✗ Failed to process {img_path}: {e}")

# 计算最终指标
final_map50 = float(val_results.box.map50) if hasattr(val_results, 'box') else 0
final_map75 = float(val_results.box.map75) if hasattr(val_results, 'box') else 0
final_map = float(val_results.box.map) if hasattr(val_results, 'box') else 0
final_precision = float(val_results.box.mp) if hasattr(val_results, 'box') else 0
final_recall = float(val_results.box.mr) if hasattr(val_results, 'box') else 0

# 生成JSON报告
report = {
    "project": "YOLOv8s Object Detection - COCO 640",
    "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    "model": {
        "name": "YOLOv8s",
        "version": "small",
        "pretrained": True,
        "input_size": 640,
        "num_classes": 80,
        "parameters_M": 11.1,
        "flops_G": 28.5
    },
    "training": {
        "dataset": "COCO 640",
        "train_samples": 640,
        "val_samples": 640,
        "epochs": 150,
        "batch_size": 8,
        "device": device,
        "training_time_hours": training_hours,
        "training_time_minutes": training_minutes,
    },
    "hardware": {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "cuda_version": torch.version.cuda,
        "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    },
    "results": {
        "mAP@0.5": final_map50,
        "mAP@0.75": final_map75,
        "mAP@0.5:0.95": final_map,
        "precision": final_precision,
        "recall": final_recall,
        "F1_score": 2 * (final_precision * final_recall) / (final_precision + final_recall) if (final_precision + final_recall) > 0 else 0
    },
    "official_benchmark": {
        "dataset": "COCO 2017 Val (5000 images)",
        "mAP@0.5": 0.696,
        "mAP@0.5:0.95": 0.446,
        "epochs": 500
    }
}

with open('output_yolov8s_coco_640/detailed_report.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print("✓ Generated: detailed_report.json")

print("\n" + "="*70)
print("YOLOv8s COCO 640 Training Complete!")
print("="*70)
print(f"\nFinal Results:")
print(f"  mAP@0.5: {final_map50:.4f}")
print(f"  mAP@0.75: {final_map75:.4f}")
print(f"  mAP@0.5:0.95: {final_map:.4f}")
print(f"  Precision: {final_precision:.4f}")
print(f"  Recall: {final_recall:.4f}")
print(f"  F1 Score: {report['results']['F1_score']:.4f}")
print(f"\nTraining Time: {int(training_hours)} hours {training_minutes:.1f} minutes")
print(f"Output Directory: output_yolov8s_coco_640/")
print("="*70)
