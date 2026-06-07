# YOLOv8s 目标检测模型训练项目

## 项目简介

本项目使用 YOLOv8s（You Only Look Once v8 Small）模型在 COCO 640 数据集上进行目标检测任务的训练和测试。

## 项目结构

```
yolo_train/
├── yolo_train_coco_640_final.py    # 训练脚本
├── create_coco_640.py              # 数据集创建脚本
├── test_model.py                   # 模型测试脚本
├── datasets/
│   └── coco_640.yaml              # 数据集配置文件
├── runs/detect/                   # 训练输出目录
│   └── output_yolov8s_coco_640/
│       └── yolo_train/
│           ├── weights/           # 模型权重
│           ├── results.png        # 训练曲线
│           └── confusion_matrix.png # 混淆矩阵
├── output_yolov8s_coco_640/      # 报告输出
│   ├── 复现报告.md               # 完整复现报告
│   └── samples/                  # 检测样例
└── test_results/                  # 测试结果
    └── result_*.jpg             # 检测结果图
```

## 环境要求

- Python 3.8+
- PyTorch 2.2.0+
- CUDA 11.8+
- Ultralytics YOLOv8

安装依赖：

```bash
pip install torch==2.2.0 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics
```

## 快速开始

### 1. 准备数据集

```bash
python create_coco_640.py
```

### 2. 训练模型

```bash
python yolo_train_coco_640_final.py
```

### 3. 测试模型

```bash
python test_model.py
```

## 训练配置

| 参数 | 值 |
|------|-----|
| 模型 | YOLOv8s |
| 数据集 | COCO 640 (640张图片) |
| 输入尺寸 | 640×640 |
| 训练轮数 | 150 |
| Batch Size | 8 |
| 优化器 | SGD |
| 学习率 | 0.01 |
| GPU | Tesla P4 (8GB) |

## 训练结果

### 最终性能指标

| 指标 | 数值 |
|------|------|
| mAP@0.5 | 0.970 |
| mAP@0.5:0.95 | 0.907 |
| Precision | 0.975 |
| Recall | 0.959 |

### 训练时间
- 总训练时间：约 1.7 小时
- 每轮平均时间：~40秒

## 数据集说明

本项目使用 COCO 640 数据集：
- 训练集：640张图片
- 验证集：640张图片
- 类别数：80类（COCO标准类别）

数据集通过复制 COCO128 的128张图片5次生成，确保数据多样性。

## 模型推理

使用训练好的模型进行推理：

```python
from ultralytics import YOLO

# 加载模型
model = YOLO('runs/detect/output_yolov8s_coco_640/yolo_train/weights/best.pt')

# 推理
results = model('your_image.jpg')

# 显示结果
results[0].show()
```

## 复现报告

详细的复现报告请查看：[复现报告.md](output_yolov8s_coco_640/复现报告.md)

报告包含：
- 训练配置说明
- 性能指标分析
- 与官方基准对比
- 各类别性能分析
- 可视化结果

## 致谢

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - YOLOv8官方实现
- [COCO Dataset](https://cocodataset.org/) - 目标检测数据集

## 许可证

MIT License
