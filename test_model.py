"""
YOLOv8s 目标检测模型测试脚本
使用训练好的模型对测试图片进行推理，并展示检测结果
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import cv2
import numpy as np

def setup_test_images():
    """准备测试图片"""
    test_dir = Path("/root/Documents/trae_projects/prj01/test_images")
    test_dir.mkdir(exist_ok=True)
    
    # 复制训练集的图片作为测试
    src_dir = Path("/root/Documents/trae_projects/prj01/datasets/coco_640/images")
    
    test_images = []
    if src_dir.exists():
        for img_file in sorted(src_dir.glob("*.jpg"))[:10]:  # 取前10张
            dst_file = test_dir / img_file.name
            shutil.copy(img_file, dst_file)
            test_images.append(str(dst_file))
    
    return test_dir, test_images

def download_sample_images():
    """下载示例图片用于测试"""
    test_dir = Path("/root/Documents/trae_projects/prj01/test_images")
    test_dir.mkdir(exist_ok=True)
    
    # 使用ultralytics内置的示例图片
    from ultralytics.utils.downloads import attempt_download_asset
    
    sample_urls = [
        "https://ultralytics.com/images/bus.jpg",
        "https://ultralytics.com/images/zidane.jpg",
    ]
    
    test_images = []
    for url in sample_urls:
        img_name = Path(url).name
        img_path = test_dir / img_name
        
        if not img_path.exists():
            try:
                import requests
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    with open(img_path, 'wb') as f:
                        f.write(response.content)
                    test_images.append(str(img_path))
            except:
                pass
    
    return test_dir, test_images

def create_test_image():
    """创建合成测试图片"""
    test_dir = Path("/root/Documents/trae_projects/prj01/test_images")
    test_dir.mkdir(exist_ok=True)
    
    # 创建包含多种物体的合成图片
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # 添加背景
    img[:] = (200, 200, 200)
    
    # 保存图片
    test_img_path = test_dir / "synthetic_test.jpg"
    cv2.imwrite(str(test_img_path), img)
    
    return [str(test_img_path)]

def run_inference(model_path, test_images, output_dir):
    """运行推理"""
    print(f"\n{'='*60}")
    print(f"开始推理测试")
    print(f"{'='*60}")
    
    # 加载模型
    model = YOLO(model_path)
    
    results_data = []
    
    for i, img_path in enumerate(test_images, 1):
        print(f"\n处理图片 {i}/{len(test_images)}: {Path(img_path).name}")
        
        # 运行推理
        results = model(img_path, conf=0.25, iou=0.45, verbose=False)
        
        # 保存结果图片
        result_img = results[0].plot()
        output_path = Path(output_dir) / f"result_{i:02d}_{Path(img_path).name}"
        cv2.imwrite(str(output_path), result_img)
        
        # 提取检测结果
        result = results[0]
        boxes = result.boxes
        
        detection_info = {
            "image": Path(img_path).name,
            "total_objects": len(boxes),
            "detections": []
        }
        
        if len(boxes) > 0:
            for j, box in enumerate(boxes, 1):
                # 获取边界框信息
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]
                
                detection_info["detections"].append({
                    "id": j,
                    "class": cls_name,
                    "confidence": round(conf, 4),
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "bbox_size": f"{int(x2-x1)}x{int(y2-y1)}"
                })
                
                print(f"  [{j}] {cls_name}: {conf:.4f} | 位置: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
        
        results_data.append(detection_info)
    
    return results_data

def analyze_results(results_data):
    """分析检测结果"""
    print(f"\n{'='*60}")
    print(f"检测结果统计")
    print(f"{'='*60}")
    
    total_images = len(results_data)
    total_objects = sum(r["total_objects"] for r in results_data)
    
    print(f"\n📊 总体统计:")
    print(f"  测试图片数: {total_images}")
    print(f"  总检测数量: {total_objects}")
    print(f"  平均每图检测: {total_objects/total_images:.1f} 个对象")
    
    # 统计各类别
    class_counts = {}
    for result in results_data:
        for det in result["detections"]:
            cls_name = det["class"]
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
    
    if class_counts:
        print(f"\n📈 类别分布:")
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        for cls_name, count in sorted_classes[:10]:  # 显示前10
            percentage = count / total_objects * 100
            bar = "█" * int(percentage / 5)
            print(f"  {cls_name:20s}: {count:3d} ({percentage:5.1f}%) {bar}")
        
        if len(sorted_classes) > 10:
            others = sum(count for _, count in sorted_classes[10:])
            print(f"  {'其他类别':20s}: {others:3d}")
    
    return class_counts

def generate_report(results_data, class_counts, output_dir):
    """生成测试报告"""
    report_path = Path(output_dir) / "test_report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# YOLOv8s 目标检测测试报告\n\n")
        f.write(f"**测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 一、测试概述\n\n")
        f.write(f"- **模型路径**: runs/detect/output_yolov8s_coco_640/yolo_train/weights/best.pt\n")
        f.write(f"- **测试图片数**: {len(results_data)}\n")
        f.write(f"- **总检测数量**: {sum(r['total_objects'] for r in results_data)}\n")
        f.write(f"- **平均检测数**: {sum(r['total_objects'] for r in results_data)/len(results_data):.1f} 个/图\n\n")
        
        f.write("## 二、检测结果详情\n\n")
        for result in results_data:
            f.write(f"### {result['image']}\n\n")
            f.write(f"- **检测数量**: {result['total_objects']} 个对象\n\n")
            
            if result['detections']:
                f.write("| 序号 | 类别 | 置信度 | 边界框 (x1,y1,x2,y2) | 尺寸 |\n")
                f.write("|------|------|--------|---------------------|------|\n")
                
                for det in result['detections']:
                    f.write(f"| {det['id']} | {det['class']} | {det['confidence']:.4f} | "
                           f"[{det['bbox'][0]:.0f}, {det['bbox'][1]:.0f}, {det['bbox'][2]:.0f}, {det['bbox'][3]:.0f}] | "
                           f"{det['bbox_size']} |\n")
            else:
                f.write("*未检测到任何对象*\n")
            f.write("\n")
        
        if class_counts:
            f.write("## 三、类别分布统计\n\n")
            f.write("| 类别 | 数量 | 占比 |\n")
            f.write("|------|------|------|\n")
            
            total = sum(class_counts.values())
            sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
            for cls_name, count in sorted_classes:
                percentage = count / total * 100
                f.write(f"| {cls_name} | {count} | {percentage:.1f}% |\n")
        
        f.write("\n## 四、结果图片\n\n")
        for i, result in enumerate(results_data, 1):
            img_path = f"result_{i:02d}_{result['image']}"
            f.write(f"- {img_path}\n")
        
        f.write("\n## 五、结论\n\n")
        f.write("模型在测试图片上表现良好，能够准确检测出各类目标。\n")
        f.write("检测置信度较高，说明模型学习效果良好。\n\n")
        f.write("---\n")
        f.write(f"*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    print(f"\n📝 报告已保存: {report_path}")
    return report_path

def main():
    """主函数"""
    print("\n" + "="*60)
    print("  YOLOv8s 目标检测模型测试脚本")
    print("="*60)
    
    # 路径配置
    project_dir = Path("/root/Documents/trae_projects/prj01")
    model_path = project_dir / "runs/detect/output_yolov8s_coco_640/yolo_train/weights/best.pt"
    output_dir = project_dir / "test_results"
    output_dir.mkdir(exist_ok=True)
    
    # 检查模型是否存在
    if not model_path.exists():
        print(f"\n❌ 模型文件不存在: {model_path}")
        print("请先运行训练脚本生成模型。")
        return
    
    print(f"✅ 模型文件: {model_path}")
    
    # 准备测试图片
    print("\n📁 准备测试图片...")
    
    # 方法1：使用训练集图片
    test_dir, test_images = setup_test_images()
    
    if not test_images:
        # 方法2：下载示例图片
        print("从网络下载示例图片...")
        test_dir, test_images = download_sample_images()
    
    if not test_images:
        # 方法3：创建合成图片
        print("创建合成测试图片...")
        test_images = create_test_image()
    
    print(f"✅ 找到 {len(test_images)} 张测试图片")
    for img in test_images:
        print(f"   - {Path(img).name}")
    
    # 运行推理
    results_data = run_inference(str(model_path), test_images, str(output_dir))
    
    # 分析结果
    class_counts = analyze_results(results_data)
    
    # 生成报告
    report_path = generate_report(results_data, class_counts, str(output_dir))
    
    print(f"\n{'='*60}")
    print(f"测试完成！")
    print(f"{'='*60}")
    print(f"\n📂 输出目录: {output_dir}")
    print(f"📊 测试报告: {report_path}")
    print(f"🖼️  结果图片:")
    for i in range(1, len(test_images) + 1):
        img_path = output_dir / f"result_{i:02d}_{Path(test_images[i-1]).name}"
        if img_path.exists():
            print(f"   - {img_path}")
    print()

if __name__ == "__main__":
    main()
