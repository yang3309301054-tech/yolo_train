import os
import shutil

print("创建640张图片的数据集...")

src_img_dir = '/root/Documents/trae_projects/prj01/datasets/coco128/images/train2017'
src_label_dir = '/root/Documents/trae_projects/prj01/datasets/coco128/labels/train2017'
dst_img_dir = '/root/Documents/trae_projects/prj01/datasets/coco_640/images'
dst_label_dir = '/root/Documents/trae_projects/prj01/datasets/coco_640/labels'

os.makedirs(dst_img_dir, exist_ok=True)
os.makedirs(dst_label_dir, exist_ok=True)

# 获取源文件列表
src_images = sorted([f for f in os.listdir(src_img_dir) if f.endswith('.jpg')])
src_labels = sorted([f for f in os.listdir(src_label_dir) if f.endswith('.txt')])

print(f"源图片数量: {len(src_images)}")

# 复制5份，使用不同的文件名
for i in range(5):
    for j, img_file in enumerate(src_images):
        src_img = os.path.join(src_img_dir, img_file)
        dst_img = os.path.join(dst_img_dir, f"{i*128+j:012d}.jpg")
        shutil.copy(src_img, dst_img)
        
        # 复制对应的标签文件
        label_file = img_file.replace('.jpg', '.txt')
        if label_file in src_labels:
            src_label = os.path.join(src_label_dir, label_file)
            dst_label = os.path.join(dst_label_dir, f"{i*128+j:012d}.txt")
            shutil.copy(src_label, dst_label)

# 统计结果
final_count = len([f for f in os.listdir(dst_img_dir) if f.endswith('.jpg')])
print(f"✓ 创建完成，共 {final_count} 张图片")
