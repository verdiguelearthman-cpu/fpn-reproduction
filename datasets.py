"""
datasets.py
COCO 数据集加载器 + 可视化工具

【升级说明】
在原有基础上补充以下内容：
  1. CocoDataset.__getitem__ 增加 area / iscrowd 字段（torchvision 检测模型训练必需）
  2. 过滤宽高为 0 的无效标注框（避免训练时报错）
  3. 处理图像无标注的边界情况
  4. 保留原有 visualize_bbox 可视化功能，无任何破坏性改动
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from pycocotools.coco import COCO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


class CocoDataset(Dataset):
    def __init__(self, root, annotation_file, transforms=None):
        self.root = root
        self.coco = COCO(annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]

        # 只取 iscrowd=False 的标注
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]["file_name"]
        img = Image.open(os.path.join(self.root, path)).convert("RGB")

        boxes, labels, areas, iscrowd = [], [], [], []
        for ann in anns:
            # COCO bbox format: [x, y, width, height] → [x_min, y_min, x_max, y_max]
            x, y, w, h = ann["bbox"]
            # 过滤无效框（宽或高为 0）
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])
            areas.append(ann["area"])
            iscrowd.append(0)  # 已过滤 iscrowd=True

        # 处理无标注图像（boxes 为空）
        if len(boxes) == 0:
            boxes   = torch.zeros((0, 4), dtype=torch.float32)
            labels  = torch.zeros((0,),   dtype=torch.int64)
            areas   = torch.zeros((0,),   dtype=torch.float32)
            iscrowd = torch.zeros((0,),   dtype=torch.int64)
        else:
            boxes   = torch.as_tensor(boxes,   dtype=torch.float32)
            labels  = torch.as_tensor(labels,  dtype=torch.int64)
            areas   = torch.as_tensor(areas,   dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes":    boxes,
            "labels":   labels,
            "image_id": torch.tensor([img_id]),
            "area":     areas,     # ← 新增，训练必需
            "iscrowd":  iscrowd,   # ← 新增，训练必需
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)


class ToTensor(object):
    """将 PIL 图像转换为 Tensor，target 保持不变"""
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def visualize_bbox(image, target, class_names=None, save_path="visualized_image_with_bbox.png"):
    """
    可视化图像上的标注框。
    image: torch.Tensor (C, H, W)
    target: dict，包含 'boxes' 和 'labels'
    """
    fig, ax = plt.subplots(1)
    ax.imshow(image.permute(1, 2, 0).numpy())

    for i, box in enumerate(target["boxes"]):
        x_min, y_min, x_max, y_max = box.tolist()
        rect = patches.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
        if class_names and i < len(target["labels"]):
            label = class_names.get(target["labels"][i].item(), str(target["labels"][i].item()))
            plt.text(x_min, y_min - 5, label, color='red', fontsize=8,
                     bbox=dict(facecolor='white', alpha=0.7))

    plt.axis("off")
    plt.savefig(save_path)
    plt.close()
    print(f"可视化结果已保存到 {save_path}")


if __name__ == "__main__":
    # 定义数据路径
    coco_root = "./coco/val2017"
    coco_annotation_file = "./coco/annotations/instances_val2017.json"

    # 实例化数据集和数据加载器
    dataset = CocoDataset(root=coco_root, annotation_file=coco_annotation_file, transforms=ToTensor())
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)

    # 获取一个批次的数据并可视化
    coco_api = COCO(coco_annotation_file)
    cats = coco_api.loadCats(coco_api.getCatIds())
    class_names = {cat["id"]: cat["name"] for cat in cats}

    for i, (imgs, targets) in enumerate(data_loader):
        if i == 0:
            img    = imgs[0]
            target = targets[0]
            print(f"图像 shape: {img.shape}")
            print(f"标注框数量: {target['boxes'].shape[0]}")
            print(f"area 字段: {target['area']}")
            print(f"iscrowd 字段: {target['iscrowd']}")
            visualize_bbox(img, target, class_names)
            break

    print("数据加载器和可视化脚本执行完毕。")
