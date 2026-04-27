"""
evaluate.py
FPN 论文复现 - 评估脚本

使用 pycocotools 计算标准 COCO AP 指标，并与论文结果对比。
对应论文 Table 3: Faster R-CNN with FPN, ResNet-50, COCO minival

用法：
  # 评估自己训练的 checkpoint
  python evaluate.py --checkpoint ./checkpoints/checkpoint_best.pth

  # 使用 torchvision 官方预训练权重（作为基准对比）
  python evaluate.py --pretrained

  # 快速验证流程（只跑 100 张图）
  python evaluate.py --pretrained --max-samples 100
"""

import os
import sys
import json
import time
import datetime
import argparse
import logging

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Subset

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 复用本项目的数据加载器
from datasets import CocoDataset, ToTensor, collate_fn


def build_eval_dataset(img_dir, ann_file, max_samples=0):
    """构建评估用数据集（不需要 area/iscrowd，只需图像和 image_id）"""
    dataset = CocoDataset(root=img_dir, annotation_file=ann_file, transforms=ToTensor())
    if max_samples > 0:
        dataset = Subset(dataset, list(range(min(max_samples, len(dataset)))))
    return dataset


def build_model(num_classes=91):
    model = fasterrcnn_resnet50_fpn(
        weights=None, weights_backbone=None,
        min_size=800, max_size=1333,
        rpn_pre_nms_top_n_test=1000, rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7, box_score_thresh=0.05,
        box_nms_thresh=0.5, box_detections_per_img=100,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def run_eval(model, data_loader, device, ann_file, logger):
    model.eval()
    coco_gt = COCO(ann_file)
    results = []
    t0 = time.time()

    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            images = [img.to(device) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                img_id = target['image_id'].item()
                for box, score, label in zip(
                    output['boxes'].cpu(), output['scores'].cpu(), output['labels'].cpu()
                ):
                    x1, y1, x2, y2 = box.tolist()
                    results.append({
                        'image_id':   img_id,
                        'category_id': int(label),
                        'bbox':  [x1, y1, x2 - x1, y2 - y1],
                        'score': float(score),
                    })

            if (i + 1) % 100 == 0:
                logger.info(f"  已推理 {i+1}/{len(data_loader)} | 耗时 {time.time()-t0:.0f}s")

    logger.info(f"推理完成，共 {len(results)} 个预测框")
    if not results:
        logger.warning("无预测结果，请检查模型或置信度阈值")
        return None

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval


def print_comparison(coco_eval, logger):
    """打印复现结果与论文结果的对比表"""
    s = coco_eval.stats
    paper = {'AP': 33.9, 'AP50': 56.9, 'APs': 17.8, 'APm': 37.7, 'APl': 45.8}
    repro = {
        'AP':   s[0] * 100, 'AP50': s[1] * 100,
        'APs':  s[3] * 100, 'APm':  s[4] * 100, 'APl': s[5] * 100,
    }
    logger.info("\n" + "=" * 55)
    logger.info("复现结果 vs 论文结果（Table 3, ResNet-50, COCO minival）")
    logger.info(f"  {'指标':<12} {'论文':>8} {'复现':>8} {'差异':>8}")
    logger.info("  " + "-" * 40)
    for k in paper:
        diff = repro[k] - paper[k]
        logger.info(f"  {k:<12} {paper[k]:>8.1f} {repro[k]:>8.1f} {diff:>+8.1f}")
    logger.info("=" * 55)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(args.log_dir, f'eval_{ts}.log')
    logging.basicConfig(
        level=logging.INFO, format='[%(asctime)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)],
    )
    logger = logging.getLogger()

    logger.info("=" * 55)
    logger.info("FPN 论文复现 - 评估脚本")
    logger.info(f"设备: {device}  |  PyTorch: {torch.__version__}")
    logger.info("=" * 55)

    # 数据集
    dataset = build_eval_dataset(args.img_dir, args.ann_file, args.max_samples)
    logger.info(f"评估集大小: {len(dataset)} 张图")
    data_loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )

    # 模型
    if args.pretrained:
        logger.info("加载 torchvision 官方预训练权重（COCO 训练）...")
        model = fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.COCO_V1')
    else:
        logger.info(f"加载 checkpoint: {args.checkpoint}")
        model = build_model(num_classes=args.num_classes)
        ckpt = torch.load(args.checkpoint, map_location=device)
        state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state)
        logger.info(f"  来自 epoch {ckpt.get('epoch', '?')}")
    model.to(device)

    # 评估
    logger.info("\n开始评估...")
    coco_eval = run_eval(model, data_loader, device, args.ann_file, logger)

    if coco_eval is not None:
        print_comparison(coco_eval, logger)

        # 保存结果
        result_dict = {
            'timestamp': ts,
            'checkpoint': args.checkpoint if not args.pretrained else 'torchvision_pretrained',
            'AP':   float(coco_eval.stats[0]),
            'AP50': float(coco_eval.stats[1]),
            'AP75': float(coco_eval.stats[2]),
            'APs':  float(coco_eval.stats[3]),
            'APm':  float(coco_eval.stats[4]),
            'APl':  float(coco_eval.stats[5]),
            'AR1':  float(coco_eval.stats[6]),
            'AR10': float(coco_eval.stats[7]),
            'AR100':float(coco_eval.stats[8]),
        }
        out_path = os.path.join(args.results_dir, f'eval_{ts}.json')
        with open(out_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
        logger.info(f"评估结果已保存: {out_path}")

    logger.info(f"评估日志已保存: {log_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-dir',      default='./coco/val2017')
    parser.add_argument('--ann-file',     default='./coco/annotations/instances_val2017.json')
    parser.add_argument('--num-classes',  type=int, default=91)
    parser.add_argument('--checkpoint',   default='./checkpoints/checkpoint_best.pth')
    parser.add_argument('--pretrained',   action='store_true',
                        help='使用 torchvision 官方预训练权重（基准对比）')
    parser.add_argument('--max-samples',  type=int, default=0)
    parser.add_argument('--num-workers',  type=int, default=4)
    parser.add_argument('--log-dir',      default='./logs')
    parser.add_argument('--results-dir',  default='./results')
    args = parser.parse_args()
    main(args)
