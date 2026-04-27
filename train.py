"""
train.py
FPN 论文复现 - Faster R-CNN with FPN 训练脚本

对应论文: Feature Pyramid Networks for Object Detection (Lin et al., CVPR 2017)

训练配置（对应论文 Section 4）:
  - 骨干: ResNet-50 + FPN, ImageNet 预训练
  - 输入: 短边 800px, 长边最大 1333px
  - 优化器: SGD, lr=0.02 (8GPU/batch=16), momentum=0.9, weight_decay=1e-4
  - 学习率调度: epoch 8 和 11 各衰减 10 倍, 共 12 epoch
  - RPN: anchor sizes {32,64,128,256,512}, aspect ratios {0.5,1,2}

用法:
  # 完整训练 (单 GPU)
  python train.py \
    --img-dir ./coco/train2017 \
    --ann-file ./coco/annotations/instances_train2017.json \
    --epochs 12 --batch-size 2 --auto-scale-lr

  # 快速验证流程 (10 张图, 1 epoch)
  python train.py \
    --img-dir ./coco/val2017 \
    --ann-file ./coco/annotations/instances_val2017.json \
    --epochs 1 --batch-size 1 --max-samples 10 --auto-scale-lr
"""

import os
import sys
import time
import json
import datetime
import argparse
import logging

import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader

from datasets import CocoDataset, ToTensor, collate_fn


# ══════════════════════════════════════════════════════════
# 模型构建
# ══════════════════════════════════════════════════════════

def build_model(num_classes=91, pretrained_backbone=True):
    """
    构建 Faster R-CNN with FPN (ResNet-50)。

    对应论文实验设置:
      - 骨干: ResNet-50 + FPN, ImageNet 预训练
      - RPN anchor: sizes per level = {32²,64²,128²,256²,512²}, ratios = {0.5,1,2}
      - RoI: 7x7 RoIAlign, 多尺度分配 (k0=4)
      - 检测头: 2 个 1024-d FC 层

    参数:
      num_classes: 类别数 (含背景), COCO=91
      pretrained_backbone: 是否使用 ImageNet 预训练骨干
    """
    model = fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone='ResNet50_Weights.IMAGENET1K_V1' if pretrained_backbone else None,
        min_size=800,
        max_size=1333,
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
    )
    # 替换检测头以匹配类别数
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# ══════════════════════════════════════════════════════════
# 训练工具
# ══════════════════════════════════════════════════════════

def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'train_{ts}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)],
    )
    return logging.getLogger(), log_file


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor=1.0 / 1000):
    """线性 warmup 学习率调度器"""
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def train_one_epoch(model, optimizer, data_loader, device, epoch, logger,
                    print_freq=20, warmup=False):
    """训练一个 epoch"""
    model.train()
    total, n = 0.0, 0
    loss_dict_sum = {}

    if warmup:
        warmup_iters = min(1000, len(data_loader) - 1)
        ws = warmup_lr_scheduler(optimizer, warmup_iters)

    t0 = time.time()
    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 过滤无标注框的图像
        valid = [(im, tg) for im, tg in zip(images, targets) if tg['boxes'].shape[0] > 0]
        if not valid:
            continue
        images, targets = zip(*valid)

        loss_dict = model(list(images), list(targets))
        losses = sum(v for v in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        if warmup:
            ws.step()

        total += losses.item()
        n += 1
        for k, v in loss_dict.items():
            loss_dict_sum[k] = loss_dict_sum.get(k, 0.0) + v.item()

        if (i + 1) % print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            detail = ' | '.join(f'{k}={v.item():.4f}' for k, v in loss_dict.items())
            logger.info(
                f"Epoch[{epoch}] [{i+1}/{len(data_loader)}] "
                f"loss={losses.item():.4f} ({detail}) lr={lr:.6f} t={time.time()-t0:.0f}s"
            )

    avg = {'total': total / max(n, 1)}
    avg.update({k: v / max(n, 1) for k, v in loss_dict_sum.items()})
    return avg


# ══════════════════════════════════════════════════════════
# 主训练流程
# ══════════════════════════════════════════════════════════

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger, log_file = setup_logger(args.log_dir)

    logger.info("=" * 60)
    logger.info("FPN 论文复现 - Faster R-CNN with FPN (ResNet-50)")
    logger.info(f"时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"设备: {device}  |  PyTorch: {torch.__version__}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info("=" * 60)

    # ── 数据集 ──
    logger.info(f"加载数据集: {args.img_dir}")
    dataset = CocoDataset(
        root=args.img_dir,
        annotation_file=args.ann_file,
        transforms=ToTensor(),
    )

    if args.max_samples > 0:
        from torch.utils.data import Subset
        dataset = Subset(dataset, list(range(min(args.max_samples, len(dataset)))))
        logger.info(f"使用子集: {len(dataset)} 张图（快速验证模式）")
    else:
        logger.info(f"数据集大小: {len(dataset)} 张图")

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # ── 模型 ──
    logger.info("构建模型: Faster R-CNN with FPN (ResNet-50, ImageNet 预训练骨干)")
    model = build_model(num_classes=args.num_classes, pretrained_backbone=True)
    model.to(device)

    total_p = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"参数量: {total_p:.1f}M")

    # ── 优化器 (论文: SGD, lr=0.02/8GPU, momentum=0.9, wd=1e-4) ──
    # 单 GPU 线性缩放: lr = 0.02 × batch_size / 16
    effective_lr = args.lr * args.batch_size / 16 if args.auto_scale_lr else args.lr
    logger.info(f"优化器: SGD | lr={effective_lr:.5f} | momentum=0.9 | wd=1e-4")
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=effective_lr, momentum=0.9, weight_decay=1e-4,
    )

    # ── 学习率调度 (论文: epoch 8 和 11 各衰减 10 倍) ──
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_steps, gamma=0.1,
    )

    history = {'epoch': [], 'loss_total': [], 'lr': []}
    best_loss = float('inf')
    os.makedirs(args.ckpt_dir, exist_ok=True)

    logger.info(f"\n开始训练: {args.epochs} epochs | batch={args.batch_size} | lr_steps={args.lr_steps}")
    logger.info("-" * 60)

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n{'='*20} Epoch {epoch}/{args.epochs} {'='*20}")
        t_ep = time.time()

        avg_losses = train_one_epoch(
            model, optimizer, data_loader, device, epoch, logger,
            print_freq=args.print_freq, warmup=(epoch == 1),
        )
        lr_scheduler.step()
        cur_lr = optimizer.param_groups[0]['lr']

        logger.info(
            f"Epoch {epoch} 完成 | 耗时 {time.time()-t_ep:.0f}s | "
            f"avg_loss={avg_losses['total']:.4f} | lr={cur_lr:.6f}"
        )

        history['epoch'].append(epoch)
        history['loss_total'].append(avg_losses['total'])
        history['lr'].append(cur_lr)

        # 保存 checkpoint
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'avg_losses': avg_losses,
        }
        torch.save(ckpt, os.path.join(args.ckpt_dir, 'checkpoint_latest.pth'))

        if avg_losses['total'] < best_loss and avg_losses['total'] > 0:
            best_loss = avg_losses['total']
            torch.save(ckpt, os.path.join(args.ckpt_dir, 'checkpoint_best.pth'))
            logger.info(f"  >> 保存最佳 checkpoint (loss={best_loss:.4f})")

        if epoch % args.save_freq == 0:
            torch.save(ckpt, os.path.join(args.ckpt_dir, f'checkpoint_ep{epoch:02d}.pth'))

    # 保存训练历史
    hist_path = os.path.join(args.log_dir, 'train_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"\n训练历史已保存: {hist_path}")
    logger.info(f"训练日志已保存: {log_file}")
    logger.info("\n训练完成！下一步请运行 evaluate.py 进行评估。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FPN Faster R-CNN 训练')
    # 数据路径
    parser.add_argument('--img-dir',  default='./coco/train2017')
    parser.add_argument('--ann-file', default='./coco/annotations/instances_train2017.json')
    # 训练超参数
    parser.add_argument('--num-classes',   type=int,   default=91)
    parser.add_argument('--epochs',        type=int,   default=12)
    parser.add_argument('--batch-size',    type=int,   default=2)
    parser.add_argument('--lr',            type=float, default=0.02,
                        help='基准学习率 (论文: 0.02, 对应 8GPU/batch=16)')
    parser.add_argument('--auto-scale-lr', action='store_true',
                        help='按 batch_size 线性缩放 lr (单 GPU 推荐开启)')
    parser.add_argument('--lr-steps',      type=int, nargs='+', default=[8, 11])
    parser.add_argument('--max-samples',   type=int, default=0,
                        help='限制样本数 (0=全部, >0 为快速验证)')
    # 输出路径
    parser.add_argument('--log-dir',    default='./logs')
    parser.add_argument('--ckpt-dir',   default='./checkpoints')
    parser.add_argument('--num-workers',type=int, default=4)
    parser.add_argument('--print-freq', type=int, default=20)
    parser.add_argument('--save-freq',  type=int, default=3)
    args = parser.parse_args()
    main(args)
