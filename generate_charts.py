"""
generate_charts.py
为 FPN 复现大作业生成高质量图表
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = '/home/ubuntu/fpn_project/charts'
os.makedirs(OUT_DIR, exist_ok=True)

# =========================================================
# 读取数据
# =========================================================
with open('/home/ubuntu/fpn_project/logs_raw/logs/train_history.json') as f:
    history = json.load(f)

with open('/home/ubuntu/fpn_project/results_raw/results/eval_20260412_214158.json') as f:
    eval_res = json.load(f)

epochs     = history['epoch']
loss_total = history['loss_total']
lr_list    = history['lr']


# =========================================================
# 图 1: 训练 Loss 曲线 + 学习率调度（高质量版）
# =========================================================
def plot_training_curves():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('FPN Faster R-CNN 训练过程 (COCO train2017, 12 Epochs)',
                 fontsize=14, fontweight='bold', y=1.02)

    # --- 左图：Loss 曲线 ---
    ax = axes[0]
    ax.plot(epochs, loss_total, 'o-', color='#2196F3', linewidth=2.5,
            markersize=7, markerfacecolor='white', markeredgewidth=2, label='Total Loss')

    # 标注关键数值
    for i in [0, 7, 10, 11]:
        ax.annotate(f'{loss_total[i]:.3f}',
                    xy=(epochs[i], loss_total[i]),
                    xytext=(epochs[i] + 0.2, loss_total[i] + 0.03),
                    fontsize=9, color='#1565C0')

    # 标注学习率衰减时刻
    for step_e in [8, 11]:
        ax.axvline(x=step_e, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.text(step_e + 0.1, max(loss_total) * 0.95, f'lr×0.1\n(epoch {step_e})',
                fontsize=8, color='red', va='top')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('训练损失曲线', fontsize=12, fontweight='bold')
    ax.set_xlim(0.5, 12.5)
    ax.set_xticks(epochs)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11)

    # 填充区域
    ax.fill_between(epochs, loss_total, alpha=0.1, color='#2196F3')

    # --- 右图：学习率调度 ---
    ax = axes[1]
    ax.step(epochs, lr_list, where='post', color='#FF5722', linewidth=2.5, label='Learning Rate')
    ax.scatter(epochs, lr_list, color='#FF5722', s=60, zorder=5)
    ax.set_yscale('log')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('学习率调度策略 (Step LR)', fontsize=12, fontweight='bold')
    ax.set_xlim(0.5, 12.5)
    ax.set_xticks(epochs)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11)

    # 标注三个阶段
    ax.axvspan(1, 7.5, alpha=0.05, color='green', label='Phase 1: lr=0.0025')
    ax.axvspan(7.5, 10.5, alpha=0.05, color='orange', label='Phase 2: lr=0.00025')
    ax.axvspan(10.5, 12, alpha=0.05, color='red', label='Phase 3: lr=0.000025')
    ax.text(4, 0.003, 'Phase 1\nlr=2.5e-3', ha='center', fontsize=9, color='green')
    ax.text(9, 0.0003, 'Phase 2\nlr=2.5e-4', ha='center', fontsize=9, color='darkorange')
    ax.text(11.5, 0.00003, 'Phase 3\nlr=2.5e-5', ha='center', fontsize=9, color='red')

    plt.tight_layout()
    out = f'{OUT_DIR}/training_curves.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✅ 训练曲线已保存: {out}')


# =========================================================
# 图 2: 性能对比柱状图（复现 vs 原论文）
# =========================================================
def plot_performance_comparison():
    metrics    = ['AP\n@0.5:0.95', 'AP\n@0.50', 'AP\n@0.75', 'APs\n(小目标)', 'APm\n(中目标)', 'APl\n(大目标)']
    paper_vals = [33.9,            56.9,         36.2,         17.8,            37.7,            45.8]
    repro_vals = [
        eval_res['AP']   * 100,
        eval_res['AP50'] * 100,
        eval_res['AP75'] * 100,
        eval_res['APs']  * 100,
        eval_res['APm']  * 100,
        eval_res['APl']  * 100,
    ]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 6))

    bars1 = ax.bar(x - width/2, paper_vals, width,
                   label='原论文 (Lin et al., CVPR 2017)', color='#9E9E9E', alpha=0.85,
                   edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x + width/2, repro_vals, width,
                   label='本次复现 (2026, RTX 4060)', color='#42A5F5', alpha=0.9,
                   edgecolor='black', linewidth=0.8)

    # 标注数值
    for bar, val in zip(bars1, paper_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, color='#424242')
    for bar, val in zip(bars2, repro_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#1565C0')

    # 标注差值
    for i, (p, r) in enumerate(zip(paper_vals, repro_vals)):
        diff = r - p
        color = '#2E7D32' if diff >= 0 else '#C62828'
        ax.text(x[i], max(p, r) + 3.5, f'Δ{diff:+.1f}',
                ha='center', fontsize=9, color=color, fontweight='bold')

    ax.set_ylabel('AP (%)', fontsize=12)
    ax.set_title('FPN Faster R-CNN 复现结果 vs 原论文对比\n(Backbone: ResNet-50, Dataset: COCO minival)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 85)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = f'{OUT_DIR}/performance_comparison.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✅ 性能对比图已保存: {out}')


# =========================================================
# 图 3: 实验配置总览表（可视化）
# =========================================================
def plot_config_table():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')

    config_data = [
        ['配置项', '本次复现设置', '原论文设置'],
        ['骨干网络', 'ResNet-50 + FPN', 'ResNet-50 + FPN'],
        ['预训练', 'ImageNet (IMAGENET1K_V1)', 'ImageNet'],
        ['输入尺寸', '短边 800px，长边 ≤ 1333px', '短边 800px，长边 ≤ 1333px'],
        ['优化器', 'SGD (momentum=0.9, wd=1e-4)', 'SGD (momentum=0.9, wd=1e-4)'],
        ['初始学习率', '0.0025 (单 GPU, batch=2)', '0.02 (8 GPU, batch=16)'],
        ['学习率调度', 'Step LR: ×0.1 at epoch 8, 11', 'Step LR: ×0.1 at epoch 8, 11'],
        ['训练轮数', '12 epochs', '12 epochs'],
        ['Batch Size', '2 (单 GPU)', '16 (8 GPU)'],
        ['FPN 输出通道', '256', '256'],
        ['RoI Align', '7×7', '7×7'],
        ['训练数据集', 'COCO train2017 (118k images)', 'COCO train2017'],
        ['评估数据集', 'COCO val2017 (5k images)', 'COCO minival (5k images)'],
        ['GPU 设备', 'NVIDIA RTX 4060 Laptop', '8× NVIDIA M40'],
        ['PyTorch 版本', '2.6.0+cu124', 'Caffe (原始)'],
    ]

    col_widths = [0.28, 0.38, 0.34]
    col_colors = [['#1565C0', '#1565C0', '#1565C0']] + \
                 [['#E3F2FD' if i % 2 == 0 else '#FAFAFA'] * 3 for i in range(len(config_data)-1)]
    text_colors = [['white', 'white', 'white']] + [['black'] * 3 for _ in range(len(config_data)-1)]

    table = ax.table(
        cellText=config_data,
        cellLoc='center',
        loc='center',
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    for (row, col), cell in table.get_celld().items():
        cell.set_facecolor(col_colors[row][col])
        cell.set_text_props(color=text_colors[row][col],
                            fontweight='bold' if row == 0 else 'normal')
        cell.set_edgecolor('#BDBDBD')

    ax.set_title('实验配置对比表', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    out = f'{OUT_DIR}/config_table.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✅ 配置表已保存: {out}')


if __name__ == '__main__':
    plot_training_curves()
    plot_performance_comparison()
    plot_config_table()
    print('\n所有图表生成完毕！')
