import os
import json
import glob
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse


def get_latest_file(pattern):
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def plot_loss_curves(history, output_dir):
    epochs = history.get('epoch', [])
    if not epochs:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('FPN Faster R-CNN Training Progress', fontsize=14, fontweight='bold')

    # Total Loss
    ax = axes[0]
    loss_key = 'loss_total' if 'loss_total' in history else 'loss'
    ax.plot(epochs, history[loss_key], 'b-o', linewidth=2, markersize=5, label='Total Loss')
    ax.set_title('Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Learning Rate
    ax = axes[1]
    ax.plot(epochs, history['lr'], 'k-o', linewidth=2, markersize=5)
    ax.set_title('Learning Rate Schedule')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(output_dir, 'loss_curves.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 损失曲线已保存: {out}")


def plot_comparison(eval_results, output_dir):
    metrics = ['AP', 'AP50', 'APs', 'APm', 'APl']
    m_labels = ['AP\n@0.5:0.95', 'AP\n@0.50', 'AP\n(small)', 'AP\n(medium)', 'AP\n(large)']
    paper_val = [33.9, 56.9, 17.8, 37.7, 45.8]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    # Paper results
    ax.bar(x - width / 2, paper_val, width, label='Original Paper (2017)', color='gray', alpha=0.6)

    # Reproduction results
    res_vals = [eval_results.get(m, 0) * 100 if eval_results.get(m, 0) < 1 else eval_results.get(m, 0) for m in metrics]
    bars = ax.bar(x + width / 2, res_vals, width, label='Our Reproduction (2026)', color='skyblue')

    for b, v in zip(bars, res_vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f'{v:.1f}', ha='center', fontweight='bold')

    ax.set_title('FPN Faster R-CNN: Reproduction vs Original Paper', fontsize=14, fontweight='bold')
    ax.set_ylabel('Value (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(m_labels)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    out = os.path.join(output_dir, 'comparison_chart.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 性能对比图已保存: {out}")


def draw_fpn_architecture(output_dir):
    plt.figure(figsize=(10, 6))
    plt.axis('off')

    # Simple schematic drawing
    # Bottom-up
    for i in range(4):
        plt.gca().add_patch(plt.Rectangle((0.1, 0.2 + i * 0.15), 0.1, 0.1, color='lightblue', alpha=0.6))
        plt.text(0.15, 0.25 + i * 0.15, f'C{i + 2}', ha='center')

    # Top-down
    for i in range(4):
        plt.gca().add_patch(plt.Rectangle((0.6, 0.2 + i * 0.15), 0.1, 0.1, color='salmon', alpha=0.6))
        plt.text(0.65, 0.25 + i * 0.15, f'P{i + 2}', ha='center')

    # Connections
    for i in range(4):
        plt.arrow(0.22, 0.25 + i * 0.15, 0.35, 0, head_width=0.01, head_length=0.02, fc='k', ec='k', linestyle='--')

    plt.text(0.15, 0.8, 'Bottom-up', ha='center', fontweight='bold')
    plt.text(0.65, 0.8, 'Top-down', ha='center', fontweight='bold')
    plt.title('FPN Architecture Schematic', fontsize=14, fontweight='bold')

    out = os.path.join(output_dir, 'fpn_architecture.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ FPN 架构示意图已保存: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--history', default=None)
    parser.add_argument('--eval-results', default=None)
    parser.add_argument('--output-dir', default='./results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Auto-discovery if not provided
    history_path = args.history or get_latest_file('./logs/train_history.json')
    eval_path = args.eval_results or get_latest_file('./results/eval_*.json')

    print(f"--- FPN 结果可视化工具 ---")

    if history_path and os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        plot_loss_curves(history, args.output_dir)

    if eval_path and os.path.exists(eval_path):
        with open(eval_path) as f:
            res = json.load(f)
        plot_comparison(res, args.output_dir)

    draw_fpn_architecture(args.output_dir)
    print(f"--- 所有图表已保存至 {args.output_dir} ---")


if __name__ == '__main__':
    main()