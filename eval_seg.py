"""
eval_seg.py — Segmentation 정량/정성 평가 스크립트

사용법:
  python eval_seg.py \
    --pretrained_weights OUTPUT/pretrain_v7/checkpoint.pth \
    --seg_head_weights   OUTPUT/finetune_v7/best_seg_head.pth \
    --block_dir          preprocessed/val/bolus \
    --label_dir          label/bolus \
    --stats_path         preprocessed/stats.json \
    --output_dir         OUTPUT/eval_v7 \
    --cfg experiments/phasor/swin_tiny_phasor.yaml

출력:
  confusion_matrix.png  — 클래스별 혼동 행렬
  pr_curve.png          — Bubble Precision-Recall 곡선
  samples/              — 입력(magnitude) + GT + 예측 시각화 (N장)
  metrics.json          — 정량 지표 (IoU, Precision, Recall, F1 per class)
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score

import utils
from models import build_model
from config import config, update_config
from datasets.block_dataset import BlockSegDataset
from finetune_seg import MultiScaleFeatureExtractor, SegHead, IGNORE_INDEX

ALL_CLASS_NAMES  = ['noise', 'tissue', 'bubble', 'unknown']
ALL_CLASS_COLORS = np.array([
    [0.2, 0.2, 0.8],   # noise    — blue
    [0.2, 0.7, 0.2],   # tissue   — green
    [0.9, 0.2, 0.2],   # bubble   — red
    [0.8, 0.6, 0.0],   # unknown  — orange
])


def colorize(label_map, colors):
    """(H, W) int → (H, W, 3) float RGB"""
    rgb = np.zeros((*label_map.shape, 3), dtype=float)
    for c, color in enumerate(colors):
        rgb[label_map == c] = color
    rgb[label_map == IGNORE_INDEX] = [0.5, 0.5, 0.5]
    return rgb


@torch.no_grad()
def run_eval(extractor, seg_head, loader, device):
    seg_head.eval()
    extractor.eval()

    all_preds, all_targets = [], []
    all_probs_bubble = []   # bubble class softmax prob (for PR curve)
    samples = []            # (image_tensor, pred, target) for visualization

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        feat_s1, feat_s3 = extractor(images)
        logits = seg_head(feat_s1, feat_s3)     # (B, C, H, W)
        probs  = torch.softmax(logits, dim=1)   # (B, C, H, W)
        pred   = logits.argmax(dim=1)           # (B, H, W)

        # 4-class면 모든 픽셀 유효, 3-class면 255 마스킹
        mask = labels != IGNORE_INDEX

        all_preds.append(pred[mask].cpu().numpy())
        all_targets.append(labels[mask].cpu().numpy())
        all_probs_bubble.append(probs[:, 2][mask].cpu().numpy())  # bubble=class2

        # 시각화용 샘플 (최대 12개)
        if len(samples) < 12:
            for i in range(min(images.size(0), 12 - len(samples))):
                samples.append((
                    images[i].cpu(),
                    pred[i].cpu().numpy(),
                    labels[i].cpu().numpy(),
                    probs[i, 2].cpu().numpy(),  # bubble prob map
                ))

    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_probs_bubble = np.concatenate(all_probs_bubble)

    return all_preds, all_targets, all_probs_bubble, samples


def compute_metrics(preds, targets, num_classes, class_names):
    metrics = {}
    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum()
        fp = ((preds == c) & (targets != c)).sum()
        fn = ((preds != c) & (targets == c)).sum()
        tn = ((preds != c) & (targets != c)).sum()

        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        iou       = tp / (tp + fp + fn + 1e-8)

        metrics[class_names[c]] = {
            'IoU': float(iou),
            'Precision': float(precision),
            'Recall': float(recall),
            'F1': float(f1),
            'TP': int(tp), 'FP': int(fp), 'FN': int(fn),
        }

    miou = np.mean([metrics[n]['IoU'] for n in class_names])
    metrics['mIoU'] = float(miou)
    return metrics


def plot_confusion_matrix(preds, targets, class_names, output_path):
    nc = len(class_names)
    cm = confusion_matrix(targets, preds, labels=list(range(nc)))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(6 * nc, 5))
    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ['Confusion Matrix (count)', 'Confusion Matrix (row-normalized)'],
        ['d', '.2f']
    ):
        im = ax.imshow(data, cmap='Blues')
        ax.set_xticks(range(nc)); ax.set_yticks(range(nc))
        ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        ax.set_title(title)
        for i in range(nc):
            for j in range(nc):
                val = data[i, j]
                text = f'{val:{fmt}}'
                ax.text(j, i, text, ha='center', va='center',
                        color='white' if (cm_norm[i, j] > 0.5) else 'black')
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_pr_curve(targets, probs_bubble, output_path):
    bubble_binary = (targets == 2).astype(int)
    precision, recall, thresholds = precision_recall_curve(bubble_binary, probs_bubble)
    ap = average_precision_score(bubble_binary, probs_bubble)

    # F1 최대 threshold
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx  = np.argmax(f1_scores[:-1])
    best_thr  = thresholds[best_idx]
    best_f1   = f1_scores[best_idx]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, 'r-', linewidth=2, label=f'AP={ap:.3f}')
    ax.scatter(recall[best_idx], precision[best_idx], s=120, zorder=5,
               color='red', label=f'Best F1={best_f1:.3f} (thr={best_thr:.2f})')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title('Bubble Class — Precision-Recall Curve')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}  (AP={ap:.3f}, best F1={best_f1:.3f} @ thr={best_thr:.2f})")
    return ap, best_thr, best_f1


def plot_samples(samples, class_names, class_colors, output_dir):
    """magnitude + GT + prediction + bubble prob map 시각화"""
    os.makedirs(output_dir, exist_ok=True)
    for idx, (img_tensor, pred, target, bubble_prob) in enumerate(samples):
        magnitude = img_tensor[0].numpy()

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle(f'Sample {idx:02d}', fontsize=12)

        axes[0].imshow(magnitude, cmap='gray')
        axes[0].set_title('Magnitude (ch0, normalized)')

        axes[1].imshow(colorize(target, class_colors))
        axes[1].set_title('Ground Truth')

        axes[2].imshow(colorize(pred, class_colors))
        axes[2].set_title('Prediction')

        axes[3].imshow(bubble_prob, cmap='hot', vmin=0, vmax=1)
        axes[3].set_title('Bubble Probability')

        patches = [mpatches.Patch(color=COLOR, label=NAME)
                   for NAME, COLOR in zip(class_names, class_colors)]
        axes[1].legend(handles=patches, loc='lower right', fontsize=7)
        axes[2].legend(handles=patches, loc='lower right', fontsize=7)

        for ax in axes:
            ax.axis('off')

        plt.tight_layout()
        path = os.path.join(output_dir, f'sample_{idx:02d}.png')
        plt.savefig(path, dpi=120)
        plt.close()

    print(f"Saved {len(samples)} sample images → {output_dir}/")


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_classes  = args.num_classes
    remap        = (num_classes == 4)
    class_names  = ALL_CLASS_NAMES[:num_classes]
    class_colors = ALL_CLASS_COLORS[:num_classes]

    # ── 데이터 ──
    stats = None
    if args.stats_path and os.path.isfile(args.stats_path):
        with open(args.stats_path) as f:
            stats = json.load(f)

    dataset = BlockSegDataset(args.block_dir, args.label_dir, stats=stats,
                              remap_unknown=remap)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=4,
        pin_memory=True, shuffle=False
    )
    print(f"Eval samples: {len(dataset)}  num_classes={num_classes}")

    # ── 모델 ──
    args.rank       = 0
    args.world_size = 1
    args.dist_url   = 'env://'
    args.local_rank = 0
    update_config(config, args)
    backbone = build_model(config, is_teacher=True).to(device)
    for p in backbone.parameters():
        p.requires_grad_(False)
    backbone.eval()

    utils.load_pretrained_weights(
        backbone, args.pretrained_weights, args.checkpoint_key, args.arch, patch_size=None
    )

    swin_spec  = config.MODEL.SPEC
    stage1_dim = swin_spec['DIM_EMBED']
    INPUT_H, INPUT_W = 80, 120
    patch_hw = (INPUT_H // swin_spec['PATCH_SIZE'],
                INPUT_W // swin_spec['PATCH_SIZE'])

    extractor = MultiScaleFeatureExtractor(backbone, patch_hw=patch_hw).to(device)

    stage3_dim = stage1_dim * 4
    seg_head   = SegHead(in_dim_s1=stage1_dim, in_dim_s3=stage3_dim,
                         num_classes=num_classes, output_hw=(INPUT_H, INPUT_W)).to(device)

    ckpt = torch.load(args.seg_head_weights, map_location='cpu', weights_only=False)
    sd = {k.replace('module.', ''): v for k, v in ckpt['seg_head'].items()}
    seg_head.load_state_dict(sd)
    print(f"SegHead loaded from {args.seg_head_weights}")

    # ── 평가 ──
    preds, targets, probs_bubble, samples = run_eval(extractor, seg_head, loader, device)

    metrics = compute_metrics(preds, targets, num_classes, class_names)
    print("\n=== Metrics ===")
    for cls in class_names:
        m = metrics[cls]
        print(f"  {cls:8s}  IoU={m['IoU']:.4f}  P={m['Precision']:.4f}  R={m['Recall']:.4f}  F1={m['F1']:.4f}  "
              f"(TP={m['TP']}, FP={m['FP']}, FN={m['FN']})")
    print(f"  {'mIoU':8s}  {metrics['mIoU']:.4f}")

    ap, best_thr, best_f1 = plot_pr_curve(
        targets, probs_bubble,
        os.path.join(args.output_dir, 'pr_curve_bubble.png')
    )
    metrics['bubble_AP']       = float(ap)
    metrics['bubble_best_thr'] = float(best_thr)
    metrics['bubble_best_F1']  = float(best_f1)

    plot_confusion_matrix(preds, targets, class_names,
                          os.path.join(args.output_dir, 'confusion_matrix.png'))

    plot_samples(samples, class_names, class_colors,
                 os.path.join(args.output_dir, 'samples'))

    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nmetrics.json saved → {args.output_dir}/metrics.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Segmentation Evaluation')
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--arch', default='swin_tiny', type=str)
    parser.add_argument('--pretrained_weights', type=str, required=True)
    parser.add_argument('--checkpoint_key', default='teacher', type=str)
    parser.add_argument('--seg_head_weights', type=str, required=True)
    parser.add_argument('--block_dir', type=str, required=True)
    parser.add_argument('--label_dir', type=str, required=True)
    parser.add_argument('--stats_path', type=str, default='preprocessed/stats.json')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_classes', default=3, type=int,
                        help='3 또는 4 (4이면 unknown을 별도 클래스로 학습)')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    main(args)
