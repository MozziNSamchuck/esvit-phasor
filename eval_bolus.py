"""
eval_bolus.py — v39/v44/sf_v1/sf3_v1 통합 평가 스크립트

val / test split에 대해 IoU, Precision, Recall, F1 계산,
confusion matrix / PR curve / GT·prediction 시각화 샘플 저장.

사용 예:
  python eval_bolus.py \
    --exp finetune_v44 \
    --dataset bolus_v5 \
    --cfg experiments/phasor/swin_tiny_bolus_v7.yaml \
    --pretrained_weights OUTPUT/pretrain_v3/checkpoint.pth \
    --seg_head_weights OUTPUT/finetune_v44/best_seg_head.pth \
    --csv_path datasets_v5/sample_index_table.csv \
    --iq_dir IQ_data/PALA_bolus \
    --gt_dir datasets_v5/gt \
    --stats_path datasets_v2/bolus/stats_v3.json \
    --split val test \
    --vis_n 20
"""

import argparse
import json
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import utils
from config import config
from config.default import _update_config_from_file
from finetune_seg import SegHead, MultiScaleFeatureExtractor
from models import build_model
from models.bubble_3d_module import BubbleUNet2D

IGNORE_INDEX = 255
NUM_CLASSES  = 3
CLASS_NAMES  = ['noise', 'tissue', 'bubble']
CLASS_COLORS = np.array([
    [0.15, 0.15, 0.15],   # noise    — dark gray
    [0.2,  0.7,  0.2 ],   # tissue   — green
    [0.2,  0.4,  0.9 ],   # bubble   — blue
], dtype=float)


# ── helpers ────────────────────────────────────────────────────────────────────

def colorize(label_map):
    """(H, W) int → (H, W, 3) float RGB, ignore=mid gray"""
    rgb = np.full((*label_map.shape, 3), 0.55, dtype=float)
    for c, color in enumerate(CLASS_COLORS):
        rgb[label_map == c] = color
    return rgb


def vis_magnitude(tensor, dataset_type, stats):
    """
    tensor: (C, H, W) normalized, on CPU
    Returns: (H, W) float [0,1] — log-scaled magnitude image
    """
    t = tensor.numpy()
    if dataset_type == 'bolus_v5':
        mag_norm = t[0::3]          # (20, H, W)  — every 3rd ch starting 0
    else:
        mag_norm = t[0::2]          # (n_frames, H, W)
    mag = mag_norm.mean(axis=0)     # (H, W)

    # denormalize
    if stats is not None:
        mag = mag * stats['mag_std'] + stats['mag_mean']
    else:
        mag = mag - mag.min()

    mag = np.clip(mag, 0, None)
    mag = np.log1p(mag)
    lo, hi = np.percentile(mag, 1), np.percentile(mag, 99)
    return np.clip((mag - lo) / (hi - lo + 1e-8), 0, 1)


def save_sample_vis(tensor, pred, gt, bubble_prob, idx, out_dir, dataset_type, stats):
    """4-panel: magnitude / GT / prediction / bubble probability."""
    mag  = vis_magnitude(tensor, dataset_type, stats)
    pred = pred.numpy() if hasattr(pred, 'numpy') else pred
    gt   = gt.numpy()   if hasattr(gt,   'numpy') else gt

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(f'Sample {idx:04d}', fontsize=12)

    axes[0].imshow(mag, cmap='gray', aspect='auto', vmin=0, vmax=1)
    axes[0].set_title('Magnitude (log, ch0)')

    axes[1].imshow(colorize(gt), aspect='auto')
    axes[1].set_title('Ground Truth')

    axes[2].imshow(colorize(pred), aspect='auto')
    axes[2].set_title('Prediction')

    axes[3].imshow(bubble_prob, cmap='hot', aspect='auto', vmin=0, vmax=1)
    axes[3].set_title('Bubble Probability')

    patches = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i])
               for i in range(NUM_CLASSES)]
    patches.append(mpatches.Patch(color=[0.55, 0.55, 0.55], label='ignore'))
    for ax in axes:
        ax.axis('off')
    fig.legend(handles=patches, loc='lower center', ncol=4, fontsize=9)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(os.path.join(out_dir, f'sample_{idx:04d}.png'), dpi=120)
    plt.close()


def plot_confusion_matrix(preds, targets, class_names, path):
    from sklearn.metrics import confusion_matrix
    nc = len(class_names)
    cm = confusion_matrix(targets, preds, labels=list(range(nc)))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(5 * nc, 5))
    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ['Count', 'Row-normalized'],
        ['d', '.2f']
    ):
        im = ax.imshow(data, cmap='Blues')
        ax.set_xticks(range(nc)); ax.set_yticks(range(nc))
        ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix ({title})')
        for i in range(nc):
            for j in range(nc):
                v = data[i, j]
                ax.text(j, i, f'{v:{fmt}}', ha='center', va='center',
                        color='white' if cm_norm[i, j] > 0.5 else 'black')
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_pr_curve(targets, probs_bubble, path):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    binary = (targets == 2).astype(int)
    prec, rec, thr = precision_recall_curve(binary, probs_bubble)
    ap = average_precision_score(binary, probs_bubble)

    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    best = np.argmax(f1[:-1])
    best_thr = thr[best]
    best_f1  = f1[best]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec, prec, 'b-', linewidth=2, label=f'AP={ap:.3f}')
    ax.scatter(rec[best], prec[best], s=120, zorder=5, color='red',
               label=f'Best F1={best_f1:.3f} (thr={best_thr:.2f})')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title('Bubble — Precision-Recall Curve')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return float(ap), float(best_thr), float(best_f1)


# ── model loading ──────────────────────────────────────────────────────────────

def load_models(args, device):
    _update_config_from_file(config, args.cfg)

    backbone = build_model(config, is_teacher=True).to(device)
    for p in backbone.parameters():
        p.requires_grad_(False)
    backbone.eval()
    utils.load_pretrained_weights(backbone, args.pretrained_weights,
                                  'teacher', 'swin_tiny', patch_size=None)

    # backbone이 실제로 받는 in_channels: patch_embed conv weight에서 감지
    backbone_in_ch = backbone.patch_embed.proj.weight.shape[1]
    print(f"  Backbone patch_embed in_channels={backbone_in_ch}")

    swin_spec = config.MODEL.SPEC
    INPUT_H, INPUT_W = 100, 108
    patch_hw = (
        math.ceil(INPUT_H / swin_spec['PATCH_SIZE']),
        math.ceil(INPUT_W / swin_spec['PATCH_SIZE']),
    )
    extractor = MultiScaleFeatureExtractor(backbone, patch_hw=patch_hw).to(device)

    s1_dim = swin_spec['DIM_EMBED']          # 96
    s2_dim = s1_dim * 2                       # 192
    s3_dim = s1_dim * 4                       # 384
    seg_head = SegHead(in_dim_s1=s1_dim, in_dim_s2=s2_dim, in_dim_s3=s3_dim,
                       num_classes=NUM_CLASSES,
                       output_hw=(INPUT_H, INPUT_W)).to(device)

    ckpt = torch.load(args.seg_head_weights, map_location='cpu', weights_only=False)
    seg_head.load_state_dict(
        {k.replace('module.', ''): v for k, v in ckpt['seg_head'].items()}
    )
    seg_head.eval()

    bubble_cnn   = None
    bubble_in_ch = None
    if 'bubble_cnn' in ckpt:
        # 첫 conv weight shape에서 실제 in_channels 자동 감지
        first_w = next(v for k, v in ckpt['bubble_cnn'].items()
                       if 'weight' in k and len(v.shape) == 4)
        bubble_in_ch = first_w.shape[1]
        bubble_cnn = BubbleUNet2D(in_channels=bubble_in_ch).to(device)
        bubble_cnn.load_state_dict(
            {k.replace('module.', ''): v for k, v in ckpt['bubble_cnn'].items()}
        )
        bubble_cnn.eval()
        print(f"  BubbleCNN loaded: in_channels={bubble_in_ch}")

    return extractor, seg_head, bubble_cnn, bubble_in_ch, backbone_in_ch


# ── evaluation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(extractor, seg_head, bubble_cnn, bubble_in_ch, backbone_in_ch,
             loader, device, vis_n, vis_dir, dataset_type, stats):

    intersection = np.zeros(NUM_CLASSES)
    union        = np.zeros(NUM_CLASSES)
    all_preds, all_targets, all_probs_bubble = [], [], []
    vis_saved = 0

    for batch in loader:
        images, _mag_std, labels, _ignore = batch
        images = images.to(device, non_blocking=True)   # (B, C, H, W)
        labels = labels.to(device, non_blocking=True)

        # backbone이 기대하는 채널 수로 슬라이싱 (v39: 60ch, v44/sf: 정확히 일치)
        backbone_input = images[:, :backbone_in_ch] if images.shape[1] > backbone_in_ch else images
        feat_s1, feat_s2, feat_s3 = extractor(backbone_input)
        logits = seg_head(feat_s1, feat_s2, feat_s3)    # (B, 3, H, W)

        if bubble_cnn is not None:
            bi = images[:, :bubble_in_ch]               # slice to trained in_ch
            bubble_logit = bubble_cnn(bi)               # (B, 1, H, W)
            logits = torch.cat([logits[:, :2],
                                logits[:, 2:3] + bubble_logit], dim=1)

        probs = torch.softmax(logits, dim=1)            # (B, 3, H, W)
        pred  = logits.argmax(dim=1)                    # (B, H, W)

        mask = (labels != IGNORE_INDEX)
        p = pred[mask].cpu().numpy()
        t = labels[mask].cpu().numpy()
        all_preds.append(p)
        all_targets.append(t)
        all_probs_bubble.append(probs[:, 2][mask].cpu().numpy())

        for c in range(NUM_CLASSES):
            pc = p == c; tc = t == c
            intersection[c] += (pc & tc).sum()
            union[c]        += (pc | tc).sum()

        # 시각화 저장
        if vis_saved < vis_n:
            for i in range(images.size(0)):
                if vis_saved >= vis_n:
                    break
                save_sample_vis(
                    images[i].cpu(), pred[i].cpu(), labels[i].cpu(),
                    probs[i, 2].cpu().numpy(),
                    vis_saved, vis_dir, dataset_type, stats
                )
                vis_saved += 1

    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_probs_bubble = np.concatenate(all_probs_bubble)

    iou = np.where(union > 0, intersection / union, 0.0)
    return iou, all_preds, all_targets, all_probs_bubble


# ── per-split runner ───────────────────────────────────────────────────────────

def run_split(split, args, extractor, seg_head, bubble_cnn, bubble_in_ch, backbone_in_ch,
              device, stats):
    out_dir = os.path.join(args.output_dir, split)
    vis_dir = os.path.join(out_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)

    # 데이터셋
    if args.dataset in ('bolus_sf', 'bolus_sf3'):
        from datasets.bolus_sf_dataset import BolusSFSegDataset
        n_frames = 3 if args.dataset == 'bolus_sf3' else 1
        ds = BolusSFSegDataset(args.csv_path, args.iq_dir, args.gt_dir,
                               split=split, stats=stats, n_frames=n_frames)
    else:
        from datasets.bolus_v5_dataset import BolusMATSegDataset
        ds = BolusMATSegDataset(args.csv_path, args.iq_dir, args.gt_dir,
                                split=split, stats=stats, jitter=0)

    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                        shuffle=False, pin_memory=True)
    print(f"\n[{split}] {len(ds):,} samples")

    iou, preds, targets, probs_bubble = evaluate(
        extractor, seg_head, bubble_cnn, bubble_in_ch, backbone_in_ch,
        loader, device, args.vis_n, vis_dir, args.dataset, stats
    )

    # ── metrics ──
    results = {}
    print(f"\n{'class':>8}  {'IoU':>6}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}")
    for c, name in enumerate(CLASS_NAMES):
        tp = ((preds == c) & (targets == c)).sum()
        fp = ((preds == c) & (targets != c)).sum()
        fn = ((preds != c) & (targets == c)).sum()
        p_val = tp / (tp + fp + 1e-8)
        r_val = tp / (tp + fn + 1e-8)
        f1    = 2 * p_val * r_val / (p_val + r_val + 1e-8)
        print(f"{name:>8}  {iou[c]:.4f}  {p_val:.4f}  {r_val:.4f}  {f1:.4f}")
        results[name] = {'IoU': float(iou[c]), 'Precision': float(p_val),
                         'Recall': float(r_val), 'F1': float(f1)}
    miou = float(iou.mean())
    print(f"{'mIoU':>8}  {miou:.4f}")
    results['mIoU'] = miou

    ap, best_thr, best_f1 = plot_pr_curve(
        targets, probs_bubble, os.path.join(out_dir, 'pr_curve_bubble.png')
    )
    results['bubble_AP']  = ap
    results['bubble_best_thr'] = best_thr
    results['bubble_best_F1']  = best_f1
    print(f"  Bubble AP={ap:.4f}  best_F1={best_f1:.4f} @ thr={best_thr:.3f}")

    plot_confusion_matrix(preds, targets, CLASS_NAMES,
                          os.path.join(out_dir, 'confusion_matrix.png'))

    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  → {out_dir}/metrics.json")
    print(f"  → {vis_dir}/ ({args.vis_n} samples)")
    return results


# ── main ───────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # stats
    stats = None
    if args.stats_path and os.path.isfile(args.stats_path):
        with open(args.stats_path) as f:
            stats = json.load(f)
        print(f"Stats: {args.stats_path}")

    print("Loading models...")
    extractor, seg_head, bubble_cnn, bubble_in_ch, backbone_in_ch = load_models(args, device)

    all_results = {}
    for split in args.split:
        r = run_split(split, args, extractor, seg_head, bubble_cnn, bubble_in_ch, backbone_in_ch,
                      device, stats)
        all_results[split] = r

    # 요약 출력
    print("\n" + "=" * 55)
    print(f"{'':10} {'':>6}  {'noise':>6}  {'tissue':>6}  {'bubble':>6}")
    for split, r in all_results.items():
        print(f"{split:10} {'mIoU':>6}  "
              f"{r.get('noise',{}).get('IoU',0):.4f}  "
              f"{r.get('tissue',{}).get('IoU',0):.4f}  "
              f"{r.get('bubble',{}).get('IoU',0):.4f}   mIoU={r['mIoU']:.4f}")
    print("=" * 55)

    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser('EsViT Bolus Segmentation Evaluation')
    p.add_argument('--exp',      type=str, required=True, help='실험 이름 (출력용)')
    p.add_argument('--dataset',  type=str, required=True,
                   choices=['bolus_v5', 'bolus_sf', 'bolus_sf3'])
    p.add_argument('--cfg',      type=str, required=True)
    p.add_argument('--pretrained_weights', type=str, required=True)
    p.add_argument('--seg_head_weights',   type=str, required=True)
    p.add_argument('--csv_path',   type=str, required=True)
    p.add_argument('--iq_dir',     type=str, required=True)
    p.add_argument('--gt_dir',     type=str, required=True)
    p.add_argument('--stats_path', type=str, default='')
    p.add_argument('--split',      type=str, nargs='+', default=['val', 'test'],
                   choices=['val', 'test', 'ft_train'])
    p.add_argument('--output_dir', type=str, required=True)
    p.add_argument('--vis_n',      type=int, default=20,
                   help='시각화 저장 샘플 수 (split당)')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_workers',type=int, default=4)
    args = p.parse_args()
    main(args)
