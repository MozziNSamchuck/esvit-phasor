"""
eval_2class_elim.py — 2-class(noise/tissue) 모델로 bubble 제거 정의 평가

2-class 모델(bubble=ignore로 학습)의 추론 결과에서:
  - tissue_prob > θ  → tissue
  - noise_prob  > θ  → noise   (= tissue_prob < 1-θ)
  - 그 사이 (불확실) → bubble  (모델이 tissue도 noise도 아니라고 판단)

여러 threshold(θ)를 동시에 평가해 최적 threshold 탐색.
GT는 원본 3-class (noise=0, tissue=1, bubble=2, ignore=255).

사용 예:
  python eval_2class_elim.py \
    --cfg experiments/phasor/swin_tiny_bolus_v7.yaml \
    --pretrained_weights OUTPUT/pretrain_v3/checkpoint.pth \
    --seg_head_weights OUTPUT/finetune_2class_v1/best_seg_head.pth \
    --csv_path datasets_v5/sample_index_table.csv \
    --iq_dir IQ_data/PALA_bolus --gt_dir datasets_v5/gt \
    --stats_path datasets_v2/bolus/stats_v3.json \
    --split val test --output_dir OUTPUT/eval_2class_v1 \
    --thresholds 0.5 0.6 0.7 0.8 0.9
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

import utils
from config import config
from config.default import _update_config_from_file
from finetune_seg import SegHead, MultiScaleFeatureExtractor
from models import build_model

IGNORE_INDEX = 255
CLASS_NAMES  = ['noise', 'tissue', 'bubble']
CLASS_COLORS = np.array([
    [0.15, 0.15, 0.15],
    [0.2,  0.7,  0.2 ],
    [0.2,  0.4,  0.9 ],
], dtype=float)


def colorize(label_map):
    rgb = np.full((*label_map.shape, 3), 0.55, dtype=float)
    for c, col in enumerate(CLASS_COLORS):
        rgb[label_map == c] = col
    return rgb


def vis_magnitude(tensor, stats):
    t = tensor.numpy()
    mag_norm = t[0::3]
    mag = mag_norm.mean(axis=0)
    if stats is not None:
        mag = mag * stats['mag_std'] + stats['mag_mean']
    else:
        mag = mag - mag.min()
    mag = np.clip(mag, 0, None)
    mag = np.log1p(mag)
    lo, hi = np.percentile(mag, 1), np.percentile(mag, 99)
    return np.clip((mag - lo) / (hi - lo + 1e-8), 0, 1)


def tissue_prob_map_vis(tissue_prob_np):
    """tissue_prob (H,W) float → (H,W,3) RGB heatmap"""
    cmap = plt.cm.RdYlGn
    return cmap(tissue_prob_np)[:, :, :3]


def elim_pred(tissue_prob, threshold):
    """
    tissue_prob: (B, H, W) float tensor [0,1]
    Returns (B, H, W) int64:  0=noise, 1=tissue, 2=bubble(uncertain)
    """
    pred = torch.full_like(tissue_prob, 2, dtype=torch.long)   # default: bubble
    pred[tissue_prob > threshold] = 1                           # tissue
    pred[tissue_prob < (1.0 - threshold)] = 0                  # noise
    return pred


def compute_iou_metrics(preds, targets):
    res = {}
    for c, name in enumerate(CLASS_NAMES):
        tp = int(((preds == c) & (targets == c)).sum())
        fp = int(((preds == c) & (targets != c)).sum())
        fn = int(((preds != c) & (targets == c)).sum())
        iou  = tp / (tp + fp + fn + 1e-8)
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        res[name] = {'IoU': float(iou), 'Precision': float(prec),
                     'Recall': float(rec), 'F1': float(f1)}
    res['mIoU'] = float(np.mean([res[n]['IoU'] for n in CLASS_NAMES]))
    return res


def plot_confusion_matrix(preds, targets, path):
    from sklearn.metrics import confusion_matrix
    nc = 3
    cm = confusion_matrix(targets, preds, labels=list(range(nc)))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, title, fmt in zip(axes, [cm, cm_norm],
                                    ['Count', 'Row-normalized'], ['d', '.2f']):
        im = ax.imshow(data, cmap='Blues')
        ax.set_xticks(range(nc)); ax.set_yticks(range(nc))
        ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix ({title})')
        for i in range(nc):
            for j in range(nc):
                v = data[i, j]
                ax.text(j, i, f'{v:{fmt}}', ha='center', va='center',
                        color='white' if cm_norm[i, j] > 0.5 else 'black')
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()


def save_vis(idx, tensor, gt, tissue_prob_np, preds_by_thr, thresholds, out_dir, stats):
    """
    preds_by_thr: list of (H,W) numpy arrays, one per threshold
    """
    mag = vis_magnitude(tensor, stats)
    n_thr = len(thresholds)
    ncols = 3 + n_thr  # mag / GT / tissue_prob / pred×n_thr

    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
    fig.suptitle(f'Sample {idx:04d}', fontsize=11)

    axes[0].imshow(mag, cmap='gray', aspect='auto')
    axes[0].set_title('Magnitude')

    axes[1].imshow(colorize(gt.numpy() if hasattr(gt, 'numpy') else gt), aspect='auto')
    axes[1].set_title('Ground Truth')

    axes[2].imshow(tissue_prob_map_vis(tissue_prob_np), aspect='auto')
    axes[2].set_title('Tissue Prob (R=low,G=high)')

    for i, (pred, thr) in enumerate(zip(preds_by_thr, thresholds)):
        axes[3 + i].imshow(colorize(pred), aspect='auto')
        axes[3 + i].set_title(f'Elim θ={thr}')

    patches = [mpatches.Patch(color=CLASS_COLORS[c], label=CLASS_NAMES[c])
               for c in range(3)]
    patches.append(mpatches.Patch(color=[0.55, 0.55, 0.55], label='ignore'))
    for ax in axes:
        ax.axis('off')
    fig.legend(handles=patches, loc='lower center', ncol=4, fontsize=8)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(os.path.join(out_dir, f'sample_{idx:04d}.png'), dpi=100)
    plt.close()


def load_models(args, device):
    _update_config_from_file(config, args.cfg)
    backbone = build_model(config, is_teacher=True).to(device)
    for p in backbone.parameters():
        p.requires_grad_(False)
    backbone.eval()
    utils.load_pretrained_weights(backbone, args.pretrained_weights,
                                  'teacher', 'swin_tiny', patch_size=None)

    backbone_in_ch = backbone.patch_embed.proj.weight.shape[1]
    print(f"  Backbone in_channels={backbone_in_ch}")

    swin_spec = config.MODEL.SPEC
    INPUT_H, INPUT_W = 100, 108
    patch_hw = (math.ceil(INPUT_H / swin_spec['PATCH_SIZE']),
                math.ceil(INPUT_W / swin_spec['PATCH_SIZE']))
    extractor = MultiScaleFeatureExtractor(backbone, patch_hw=patch_hw).to(device)

    s1, s2, s3 = swin_spec['DIM_EMBED'], swin_spec['DIM_EMBED']*2, swin_spec['DIM_EMBED']*4
    seg_head = SegHead(in_dim_s1=s1, in_dim_s2=s2, in_dim_s3=s3,
                       num_classes=2,                    # 2-class 모델
                       output_hw=(INPUT_H, INPUT_W)).to(device)

    ckpt = torch.load(args.seg_head_weights, map_location='cpu', weights_only=False)
    seg_head.load_state_dict(
        {k.replace('module.', ''): v for k, v in ckpt['seg_head'].items()})
    seg_head.eval()
    return extractor, seg_head, backbone_in_ch


@torch.no_grad()
def evaluate_split(split, args, extractor, seg_head, backbone_in_ch, device, stats):
    out_dir = os.path.join(args.output_dir, split)
    vis_dir = os.path.join(out_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)

    # 원본 3-class GT로 평가 (bubble=2 포함)
    from datasets.bolus_v5_dataset import BolusMATSegDataset
    ds = BolusMATSegDataset(args.csv_path, args.iq_dir, args.gt_dir,
                            split=split, stats=stats, jitter=0,
                            two_class=False)   # GT는 3-class 그대로 사용
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                        shuffle=False, pin_memory=True)
    print(f"\n[{split}] {len(ds):,} samples, thresholds={args.thresholds}")

    thresholds = args.thresholds
    all_targets = []
    all_tissue_probs = []
    all_preds = {thr: [] for thr in thresholds}
    vis_saved = 0

    for batch in loader:
        images, _ms, labels, _ig = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        bi = images[:, :backbone_in_ch] if images.shape[1] > backbone_in_ch else images
        feat_s1, feat_s2, feat_s3 = extractor(bi)
        logits = seg_head(feat_s1, feat_s2, feat_s3)    # (B, 2, H, W)
        probs  = torch.softmax(logits, dim=1)            # (B, 2, H, W)
        tissue_prob = probs[:, 1]                        # (B, H, W)

        mask = labels != IGNORE_INDEX
        all_targets.append(labels[mask].cpu().numpy())
        all_tissue_probs.append(tissue_prob[mask].cpu().numpy())

        for thr in thresholds:
            pred = elim_pred(tissue_prob, thr)
            all_preds[thr].append(pred[mask].cpu().numpy())

        # 시각화
        if vis_saved < args.vis_n:
            for i in range(images.size(0)):
                if vis_saved >= args.vis_n:
                    break
                preds_list = [elim_pred(tissue_prob[i:i+1], thr)[0].cpu().numpy()
                              for thr in thresholds]
                save_vis(vis_saved,
                         images[i].cpu(), labels[i].cpu(),
                         tissue_prob[i].cpu().numpy(),
                         preds_list, thresholds, vis_dir, stats)
                vis_saved += 1

    all_targets = np.concatenate(all_targets)
    all_tissue_probs = np.concatenate(all_tissue_probs)

    # ── metrics per threshold ──
    results = {}
    print(f"\n  {'θ':>5}  {'noise':>6}  {'tissue':>6}  {'bubble':>6}  {'mIoU':>6}  "
          f"{'bubble_P':>8}  {'bubble_R':>8}  {'bubble_F1':>9}")
    for thr in thresholds:
        preds = np.concatenate(all_preds[thr])
        m = compute_iou_metrics(preds, all_targets)
        b = m['bubble']
        print(f"  {thr:>5.2f}  {m['noise']['IoU']:.4f}  {m['tissue']['IoU']:.4f}  "
              f"{b['IoU']:.4f}  {m['mIoU']:.4f}  "
              f"{b['Precision']:.4f}    {b['Recall']:.4f}    {b['F1']:.4f}")
        results[str(thr)] = m

        plot_confusion_matrix(preds, all_targets,
                              os.path.join(out_dir, f'cm_thr{thr:.1f}.png'))

    # bubble PR curve
    from sklearn.metrics import precision_recall_curve, average_precision_score
    bubble_binary = (all_targets == 2).astype(int)
    # tissue_prob가 낮을수록 bubble → bubble score = 1 - tissue_prob (when between thresholds)
    # uncertainty score: 거리에서 0.5에 가까울수록 높음
    uncertainty = 1.0 - 2.0 * np.abs(all_tissue_probs - 0.5)  # 0.5 근처=1, 0/1 근처=0
    prec_curve, rec_curve, _ = precision_recall_curve(bubble_binary, uncertainty)
    ap = average_precision_score(bubble_binary, uncertainty)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec_curve, prec_curve, 'b-', linewidth=2, label=f'AP={ap:.3f}')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title('Bubble PR Curve (uncertainty score = 1 - 2|tissue_prob - 0.5|)')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xlim([0,1]); ax.set_ylim([0,1])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'pr_curve_bubble.png'), dpi=150)
    plt.close()
    results['bubble_AP_uncertainty'] = float(ap)

    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  → {out_dir}/  (vis: {vis_saved} samples)")
    return results


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    stats = None
    if args.stats_path and os.path.isfile(args.stats_path):
        with open(args.stats_path) as f:
            stats = json.load(f)

    print("Loading models...")
    extractor, seg_head, backbone_in_ch = load_models(args, device)

    all_results = {}
    for split in args.split:
        r = evaluate_split(split, args, extractor, seg_head, backbone_in_ch, device, stats)
        all_results[split] = r

    # threshold별 bubble IoU 요약
    print("\n" + "=" * 60)
    print("bubble IoU by threshold:")
    thrs = [str(t) for t in args.thresholds]
    for split, r in all_results.items():
        row = "  " + split + "  " + "  ".join(
            f"θ={t}: {r[t]['bubble']['IoU']:.4f}" for t in thrs)
        print(row)
    print("=" * 60)

    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser('2-class Elimination Evaluation')
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
    p.add_argument('--thresholds', type=float, nargs='+',
                   default=[0.5, 0.6, 0.7, 0.8, 0.9],
                   help='tissue/noise threshold 후보들 (θ). tissue_prob>θ→tissue, <1-θ→noise, else→bubble')
    p.add_argument('--vis_n',      type=int, default=20)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_workers',type=int, default=4)
    args = p.parse_args()
    main(args)
