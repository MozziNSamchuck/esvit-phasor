"""
eval_elimination.py — Tissue/Noise Elimination으로 Bubble 정의하는 평가

아이디어:
  모델이 확신하는 tissue / noise 픽셀을 제거(eliminate)한 나머지를 bubble로 정의.
  - standard  : argmax(SegHead + BubbleCNN logits)  — 기존 방식
  - elim_seg  : SegHead logits만 사용. tissue/noise 각각의 확률이 threshold 미만이면 → bubble
  - elim_full : SegHead + BubbleCNN logits 사용. 동일 thresholding

  시각화: magnitude / GT / standard / elim_seg / elim_full / 차이맵

사용 예:
  python eval_elimination.py \
    --exp finetune_v44 \
    --cfg experiments/phasor/swin_tiny_bolus_v7.yaml \
    --pretrained_weights OUTPUT/pretrain_v3/checkpoint.pth \
    --seg_head_weights OUTPUT/finetune_v44/best_seg_head.pth \
    --csv_path datasets_v5/sample_index_table.csv \
    --iq_dir IQ_data/PALA_bolus --gt_dir datasets_v5/gt \
    --stats_path datasets_v2/bolus/stats_v3.json \
    --split val test --output_dir OUTPUT/elim_v44 --threshold 0.5
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
from models.bubble_3d_module import BubbleUNet2D

IGNORE_INDEX = 255
NUM_CLASSES  = 3
CLASS_NAMES  = ['noise', 'tissue', 'bubble']
CLASS_COLORS = np.array([
    [0.15, 0.15, 0.15],   # noise  — dark gray
    [0.2,  0.7,  0.2 ],   # tissue — green
    [0.2,  0.4,  0.9 ],   # bubble — blue
], dtype=float)
DIFF_COLOR   = np.array([0.9, 0.2, 0.2], dtype=float)   # red: 예측 불일치


# ── helpers ────────────────────────────────────────────────────────────────────

def colorize(label_map):
    """(H, W) int → (H, W, 3) float; ignore=mid gray"""
    rgb = np.full((*label_map.shape, 3), 0.55, dtype=float)
    for c, col in enumerate(CLASS_COLORS):
        rgb[label_map == c] = col
    return rgb


def diff_map(pred_a, pred_b, gt):
    """
    pred_a vs pred_b 차이 시각화.
    - agree & correct  : class color
    - agree & wrong    : dark gray
    - disagree & a_right: 파랑(b가 틀림)
    - disagree & b_right: 빨강(a가 틀림)
    - disagree & both wrong: 노랑
    """
    h, w = pred_a.shape
    rgb = np.full((h, w, 3), 0.55, dtype=float)
    agree    = pred_a == pred_b
    a_right  = pred_a == gt
    b_right  = pred_b == gt
    valid    = gt != IGNORE_INDEX

    # agree & correct → class color
    for c, col in enumerate(CLASS_COLORS):
        mask = agree & a_right & (gt == c) & valid
        rgb[mask] = col
    # agree & wrong → dark
    rgb[agree & ~a_right & valid] = [0.2, 0.2, 0.2]
    # disagree: only b right (standard wrong, elim right)
    rgb[~agree & ~a_right & b_right & valid] = [0.0, 0.8, 0.4]   # green
    # disagree: only a right (standard right, elim wrong)
    rgb[~agree & a_right & ~b_right & valid] = [0.9, 0.3, 0.1]   # red-orange
    # disagree: both wrong
    rgb[~agree & ~a_right & ~b_right & valid] = [0.9, 0.8, 0.0]  # yellow
    return rgb


def vis_magnitude(tensor, stats):
    """(C, H, W) normalized tensor → (H, W) float [0,1] log-scale magnitude."""
    t = tensor.numpy()
    mag_norm = t[0::3]    # magnitude channels (every 3rd ch for 20-frame)
    if mag_norm.shape[0] == 0:
        mag_norm = t[0::2]
    mag = mag_norm.mean(axis=0)
    if stats is not None:
        mag = mag * stats['mag_std'] + stats['mag_mean']
    else:
        mag = mag - mag.min()
    mag = np.clip(mag, 0, None)
    mag = np.log1p(mag)
    lo, hi = np.percentile(mag, 1), np.percentile(mag, 99)
    return np.clip((mag - lo) / (hi - lo + 1e-8), 0, 1)


def save_vis(idx, tensor, gt, pred_std, pred_elim_seg, pred_elim_full,
             out_dir, stats):
    mag = vis_magnitude(tensor, stats)
    arrays = {
        'gt':             gt.numpy(),
        'std':            pred_std.numpy(),
        'elim_seg':       pred_elim_seg.numpy(),
        'elim_full':      pred_elim_full.numpy(),
    }

    titles = [
        'Magnitude', 'Ground Truth',
        'Standard (argmax+BubbleCNN)',
        'Elim-SegHead (no BubbleCNN)',
        'Elim-Full (SegHead+BubbleCNN)',
        'Elim-Full vs Standard\n(green=elim↑ red=elim↓)',
    ]
    imgs = [
        mag,
        colorize(arrays['gt']),
        colorize(arrays['std']),
        colorize(arrays['elim_seg']),
        colorize(arrays['elim_full']),
        diff_map(arrays['std'], arrays['elim_full'], arrays['gt']),
    ]

    fig, axes = plt.subplots(1, 6, figsize=(26, 4))
    fig.suptitle(f'Sample {idx:04d}', fontsize=11)
    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img if img.ndim == 3 else img, cmap='gray' if img.ndim == 2 else None,
                  aspect='auto', vmin=0 if img.ndim == 2 else None, vmax=1 if img.ndim == 2 else None)
        ax.set_title(title, fontsize=8)
        ax.axis('off')

    patches = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i])
               for i in range(NUM_CLASSES)]
    patches.append(mpatches.Patch(color=[0.55, 0.55, 0.55], label='ignore'))
    fig.legend(handles=patches, loc='lower center', ncol=4, fontsize=8)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(os.path.join(out_dir, f'sample_{idx:04d}.png'), dpi=110)
    plt.close()


# ── IoU helper ─────────────────────────────────────────────────────────────────

def compute_iou_metrics(preds, targets):
    """Returns dict with per-class IoU, Precision, Recall, F1 + mIoU."""
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


def print_metrics(label, m):
    print(f"  [{label}]")
    for name in CLASS_NAMES:
        r = m[name]
        print(f"    {name:>8}  IoU={r['IoU']:.4f}  P={r['Precision']:.4f}"
              f"  R={r['Recall']:.4f}  F1={r['F1']:.4f}")
    print(f"    {'mIoU':>8}  {m['mIoU']:.4f}")


def plot_confusion_matrix(preds, targets, class_names, path):
    from sklearn.metrics import confusion_matrix
    nc = len(class_names)
    cm = confusion_matrix(targets, preds, labels=list(range(nc)))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    fig, axes = plt.subplots(1, 2, figsize=(5 * nc, 5))
    for ax, data, title, fmt in zip(axes, [cm, cm_norm],
                                    ['Count', 'Row-normalized'], ['d', '.2f']):
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
    plt.savefig(path, dpi=150); plt.close()


# ── model loading ──────────────────────────────────────────────────────────────

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
                       num_classes=NUM_CLASSES,
                       output_hw=(INPUT_H, INPUT_W)).to(device)

    ckpt = torch.load(args.seg_head_weights, map_location='cpu', weights_only=False)
    seg_head.load_state_dict(
        {k.replace('module.', ''): v for k, v in ckpt['seg_head'].items()})
    seg_head.eval()

    bubble_cnn = bubble_in_ch = None
    if 'bubble_cnn' in ckpt:
        first_w = next(v for k, v in ckpt['bubble_cnn'].items()
                       if 'weight' in k and len(v.shape) == 4)
        bubble_in_ch = first_w.shape[1]
        bubble_cnn = BubbleUNet2D(in_channels=bubble_in_ch).to(device)
        bubble_cnn.load_state_dict(
            {k.replace('module.', ''): v for k, v in ckpt['bubble_cnn'].items()})
        bubble_cnn.eval()
        print(f"  BubbleCNN in_channels={bubble_in_ch}")

    return extractor, seg_head, bubble_cnn, bubble_in_ch, backbone_in_ch


# ── elimination prediction ─────────────────────────────────────────────────────

def elimination_pred(logits, threshold):
    """
    logits: (B, 3, H, W)
    threshold: float — tissue/noise 확률이 이 값 미만이면 bubble로 분류

    반환: (B, H, W) int64
    - tissue if tissue_prob > threshold
    - noise  if noise_prob  > threshold (and not tissue)
    - bubble otherwise (elimination: NOT confident tissue AND NOT confident noise)
    """
    probs = torch.softmax(logits, dim=1)          # (B, 3, H, W)
    noise_p  = probs[:, 0]                         # (B, H, W)
    tissue_p = probs[:, 1]

    pred = torch.full(noise_p.shape, 2, dtype=torch.long, device=logits.device)  # 기본: bubble
    pred[tissue_p > threshold] = 1                 # tissue
    pred[(noise_p > threshold) & (tissue_p <= threshold)] = 0  # noise
    return pred


# ── evaluation loop ────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_split(split, args, extractor, seg_head, bubble_cnn,
                   bubble_in_ch, backbone_in_ch, device, stats):
    out_dir = os.path.join(args.output_dir, split)
    vis_dir = os.path.join(out_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)

    from datasets.bolus_v5_dataset import BolusMATSegDataset
    ds = BolusMATSegDataset(args.csv_path, args.iq_dir, args.gt_dir,
                            split=split, stats=stats, jitter=0)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                        shuffle=False, pin_memory=True)
    print(f"\n[{split}] {len(ds):,} samples  threshold={args.threshold}")

    all_std  = []
    all_eseg = []
    all_efull = []
    all_targets = []
    vis_saved = 0

    for batch in loader:
        images, _ms, labels, _ig = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # backbone forward (60ch slice for v40)
        bi = images[:, :backbone_in_ch] if images.shape[1] > backbone_in_ch else images
        feat_s1, feat_s2, feat_s3 = extractor(bi)
        logits_seg = seg_head(feat_s1, feat_s2, feat_s3)    # (B, 3, H, W)

        # BubbleCNN logits
        if bubble_cnn is not None:
            bub_in = images[:, :bubble_in_ch]
            bubble_logit = bubble_cnn(bub_in)               # (B, 1, H, W)
            logits_full  = torch.cat([logits_seg[:, :2],
                                      logits_seg[:, 2:3] + bubble_logit], dim=1)
        else:
            logits_full = logits_seg

        # predictions
        pred_std       = logits_full.argmax(dim=1)                         # standard
        pred_elim_seg  = elimination_pred(logits_seg,  args.threshold)     # SegHead only
        pred_elim_full = elimination_pred(logits_full, args.threshold)     # SegHead+BubbleCNN

        mask = labels != IGNORE_INDEX
        all_std.append(pred_std[mask].cpu().numpy())
        all_eseg.append(pred_elim_seg[mask].cpu().numpy())
        all_efull.append(pred_elim_full[mask].cpu().numpy())
        all_targets.append(labels[mask].cpu().numpy())

        # 시각화
        if vis_saved < args.vis_n:
            probs_full = torch.softmax(logits_full, dim=1)
            for i in range(images.size(0)):
                if vis_saved >= args.vis_n:
                    break
                save_vis(vis_saved,
                         images[i].cpu(), labels[i].cpu(),
                         pred_std[i].cpu(),
                         pred_elim_seg[i].cpu(),
                         pred_elim_full[i].cpu(),
                         vis_dir, stats)
                vis_saved += 1

    all_targets = np.concatenate(all_targets)
    m_std  = compute_iou_metrics(np.concatenate(all_std),  all_targets)
    m_eseg = compute_iou_metrics(np.concatenate(all_eseg), all_targets)
    m_efull= compute_iou_metrics(np.concatenate(all_efull),all_targets)

    print("\n  Metrics:")
    print_metrics("Standard (argmax+BubbleCNN)", m_std)
    print_metrics(f"Elim-SegHead (thr={args.threshold})", m_eseg)
    print_metrics(f"Elim-Full    (thr={args.threshold})", m_efull)

    # confusion matrices
    plot_confusion_matrix(np.concatenate(all_std), all_targets, CLASS_NAMES,
                          os.path.join(out_dir, 'cm_standard.png'))
    plot_confusion_matrix(np.concatenate(all_efull), all_targets, CLASS_NAMES,
                          os.path.join(out_dir, 'cm_elim_full.png'))

    results = {
        'threshold': args.threshold,
        'standard':  m_std,
        'elim_seg':  m_eseg,
        'elim_full': m_efull,
    }
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  → {out_dir}/metrics.json  |  vis: {vis_saved} samples")
    return results


# ── main ───────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    stats = None
    if args.stats_path and os.path.isfile(args.stats_path):
        with open(args.stats_path) as f:
            stats = json.load(f)
        print(f"Stats: {args.stats_path}")

    print("Loading models...")
    extractor, seg_head, bubble_cnn, bubble_in_ch, backbone_in_ch = load_models(args, device)

    all_results = {}
    for split in args.split:
        r = evaluate_split(split, args, extractor, seg_head, bubble_cnn,
                           bubble_in_ch, backbone_in_ch, device, stats)
        all_results[split] = r

    # 요약
    print("\n" + "=" * 70)
    print(f"{'':15} {'method':20} {'noise':>6} {'tissue':>6} {'bubble':>6} {'mIoU':>6}")
    for split, r in all_results.items():
        for mkey in ['standard', 'elim_seg', 'elim_full']:
            m = r[mkey]
            print(f"  {split:10} {mkey:20} "
                  f"{m['noise']['IoU']:.4f} {m['tissue']['IoU']:.4f} "
                  f"{m['bubble']['IoU']:.4f} {m['mIoU']:.4f}")
        print()
    print("=" * 70)

    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser('Tissue/Noise Elimination Evaluation')
    p.add_argument('--exp',      type=str, required=True)
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
    p.add_argument('--threshold',  type=float, default=0.5,
                   help='tissue/noise 확률 threshold. 미만이면 bubble로 분류 (기본 0.5)')
    p.add_argument('--vis_n',      type=int, default=20)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_workers',type=int, default=4)
    args = p.parse_args()
    main(args)
