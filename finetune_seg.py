"""
EsViT Phasor Segmentation Fine-tuning (Linear Probe)

사용법:
  torchrun --nproc_per_node=8 finetune_seg.py \
    --pretrained_weights OUTPUT/phasor_pretrain_v7/checkpoint0099.pth \
    --block_dir_train preprocessed/train/bolus \
    --block_dir_val   preprocessed/val/bolus \
    --label_dir       label/bolus \
    --stats_path      preprocessed/stats.json \
    --output_dir      OUTPUT/seg_finetune_v1 \
    --cfg experiments/phasor/swin_tiny_phasor.yaml

세부 사항:
  - Backbone (Swin-Tiny) 완전 freeze
  - Stage 1 마지막 블록 출력 (B, 96, 20, 30) 을 hook 으로 캡처
  - SegHead (1×1 Conv) 로 3클래스 예측 → bilinear upsample → (B, 3, 80, 120)
  - CrossEntropyLoss(ignore_index=255)
  - 평가지표: mean IoU (3클래스)
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset

import math
import utils
from models import build_model
from config import config, update_config
from datasets.block_dataset import BlockSegDataset

NUM_CLASSES  = 3   # 4-class 실험 시 args.num_classes로 덮어씀
IGNORE_INDEX = 255


class DistributedWeightedSampler(torch.utils.data.Sampler):
    """
    가중치 기반 샘플링 + 분산 학습 지원.
    모든 rank가 동일한 시드로 전체 인덱스를 샘플링한 뒤 rank별로 분배.
    """
    def __init__(self, weights, num_samples, num_replicas, rank, replacement=True):
        self.weights        = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples    = num_samples
        self.num_replicas   = num_replicas
        self.rank           = rank
        self.replacement    = replacement
        self.epoch          = 0
        self.num_per_rank   = math.ceil(num_samples / num_replicas)
        self.total_size     = self.num_per_rank * num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.num_per_rank

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.multinomial(self.weights, self.total_size,
                                    replacement=self.replacement, generator=g).tolist()
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)


class DiceLoss(nn.Module):
    """
    Multiclass Dice Loss.
    각 클래스별 Dice를 계산한 뒤 평균.
    ignore_index 위치는 마스킹해서 제외.
    class_weights: (num_classes,) tensor — 클래스별 가중치 (None이면 균등)
    """
    def __init__(self, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX,
                 class_weights=None, smooth=1e-6):
        super().__init__()
        self.num_classes   = num_classes
        self.ignore_index  = ignore_index
        self.class_weights = class_weights   # (C,) tensor or None
        self.smooth        = smooth

    def forward(self, logits, targets):
        # logits: (B, C, H, W)  targets: (B, H, W)
        probs = torch.softmax(logits, dim=1)          # (B, C, H, W)
        mask  = (targets != self.ignore_index)        # (B, H, W)

        dice_total, weight_total = 0.0, 0.0
        for c in range(self.num_classes):
            p = probs[:, c][mask]                     # 유효 픽셀만
            t = (targets[mask] == c).float()
            inter = (p * t).sum()
            union = p.sum() + t.sum()
            dice_c = (2 * inter + self.smooth) / (union + self.smooth)

            w = self.class_weights[c].item() if self.class_weights is not None else 1.0
            dice_total  += w * (1 - dice_c)
            weight_total += w

        return dice_total / weight_total


def binary_dice_loss(logit, target, mask, smooth=1.0):
    """
    Binary Dice Loss for BubbleCNN.
    logit:  (N,) — masked 1D, raw logit
    target: (N,) — masked 1D, float 0/1
    mask:   ignore(255) 제외 후 전달된 유효 픽셀 기준
    """
    pred  = torch.sigmoid(logit)
    inter = (pred * target).sum()
    return 1.0 - (2.0 * inter + smooth) / (pred.sum() + target.sum() + smooth)


class FocalLoss(nn.Module):
    """
    Focal Loss with per-class weights.
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """
    def __init__(self, gamma=2.0, weight=None, ignore_index=IGNORE_INDEX):
        super().__init__()
        self.gamma        = gamma
        self.weight       = weight       # (num_classes,) tensor or None
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # logits: (B, C, H, W)  targets: (B, H, W)
        ce = F.cross_entropy(logits, targets,
                             weight=self.weight,
                             ignore_index=self.ignore_index,
                             reduction='none')          # (B, H, W)
        # p_t: probability of the correct class
        with torch.no_grad():
            p_t = torch.exp(-ce)                        # (B, H, W)
        focal_weight = (1 - p_t) ** self.gamma
        loss = focal_weight * ce
        # ignore_index 위치 마스킹
        mask = targets != self.ignore_index
        return loss[mask].mean()


# ── Segmentation head ─────────────────────────────────────────────────────────

class SegHead(nn.Module):
    """
    Multi-scale decoder: Stage1(96ch) + Stage2(192ch) + Stage3(384ch) 융합.
    Stage2/3 → 1×1 Conv(96ch) → upsample to Stage1 해상도
    → concat(96×3=288ch) → 3×3 Conv × 2 → 1×1 Conv(num_classes) → upsample to output
    """
    def __init__(self, in_dim_s1=96, in_dim_s2=192, in_dim_s3=384,
                 num_classes=NUM_CLASSES, output_hw=(80, 120), mid_dim=128):
        super().__init__()
        self.s2_reduce = nn.Conv2d(in_dim_s2, in_dim_s1, kernel_size=1)
        self.s3_reduce = nn.Conv2d(in_dim_s3, in_dim_s1, kernel_size=1)
        fused_dim = in_dim_s1 * 3
        self.decoder = nn.Sequential(
            nn.Conv2d(fused_dim, mid_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, mid_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, num_classes, kernel_size=1),
        )
        self.output_hw = output_hw

    def forward(self, feat_s1, feat_s2, feat_s3):
        s2 = self.s2_reduce(feat_s2)
        s2 = F.interpolate(s2, size=feat_s1.shape[-2:],
                           mode='bilinear', align_corners=False)
        s3 = self.s3_reduce(feat_s3)
        s3 = F.interpolate(s3, size=feat_s1.shape[-2:],
                           mode='bilinear', align_corners=False)
        x = torch.cat([feat_s1, s2, s3], dim=1)
        x = self.decoder(x)
        x = F.interpolate(x, size=self.output_hw,
                          mode='bilinear', align_corners=False)
        return x


class ClassSpecificSegHead(nn.Module):
    """
    클래스별로 다른 feature scale을 사용하는 decoder.
    noise:  s3 (7×7, 384ch)        — 전체 배경 패턴
    tissue: s1 (25×27, 96ch)       — 고정 위치, 세밀한 경계
    bubble: s1+s2+s3 (all scales)  — 분산된 위치, 다양한 스케일
    각 head → 1ch logit → concat → (B, 3, H, W)
    """
    def __init__(self, in_dim_s1=96, in_dim_s2=192, in_dim_s3=384,
                 output_hw=(100, 108), mid_dim=64):
        super().__init__()
        self.noise_head = nn.Sequential(
            nn.Conv2d(in_dim_s3, mid_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, 1, kernel_size=1),
        )
        self.tissue_head = nn.Sequential(
            nn.Conv2d(in_dim_s1, mid_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, 1, kernel_size=1),
        )
        self.bubble_s2_reduce = nn.Conv2d(in_dim_s2, in_dim_s1, kernel_size=1)
        self.bubble_s3_reduce = nn.Conv2d(in_dim_s3, in_dim_s1, kernel_size=1)
        self.bubble_head = nn.Sequential(
            nn.Conv2d(in_dim_s1 * 3, mid_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, 1, kernel_size=1),
        )
        self.output_hw = output_hw

    def forward(self, feat_s1, feat_s2, feat_s3):
        noise = self.noise_head(feat_s3)
        noise = F.interpolate(noise, size=self.output_hw, mode='bilinear', align_corners=False)

        tissue = self.tissue_head(feat_s1)
        tissue = F.interpolate(tissue, size=self.output_hw, mode='bilinear', align_corners=False)

        s2 = self.bubble_s2_reduce(feat_s2)
        s2 = F.interpolate(s2, size=feat_s1.shape[-2:], mode='bilinear', align_corners=False)
        s3 = self.bubble_s3_reduce(feat_s3)
        s3 = F.interpolate(s3, size=feat_s1.shape[-2:], mode='bilinear', align_corners=False)
        bubble = self.bubble_head(torch.cat([feat_s1, s2, s3], dim=1))
        bubble = F.interpolate(bubble, size=self.output_hw, mode='bilinear', align_corners=False)

        return torch.cat([noise, tissue, bubble], dim=1)


# ── Feature extractor (backbone + hook) ──────────────────────────────────────

class MultiScaleFeatureExtractor(nn.Module):
    """
    Swin backbone Stage1 + Stage2 + Stage3 출력을 hook으로 캡처.
    Backbone은 완전 freeze (no_grad).
    """
    def __init__(self, backbone, patch_hw=(20, 30)):
        super().__init__()
        self.backbone  = backbone
        self.patch_hw  = patch_hw
        self._feat_s1  = None
        self._feat_s2  = None
        self._feat_s3  = None
        backbone.layers[0].blocks[-1].register_forward_hook(self._hook_s1)
        backbone.layers[1].blocks[-1].register_forward_hook(self._hook_s2)
        backbone.layers[2].blocks[-1].register_forward_hook(self._hook_s3)

    def _hook_s1(self, module, input, output):
        x, _ = output
        B, L, C = x.shape
        H, W = self.patch_hw
        self._feat_s1 = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

    def _hook_s2(self, module, input, output):
        x, _ = output
        B, L, C = x.shape
        H = math.ceil(self.patch_hw[0] / 2)
        W = L // H
        self._feat_s2 = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

    def _hook_s3(self, module, input, output):
        x, _ = output
        B, L, C = x.shape
        # Swin PatchMerging은 홀수 크기를 패딩 후 절반으로 줄임 → ceil 사용
        H = math.ceil(math.ceil(self.patch_hw[0] / 2) / 2)
        W = L // H
        self._feat_s3 = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

    def forward(self, x):
        if any(p.requires_grad for p in self.backbone.parameters()):
            self.backbone.forward_features(x)
        else:
            with torch.no_grad():
                self.backbone.forward_features(x)
        return self._feat_s1, self._feat_s2, self._feat_s3


# ── mIoU metric ───────────────────────────────────────────────────────────────

class IoUMetric:
    def __init__(self, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX):
        self.num_classes    = num_classes
        self.ignore_index   = ignore_index
        self.intersection   = np.zeros(num_classes)
        self.union          = np.zeros(num_classes)

    def update(self, pred, target):
        """
        pred:   (B, H, W) int64, 예측 클래스
        target: (B, H, W) int64
        """
        mask = (target != self.ignore_index)
        pred   = pred[mask].cpu().numpy()
        target = target[mask].cpu().numpy()
        for c in range(self.num_classes):
            p = pred   == c
            t = target == c
            self.intersection[c] += (p & t).sum()
            self.union[c]        += (p | t).sum()

    def iou_per_class(self):
        iou = np.zeros(self.num_classes)
        for c in range(self.num_classes):
            if self.union[c] > 0:
                iou[c] = self.intersection[c] / self.union[c]
        return iou

    def mean_iou(self):
        return self.iou_per_class().mean()

    def reset(self):
        self.intersection[:] = 0
        self.union[:]        = 0


# ── Training / validation ─────────────────────────────────────────────────────

def train_one_epoch(extractor, seg_head, loader, optimizer, epoch, args, criterion,
                    bubble_cnn=None):
    seg_head.train()
    extractor.eval()
    if bubble_cnn is not None:
        bubble_cnn.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    use_mag_std   = getattr(args, 'use_mag_std', False)
    soft_ignore   = getattr(args, 'soft_ignore', False)
    soft_weight   = getattr(args, 'soft_ignore_weight', 0.3)

    for images, mag_std, labels, ignore_mask in metric_logger.log_every(loader, 20, f'Epoch [{epoch}]'):
        images      = images.cuda(non_blocking=True)
        mag_std     = mag_std.cuda(non_blocking=True)
        labels      = labels.cuda(non_blocking=True)
        ignore_mask = ignore_mask.cuda(non_blocking=True)   # (B, H, W) bool

        bubble_input = torch.cat([images, mag_std], dim=1) if use_mag_std else images

        feat_s1, feat_s2, feat_s3 = extractor(images)
        logits = seg_head(feat_s1, feat_s2, feat_s3)

        if args.separate_bubble and bubble_cnn is not None:
            # SegHead: bubble 픽셀은 ignore로 마스킹, noise/tissue만 학습
            seg_label = labels.clone()
            seg_label[labels == 2] = IGNORE_INDEX
            seg_loss = criterion(logits, seg_label, ignore_mask, soft_weight)

            # BubbleCNN: binary BCE (bubble=1, 나머지=0)
            bubble_logit = bubble_cnn(bubble_input).squeeze(1)  # (B, H, W)
            bubble_gt    = (labels == 2).float()
            bubble_mask  = (labels != IGNORE_INDEX)
            bubble_loss  = F.binary_cross_entropy_with_logits(
                bubble_logit[bubble_mask], bubble_gt[bubble_mask]
            )
            loss = seg_loss + args.bubble_lambda * bubble_loss
        else:
            if bubble_cnn is not None:
                bubble_logit = bubble_cnn(bubble_input)         # (B, 1, H, W)
                if NUM_CLASSES == 4:
                    logits = torch.cat([logits[:, :2],
                                        logits[:, 2:3] + bubble_logit,
                                        logits[:, 3:4]], dim=1)
                else:
                    logits = torch.cat([logits[:, :2], logits[:, 2:3] + bubble_logit], dim=1)
                if getattr(args, 'bubble_dice', False):
                    bl          = bubble_logit.squeeze(1)       # (B, H, W)
                    bubble_gt   = (labels == 2).float()
                    bubble_mask = (labels != IGNORE_INDEX)
                    bce  = F.binary_cross_entropy_with_logits(
                        bl[bubble_mask], bubble_gt[bubble_mask])
                    dice = binary_dice_loss(bl[bubble_mask], bubble_gt[bubble_mask],
                                            bubble_mask)
                    loss = criterion(logits, labels, ignore_mask, soft_weight) + bce + dice
                else:
                    loss = criterion(logits, labels, ignore_mask, soft_weight)
            else:
                loss = criterion(logits, labels, ignore_mask, soft_weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

    metric_logger.synchronize_between_processes()
    return {k: m.global_avg for k, m in metric_logger.meters.items()}


@torch.no_grad()
def validate(extractor, seg_head, loader, criterion, bubble_cnn=None, args=None):
    seg_head.eval()
    extractor.eval()
    if bubble_cnn is not None:
        bubble_cnn.eval()
    metric    = IoUMetric(num_classes=NUM_CLASSES)
    total_loss, n = 0.0, 0
    separate_bubble    = args is not None and getattr(args, 'separate_bubble', False)
    hard_bubble_fusion = args is not None and getattr(args, 'hard_bubble_fusion', False)

    use_mag_std = args is not None and getattr(args, 'use_mag_std', False)

    for images, mag_std, labels, ignore_mask in loader:
        images      = images.cuda(non_blocking=True)
        mag_std     = mag_std.cuda(non_blocking=True)
        labels      = labels.cuda(non_blocking=True)
        ignore_mask = ignore_mask.cuda(non_blocking=True)   # val: 모두 False

        bubble_input = torch.cat([images, mag_std], dim=1) if use_mag_std else images

        feat_s1, feat_s2, feat_s3 = extractor(images)
        logits = seg_head(feat_s1, feat_s2, feat_s3)

        if separate_bubble and bubble_cnn is not None:
            seg_label = labels.clone()
            seg_label[labels == 2] = IGNORE_INDEX
            seg_loss = criterion(logits, seg_label, ignore_mask, 0.0)

            bubble_logit = bubble_cnn(bubble_input).squeeze(1)  # (B, H, W)
            bubble_gt    = (labels == 2).float()
            bubble_mask  = (labels != IGNORE_INDEX)
            bubble_loss  = F.binary_cross_entropy_with_logits(
                bubble_logit[bubble_mask], bubble_gt[bubble_mask]
            )
            loss = seg_loss + getattr(args, 'bubble_lambda', 1.0) * bubble_loss

            if hard_bubble_fusion:
                # noise/tissue argmax → bubble_prob > 0.5인 위치를 2로 덮어씀
                pred = logits[:, :2].argmax(dim=1)              # (B, H, W): 0 or 1
                bubble_prob = torch.sigmoid(bubble_logit)       # (B, H, W)
                pred = pred.clone()
                pred[bubble_prob > 0.5] = 2
            else:
                logits = torch.cat([logits[:, :2], bubble_logit.unsqueeze(1)], dim=1)
                pred = logits.argmax(dim=1)
        else:
            if bubble_cnn is not None:
                bubble_logit = bubble_cnn(bubble_input)         # (B, 1, H, W)
                if NUM_CLASSES == 4:
                    logits = torch.cat([logits[:, :2],
                                        logits[:, 2:3] + bubble_logit,
                                        logits[:, 3:4]], dim=1)
                else:
                    logits = torch.cat([logits[:, :2], logits[:, 2:3] + bubble_logit], dim=1)
                if getattr(args, 'bubble_dice', False):
                    bl          = bubble_logit.squeeze(1)
                    bubble_gt   = (labels == 2).float()
                    bubble_mask = (labels != IGNORE_INDEX)
                    bce  = F.binary_cross_entropy_with_logits(
                        bl[bubble_mask], bubble_gt[bubble_mask])
                    dice = binary_dice_loss(bl[bubble_mask], bubble_gt[bubble_mask],
                                            bubble_mask)
                    loss = criterion(logits, labels, ignore_mask, 0.0) + bce + dice
                else:
                    loss = criterion(logits, labels, ignore_mask, 0.0)
            else:
                loss = criterion(logits, labels, ignore_mask, 0.0)
            pred = logits.argmax(dim=1)

        metric.update(pred, labels)

        total_loss += loss.item() * images.size(0)
        n          += images.size(0)

    iou = metric.iou_per_class()
    names  = ['noise', 'tissue', 'bubble', 'ignore'][:NUM_CLASSES]
    detail = '  '.join(f'{nm}={iou[i]:.3f}' for i, nm in enumerate(names))

    if NUM_CLASSES == 4:
        # 3cls mIoU (noise/tissue/bubble) for best model selection; 4cls for reference
        miou_3cls = float(iou[:3].mean())
        miou_4cls = float(iou.mean())
        print(f"  Val loss={total_loss/n:.4f}  {detail}  mIoU(3cls)={miou_3cls:.3f}  mIoU(4cls)={miou_4cls:.3f}")
        result = {'val_loss': total_loss / n, 'mIoU': miou_3cls, 'mIoU_4cls': miou_4cls}
    else:
        print(f"  Val loss={total_loss/n:.4f}  {detail}  mIoU={iou.mean():.3f}")
        result = {'val_loss': total_loss / n, 'mIoU': float(iou.mean())}
    for i, nm in enumerate(names):
        result[f'iou_{nm}'] = float(iou[i])
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    utils.init_distributed_mode(args)
    cudnn.benchmark = True

    # num_classes 전역 덮어쓰기
    global NUM_CLASSES
    NUM_CLASSES = args.num_classes
    remap = (args.num_classes == 4)   # 4-class면 255→3 리매핑

    # ── 데이터셋 ──
    stats = None
    if args.stats_path and os.path.isfile(args.stats_path):
        with open(args.stats_path) as f:
            stats = json.load(f)
        print(f"Stats loaded: {args.stats_path}")

    if args.dataset in ('bolus_sf', 'bolus_sf3'):
        from datasets.bolus_sf_dataset import BolusSFSegDataset
        _csv      = getattr(args, 'csv_path', 'datasets_sf/sample_index_table.csv')
        _gt       = getattr(args, 'gt_dir',   'GT_v1/gt')
        _n_frames = 3 if args.dataset == 'bolus_sf3' else 1
        train_ds = BolusSFSegDataset(_csv, args.iq_dir, _gt,
                                     split='ft_train', stats=stats, cache=True,
                                     n_frames=_n_frames)
        if getattr(args, 'include_pretrain_gt', False):
            pretrain_ds = BolusSFSegDataset(_csv, args.iq_dir, _gt,
                                            split='pretrain', stats=stats, cache=True,
                                            n_frames=_n_frames)
            train_ds = torch.utils.data.ConcatDataset([train_ds, pretrain_ds])
            print(f"Pretrain GT included: ft_train + pretrain = {len(train_ds):,}개")
        val_ds = BolusSFSegDataset(_csv, args.iq_dir, _gt,
                                   split='val', stats=stats, cache=True,
                                   n_frames=_n_frames)
    elif args.dataset == 'bolus_v5':
        from datasets.bolus_v5_dataset import BolusMATSegDataset
        _soft      = getattr(args, 'soft_ignore', False)
        _four_cls  = (NUM_CLASSES == 4)
        _two_cls   = getattr(args, 'two_class', False)
        train_ds = BolusMATSegDataset(args.csv_path, args.iq_dir, args.gt_dir,
                                      split='ft_train', stats=stats,
                                      cache=True, jitter=9,
                                      soft_ignore=_soft, four_class=_four_cls,
                                      two_class=_two_cls)
        if args.include_pretrain_gt:
            pretrain_ds = BolusMATSegDataset(args.csv_path, args.iq_dir, args.gt_dir,
                                             split='pretrain', stats=stats,
                                             cache=True, jitter=9,
                                             soft_ignore=_soft, four_class=_four_cls,
                                             two_class=_two_cls)
            train_ds = torch.utils.data.ConcatDataset([train_ds, pretrain_ds])
            print(f"Pretrain GT included: ft_train + pretrain = {len(train_ds)}개")
        val_ds   = BolusMATSegDataset(args.csv_path, args.iq_dir, args.gt_dir,
                                      split='val', stats=stats,
                                      cache=True, jitter=0,
                                      soft_ignore=False, four_class=_four_cls,
                                      two_class=_two_cls)
    else:
        train_ds = BlockSegDataset(args.block_dir_train, args.label_dir, stats=stats,
                                   remap_unknown=remap)
        val_ds   = BlockSegDataset(args.block_dir_val,   args.label_dir, stats=stats,
                                   remap_unknown=remap)
    print(f"Train samples: {len(train_ds)}  Val samples: {len(val_ds)}")

    if args.oversample_bubble and args.dataset == 'bolus_v5':
        print("Computing bubble sample weights for oversampling...")
        bubble_weights = train_ds.get_bubble_weights()
        train_sampler  = DistributedWeightedSampler(
            bubble_weights, len(train_ds),
            num_replicas=utils.get_world_size(),
            rank=utils.get_rank())
        print(f"Oversampling enabled: {sum(w > 1.0 for w in bubble_weights)} / {len(bubble_weights)} samples have bubble pixels")
    else:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, shuffle=True)
    train_loader  = DataLoader(train_ds, batch_size=args.batch_size_per_gpu,
                               sampler=train_sampler, num_workers=args.num_workers,
                               pin_memory=True, drop_last=True)
    val_loader    = DataLoader(val_ds, batch_size=args.batch_size_per_gpu,
                               num_workers=args.num_workers, pin_memory=True)

    # ── 모델 ──
    update_config(config, args)
    backbone = build_model(config, is_teacher=True)
    backbone.cuda()

    # 전체 freeze 후 Stage 1만 선택적 unfreeze
    for p in backbone.parameters():
        p.requires_grad_(False)
    if args.unfreeze_stage1:
        for p in backbone.layers[0].parameters():
            p.requires_grad_(True)
        print("Backbone Stage 1 unfrozen (lr={})".format(args.backbone_lr))
    backbone.eval()   # BN/dropout 고정 (파라미터만 학습)

    utils.load_pretrained_weights(
        backbone, args.pretrained_weights, args.checkpoint_key, args.arch, patch_size=None
    )

    swin_spec  = config.MODEL.SPEC
    stage1_dim = swin_spec['DIM_EMBED']         # 96
    if args.dataset in ('bolus_v5', 'bolus_sf', 'bolus_sf3'):
        INPUT_H, INPUT_W = 100, 108
    else:
        INPUT_H, INPUT_W = 80, 120
    patch_hw = (INPUT_H // swin_spec['PATCH_SIZE'],
                INPUT_W // swin_spec['PATCH_SIZE'])

    extractor = MultiScaleFeatureExtractor(backbone, patch_hw=patch_hw).cuda()

    stage2_dim = stage1_dim * 2
    stage3_dim = stage1_dim * 4
    if args.class_specific_head:
        seg_head = ClassSpecificSegHead(in_dim_s1=stage1_dim, in_dim_s2=stage2_dim,
                                        in_dim_s3=stage3_dim,
                                        output_hw=(INPUT_H, INPUT_W)).cuda()
        print("Using ClassSpecificSegHead (noise=s3, tissue=s1, bubble=s1+s2+s3)")
    else:
        seg_head = SegHead(in_dim_s1=stage1_dim, in_dim_s2=stage2_dim, in_dim_s3=stage3_dim,
                           num_classes=NUM_CLASSES,
                           output_hw=(INPUT_H, INPUT_W)).cuda()
    seg_head  = nn.parallel.DistributedDataParallel(
        seg_head, device_ids=[args.gpu]
    )

    bubble_cnn = None
    if args.use_bubble_cnn:
        from models.bubble_3d_module import build_bubble_cnn
        extra = 1 if getattr(args, 'use_mag_std', False) else 0
        in_ch = config.MODEL.SPEC['IN_CHANS'] + extra if args.bubble_cnn_type == 'unet2d' else 2
        # unet3d는 build_bubble_cnn 내부에서 reshape 처리 (in_ch 무관)
        bubble_cnn = build_bubble_cnn(args.bubble_cnn_type, in_channels=in_ch,
                                      channels=(16, 32, 64)).cuda()
        bubble_cnn = nn.parallel.DistributedDataParallel(bubble_cnn, device_ids=[args.gpu])
        print(f"BubbleCNN enabled: type={args.bubble_cnn_type}, in_channels={in_ch}")

    # SegHead freeze: pretrained seg head 로드 후 파라미터 고정, BubbleCNN만 학습
    if args.freeze_seg_head:
        assert args.pretrained_seg_head, "--freeze_seg_head 사용 시 --pretrained_seg_head 경로 필요"
        ckpt_sh = torch.load(args.pretrained_seg_head, map_location='cpu')
        seg_head.load_state_dict(ckpt_sh['seg_head'])
        for p in seg_head.parameters():
            p.requires_grad_(False)
        seg_head.eval()
        print(f"SegHead frozen from {args.pretrained_seg_head}")

    # ── Loss ──
    if args.class_weights is not None:
        cw = args.class_weights
    elif NUM_CLASSES == 4:
        cw = [1.0, 2.0, 5.0, 0.5]   # unknown은 낮은 가중치
    else:
        cw = [1.0, 2.0, 5.0]
    class_weights = torch.tensor(cw, dtype=torch.float32).cuda()
    ce_loss_none  = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, weight=class_weights,
                                        reduction='none')
    ce_loss       = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, weight=class_weights)
    dice_loss     = DiceLoss(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX,
                             class_weights=class_weights)

    def criterion(logits, targets, ignore_mask=None, soft_w=0.0):
        # soft_ignore 활성 시: ignore_mask 위치에 soft_w 가중치 적용한 per-pixel CE
        if ignore_mask is not None and ignore_mask.any():
            ce_px = ce_loss_none(logits, targets)           # (B, H, W)
            w     = torch.ones_like(targets, dtype=torch.float32)
            w[ignore_mask] = soft_w
            valid = targets != IGNORE_INDEX
            ce    = (ce_px * w)[valid].mean() if valid.any() else ce_px.mean()
        else:
            ce = ce_loss(logits, targets)
        return ce + args.dice_weight * dice_loss(logits, targets)

    soft_ignore = getattr(args, 'soft_ignore', False)
    soft_w_str  = f", soft_ignore_weight={args.soft_ignore_weight}" if soft_ignore else ""
    weight_str = "  ".join(f"cls{i}:{w:.1f}" for i, w in enumerate(class_weights.tolist()))
    print(f"Loss: CE + {args.dice_weight}×Dice  weights — {weight_str}{soft_w_str}")

    # ── Optimizer ──
    param_groups = []
    if not args.freeze_seg_head:
        param_groups.append({'params': seg_head.parameters(), 'lr': args.lr})
    if bubble_cnn is not None:
        param_groups.append({'params': bubble_cnn.parameters(), 'lr': args.lr})
    if args.unfreeze_stage1:
        backbone_params = [p for p in backbone.layers[0].parameters() if p.requires_grad]
        param_groups.append({'params': backbone_params, 'lr': args.backbone_lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.min_lr
    )

    # ── 체크포인트 복원 ──
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path  = os.path.join(args.output_dir, 'checkpoint.pth')
    start_epoch, best_miou = 0, 0.0
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        seg_head.load_state_dict(ckpt['seg_head'])
        if bubble_cnn is not None and 'bubble_cnn' in ckpt:
            bubble_cnn.load_state_dict(ckpt['bubble_cnn'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch']
        best_miou   = ckpt.get('best_miou', 0.0)
        print(f"Resumed from epoch {start_epoch}, best mIoU={best_miou:.4f}")

    # ── 학습 루프 ──
    log_path = Path(args.output_dir) / 'log.txt'
    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(extractor, seg_head, train_loader, optimizer, epoch, args, criterion,
                                      bubble_cnn=bubble_cnn)
        scheduler.step()

        log = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}

        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            val_stats = validate(extractor, seg_head, val_loader, criterion, bubble_cnn=bubble_cnn, args=args)
            log.update(val_stats)

            if utils.is_main_process() and val_stats['mIoU'] > best_miou:
                best_miou = val_stats['mIoU']
                best_ckpt = {'seg_head': seg_head.state_dict()}
                if bubble_cnn is not None:
                    best_ckpt['bubble_cnn'] = bubble_cnn.state_dict()
                torch.save(best_ckpt, os.path.join(args.output_dir, 'best_seg_head.pth'))
                print(f"  → Best mIoU updated: {best_miou:.4f}")

        if utils.is_main_process():
            save = {
                'epoch': epoch + 1,
                'seg_head': seg_head.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_miou': best_miou,
            }
            if bubble_cnn is not None:
                save['bubble_cnn'] = bubble_cnn.state_dict()
            torch.save(save, ckpt_path)
            with open(log_path, 'a') as f:
                f.write(json.dumps(log) + '\n')

    print(f"\nFinished. Best mIoU: {best_miou:.4f}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Phasor Segmentation Fine-tuning')

    parser.add_argument('--cfg', type=str, required=True,
                        help='YAML config (e.g. experiments/imagenet/swin/swin_tiny_phasor.yaml)')
    parser.add_argument('--arch', default='swin_tiny', type=str)
    parser.add_argument('--pretrained_weights', type=str, required=True,
                        help='사전학습 checkpoint 경로')
    parser.add_argument('--checkpoint_key', default='teacher', type=str)
    parser.add_argument('--dataset', type=str, default='phasor',
                        choices=['phasor', 'bolus_v5', 'bolus_sf', 'bolus_sf3'],
                        help='데이터셋 종류 (bolus_v5: 원본 IQ 직접 읽기)')
    parser.add_argument('--csv_path', type=str, default='datasets_v5/sample_index_table.csv',
                        help='[bolus_v5] sample_index_table.csv 경로')
    parser.add_argument('--iq_dir', type=str, default='IQ_data/PALA_bolus',
                        help='[bolus_v5] 원본 IQ MAT 파일 디렉토리')
    parser.add_argument('--block_dir_train', type=str, default='',
                        help='학습용 bolus block_*.pt 디렉토리 (phasor 전용)')
    parser.add_argument('--block_dir_val', type=str, default='',
                        help='검증용 bolus block_*.pt 디렉토리 (phasor 전용)')
    parser.add_argument('--label_dir', type=str, default='',
                        help='label_001.mat ~ label_213.mat 디렉토리 (phasor 전용)')
    parser.add_argument('--gt_dir', type=str, default='datasets_v5/gt',
                        help='[bolus_v5] 샘플별 GT MAT 디렉토리 (datasets_v5/gt/)')
    parser.add_argument('--stats_path', type=str, default='datasets_v2/bolus/stats.json',
                        help='정규화 통계값 JSON 경로')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--min_lr', default=1e-6, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--batch_size_per_gpu', default=32, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--val_freq', default=5, type=int)
    parser.add_argument('--num_classes', default=3, type=int,
                        help='분류 클래스 수. 3=noise/tissue/bubble, 4=+unknown. 기본: 3')
    parser.add_argument('--two_class', action='store_true',
                        help='2-class(noise/tissue) 학습: bubble→255(ignore). --num_classes 2와 함께 사용')
    parser.add_argument('--class_weights', default=None, nargs='+', type=float,
                        help='CE loss 클래스 가중치. 기본: 3cls=[1,2,5], 4cls=[1,2,5,0.5]')
    parser.add_argument('--dice_weight', default=1.0, type=float,
                        help='Dice Loss 가중치. Loss = CE + dice_weight×Dice. 기본: 1.0')
    parser.add_argument('--unfreeze_stage1', action='store_true',
                        help='Backbone Stage 1을 학습에 참여시킴')
    parser.add_argument('--backbone_lr', default=1e-5, type=float,
                        help='Backbone Stage 1 learning rate. 기본: 1e-5')
    parser.add_argument('--oversample_bubble', action='store_true',
                        help='bubble pixel 수에 비례한 weighted sampling (bolus_v5 전용)')
    parser.add_argument('--class_specific_head', action='store_true',
                        help='클래스별 다른 feature scale 사용 (noise=s3, tissue=s1, bubble=all)')
    parser.add_argument('--include_pretrain_gt', action='store_true',
                        help='pretrain split GT를 ft_train에 합쳐서 학습 데이터 확장 (bolus_v5 전용)')
    parser.add_argument('--use_bubble_cnn', action='store_true',
                        help='BubbleCNN 추가 — bubble 채널 logit 보강')
    parser.add_argument('--bubble_cnn_type', type=str, default='2plus1d',
                        choices=['2plus1d', '3d_dws', 'unet2d', 'unet3d'],
                        help='BubbleCNN 구조: 2plus1d=(2+1)D Conv, 3d_dws=3D Depthwise Separable, unet2d=2D UNet, unet3d=3D UNet')
    parser.add_argument('--freeze_seg_head', action='store_true',
                        help='SegHead를 freeze하고 BubbleCNN만 학습 (순차 학습 Phase 2)')
    parser.add_argument('--pretrained_seg_head', type=str, default='',
                        help='freeze_seg_head 시 로드할 SegHead checkpoint 경로')
    parser.add_argument('--separate_bubble', action='store_true',
                        help='SegHead는 noise/tissue만 학습(bubble=ignore), BubbleCNN은 binary BCE로 독립 학습')
    parser.add_argument('--bubble_lambda', default=1.0, type=float,
                        help='separate_bubble 모드에서 bubble_loss 가중치. Loss=seg_loss + lambda*bubble_loss')
    parser.add_argument('--bubble_dice', action='store_true',
                        help='BubbleCNN에 binary BCE+Dice loss 추가 (additive 모드 전용)')
    parser.add_argument('--use_mag_std', action='store_true',
                        help='BubbleUNet 입력에 20frame magnitude 표준편차 채널(1ch) 추가 (unet2d 전용)')
    parser.add_argument('--soft_ignore', action='store_true',
                        help='학습 시 ignore(gt==0) 픽셀을 bubble(2)로 변환 후 낮은 가중치로 학습')
    parser.add_argument('--soft_ignore_weight', default=0.3, type=float,
                        help='soft_ignore 픽셀에 부여할 loss 가중치 (기본: 0.3)')
    parser.add_argument('--hard_bubble_fusion', action='store_true',
                        help='separate_bubble 모드 추론 시 sigmoid>0.5 하드 임계값으로 bubble 덮어씀')
    parser.add_argument('--dist_url', default='env://', type=str)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    main(args)
