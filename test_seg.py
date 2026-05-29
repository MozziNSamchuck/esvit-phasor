"""
Segmentation Test Script (파라미터 고정, 평가만)

사용법:
  python test_seg.py \
    --cfg experiments/imagenet/swin/swin_tiny_phasor.yaml \
    --pretrained_weights OUTPUT/phasor_pretrain_v3/checkpoint0090.pth \
    --seg_head_weights   OUTPUT/seg_finetune_v1/best_seg_head.pth \
    --phasor_path USdataset/phasor/HJ \
    --label_path  USdataset/label/HJ \
    --output_dir  OUTPUT/test_HJ_v1 \
    --vis_n 5
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import utils
from models import build_model
from config import config
from config.default import _update_config_from_file
from datasets.bolus_dataset import BolusSegDataset

NUM_CLASSES  = 3
IGNORE_INDEX = 255
CLASS_NAMES  = ['noise', 'tissue', 'bubble']
# 시각화 색상: noise=검정, tissue=빨강, bubble=파랑, ignore=회색
CLASS_COLORS = np.array([
    [0,   0,   0  ],   # 0: noise
    [220, 50,  50 ],   # 1: tissue
    [50,  100, 220],   # 2: bubble
], dtype=np.uint8)


# ── 모델 구성 (finetune_seg.py와 동일) ────────────────────────────────────────

class SegHead(nn.Module):
    def __init__(self, in_dim_s1=96, in_dim_s3=384,
                 num_classes=NUM_CLASSES, output_hw=(80, 120), mid_dim=128):
        super().__init__()
        self.s3_reduce = nn.Conv2d(in_dim_s3, in_dim_s1, kernel_size=1)
        fused_dim = in_dim_s1 * 2
        self.decoder = nn.Sequential(
            nn.Conv2d(fused_dim, mid_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, mid_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, num_classes, kernel_size=1),
        )
        self.output_hw = output_hw

    def forward(self, feat_s1, feat_s3):
        s3 = self.s3_reduce(feat_s3)
        s3 = F.interpolate(s3, size=feat_s1.shape[-2:],
                           mode='bilinear', align_corners=False)
        x = torch.cat([feat_s1, s3], dim=1)
        x = self.decoder(x)
        x = F.interpolate(x, size=self.output_hw,
                          mode='bilinear', align_corners=False)
        return x


class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, backbone, patch_hw=(20, 30)):
        super().__init__()
        self.backbone  = backbone
        self.patch_hw  = patch_hw
        self._feat_s1  = None
        self._feat_s3  = None
        backbone.layers[0].blocks[-1].register_forward_hook(self._hook_s1)
        backbone.layers[2].blocks[-1].register_forward_hook(self._hook_s3)

    def _hook_s1(self, module, input, output):
        x, _ = output
        B, L, C = x.shape
        H, W = self.patch_hw
        self._feat_s1 = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

    def _hook_s3(self, module, input, output):
        x, _ = output
        B, L, C = x.shape
        # Stage3 해상도 = patch_hw / 4, 비율 유지
        H1, W1 = self.patch_hw
        H = H1 // 4
        W = W1 // 4
        # 실제 L과 맞는지 확인, 안 맞으면 비율로 재계산
        if H * W != L:
            H = int(round((L * H1 / W1) ** 0.5))
            W = L // H
        self._feat_s3 = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

    def forward(self, x):
        with torch.no_grad():
            self.backbone.forward_features(x)
        return self._feat_s1, self._feat_s3


# ── IoU metric ────────────────────────────────────────────────────────────────

class IoUMetric:
    def __init__(self):
        self.intersection = np.zeros(NUM_CLASSES)
        self.union        = np.zeros(NUM_CLASSES)

    def update(self, pred, target):
        mask   = (target != IGNORE_INDEX)
        pred   = pred[mask].cpu().numpy()
        target = target[mask].cpu().numpy()
        for c in range(NUM_CLASSES):
            p = pred   == c
            t = target == c
            self.intersection[c] += (p & t).sum()
            self.union[c]        += (p | t).sum()

    def result(self):
        iou = np.zeros(NUM_CLASSES)
        for c in range(NUM_CLASSES):
            if self.union[c] > 0:
                iou[c] = self.intersection[c] / self.union[c]
        return iou


# ── 시각화 ────────────────────────────────────────────────────────────────────

def colorize(label_map):
    """(H, W) int → (H, W, 3) uint8 RGB"""
    rgb = np.full((*label_map.shape, 3), 128, dtype=np.uint8)  # ignore=회색
    for c, color in enumerate(CLASS_COLORS):
        rgb[label_map == c] = color
    return rgb


def save_vis(image, pred, label, sample_idx, out_dir, stats=None):
    """
    image: (20, H, W) tensor  → magnitude 첫 프레임만 표시
    pred:  (H, W) numpy int
    label: (H, W) numpy int
    stats: dict with mag_mean, mag_std (역정규화용)
    """
    # 짝수 채널 = magnitude (10개 프레임) → 시간 평균으로 speckle 감소
    mag_frames = image[0::2].cpu().numpy()  # (10, H, W)
    mag = mag_frames.mean(axis=0)           # (H, W)

    # 역정규화 → 원본 magnitude 복원 후 로그 스케일
    if stats is not None:
        mag = mag * stats['mag_std'] + stats['mag_mean']  # 원본 범위로 복원
    else:
        mag = mag - mag.min()  # fallback

    mag = np.clip(mag, 0, None)          # 음수 제거
    mag = np.log1p(mag)                  # log(1+x): 0~30000 → 자연스러운 범위
    # percentile clipping: 극단값 제거 후 0~1
    lo, hi = np.percentile(mag, 1), np.percentile(mag, 99)
    mag = np.clip((mag - lo) / (hi - lo + 1e-8), 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(mag, cmap='gray', aspect='auto')
    axes[0].set_title('Magnitude (ch0)')
    axes[0].axis('off')

    axes[1].imshow(colorize(pred), aspect='auto')
    axes[1].set_title('Prediction')
    axes[1].axis('off')

    axes[2].imshow(colorize(label), aspect='auto')
    axes[2].set_title('GT Label')
    axes[2].axis('off')

    # 범례
    patches = [
        matplotlib.patches.Patch(color=CLASS_COLORS[i]/255, label=CLASS_NAMES[i])
        for i in range(NUM_CLASSES)
    ]
    patches.append(matplotlib.patches.Patch(color=[0.5,0.5,0.5], label='ignore'))
    fig.legend(handles=patches, loc='lower center', ncol=4, fontsize=9)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(os.path.join(out_dir, f'vis_{sample_idx:04d}.png'), dpi=120)
    plt.close()


# ── Test ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def test(args):
    cudnn.benchmark = True
    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)

    # stats.json 로드 (역정규화용)
    import json as _json
    stats_path = os.path.join(args.phasor_path, 'stats.json')
    vis_stats = None
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            vis_stats = _json.load(f)

    # 데이터셋
    dataset = BolusSegDataset(
        phasor_path=args.phasor_path,
        label_path=args.label_path,
        center_crop_hw=tuple(args.center_crop_hw),
        frame_stride=args.frame_stride,
        downscale=args.downscale,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=4,
        shuffle=False, pin_memory=True,
    )
    print(f"테스트 샘플 수: {len(dataset)}")

    # 모델
    _update_config_from_file(config, args.cfg)
    config.defrost()
    config.freeze()
    backbone = build_model(config, is_teacher=True)
    backbone.cuda().eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    utils.load_pretrained_weights(
        backbone, args.pretrained_weights, args.checkpoint_key, args.arch, patch_size=None
    )

    swin_spec  = config.MODEL.SPEC
    stage1_dim = swin_spec['DIM_EMBED']
    patch_size = swin_spec['PATCH_SIZE']
    # center_crop_hw → pad to multiple of 4 → actual input size
    ch, cw   = args.center_crop_hw
    input_H  = ch + (patch_size - ch % patch_size) % patch_size
    input_W  = cw + (patch_size - cw % patch_size) % patch_size
    patch_hw = (input_H // patch_size, input_W // patch_size)

    extractor = MultiScaleFeatureExtractor(backbone, patch_hw=patch_hw).cuda()
    stage3_dim = stage1_dim * 4
    seg_head  = SegHead(in_dim_s1=stage1_dim, in_dim_s3=stage3_dim,
                        num_classes=NUM_CLASSES,
                        output_hw=(input_H, input_W)).cuda()

    # fine-tuning된 seg_head 가중치 로드
    ckpt = torch.load(args.seg_head_weights, map_location='cpu')
    # DDP로 저장된 경우 'module.' prefix 제거
    state = {k.replace('module.', ''): v for k, v in ckpt['seg_head'].items()}
    seg_head.load_state_dict(state)
    seg_head.eval()

    # 평가
    metric     = IoUMetric()
    criterion  = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    total_loss, n = 0.0, 0
    vis_saved  = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        feat_s1, feat_s3 = extractor(images)
        logits = seg_head(feat_s1, feat_s3)
        loss   = criterion(logits, labels)
        pred   = logits.argmax(dim=1)

        metric.update(pred, labels)
        total_loss += loss.item() * images.size(0)
        n          += images.size(0)

        # 시각화 저장 (처음 vis_n 샘플)
        if vis_saved < args.vis_n:
            for i in range(images.size(0)):
                if vis_saved >= args.vis_n:
                    break
                save_vis(
                    images[i],
                    pred[i].cpu().numpy(),
                    labels[i].cpu().numpy(),
                    vis_saved,
                    vis_dir,
                    stats=vis_stats,
                )
                vis_saved += 1

        if batch_idx % 50 == 0:
            print(f"  [{batch_idx}/{len(loader)}]")

    # 결과 출력
    iou = metric.result()
    results = {
        'test_loss': total_loss / n,
        'mIoU':      float(iou.mean()),
        'iou_noise':  float(iou[0]),
        'iou_tissue': float(iou[1]),
        'iou_bubble': float(iou[2]),
    }

    print("\n===== Test Results =====")
    print(f"  Loss:         {results['test_loss']:.4f}")
    print(f"  noise  IoU:   {results['iou_noise']:.4f}")
    print(f"  tissue IoU:   {results['iou_tissue']:.4f}")
    print(f"  bubble IoU:   {results['iou_bubble']:.4f}")
    print(f"  mIoU:         {results['mIoU']:.4f}")
    print(f"  시각화:        {vis_dir}/")
    print("========================")

    with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Phasor Segmentation Test')
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--arch', default='swin_tiny', type=str)
    parser.add_argument('--pretrained_weights', type=str, required=True)
    parser.add_argument('--checkpoint_key', default='teacher', type=str)
    parser.add_argument('--seg_head_weights', type=str, required=True,
                        help='fine-tuning된 best_seg_head.pth 경로')
    parser.add_argument('--phasor_path', type=str, required=True)
    parser.add_argument('--label_path',  type=str, required=True)
    parser.add_argument('--output_dir',  type=str, required=True)
    parser.add_argument('--batch_size',  default=32, type=int)
    parser.add_argument('--vis_n',       default=10, type=int,
                        help='시각화 저장할 샘플 수')
    parser.add_argument('--frame_stride', default=50, type=int,
                        help='프레임 샘플링 간격. 1000fps=50, 500fps=25')
    parser.add_argument('--center_crop_hw', default=[78, 118], nargs=2, type=int,
                        help='center crop 크기 [H, W]. 기본: 78 118 (bolus 기준)')
    parser.add_argument('--downscale', default=1.0, type=float,
                        help='로드 후 공간 축소 비율. 0.5이면 절반. 기본: 1.0 (원본)')
    parser.add_argument('--opts', default=[], nargs=argparse.REMAINDER)
    args = parser.parse_args()
    test(args)
