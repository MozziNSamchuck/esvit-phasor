"""
bolus_sf_dataset.py

Single-frame (and multi-frame) dataset for pretrain and fine-tuning.

입력: 원본 IQ MAT (107×128×800) → 단일/인접 프레임 슬라이스 → crop 100×108
변환: magnitude + phase → [2*n_frames, 100, 108] (인터리브: mag_t-k, ph_t-k, ..., mag_t, ph_t, ...)
GT:   GT_v1/gt/{block}_f{frame:04d}.mat  (107×128, 0/1/2/3)
      0=ignore→255, 1=tissue, 2=bubble, 3=noise→0

채널 순서 (n_frames=3 예시): [mag_t-1, ph_t-1, mag_t, ph_t, mag_t+1, ph_t+1]
"""

import csv
import os

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


class BolusSFDirectDataset(Dataset):
    """
    Pretrain용 단일 프레임 Dataset.
    원본 IQ MAT에서 단일 프레임을 읽어 2ch (mag, phase) 텐서 반환.

    Args:
        csv_path:  datasets_sf/sample_index_table.csv
        iq_dir:    IQ MAT 디렉토리 (IQ_data/PALA_bolus)
        split:     'pretrain'
        transform: DataAugmentationPhasor 인스턴스
        stats:     정규화 통계 dict (mag_mean, mag_std, phase_mean, phase_std)
        cache:     True이면 블록 IQ를 메모리에 캐싱
    """

    ROW_START, ROW_END = 7, 107
    COL_START, COL_END = 12, 120

    def __init__(self, csv_path, iq_dir, split='pretrain',
                 transform=None, stats=None, cache=True, n_frames=1):
        self.iq_dir    = iq_dir
        self.transform = transform
        self.stats     = stats
        self.cache     = cache
        self.n_frames  = n_frames
        self._block_cache = {}

        self.samples = []
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['split'] != split:
                    continue
                self.samples.append({
                    'block_name': row['block_name'],
                    'frame':      int(row['frame']),  # 1-indexed
                })

        if not self.samples:
            raise ValueError(f"split='{split}' 샘플 없음: {csv_path}")
        print(f"[BolusSFDirectDataset] split={split}, {len(self.samples):,}개 샘플, n_frames={n_frames}")

    def __len__(self):
        return len(self.samples)

    def _load_block(self, block_name):
        if self.cache and block_name in self._block_cache:
            return self._block_cache[block_name]
        mat_path = os.path.join(self.iq_dir, block_name + '.mat')
        d = sio.loadmat(mat_path)
        iq = d['IQ'][self.ROW_START:self.ROW_END,
                     self.COL_START:self.COL_END, :].astype(np.complex64)
        if self.cache:
            self._block_cache[block_name] = iq
        return iq

    def _frame_to_channels(self, iq, fi):
        """fi 기준 n_frames 인접 프레임을 (2*n_frames, H, W) 인터리브 텐서로 반환."""
        T = iq.shape[2]
        half = self.n_frames // 2
        indices = [min(max(fi + d, 0), T - 1) for d in range(-half, half + 1)]

        channels = []
        for i in indices:
            frm = iq[:, :, i]
            channels.append(np.abs(frm).astype(np.float32))    # mag
            channels.append(np.angle(frm).astype(np.float32))  # phase
        return torch.from_numpy(np.stack(channels, axis=0))    # (2*n_frames, H, W)

    def __getitem__(self, idx):
        s      = self.samples[idx]
        iq     = self._load_block(s['block_name'])   # (100, 108, 800)
        fi     = s['frame'] - 1                       # 0-indexed
        tensor = self._frame_to_channels(iq, fi)      # (2*n_frames, 100, 108)

        if self.stats is not None:
            tensor[0::2] = (tensor[0::2] - self.stats['mag_mean'])   / (self.stats['mag_std']   + 1e-8)
            tensor[1::2] = (tensor[1::2] - self.stats['phase_mean']) / (self.stats['phase_std'] + 1e-8)

        if self.transform is not None:
            return self.transform(tensor), 0
        return tensor, 0


class BolusSFSegDataset(Dataset):
    """
    Fine-tuning / val / test용 단일 프레임 Dataset.
    GT: GT_v1/gt/{block}_f{frame:04d}.mat

    GT 인코딩 (원본→학습용):
        0 (ignore) → 255
        1 (tissue) → 1
        2 (bubble) → 2
        3 (noise)  → 0

    Returns: (tensor[2,100,108], dummy_mag_std[1,100,108], gt[100,108], ignore_mask[100,108])
    dummy_mag_std: 기존 finetune_seg.py 4-tuple 인터페이스와의 호환용 zero tensor
    """

    ROW_START, ROW_END = 7, 107
    COL_START, COL_END = 12, 120

    def __init__(self, csv_path, iq_dir, gt_dir, split,
                 stats=None, cache=True, n_frames=1):
        self.iq_dir   = iq_dir
        self.gt_dir   = gt_dir
        self.stats    = stats
        self.cache    = cache
        self.n_frames = n_frames
        self._iq_cache = {}

        self.samples = []
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['split'] != split:
                    continue
                self.samples.append({
                    'block_name': row['block_name'],
                    'frame':      int(row['frame']),
                })

        if not self.samples:
            raise ValueError(f"split='{split}' 샘플 없음: {csv_path}")
        print(f"[BolusSFSegDataset] split={split}, {len(self.samples):,}개 샘플, n_frames={n_frames}")

    def __len__(self):
        return len(self.samples)

    def _load_iq(self, block_name):
        if self.cache and block_name in self._iq_cache:
            return self._iq_cache[block_name]
        d = sio.loadmat(os.path.join(self.iq_dir, block_name + '.mat'))
        iq = d['IQ'][self.ROW_START:self.ROW_END,
                     self.COL_START:self.COL_END, :].astype(np.complex64)
        if self.cache:
            self._iq_cache[block_name] = iq
        return iq

    def _load_gt(self, block_name, frame):
        fn = f"{block_name}_f{frame:04d}.mat"
        d  = sio.loadmat(os.path.join(self.gt_dir, fn))
        gt = d['gt'][self.ROW_START:self.ROW_END,
                     self.COL_START:self.COL_END]   # (100, 108)
        out = np.full_like(gt, 255, dtype=np.uint8)
        out[gt == 3] = 0   # noise
        out[gt == 1] = 1   # tissue
        out[gt == 2] = 2   # bubble
        # gt == 0 → 255 (ignore)
        return out

    def get_bubble_weights(self):
        """bubble pixel 수 비례 가중치 (WeightedSampler용)."""
        weights = []
        for s in self.samples:
            fn = f"{s['block_name']}_f{s['frame']:04d}.mat"
            d  = sio.loadmat(os.path.join(self.gt_dir, fn))
            gt = d['gt'][self.ROW_START:self.ROW_END, self.COL_START:self.COL_END]
            weights.append(float((gt == 2).sum()) + 1.0)
        return weights

    def _frame_to_channels(self, iq, fi):
        T    = iq.shape[2]
        half = self.n_frames // 2
        indices = [min(max(fi + d, 0), T - 1) for d in range(-half, half + 1)]
        channels = []
        for i in indices:
            frm = iq[:, :, i]
            channels.append(np.abs(frm).astype(np.float32))
            channels.append(np.angle(frm).astype(np.float32))
        return torch.from_numpy(np.stack(channels, axis=0))  # (2*n_frames, H, W)

    def __getitem__(self, idx):
        s      = self.samples[idx]
        iq     = self._load_iq(s['block_name'])
        fi     = s['frame'] - 1
        tensor = self._frame_to_channels(iq, fi)  # (2*n_frames, 100, 108)

        if self.stats is not None:
            tensor[0::2] = (tensor[0::2] - self.stats['mag_mean'])   / (self.stats['mag_std']   + 1e-8)
            tensor[1::2] = (tensor[1::2] - self.stats['phase_mean']) / (self.stats['phase_std'] + 1e-8)

        gt = self._load_gt(s['block_name'], s['frame'])

        dummy_mag_std = torch.zeros(1, tensor.shape[1], tensor.shape[2])
        ignore_mask   = torch.zeros(tensor.shape[1], tensor.shape[2], dtype=torch.bool)

        return (tensor,
                dummy_mag_std,
                torch.from_numpy(gt.astype(np.int64)),
                ignore_mask)
