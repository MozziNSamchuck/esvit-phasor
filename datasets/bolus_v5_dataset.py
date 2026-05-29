"""
bolus_v5_dataset.py

datasets_v5/sample_index_table.csv 기반 pretrain Dataset.

샘플 정의:
  - 연속 20프레임, stride 10 (CSV에 frame_start/frame_end로 정의)
  - 원본 IQ MAT에서 직접 읽기 (IQ key, complex64, shape 107×128×800)
  - 크롭: row 7:107, col 12:120 → 100×108
  - 변환: magnitude + phase + phase_diff → [60, 100, 108] (20프레임 × 3채널)
  - temporal jitter: 매 호출마다 시작 프레임을 균등 랜덤 오프셋
    경계 샘플은 블록 범위 내에서 유효한 δ 구간을 동적으로 계산해 균등 분포 유지

정규화: stats.json (mag_mean/std, phase_mean/std, phase_diff_mean/std)
채널 순서: (magnitude, phase, phase_diff) × 20frame
  - tensor[0::3] → magnitude
  - tensor[1::3] → phase
  - tensor[2::3] → phase_diff  (첫 frame = 0으로 패딩)
"""

import os
import csv

import h5py
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


class BolusMATDirectDataset(Dataset):
    """
    원본 IQ MAT 파일에서 연속 20프레임을 직접 읽어 SSL 사전학습용 샘플을 생성.

    Args:
        csv_path:    sample_index_table.csv 경로
        iq_dir:      IQ MAT 파일 디렉토리 (e.g. IQ_data/PALA_bolus/)
        split:       사용할 split 이름 (e.g. 'pretrain')
        transform:   DataAugmentationPhasor 인스턴스
        stats:       정규화 통계 dict (mag_mean, mag_std, phase_mean, phase_std)
        cache:       True이면 블록을 메모리에 캐싱 (워커별 캐시)
        jitter:      temporal jitter 최대 범위 (프레임 수). 0이면 비활성화.
                     샘플마다 블록 경계를 고려한 유효 범위 내에서 균등 랜덤 δ 적용.
    """

    ROW_START, ROW_END = 7, 107
    COL_START, COL_END = 12, 120
    T_TOTAL = 800
    N_FRAMES = 20

    def __init__(self, csv_path, iq_dir, split='pretrain',
                 transform=None, stats=None, cache=True, jitter=9):
        self.iq_dir    = iq_dir
        self.transform = transform
        self.stats     = stats
        self.cache     = cache
        self.jitter    = jitter
        self._block_cache = {}

        self.samples = []
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['split'] != split:
                    continue
                self.samples.append({
                    'block_name':  row['block_name'],
                    'frame_start': int(row['frame_start']),  # 1-indexed
                })

        if not self.samples:
            raise ValueError(f"split='{split}' 샘플 없음: {csv_path}")
        print(f"[BolusMATDirectDataset] split={split}, {len(self.samples)}개 샘플, jitter={jitter}")

    def __len__(self):
        return len(self.samples)

    def _load_block(self, block_name):
        if self.cache and block_name in self._block_cache:
            return self._block_cache[block_name]

        mat_path = os.path.join(self.iq_dir, block_name + '.mat')
        d = sio.loadmat(mat_path)
        iq = d['IQ']  # (107, 128, 800) complex64

        iq_crop = iq[self.ROW_START:self.ROW_END,
                     self.COL_START:self.COL_END, :].astype(np.complex64)

        if self.cache:
            self._block_cache[block_name] = iq_crop
        return iq_crop

    def __getitem__(self, idx):
        s = self.samples[idx]
        iq_crop = self._load_block(s['block_name'])  # (100, 108, 800)

        fs = s['frame_start'] - 1  # 0-indexed

        # temporal jitter: 블록 경계를 고려한 유효 δ 구간에서 균등 샘플링
        if self.jitter > 0:
            delta_min = max(-self.jitter, -fs)
            delta_max = min(self.jitter, self.T_TOTAL - self.N_FRAMES - fs)
            delta = int(np.random.randint(delta_min, delta_max + 1))
            fs = fs + delta

        fe = fs + self.N_FRAMES
        frames = iq_crop[:, :, fs:fe]  # (100, 108, 20)

        n_frames = frames.shape[2]
        channels = []
        for fi in range(n_frames):
            f    = frames[:, :, fi]
            mag  = np.abs(f)
            ph   = np.angle(f)
            channels.append(mag)
            channels.append(ph)
            if fi == 0:
                channels.append(np.zeros_like(ph))   # 첫 frame: phase_diff = 0
            else:
                prev = frames[:, :, fi - 1]
                diff = ph - np.angle(prev)
                channels.append(np.arctan2(np.sin(diff), np.cos(diff)))

        # [60, 100, 108]
        tensor = torch.from_numpy(
            np.stack(channels, axis=0).astype(np.float32)
        )

        # mag_std: 20프레임 magnitude 픽셀별 표준편차 (1ch)
        mags    = np.abs(frames)                           # (100, 108, 20)
        mag_std = mags.std(axis=2).astype(np.float32)     # (100, 108)

        if self.stats is not None:
            tensor[0::3] = (tensor[0::3] - self.stats['mag_mean']) \
                           / (self.stats['mag_std']        + 1e-8)
            tensor[1::3] = (tensor[1::3] - self.stats['phase_mean']) \
                           / (self.stats['phase_std']      + 1e-8)
            tensor[2::3] = (tensor[2::3] - self.stats['phase_diff_mean']) \
                           / (self.stats['phase_diff_std'] + 1e-8)
            if 'mag_std_mean' in self.stats:
                mag_std = (mag_std - self.stats['mag_std_mean']) \
                          / (self.stats['mag_std_std'] + 1e-8)

        # mag_std를 61번째 채널로 추가 → [61, 100, 108]
        mag_std_t = torch.from_numpy(mag_std).unsqueeze(0)   # (1, 100, 108)
        tensor    = torch.cat([tensor, mag_std_t], dim=0)    # (61, 100, 108)

        if self.transform is not None:
            return self.transform(tensor), 0
        return tensor, 0


class BolusMATSegDataset(Dataset):
    """
    원본 IQ MAT + 샘플별 GT를 읽어 fine-tuning / val용 샘플 생성.

    GT: datasets_v5/gt/{sample_name}.mat  (scipy, key='gt', shape (107,128), 0/1/2/3)

    Args:
        csv_path:   sample_index_table.csv 경로
        iq_dir:     IQ MAT 파일 디렉토리
        gt_dir:     샘플별 GT MAT 디렉토리 (datasets_v5/gt/)
        split:      'ft_train' or 'val' or 'test'
        stats:      정규화 통계 dict
        cache:      IQ 블록 메모리 캐싱 여부
        jitter:     temporal jitter 최대 프레임 수
    """

    ROW_START, ROW_END = 7, 107
    COL_START, COL_END = 12, 120
    T_TOTAL  = 800
    N_FRAMES = 20

    def __init__(self, csv_path, iq_dir, gt_dir, split,
                 stats=None, cache=True, jitter=0, soft_ignore=False, four_class=False,
                 two_class=False):
        self.iq_dir      = iq_dir
        self.gt_dir      = gt_dir
        self.stats       = stats
        self.cache       = cache
        self.jitter      = jitter
        self.soft_ignore = soft_ignore  # True: gt==0 → bubble(2), ignore_mask 반환
        self.four_class  = four_class   # True: gt==0 → class 3 (no 255)
        self.two_class   = two_class    # True: bubble→255(ignore), noise/tissue만 학습
        self._iq_cache   = {}

        self.samples = []
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['split'] != split:
                    continue
                self.samples.append({
                    'block_name':  row['block_name'],
                    'sample_name': row['sample_name'],
                    'frame_start': int(row['frame_start']),
                })

        if not self.samples:
            raise ValueError(f"split='{split}' 샘플 없음: {csv_path}")
        print(f"[BolusMATSegDataset] split={split}, {len(self.samples)}개 샘플, "
              f"jitter={jitter}, soft_ignore={soft_ignore}, four_class={four_class}, "
              f"two_class={two_class}")

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

    def _load_gt(self, sample_name):
        d = sio.loadmat(os.path.join(self.gt_dir, sample_name + '.mat'))
        gt = d['gt'][self.ROW_START:self.ROW_END,
                     self.COL_START:self.COL_END]             # (100, 108), 0/1/2/3

        # per-sample GT 인코딩: 0=ignore, 1=tissue, 2=bubble, 3=noise
        if self.two_class:
            # bubble → 255 (ignore), noise=0, tissue=1
            ignore_mask = np.zeros_like(gt, dtype=bool)
            out = np.full_like(gt, 255, dtype=np.uint8)
            out[gt == 3] = 0  # noise
            out[gt == 1] = 1  # tissue
            # gt==2 (bubble) → 255, gt==0 (ignore) → 255
        elif self.soft_ignore:
            # ignore → bubble(2)로 변환, ignore_mask에 원본 위치 기록
            ignore_mask = (gt == 0)
            out = np.full_like(gt, 0, dtype=np.uint8)
            out[gt == 3] = 0  # noise
            out[gt == 1] = 1  # tissue
            out[gt == 2] = 2  # bubble
            out[gt == 0] = 2  # ignore → bubble (soft label, weight=soft_ignore_weight)
        elif self.four_class:
            # ignore → class 3 (4번째 클래스), 255 없음
            ignore_mask = np.zeros_like(gt, dtype=bool)
            out = np.full_like(gt, 0, dtype=np.uint8)
            out[gt == 3] = 0  # noise
            out[gt == 1] = 1  # tissue
            out[gt == 2] = 2  # bubble
            out[gt == 0] = 3  # ignore → class 3
        else:
            # 기본: ignore → 255 (학습/평가 모두 제외), ignore_mask는 빈 마스크
            ignore_mask = np.zeros_like(gt, dtype=bool)
            out = np.full_like(gt, 255, dtype=np.uint8)
            out[gt == 3] = 0  # noise
            out[gt == 1] = 1  # tissue
            out[gt == 2] = 2  # bubble
            # gt == 0 → 255 유지

        return out, ignore_mask

    def get_bubble_weights(self):
        """각 샘플의 bubble pixel 수에 비례한 샘플링 가중치 반환 (bubble class=2)."""
        weights = []
        for s in self.samples:
            d = sio.loadmat(os.path.join(self.gt_dir, s['sample_name'] + '.mat'))
            gt = d['gt'][self.ROW_START:self.ROW_END, self.COL_START:self.COL_END]
            bubble_count = int((gt == 2).sum())
            weights.append(float(bubble_count) + 1.0)  # +1: bubble 없는 샘플도 최소 1
        return weights

    def __getitem__(self, idx):
        s  = self.samples[idx]
        iq = self._load_iq(s['block_name'])
        fs = s['frame_start'] - 1

        if self.jitter > 0:
            delta_min = max(-self.jitter, -fs)
            delta_max = min(self.jitter, self.T_TOTAL - self.N_FRAMES - fs)
            fs = fs + int(np.random.randint(delta_min, delta_max + 1))

        frames = iq[:, :, fs:fs + self.N_FRAMES]

        channels = []
        for fi in range(self.N_FRAMES):
            fr  = frames[:, :, fi]
            mag = np.abs(fr)
            ph  = np.angle(fr)
            channels.append(mag)
            channels.append(ph)
            if fi == 0:
                channels.append(np.zeros_like(ph))   # 첫 frame: phase_diff = 0
            else:
                prev = frames[:, :, fi - 1]
                diff = ph - np.angle(prev)
                channels.append(np.arctan2(np.sin(diff), np.cos(diff)))

        tensor = torch.from_numpy(np.stack(channels, axis=0).astype(np.float32))

        # mag_std: 20프레임 magnitude의 픽셀별 표준편차 (1ch, 100×108)
        mags        = np.abs(frames)
        mag_std_map = mags.std(axis=2).astype(np.float32)   # (100, 108)

        if self.stats is not None:
            tensor[0::3] = (tensor[0::3] - self.stats['mag_mean']) \
                           / (self.stats['mag_std']        + 1e-8)
            tensor[1::3] = (tensor[1::3] - self.stats['phase_mean']) \
                           / (self.stats['phase_std']      + 1e-8)
            tensor[2::3] = (tensor[2::3] - self.stats['phase_diff_mean']) \
                           / (self.stats['phase_diff_std'] + 1e-8)
            if 'mag_std_mean' in self.stats:
                mag_std_map = (mag_std_map - self.stats['mag_std_mean']) \
                              / (self.stats['mag_std_std'] + 1e-8)

        mag_std_tensor = torch.from_numpy(mag_std_map).unsqueeze(0)  # (1, 100, 108)

        # mag_std를 61번째 채널로 추가 → [61, 100, 108]
        tensor = torch.cat([tensor, mag_std_tensor], dim=0)

        gt, ignore_mask = self._load_gt(s['sample_name'])
        # 4-tuple: (image, mag_std, gt, ignore_mask)
        # ignore_mask: bool tensor, soft_ignore=True 시 원본 gt==0 위치, 아니면 모두 False
        return (tensor,
                mag_std_tensor,
                torch.from_numpy(gt.astype(np.int64)),
                torch.from_numpy(ignore_mask))
