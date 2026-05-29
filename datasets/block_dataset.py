"""
block_dataset.py

preprocessed/ 폴더의 block_XXX.pt 파일을 읽는 Dataset 클래스.

블록 포맷: [T, 2, 80, 120] float32  (T=800 perfusion/bolus, T=790 kidney)
  dim0: 프레임 수
  dim1: 채널 [magnitude, phase]
  dim2-3: spatial (80×120)

샘플 생성 방식 (v9 방식):
  블록 하나 = 독립 시퀀스 (블록 경계를 절대 넘지 않음)
  → start_step=5 간격으로 시작 프레임 선택
  → 각 시작점에서 10프레임을 50프레임 간격으로 추출
  → [20, 80, 120] 텐서 반환

Classes:
  BlockPretrainDataset  -- SSL pre-training용 (라벨 없음)
  BlockSegDataset       -- fine-tuning / val / test용 (라벨 포함)
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


# ── 공통 상수 ──────────────────────────────────────────────────────────────────
N_FRAMES     = 10
FRAME_STRIDE = 50
START_STEP   = 5

# 레이블 원본 크기 (bolus IQ 기준)
LABEL_ORIG_H = 107
LABEL_ORIG_W = 128
TARGET_H     = 80
TARGET_W     = 120
LABEL_H0 = (LABEL_ORIG_H - TARGET_H) // 2   # 13
LABEL_W0 = (LABEL_ORIG_W - TARGET_W) // 2   # 4

IGNORE_INDEX = 255
FRAMES_PER_BLOCK = 800  # perfusion/bolus 기준 (BlockSegDataset용)


def load_block(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def load_label(path, remap_unknown=False):
    with h5py.File(path, 'r') as f:
        label = np.array(f['label']).T   # (107, 128)
    label = label[LABEL_H0:LABEL_H0 + TARGET_H,
                  LABEL_W0:LABEL_W0 + TARGET_W]   # (80, 120)
    if remap_unknown:
        label[label == 255] = 3   # unknown → class 3
    return label


def normalize(tensor, stats):
    if stats is None:
        return tensor
    tensor = tensor.clone()
    tensor[0::2] = (tensor[0::2] - stats['mag_mean'])   / (stats['mag_std']   + 1e-8)
    tensor[1::2] = (tensor[1::2] - stats['phase_mean']) / (stats['phase_std'] + 1e-8)
    return tensor


# ── BlockPretrainDataset ───────────────────────────────────────────────────────

class BlockPretrainDataset(Dataset):
    """
    SSL pre-training용 Dataset.
    블록 하나씩 독립 시퀀스로 샘플 생성 (v9 방식).
    블록 경계를 절대 넘지 않음.

    Args:
        block_dirs: 블록 폴더 경로(들). 문자열 하나 또는 리스트.
        transform:  DataAugmentationPhasor 인스턴스
        stats:      정규화 통계값 dict
        n_frames:   샘플당 프레임 수 (기본 10)
        frame_stride: 프레임 간격 (기본 50)
        start_step: 시작 프레임 후보 간격 (기본 5)
    """

    def __init__(self, block_dirs, transform=None, stats=None,
                 n_frames=N_FRAMES, frame_stride=FRAME_STRIDE, start_step=START_STEP):
        self.transform    = transform
        self.stats        = stats
        self.n_frames     = n_frames
        self.frame_stride = frame_stride

        if isinstance(block_dirs, str):
            block_dirs = [block_dirs]

        self.all_blocks = []   # list of [T, 2, 80, 120] 텐서
        self.samples    = []   # (block_id, start_frame)

        min_frames = (n_frames - 1) * frame_stride + 1

        for folder in block_dirs:
            block_paths = sorted([
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.startswith('block_') and f.endswith('.pt')
            ])
            if not block_paths:
                raise FileNotFoundError(f"block_*.pt 파일 없음: {folder}")

            print(f"[BlockPretrainDataset] {folder}: {len(block_paths)}개 블록 로딩 중...")
            for p in block_paths:
                block    = load_block(p)
                block_id = len(self.all_blocks)
                self.all_blocks.append(block)

                T          = block.shape[0]
                last_start = T - min_frames
                for s in range(0, last_start + 1, start_step):
                    self.samples.append((block_id, s))

        print(f"[BlockPretrainDataset] 총 {len(self.samples)}개 샘플")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        block_id, start_frame = self.samples[idx]
        block = self.all_blocks[block_id]

        channels = []
        for i in range(self.n_frames):
            fi = start_frame + i * self.frame_stride
            channels.append(block[fi, 0])  # magnitude
            channels.append(block[fi, 1])  # phase
        tensor = torch.stack(channels, dim=0)  # [20, 80, 120]

        tensor = normalize(tensor, self.stats)
        if self.transform is not None:
            return self.transform(tensor), 0
        return tensor, 0


# ── BlockSegDataset ────────────────────────────────────────────────────────────

class BlockSegDataset(Dataset):
    """
    Segmentation fine-tuning / val / test용 Dataset.
    bolus 블록 + 레이블(mat 파일)을 함께 로드.
    """

    def __init__(self, block_dir, label_dir, stats=None,
                 finetune_indices=None, n_perfusion_blocks=192,
                 n_frames=N_FRAMES, frame_stride=FRAME_STRIDE, start_step=START_STEP,
                 remap_unknown=False):
        self.stats          = stats
        self.n_frames       = n_frames
        self.frame_stride   = frame_stride
        self.remap_unknown  = remap_unknown

        block_paths = sorted([
            os.path.join(block_dir, f)
            for f in os.listdir(block_dir)
            if f.startswith('block_') and f.endswith('.pt')
        ])
        if not block_paths:
            raise FileNotFoundError(f"block_*.pt 파일 없음: {block_dir}")

        print(f"[BlockSegDataset] {block_dir}: {len(block_paths)}개 블록 로딩 중...")
        self.blocks   = [load_block(p) for p in block_paths]
        self.n_blocks = len(self.blocks)

        # 누적 오프셋 (bolus는 800프레임 고정이지만 일관성 유지)
        self.offsets = [0]
        for b in self.blocks:
            self.offsets.append(self.offsets[-1] + b.shape[0])

        total_frames = self.offsets[-1]
        min_frames   = (n_frames - 1) * frame_stride + 1
        last_start   = total_frames - min_frames

        self.label_paths = []
        for bp in block_paths:
            fname = os.path.basename(bp)
            num   = fname.replace('block_', '').replace('.pt', '')
            lpath = os.path.join(label_dir, f'label_{num}.mat')
            if not os.path.exists(lpath):
                raise FileNotFoundError(f"레이블 파일 없음: {lpath}")
            self.label_paths.append(lpath)

        if finetune_indices is not None:
            bolus_offset = n_perfusion_blocks * FRAMES_PER_BLOCK
            bolus_idx    = finetune_indices[finetune_indices >= bolus_offset] - bolus_offset
            bolus_idx    = bolus_idx[bolus_idx <= last_start]
            self.start_frames = bolus_idx.tolist()
            print(f"[BlockSegDataset] finetune 샘플: {len(self.start_frames)}개")
        else:
            self.start_frames = list(range(0, last_start + 1, start_step))
            print(f"[BlockSegDataset] 전체 샘플: {len(self.start_frames)}개")

    def __len__(self):
        return len(self.start_frames)

    def __getitem__(self, idx):
        global_start = self.start_frames[idx]

        block_idx = next(j for j in range(self.n_blocks)
                         if self.offsets[j] <= global_start < self.offsets[j + 1])
        local_start = global_start - self.offsets[block_idx]

        channels = []
        for i in range(self.n_frames):
            fi = local_start + i * self.frame_stride
            # 블록 경계 처리
            bi = block_idx
            while fi >= self.blocks[bi].shape[0]:
                fi -= self.blocks[bi].shape[0]
                bi += 1
            channels.append(self.blocks[bi][fi, 0])
            channels.append(self.blocks[bi][fi, 1])
        tensor = torch.stack(channels, dim=0)
        tensor = normalize(tensor, self.stats)

        label = load_label(self.label_paths[block_idx], remap_unknown=self.remap_unknown)
        label = torch.from_numpy(label.astype(np.int64))

        return tensor, label
